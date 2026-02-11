import os
import sys
import time
import json
import tqdm
import cv2
import torch
import argparse
import deepspeed
import numpy as np
from transformers import CLIPImageProcessor
import transformers
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

from model.GLaMM import GLaMMForCausalLM 
from model.llava import conversation as conversation_lib
from dataset.dataset import custom_collate_fn
from tools.utils import AverageMeter, ProgressMeter, dict_to_cuda, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.model.language_model.llava_llama import LlamaConfig
from peft.tuners.lora import Linear as LoraLinear
from peft.tuners.lora import LoraLayer

# =============================================================================
# 🕵️‍♂️ [디버깅 도우미] 모델/데이터 상태 출력
# =============================================================================
def print_model_status(model, stage_name):
    print(f"\n{'='*20} [DEBUG: {stage_name}] {'='*20}")
    
    # 1. Parameter Dtype 통계
    dtypes = {}
    fp32_params = []
    for name, p in model.named_parameters():
        dtype = str(p.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
        if p.dtype == torch.float32 and p.requires_grad:
            fp32_params.append(name)
            
    print(f"📊 Parameter Dtypes Stats: {dtypes}")
    
    # 2. SAM(Grounding Encoder) 상태 정밀 검사
    if hasattr(model, "base_model") and hasattr(model.base_model.model.model, "grounding_encoder"):
        sam = model.base_model.model.model.grounding_encoder
        print(f"🔎 Checking SAM (Grounding Encoder)...")
        
        # SAM 내부의 첫번째 Linear 레이어 확인
        for name, mod in sam.named_modules():
            if isinstance(mod, torch.nn.Linear):
                print(f"   - SAM Linear Layer '{name}': weight={mod.weight.dtype}, bias={mod.bias.dtype if mod.bias is not None else 'None'}")
                break
        
        # SAM 내부에 LoRA가 있는지 확인
        lora_found = []
        for name, mod in sam.named_modules():
            if "Lora" in mod.__class__.__name__:
                lora_found.append(name)
        
        if lora_found:
            print(f"   ⚠️ WARNING: LoRA found in SAM! Count: {len(lora_found)}")
            print(f"   -> Example: {lora_found[0]}")
        else:
            print(f"   ✅ SAM is Clean (No LoRA found)")

    # 3. 남아있는 FP32 학습 파라미터 경고
    if len(fp32_params) > 0:
        print(f"⚠️ WARNING: {len(fp32_params)} trainable parameters are still Float32!")
        print(f"   -> First 3 culprits: {fp32_params[:3]}")
    else:
        print("✅ No Float32 trainable parameters found.")
    
    print("="*60 + "\n")

# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Forest Finetuning")
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to train.json")
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--output_dir", default="./checkpoints", type=str)
    parser.add_argument("--lora_r", default=128, type=int)
    parser.add_argument("--lora_alpha", default=256, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--deepspeed_config", type=str)
    return parser.parse_args()

class ForestDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, image_processor, model_args):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_args = model_args
        self.sam_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.sam_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def preprocess_for_sam(self, image):
        img_res = image.resize((1024, 1024)) 
        img_np = np.array(img_res)
        if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[2] == 4: img_np = img_np[:, :, :3]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        img_tensor = (img_tensor - self.sam_mean) / self.sam_std
        return img_tensor
    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item['image']
        image_path = os.path.join(self.image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = image.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        if self.image_processor:
            clip_image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            clip_image = torch.zeros(3, 336, 336)
        sam_image = self.preprocess_for_sam(image)
        mask_path = item.get('mask_path', None)
        masks = torch.zeros((0, 1024, 1024)).float()
        if mask_path:
            if isinstance(mask_path, str): mask_paths = [mask_path]
            else: mask_paths = mask_path
            mask_list = []
            for mp in mask_paths:
                full_mp = os.path.join(self.image_folder, mp)
                try:
                    mask_np = cv2.imread(full_mp, 0)
                    if mask_np is None: continue
                    mask_resized = cv2.resize(mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    obj_ids = np.unique(mask_resized)
                    obj_ids = obj_ids[obj_ids > 0]
                    if len(obj_ids) > 0:
                        for obj_id in obj_ids:
                            binary_mask = (mask_resized == obj_id).astype(np.float32)
                            mask_tensor = torch.from_numpy(binary_mask)
                            mask_list.append(mask_tensor)
                except Exception as e:
                    print(f"Skipping mask: {e}")
            if len(mask_list) > 0:
                masks = torch.stack(mask_list)
        return {
            'image': clip_image, 'grounding_enc_images': sam_image,
            'conversations': [item['conversations']], 'image_path': image_path,
            'masks': masks, 'region': item.get('bboxes', None),
            'resize_list': [orig_w, orig_h]
        }

# SAM을 피해서 LoRA 타겟을 정하는 안전한 함수
def find_safe_target_modules(model):
    target_names = []
    keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    blacklist = ["grounding_encoder", "mask_decoder", "mm_projector", "text_hidden_fcs", "region_encoder"]
    
    for name, module in model.named_modules():
        if any(name.endswith(k) for k in keywords):
            if any(b in name for b in blacklist):
                continue
            if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
                target_names.append(name)
    return target_names

def main():
    args = parse_args()

    # 1. GPU 설정
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    # 2. 토크나이저 설정
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    temp_config = transformers.AutoConfig.from_pretrained(args.version)
    max_pos_len = getattr(temp_config, "max_position_embeddings", 4096)
    tokenizer.model_max_length = max_pos_len
    tokenizer.pad_token = tokenizer.unk_token
    
    # Special Tokens
    special_tokens = ['[SEG]', '<bbox>', '<point>', '<p>', '</p>']
    if args.use_mm_start_end:
        special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # 3. 모델 로드 (4-bit)
    skip_modules = ["vision_tower", "grounding_encoder", "mm_projector", 
                    "text_hidden_fcs", "region_encoder", "lm_head", "embed_tokens"]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules
    )

    print(f"Loading GLaMM from {args.version}...")
    
    model_kwargs = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "mm_vision_select_layer": -2,
        "with_region": True
    }
    
    model = GLaMMForCausalLM.from_pretrained(
        args.version,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map = {"": args.local_rank},
        **model_kwargs
    )

    # 4. 임베딩 리사이즈
    target_vocab_size = len(tokenizer)
    model.config.vocab_size = target_vocab_size
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.vocab_size = target_vocab_size
    model.resize_token_embeddings(len(tokenizer))
    
    # 5. 모델 전처리
    model = prepare_model_for_kbit_training(model)

    # 6. LoRA 적용 (Whitelist 방식 -> SAM 원천 차단)
    print("🔍 Generating safe LoRA target list (Avoiding SAM)...")
    target_modules = find_safe_target_modules(model)
    print(f"✅ Found {len(target_modules)} safe LoRA targets (LLM + CLIP only).")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    model = get_peft_model(model, lora_config)

    # ==============================================================================
    # 7. [Type Casting] SAM을 BFloat16으로 변환
    # ==============================================================================
    print("🚑 CASTING: Converting SAM & Projectors to BFloat16...")
    base_glamm = model.base_model.model.model

    # (A) SAM & Projector -> .to(BF16)
    if hasattr(base_glamm, "grounding_encoder"):
        base_glamm.grounding_encoder.to(device=device, dtype=torch.bfloat16)
    for mod_name in ["mm_projector", "text_hidden_fcs", "region_encoder"]:
        if hasattr(base_glamm, mod_name):
            getattr(base_glamm, mod_name).to(device=device, dtype=torch.bfloat16)

    # (B) LLM & CLIP의 LoRA -> param.data.to(BF16)
    count_casted = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            count_casted += 1
    print(f"✅ Casted {count_casted} remaining LoRA parameters to BFloat16.")

    # (C) Unfreeze (FFT 대상 학습 활성화)
    if hasattr(base_glamm, "grounding_encoder"):
        for param in base_glamm.grounding_encoder.parameters(): param.requires_grad = True
    for mod_name in ["mm_projector", "text_hidden_fcs", "region_encoder"]:
        if hasattr(base_glamm, mod_name):
            for param in getattr(base_glamm, mod_name).parameters(): param.requires_grad = True

    # ⚠️ [수정됨] Gaussian Matrix를 FP32로 되돌리는 코드 삭제됨!
    #    -> SAM 내부 연산을 BF16으로 유지하기 위함.
    
    # ==============================================================================

    # 🔥 [DEBUG] 학습 시작 전 모델 상태 최종 점검
    if args.local_rank == 0:
        print_model_status(model, "Before Training Loop")

    # [6] 데이터셋 로드
    print(f"Loading Dataset from {args.dataset_path}")
    train_dataset = ForestDataset(
        json_path=args.dataset_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336"),
        model_args=args
    )
    
    collate_fn = partial(
        custom_collate_fn, 
        tokenizer=tokenizer, 
        use_mm_start_end=args.use_mm_start_end, 
        local_rank=args.local_rank,
        inference=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    # [7] DeepSpeed Init
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": { "lr": args.lr, "weight_decay": 0.0, "betas": [0.9, 0.95] }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * len(train_loader),
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100
            }
        },
        "bf16": { "enabled": True },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8
        }
    }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
    
    # [8] 학습 루프
    print("Starting Training Loop")
    global_step = 0
    final_vocab_size = len(tokenizer) 

    if args.local_rank == 0:
        writer = SummaryWriter(args.output_dir)
    
    for epoch in range(args.epochs):
        model_engine.train()
        progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(args.local_rank != 0))
        
        for step, batch in enumerate(progress):
            batch = dict_to_cuda(batch)

            # --- [DEBUG] 배치 데이터 상태 확인 (첫 스텝만) ---
            if global_step == 0 and args.local_rank == 0:
                print(f"\n{'='*20} [DEBUG: First Batch] {'='*20}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f" - Input [{k}]: dtype={v.dtype}, shape={v.shape}")
                print("="*60 + "\n")
            # ------------------------------------------------

            # [Labels 정화]
            if 'labels' in batch:
                batch['labels'][batch['labels'] == -200] = -100
                batch['labels'][(batch['labels'] >= final_vocab_size) & (batch['labels'] != -100)] = -100

            # [Smart Clamping]
            if 'input_ids' in batch:
                bsz = batch['input_ids'].shape[0]
                batch['offset'] = torch.arange(bsz + 1, dtype=torch.long, device=device)
                
                # Seg token 처리
                if args.seg_token_idx is not None:
                    new_seg_mask = (batch['input_ids'] == args.seg_token_idx)
                    if new_seg_mask.any(): batch['seg_token_mask'] = new_seg_mask
                    else: 
                        if 'seg_token_mask' in batch: del batch['seg_token_mask']

                # Index Error 방지
                is_image_token = (batch['input_ids'] == -200)
                clamped_ids = batch['input_ids'].clamp(0, final_vocab_size - 1)
                batch['input_ids'] = torch.where(is_image_token, batch['input_ids'], clamped_ids)
            
            # ==================================================================
            # 🔥 [Input Data Casting] 모든 Float 입력을 BFloat16으로 변환
            #    (이미지뿐만 아니라 region, mask 등 Float32일 수 있는 모든 것을 변환)
            # ==================================================================
            for key, val in batch.items():
                if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
                    if val.dtype != torch.bfloat16:
                        batch[key] = val.to(torch.bfloat16)
            # ==================================================================
                
            outputs = model_engine(**batch)
            loss = outputs['loss']
            
            model_engine.backward(loss)
            model_engine.step()
            
            if args.local_rank == 0 and step % args.print_freq == 0:
                current_lr = model_engine.get_lr()[0]
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/LR", current_lr, global_step)
                if 'ce_loss' in outputs: writer.add_scalar("Train/CE_Loss", outputs['ce_loss'].item(), global_step)
                if 'mask_loss' in outputs: writer.add_scalar("Train/Mask_Loss", outputs['mask_loss'].item(), global_step)
                
            global_step += 1
            
        if args.local_rank == 0:
            save_checkpoint(model_engine, args, epoch)

def save_checkpoint(model_engine, args, epoch):
    save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
    os.makedirs(save_path, exist_ok=True)
    model_engine.module.save_pretrained(save_path)
    print(f"Saving non-LoRA weights to {save_path}...")
    non_lora_state = {}
    for name, param in model_engine.module.named_parameters():
        if param.requires_grad and "lora_" not in name:
            non_lora_state[name] = param.cpu()
    torch.save(non_lora_state, os.path.join(save_path, "non_lora_trainables.bin"))
    print("Save complete.")

if __name__ == "__main__":
    main()