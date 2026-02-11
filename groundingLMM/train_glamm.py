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

from model.GLaMM import GLaMMForCausalLM 
from model.llava import conversation as conversation_lib
from dataset.dataset import custom_collate_fn
from tools.utils import AverageMeter, ProgressMeter, dict_to_cuda, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.model.language_model.llava_llama import LlamaConfig
from peft.tuners.lora import Linear as LoraLinear
from peft.tuners.lora import LoraLayer

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

def find_target_linear_modules(model, exclude_keywords=[]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(keyword in name for keyword in exclude_keywords): continue
        if isinstance(module, cls):
            lora_module_names.add(name.split('.')[-1])
    if 'lm_head' in lora_module_names: lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = ['[SEG]', '<bbox>', '<point>', '<p>', '</p>']
    if args.use_mm_start_end:
        special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # [1] 모델 로드
    skip_modules = ["vision_tower", "grounding_encoder", "mm_projector", 
                    "text_hidden_fcs", "region_encoder", "lm_head", "embed_tokens"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules
    )
    print(f"Loading GLaMM from {args.version}...")
    model = GLaMMForCausalLM.from_pretrained(
        args.version, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map = {"": args.local_rank},
        train_mask_decoder=args.train_mask_decoder, out_dim=args.out_dim,
        ce_loss_weight=args.ce_loss_weight, dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight, seg_token_idx=args.seg_token_idx,
        vision_pretrained=args.vision_pretrained, vision_tower=args.vision_tower,
        use_mm_start_end=args.use_mm_start_end, mm_vision_select_layer=-2, with_region=True
    )

    target_vocab_size = len(tokenizer)
    model.config.vocab_size = target_vocab_size
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.vocab_size = target_vocab_size
    model.resize_token_embeddings(len(tokenizer))
    
    # [2] QLoRA 준비
    model = prepare_model_for_kbit_training(model)

    # [3] LoRA 적용 (이때 SAM에도 LoRA가 붙어버림)
    # exclude_keywords가 잘 안 먹히는 경우가 많아서, 일단 다 붙이고 나중에 뗄겁니다.
    target_modules = find_target_linear_modules(model, exclude_keywords=[]) 
    
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]
    )
    model = get_peft_model(model, lora_config)

    # ==============================================================================
    # [4] 🔥 [LoRA 광역 박리 (Global Exorcism)] 키워드 기반으로 확실하게 제거
    #     - 모델 구조를 몰라도 'grounding'이나 'mask_decoder'가 들어간 곳은 무조건 복구
    # ==============================================================================
    print("🚑 GLOBAL EXORCISM: Searching and Destroying LoRA in SAM/Projectors...")
    
    # LoRA 제거 함수 (재귀)
    def recursive_strip_lora(module, module_name=""):
        for name, child in module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            
            # (1) SAM이나 Projector 관련 모듈인지 확인 (키워드 매칭)
            is_fft_target = any(k in full_name for k in ["grounding_encoder", "mask_decoder", "mm_projector", "text_hidden_fcs", "region_encoder"])
            
            # (2) LoRA 레이어인지 확인
            is_lora = isinstance(child, (LoraLinear, LoraLayer)) or "Lora" in child.__class__.__name__
            
            if is_fft_target and is_lora:
                if hasattr(child, "base_layer"):
                    print(f"   -> ✂️ Stripping LoRA from: {full_name}")
                    setattr(module, name, child.base_layer) # 원본 Linear로 교체
                elif hasattr(child, "linear"): # 일부 버전에선 linear 속성
                    print(f"   -> ✂️ Stripping LoRA from: {full_name}")
                    setattr(module, name, child.linear)
            else:
                recursive_strip_lora(child, full_name)

    # 모델 전체를 돌면서 키워드가 포함된 곳의 LoRA를 제거
    recursive_strip_lora(model)
    print("✅ LoRA stripping complete.")

    # ==============================================================================
    # [5] 🔥 [Type Casting] SAM은 이제 순수 Linear이므로 .to(BF16) 가능
    # ==============================================================================
    print("🚑 HYBRID CASTING: Converting modules to BFloat16...")

    # (A) 키워드 기반으로 SAM & Projector 모듈 찾아서 .to(BF16)
    # 모델 구조가 복잡해도 키워드로 찾아서 바꿈
    for name, module in model.named_modules():
        if any(k in name for k in ["grounding_encoder", "mask_decoder", "mm_projector", "text_hidden_fcs", "region_encoder"]):
            # 이미 변환된 상위 모듈의 하위일 수 있으므로 try-except
            try:
                module.to(device=device, dtype=torch.bfloat16)
            except:
                pass

    # (B) LLM & CLIP에 남은 LoRA -> param.data.to(BF16)
    count_casted = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            count_casted += 1
    print(f"✅ Casted {count_casted} remaining LoRA parameters to BFloat16.")

    # (C) Unfreeze (FFT 대상 학습 활성화)
    print("🔓 Unfreezing SAM and Projectors...")
    for name, param in model.named_parameters():
        if any(k in name for k in ["grounding_encoder", "mm_projector", "text_hidden_fcs", "region_encoder"]):
            param.requires_grad = True

    # (D) SAM Gaussian Matrix 복구
    count_reset = 0
    for name, module in model.named_modules():
        if hasattr(module, "positional_encoding_gaussian_matrix"):
            module.positional_encoding_gaussian_matrix = module.positional_encoding_gaussian_matrix.to(torch.float32)
            count_reset += 1
    print(f"✅ Reset {count_reset} Gaussian matrices to FP32.")
    # ==============================================================================

    # [Debug] 최종 확인 (로그에 이거 뜨는지 꼭 확인하세요!)
    if args.local_rank == 0:
        print("\n" + "="*50)
        print("🔍 FINAL CHECK")
        found_lora_in_sam = False
        for name, mod in model.named_modules():
            if "mask_decoder" in name and isinstance(mod, (LoraLinear, LoraLayer)):
                print(f"⚠️ [CRITICAL] STILL FOUND LORA IN SAM: {name}")
                found_lora_in_sam = True
        
        if not found_lora_in_sam:
            print("✅ SAM is Clean (No LoRA found in mask_decoder)")
        else:
            print("❌ SAM still has LoRA. Exorcism Failed.")
        print("="*50 + "\n")

    # [6] 데이터셋 로드
    print(f"Loading Dataset from {args.dataset_path}")
    train_dataset = ForestDataset(
        json_path=args.dataset_path, image_folder=args.image_folder,
        tokenizer=tokenizer, image_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336"),
        model_args=args
    )
    collate_fn = partial(custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank, inference=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # [7] DeepSpeed Init
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": { "type": "AdamW", "params": { "lr": args.lr, "weight_decay": 0.0, "betas": [0.9, 0.95] } },
        "scheduler": { "type": "WarmupDecayLR", "params": { "total_num_steps": args.epochs * len(train_loader), "warmup_min_lr": 0, "warmup_max_lr": args.lr, "warmup_num_steps": 100 } },
        "bf16": { "enabled": True },
        "zero_optimization": { "stage": 2, "contiguous_gradients": True, "overlap_comm": True, "reduce_scatter": True, "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8 }
    }
    model_engine, optimizer, _, scheduler = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    
    # [8] 학습 루프
    print("Starting Training Loop")
    global_step = 0
    final_vocab_size = len(tokenizer) 
    if args.local_rank == 0: writer = SummaryWriter(args.output_dir)
    
    for epoch in range(args.epochs):
        model_engine.train()
        progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(args.local_rank != 0))
        for step, batch in enumerate(progress):
            batch = dict_to_cuda(batch)

            if 'labels' in batch:
                batch['labels'][batch['labels'] == -200] = -100
                batch['labels'][(batch['labels'] >= final_vocab_size) & (batch['labels'] != -100)] = -100
            
            if 'input_ids' in batch:
                bsz = batch['input_ids'].shape[0]
                batch['offset'] = torch.arange(bsz + 1, dtype=torch.long, device=device)

            if 'input_ids' in batch and args.seg_token_idx is not None:
                new_seg_mask = (batch['input_ids'] == args.seg_token_idx)
                if new_seg_mask.any(): batch['seg_token_mask'] = new_seg_mask
                else: 
                    if 'seg_token_mask' in batch: del batch['seg_token_mask']

            if 'input_ids' in batch:
                is_image_token = (batch['input_ids'] == -200)
                clamped_ids = batch['input_ids'].clamp(0, final_vocab_size - 1)
                batch['input_ids'] = torch.where(is_image_token, batch['input_ids'], clamped_ids)
            
            if "global_enc_images" in batch: batch["global_enc_images"] = batch["global_enc_images"].bfloat16()
            if "grounding_enc_images" in batch: batch["grounding_enc_images"] = batch["grounding_enc_images"].bfloat16()
                
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
            
        if args.local_rank == 0: save_checkpoint(model_engine, args, epoch)

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