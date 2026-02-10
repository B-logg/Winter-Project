# train_ft.py의 로직을 기반으로 함

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

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Forest Finetuning")
    
    # 기본 설정
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to train.json")
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    
    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--output_dir", default="./checkpoints", type=str)
    
    # LoRA 설정
    parser.add_argument("--lora_r", default=128, type=int)
    parser.add_argument("--lora_alpha", default=256, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    
    # Loss 가중치 (기본값 유지)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    
    # 기타 모델 설정 (GLaMM 필수 인자)
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)

    # DeepSpeed가 자동으로 넣어주는 인자 무시용
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--deepspeed_config", type=str)

    return parser.parse_args()

def force_cast_gaussian_matrix(model, device):
    """
    SAM 모델 깊숙이 숨어있는 positional_encoding_gaussian_matrix를 찾아내서
    강제로 BF16으로 변환하는 함수 (재귀 탐색)
    """
    print("🔍 Searching for 'positional_encoding_gaussian_matrix' to cast...")
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "positional_encoding_gaussian_matrix"):
            target = module.positional_encoding_gaussian_matrix
            if isinstance(target, torch.Tensor):
                # 강제 변환 및 재할당
                module.positional_encoding_gaussian_matrix = target.to(device=device, dtype=torch.bfloat16)
                print(f"   ✅ Casted: {name}.positional_encoding_gaussian_matrix -> {module.positional_encoding_gaussian_matrix.dtype}")
                count += 1
    
    if count == 0:
        print("⚠️ Warning: Gaussian Matrix를 찾지 못했습니다! 에러가 날 수 있습니다.")
    else:
        print(f"🎉 Total {count} matrices casted to BF16.")

# Custom Dataset Class
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
        if img_np.ndim == 2:
             img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[2] == 4:
             img_np = img_np[:, :, :3]
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

        # --- [핵심 수정] 마스크 인스턴스 분리 로직 ---
        mask_path = item.get('mask_path', None)
        masks = torch.zeros((0, 1024, 1024)).float()

        if mask_path:
            if isinstance(mask_path, str): mask_paths = [mask_path]
            else: mask_paths = mask_path
                
            mask_list = []
            for mp in mask_paths:
                full_mp = os.path.join(self.image_folder, mp)
                try:
                    # 1. 마스크 로드 (Grayscale)
                    # 파일에 1, 2, 3... 처럼 객체 ID가 들어있다고 가정
                    mask_np = cv2.imread(full_mp, 0)
                    if mask_np is None: continue
                    
                    # 2. 리사이즈 (Nearest Neighbor 필수! ID값 변형 방지)
                    mask_resized = cv2.resize(mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    
                    # 3. 고유한 객체 ID 추출 (0은 배경이므로 제외)
                    obj_ids = np.unique(mask_resized)
                    obj_ids = obj_ids[obj_ids > 0] # 0보다 큰 값만 추출
                    
                    # 4. ID 별로 마스크 쪼개기
                    if len(obj_ids) > 0:
                        for obj_id in obj_ids:
                            # 해당 ID만 1로 만들고 나머지는 0
                            binary_mask = (mask_resized == obj_id).astype(np.float32)
                            mask_tensor = torch.from_numpy(binary_mask)
                            mask_list.append(mask_tensor)
                    else:
                        # 만약 0밖에 없다면(빈 마스크) 건너뜀
                        pass

                except Exception as e:
                    print(f"Skipping mask: {e}")
            
            # 5. 스택 (이제 [11, 1024, 1024] 처럼 객체 수만큼 쌓임)
            if len(mask_list) > 0:
                masks = torch.stack(mask_list)

        return {
            'image': clip_image,
            'grounding_enc_images': sam_image,
            'conversations': [item['conversations']],
            'image_path': image_path,
            'masks': masks, 
            'region': item.get('bboxes', None),
            'resize_list': [orig_w, orig_h]
        }
# Main
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

    # 현재 프로세스의 GPU ID를 확실하게 설정
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    # 1. 토크나이저 설정
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )

    # [Fix] Force Context Length
    tokenizer.model_max_length = 8192
    print(f'Overriding tokenizer model_max_length to {tokenizer.model_max_length}')
    tokenizer.pad_token = tokenizer.unk_token
    
    # Special Tokens 추가
    special_tokens = ['[SEG]', '<bbox>', '<point>', '<p>', '</p>']
    if args.use_mm_start_end:
        special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # 2. 모델 로드 및 4-bit 양자화
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

    # =================================================================================
    # 🔥 [핵심 수정] 강제 리사이즈 (Force Resize) 적용 🔥
    # 함수 호출(model.resize_token_embeddings) 대신 직접 할당하여 씹힘 방지
    # =================================================================================
    print(f"🔄 [Force Resize] Target Tokenizer Len: {len(tokenizer)}")
    
    # 1. 모델 설정 강제 업데이트
    model.config.vocab_size = len(tokenizer)
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.vocab_size = len(tokenizer)

    # 2. 임베딩 레이어 직접 확장
    # GLaMM 구조상 model.model.embed_tokens에 위치함
    current_embed = model.model.embed_tokens
    if current_embed.weight.shape[0] != len(tokenizer):
        print(f"   ↳ Expanding embedding from {current_embed.weight.shape[0]} to {len(tokenizer)}...")
        
        # 새 레이어 생성 (BF16)
        new_embed = torch.nn.Embedding(len(tokenizer), current_embed.embedding_dim, padding_idx=current_embed.padding_idx)
        new_embed.to(device=device, dtype=torch.bfloat16)
        
        # 기존 가중치 복사 (매우 중요)
        with torch.no_grad():
            new_embed.weight[:current_embed.weight.shape[0]] = current_embed.weight
            
        # 모델에 갈아끼우기
        model.model.embed_tokens = new_embed
        
        # LM Head도 확장 필요 (출력층)
        current_head = model.lm_head
        if current_head.out_features != len(tokenizer):
            print(f"   ↳ Expanding LM Head from {current_head.out_features} to {len(tokenizer)}...")
            new_head = torch.nn.Linear(current_head.in_features, len(tokenizer), bias=False)
            new_head.to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                new_head.weight[:current_head.out_features, :] = current_head.weight
            model.lm_head = new_head

    print(f"✅ [Force Resize] Final Embed Size: {model.model.embed_tokens.weight.shape[0]}")
    # =================================================================================
    
    # 3. 모델 전처리 (Q-LoRA & Casting)
    model = prepare_model_for_kbit_training(model)
    
    # BF16 Casting for Full-Tuning Modules
    glamm_model = model.model
    modules_to_cast = ["vision_tower", "grounding_encoder", "mm_projector", "text_hidden_fcs", "region_encoder"]
    
    print("Casting modules (Params & Buffers) to BF16...")
    for mod_name in modules_to_cast:
        if hasattr(glamm_model, mod_name):
            module = getattr(glamm_model, mod_name)
            if isinstance(module, list): module = module[0]
            
            # 파라미터 변환
            for param in module.parameters():
                param.data = param.data.to(torch.bfloat16)
            
            # [중요] 버퍼 변환 (CUBLAS 에러 방지)
            for buffer in module.buffers():
                buffer.data = buffer.data.to(torch.bfloat16)
    
    force_cast_gaussian_matrix(model, device)
            
    if hasattr(glamm_model, "grounding_encoder"):
        prompt_encoder = glamm_model.grounding_encoder.prompt_encoder
        if hasattr(prompt_encoder, "positional_encoding_gaussian_matrix"):
            prompt_encoder.positional_encoding_gaussian_matrix = \
                prompt_encoder.positional_encoding_gaussian_matrix.to(device=device, dtype=torch.bfloat16)

    # 4. LoRA 설정
    exclude_keywords = ["grounding_encoder", "mm_projector", "text_hidden_fcs", "region_encoder"]
    target_modules = find_target_linear_modules(model, exclude_keywords)
    
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

    # 5. Full-Tuning 모듈 Unfreeze
    base_glamm = model.base_model.model.model
    
    if hasattr(base_glamm, "grounding_encoder"):
        mask_decoder = base_glamm.grounding_encoder.mask_decoder
        for param in mask_decoder.parameters(): param.requires_grad = True
        for param in base_glamm.grounding_encoder.image_encoder.parameters(): param.requires_grad = False
        
    for mod_name in ["mm_projector", "text_hidden_fcs", "region_encoder"]:
        if hasattr(base_glamm, mod_name):
            for param in getattr(base_glamm, mod_name).parameters():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params: {trainable_params:,}")

    # 6. 데이터셋 & 로더 준비
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
    
    # Training Loop
    print("Starting Training Loop")
    global_step = 0
    
    if args.local_rank == 0:
        writer = SummaryWriter(args.output_dir)
    
    for epoch in range(args.epochs):
        model_engine.train()
        progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(args.local_rank != 0))
        
        for step, batch in enumerate(progress):
            batch = dict_to_cuda(batch)
            
            if "global_enc_images" in batch and batch["global_enc_images"] is not None:
                batch["global_enc_images"] = batch["global_enc_images"].bfloat16()
            if "grounding_enc_images" in batch and batch["grounding_enc_images"] is not None:
                batch["grounding_enc_images"] = batch["grounding_enc_images"].bfloat16()
                
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
    
    # 1. LoRA Adapters 저장
    # model_engine.module 은 PEFT 모델
    model_engine.module.save_pretrained(save_path)
    
    # 2. Non-LoRA Trainable Weights 저장 (Mask Decoder 등)
    print(f"Saving non-LoRA weights to {save_path}...")
    non_lora_state = {}
    for name, param in model_engine.module.named_parameters():
        if param.requires_grad and "lora_" not in name:
            non_lora_state[name] = param.cpu()
            
    torch.save(non_lora_state, os.path.join(save_path, "non_lora_trainables.bin"))
    print("Save complete.")

if __name__ == "__main__":
    main()