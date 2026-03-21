import os
import json
import tqdm
import cv2
import torch
import random
import argparse
import deepspeed
import numpy as np
from PIL import Image
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import CLIPImageProcessor
from peft import LoraConfig, get_peft_model

from model.GLaMM import GLaMMForCausalLM 
from dataset.dataset import custom_collate_fn
from tools.utils import dict_to_cuda, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# 1. 인자 및 경로 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Optimal GLaMM Forest Finetuning (Pure LoRA)")
    parser.add_argument("--version", default=os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/checkpoints/GLaMM-GCG"))
    
    # 학습용 JSON
    parser.add_argument("--dataset_path", type=str, default=os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/dataset/glamm_train.json"))
    parser.add_argument("--image_folder", type=str, default=os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/dataset/glamm_images_train"))
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/checkpoints"))
    
    # 하이퍼파라미터
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accumulation_steps", default=2, type=int)
    parser.add_argument("--lr", default=2e-5, type=float) # (참고용) 스크립트에서 제어되므로 여기서 바꿔도 무방함
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--val_ratio", default=0.05, type=float, help="학습 데이터 내 검증(Val) 데이터 비율 (기본 5%)")
    
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=8, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)

    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    
    parser.add_argument("--vision_pretrained", default=os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/checkpoints"), type=str)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    return parser.parse_args()

# 2. 유연한 데이터셋 클래스 (List 기반)
class ForestDataset(Dataset):
    def __init__(self, data_list, image_folder, tokenizer, image_processor):
        self.data = data_list
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sam_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.sam_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    def __len__(self):
        return len(self.data)

    def preprocess_for_sam(self, image):
        img_res = image.resize((1024, 1024)) 
        img_np = np.array(img_res)
        if img_np.ndim == 2: img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[2] == 4: img_np = img_np[:, :, :3]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return (img_tensor - self.sam_mean) / self.sam_std

    def __getitem__(self, idx):
        item = self.data[idx]
        
        raw_image_path = item['image'].strip()
        image_path = raw_image_path.replace('~', '/shared/home/naislab')

        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        
        clip_image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        sam_image = self.preprocess_for_sam(image)
        
        mask_paths = item.get('mask_path', [])
        mask_list = []
        for mp in mask_paths:
            raw_mp = mp.strip()
            mp_expanded = raw_mp.replace('~', '/shared/home/naislab')

            mask_np = cv2.imread(mp_expanded, cv2.IMREAD_GRAYSCALE)
            if mask_np is not None:
                mask_resized = cv2.resize(mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0).astype(np.float32)
                mask_list.append(torch.from_numpy(binary_mask))
                
        masks = torch.stack(mask_list) if mask_list else torch.zeros((0, 1024, 1024)).float()

        return {
            'image': clip_image, 
            'grounding_enc_images': sam_image,
            'conversations': [item['conversations']], 
            'image_path': image_path,
            'masks': masks, 
            'resize_list': [orig_w, orig_h]
        }

def find_all_linear_names(model):
    target_names = []
    # SAM 등 시각 모듈에는 절대 LoRA가 붙지 못하게 차단
    blacklist = ["grounding_encoder", "mask_decoder", "mm_projector", "text_hidden_fcs", "region_encoder", "vision_tower", "lm_head", "embed_tokens"]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not any(b in name for b in blacklist):
                target_names.append(name) 
    return target_names

# 3. 메인 학습 및 검증 루프
def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    # [A] 데이터 준비 및 Train/Val 동적 분할
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        
    random.seed(42)
    random.shuffle(full_data)
    
    val_size = int(len(full_data) * args.val_ratio)
    val_data = full_data[:val_size]
    train_data = full_data[val_size:]
    
    if args.local_rank == 0:
        print(f"Dataset Split: Train {len(train_data)} / Val {len(val_data)} (from {args.dataset_path})")

    # Tokenizer 및 모델 세팅
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

    
    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": args.local_rank},
        train_mask_decoder=True, out_dim=256,
        ce_loss_weight=args.ce_loss_weight, dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight, seg_token_idx=args.seg_token_idx,
        vision_pretrained=args.vision_pretrained, vision_tower=args.vision_tower,
        use_mm_start_end=True, mm_vision_select_layer=-2
    )
    

    # [C] LoRA 설정 (순수 LoRA만 학습하도록 수정됨!)
    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    base_model = model.base_model.model.model
    
    # 시각-언어 다리와 픽셀 디코더 전면 학습
    modules_to_train = ["mm_projector", "text_hidden_fcs"]
    
    if hasattr(base_model, "grounding_encoder"):
        if hasattr(base_model.grounding_encoder, "mask_decoder"):
            modules_to_train.append("grounding_encoder.mask_decoder")

    # 선택된 모듈들의 가중치 업데이트(학습)를 활성화
    for name, param in base_model.named_parameters():
        if any(mod in name for mod in modules_to_train):
            param.requires_grad = True
            
    # 타입 캐스팅 안정화
    for mod_name in ["mm_projector", "text_hidden_fcs"]:
        if hasattr(base_model, mod_name):
            module = getattr(base_model, mod_name)
            for name, buf in module.named_buffers():
                if buf.dtype != torch.bfloat16 and torch.is_floating_point(buf):
                    buf.data = buf.data.to(torch.bfloat16)
    
    # 몽키 패치: SAM Mask decoder 입구에서 무조건 BF16으로 변환
    if hasattr(base_model, "grounding_encoder"):
        mask_decoder = base_model.grounding_encoder.mask_decoder
        original_forward = mask_decoder.forward

        def mask_decoder_forward_wrapper(*args, **kwargs):
            new_args = [a.to(torch.bfloat16) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args]
            new_kwargs = {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v for k, v in kwargs.items()}
            return original_forward(*new_args, **new_kwargs)

        mask_decoder.forward = mask_decoder_forward_wrapper

    # [D] DataLoader 설정
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    
    train_dataset = ForestDataset(train_data, args.image_folder, tokenizer, image_processor)
    val_dataset = ForestDataset(val_data, args.image_folder, tokenizer, image_processor)
    
    collate_fn = partial(custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=True, local_rank=args.local_rank, inference=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # [E] DeepSpeed 초기화
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": { "type": "AdamW", "params": { "lr": args.lr, "weight_decay": 0.05, "betas": [0.9, 0.95] } },
        "scheduler": { "type": "WarmupDecayLR", "params": { "total_num_steps": args.epochs * len(train_loader), "warmup_num_steps": 100, "warmup_max_lr": args.lr, "warmup_min_lr": 0.0} },
        "bf16": { "enabled": True },
        "zero_optimization": { "stage": 2, "overlap_comm": True, "contiguous_gradients": True }
    }
    model_engine, optimizer, _, scheduler = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    
    if args.local_rank == 0: writer = SummaryWriter(args.output_dir)
    global_step = 0

    # [F] 훈련 및 에포크 검증 루프
    for epoch in range(args.epochs):
        # 1. Training Loop
        model_engine.train()
        progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]", disable=(args.local_rank != 0))
        
        train_loss_sum = 0
        for step, batch in enumerate(progress):
            batch = dict_to_cuda(batch)

            if 'input_ids' in batch:
                bsz = batch['input_ids'].shape[0]
                batch['offset'] = torch.arange(bsz + 1, dtype=torch.long, device=device)

                if args.seg_token_idx is not None:
                    new_seg_mask = (batch['input_ids'] == args.seg_token_idx)
                    if new_seg_mask.any():
                        batch['seg_token_mask'] = new_seg_mask

            if 'labels' in batch:
                batch['labels'][batch['labels'] == -200] = -100
                batch['labels'][(batch['labels'] >= len(tokenizer)) &(batch['labels'] != -100)] = -100
                
            if 'input_ids' in batch:
                is_image_token = (batch['input_ids'] == -200)
                clamped_ids = batch['input_ids'].clamp(0, len(tokenizer) - 1)
                batch['input_ids'] = torch.where(is_image_token, batch['input_ids'], clamped_ids)
            
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    batch[k] = v.to(torch.bfloat16)

            outputs = model_engine(**batch)
            loss = outputs['loss']
            model_engine.backward(loss)
            model_engine.step()
            
            train_loss_sum += loss.item()
            
            if args.local_rank == 0 and step % 10 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/LR", model_engine.get_lr()[0], global_step)

                progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{model_engine.get_lr()[0]:.6f}"})

            global_step += 1
            
        # 3. 로깅 및 체크포인트 저장
        if args.local_rank == 0:
            avg_train_loss = train_loss_sum / len(train_loader)
            
            print(f"\nEpoch {epoch+1} Results: Train Loss = {avg_train_loss:.4f}")
            
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            
            # 1. LoRA 가중치 저장
            model_engine.module.save_pretrained(save_path)
            
            # SAM 마스크 디코더 등 LoRA가 안 붙은 '진짜 학습된 부품'들 저장
            non_lora_state = {n: p.cpu() for n, p in model_engine.module.named_parameters() if p.requires_grad and "lora_" not in n}
            torch.save(non_lora_state, os.path.join(save_path, "non_lora_trainables.bin"))
            
            print(f"Checkpoint saved at {save_path}\n")

if __name__ == "__main__":
    main()