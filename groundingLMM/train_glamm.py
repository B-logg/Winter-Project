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
from transformers import CLIPImageProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

from model.GLaMM import GLaMMForCausalLM 
from dataset.dataset import custom_collate_fn
from tools.utils import dict_to_cuda, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# 1. 인자 및 경로 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Optimal GLaMM Forest Finetuning")
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    
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
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--val_ratio", default=0.05, type=float, help="학습 데이터 내 검증(Val) 데이터 비율 (기본 5%)")
    
    parser.add_argument("--lora_r", default=128, type=int)
    parser.add_argument("--lora_alpha", default=256, type=int)
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
    target_names = set()
    blacklist = ["grounding_encoder", "mask_decoder", "mm_projector", "text_hidden_fcs", "region_encoder", "vision_tower"]
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
            if not any(b in name for b in blacklist):
                target_names.add(name.split('.')[-1])
    return list(target_names)

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
    special_tokens = ['[SEG]', '<p>', '</p>', DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision_tower", "grounding_encoder", "mm_projector", "text_hidden_fcs", "lm_head"]
    )
    
    model = GLaMMForCausalLM.from_pretrained(
        args.version, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": args.local_rank},
        train_mask_decoder=True, out_dim=256,
        ce_loss_weight=args.ce_loss_weight, dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight, seg_token_idx=args.seg_token_idx,
        vision_pretrained=args.vision_pretrained, vision_tower=args.vision_tower,
        use_mm_start_end=True, mm_vision_select_layer=-2
    )
    
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    # [C] LoRA 및 Unfreeze 설정
    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"] 
    )
    model = get_peft_model(model, lora_config)

    base_model = model.base_model.model.model
    for mod_name in ["grounding_encoder", "mm_projector", "text_hidden_fcs"]:
        if hasattr(base_model, mod_name):
            module = getattr(base_model, mod_name)
            module.to(device=device, dtype=torch.bfloat16) 
            for param in module.parameters(): param.requires_grad = True

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
        "optimizer": { "type": "AdamW", "params": { "lr": args.lr, "weight_decay": 0.0, "betas": [0.9, 0.95] } },
        "scheduler": { "type": "WarmupDecayLR", "params": { "total_num_steps": args.epochs * len(train_loader), "warmup_num_steps": 100 } },
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
                bsz = batch['input_ids'].shape(0)
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
            
            # 실수형(float 32) 텐서인 이미지, 마스크를 모두 BF16으로 변환 -> DataLoader에서 전처리하여 나온 이미지와 마스크는 기본적으로 float32
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
            global_step += 1
            
        # 2. Validation Loop (에포크 종료 시점)
        model_engine.eval() # 모델을 평가 모드로 전환 (Dropout 등 비활성화)
        val_loss_sum = 0
        val_progress = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]", disable=(args.local_rank != 0))
        
        with torch.no_grad(): # 역전파(기울기 계산) 비활성화하여 메모리 절약 및 속도 향상
            for batch in val_progress:
                batch = dict_to_cuda(batch)

                if 'input_ids' in batch:
                    bsz = batch['input_ids'].shape(0)
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
                val_loss_sum += outputs['loss'].item()
        
        # 3. 로깅 및 체크포인트 저장
        if args.local_rank == 0:
            avg_train_loss = train_loss_sum / len(train_loader)
            avg_val_loss = val_loss_sum / len(val_loader)
            
            print(f"\nEpoch {epoch+1} Results: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
            writer.add_scalar("Val/Loss_Epoch", avg_val_loss, epoch)
            
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model_engine.module.save_pretrained(save_path)
            
            non_lora_state = {n: p.cpu() for n, p in model_engine.module.named_parameters() if p.requires_grad and "lora_" not in n}
            torch.save(non_lora_state, os.path.join(save_path, "non_lora_trainables.bin"))
            print(f"Checkpoint saved at {save_path}\n")

if __name__ == "__main__":
    main()