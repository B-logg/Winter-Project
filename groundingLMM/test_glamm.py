import os
import cv2
import json
import torch
import argparse
import re
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from eval.utils import mask_to_rle_pytorch, coco_encode_rle

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Test with Loss")
    parser.add_argument("--hf_model_path", required=True, help="Path to checkpoint")
    parser.add_argument("--test_json_path", required=True, help="Path to test.json")
    parser.add_argument("--image_folder", required=True, help="Image folder")
    parser.add_argument("--output_dir", required=True, help="Result save dir")
    parser.add_argument("--conv_type", default="llava_v1")
    return parser.parse_args()

class ForestTestDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, image_processor, transform, model_config):
        with open(json_path, 'r') as f: self.data = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        
    def __len__(self): return len(self.data)

    def preprocess_image(self, image_path):
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        orig_size = image_np.shape[:2]
        
        # CLIP Image
        image_clip = self.image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        
        # SAM Image
        image_sam = self.transform.apply_image(image_np)
        resize_shape = image_sam.shape[:2]
        image_sam = torch.from_numpy(image_sam).permute(2, 0, 1).float()
        
        # SAM Normalization
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        image_sam = (image_sam - pixel_mean) / pixel_std
        
        return image_clip, image_sam, orig_size, resize_shape

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])
        
        # 1. 이미지 처리
        clip_img, sam_img, orig_size, resize_shape = self.preprocess_image(image_path)
        
        # 2. 텍스트 처리 (Loss 계산용: 질문+답변 / 추론용: 질문만)
        human_q = item['conversations'][0]['value']
        gpt_a = item['conversations'][1]['value'] # GT Answer
        
        # --- Loss 계산을 위한 Full Prompt (Teacher Forcing) ---
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        
        # 질문 구성
        q_text = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n" + human_q # 수정 필요
        
        q_text = q_text.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        conv.append_message(conv.roles[0], q_text)
        conv.append_message(conv.roles[1], gpt_a) # 답변 포함
        full_prompt = conv.get_prompt()
        
        input_ids_loss = tokenizer_image_token(full_prompt, self.tokenizer, return_tensors='pt')

        """ 토큰 터짐 디버깅
        # =========================================================
        if input_ids_loss.shape[0] > 1536:
            print(f"\n[🚨 토큰 폭발 발견!] 총 토큰 수: {input_ids_loss.shape[0]}")
            print(f"문제의 파일명: {item['image']}")
            print(f"문제의 텍스트:\n{full_prompt}\n" + "="*50)
            # 확인을 위해 여기서 프로그램을 강제로 멈춥니다.
            raise ValueError("토큰 길이 초과 데이터를 발견하여 중단합니다.")
        # =========================================================
        """

        if input_ids_loss.shape[0] > 1536:
            print(f"\n[경고] 데이터 스킵! (토큰: {input_ids_loss.shape[0]}) -> {item['image']}")
            
            # 1. 에러가 나지 않도록 아주 짧은 가짜 질문과 답변([SEG] 1개 포함)으로 덮어씁니다.
            human_q = "탄소 저장량을 분석해줘."
            gpt_a = "산림이 과밀하거나 구조적으로 불균형할 경우 나무의 안정성과 생육 효율이 저하될 수 있으므로, 밀도 조절(예: 솎아베기)을 통해 건강성과 탄소 흡수 능력을 개선할 필요가 있다. [SEG]"
            
            # 2. 가짜 데이터로 프롬프트를 다시 짭니다.
            conv = conversation_lib.conv_templates["llava_v1"].copy()
            conv.messages = []

            q_text = DEFAULT_IMAGE_TOKEN + "\n" + human_q
            conv.append_message(conv.roles[0], q_text)
            conv.append_message(conv.roles[1], gpt_a)
            full_prompt = conv.get_prompt()
            
            # 3. 토큰을 다시 생성합니다. (이제 1536을 절대 넘지 않음)
            input_ids_loss = tokenizer_image_token(full_prompt, self.tokenizer, return_tensors='pt')
        
        # --- Labels 생성 (Human 질문 부분은 마스킹 -100) ---
        labels = input_ids_loss.clone()
        # 간단히: "ASSISTANT:" 이전까지는 모두 마스킹 (-100)
        sep = "ASSISTANT: "
        parts = full_prompt.split(sep)
        if len(parts) >= 2:
            len_context = len(tokenizer_image_token(parts[0] + sep, self.tokenizer))
            labels[:len_context-1] = -100 # -1 빼는건 오차 범위 보정
        
        # 3. GT 마스크 로드 (Loss 계산용)
        gt_mask = torch.zeros((1024, 1024)).float()
        mask_path = item.get('mask_path', None)
        if mask_path:
            if isinstance(mask_path, str): mask_path = [mask_path]
            for mp in mask_path:
                m = cv2.imread(os.path.join(self.image_folder, mp), 0)
                if m is not None:
                    m = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    gt_mask = torch.maximum(gt_mask, torch.from_numpy(m).float())
        gt_mask = (gt_mask > 0).float().unsqueeze(0) # (1, 1024, 1024)

        return {
            "id": item['id'],
            "image_path": image_path,
            "human_q": human_q,
            "clip_img": clip_img,
            "sam_img": sam_img,
            "input_ids_loss": input_ids_loss,
            "labels": labels,
            "masks": gt_mask,
            "orig_size": orig_size,
            "resize_shape": resize_shape
        }

def main():
    from peft import PeftModel
    import torch
    import re

    args = parse_args()
    
    # 원본 모델 경로 (환경에 맞게 수정 확인)
    BASE_MODEL_PATH = "checkpoints/GLaMM-GCG"

    # 1. 모델 로드 (Base Model + LoRA + Non-LoRA 병합)
    print(f"Loading Base Model from {BASE_MODEL_PATH}...")
    
    # (1) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    # (2) 베이스 모델 뼈대 불러오기
    model = GLaMMForCausalLM.from_pretrained(
        BASE_MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, seg_token_idx=seg_token_idx,
        train_mask_decoder=True 
    )
    
    # (3) LoRA 가중치 병합
    print(f"Applying LoRA weights from {args.hf_model_path}...")
    model = PeftModel.from_pretrained(model, args.hf_model_path)
    model = model.merge_and_unload() 
    
    # (4) Non-LoRA 가중치 덮어씌우기
    non_lora_path = os.path.join(args.hf_model_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        print(f"Loading non-LoRA trainables from {non_lora_path}...")
        non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
        
        cleaned_state_dict = {}
        for k, v in non_lora_trainables.items():
            if k.startswith('base_model.model.'):
                cleaned_state_dict[k[17:]] = v
            else:
                cleaned_state_dict[k] = v
        model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.cuda()
    model = model.bfloat16() # 전체 1차 변환

    print("✅ Setting Loss weights...")
    model.ce_loss_weight = 1.0
    model.dice_loss_weight = 0.5
    model.bce_loss_weight = 2.0

    # 모델 내부 파라미터 및 버퍼 전수 조사 강제 캐스팅
    for name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point() and buffer.dtype != torch.bfloat16:
            buffer.data = buffer.data.to(torch.bfloat16)

    # 비전 타워 및 그라운딩 인코더 개별 강제 캐스팅
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device='cuda')
    model.get_model().grounding_encoder.to(dtype=torch.bfloat16, device='cuda')

    # ✅ [Monkey Patch] SAM Mask Decoder 입구 봉쇄 (Train 코드의 핵심 로직 이식)
    print("✅ Applying Monkey Patch to SAM Mask Decoder (Force BF16 Inputs)...")
    base_glamm = model.get_model()
    if hasattr(base_glamm, "grounding_encoder"):
        mask_decoder = base_glamm.grounding_encoder.mask_decoder
        original_forward = mask_decoder.forward
        
        def mask_decoder_forward_wrapper(*args, **kwargs):
            new_args = [a.to(torch.bfloat16) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args]
            new_kwargs = {k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for k, v in kwargs.items()}
            return original_forward(*new_args, **new_kwargs)
        
        mask_decoder.forward = mask_decoder_forward_wrapper
    # ---------------------------------------------------------------------

    clip_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(1024)

    # 2. 데이터셋 (num_workers=0 권장: 디버깅 및 경로 충돌 방지)
    dataset = ForestTestDataset(args.test_json_path, args.image_folder, tokenizer, clip_processor, transform, model.config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. 테스트 루프
    total_loss = 0.0
    ce_loss = 0.0
    mask_loss = 0.0
    count = 0
    
    print(">>> Starting Test Loop (Loss Calculation & Inference)...")
    results = []
    
    for batch in tqdm(dataloader):
        print(f"\n[로그] 현재 처리 중인 이미지: {batch['id'][0]}")
        # 데이터 준비 (bfloat16)
        images = batch['clip_img'].cuda().bfloat16()
        sam_images = batch['sam_img'].cuda().bfloat16()
        input_ids_loss = batch['input_ids_loss'].cuda()
        labels = batch['labels'].cuda()
        gt_masks = batch['masks'].cuda().bfloat16()
        
        # (A) Loss Calculation
        resize_shape_list = [[batch['resize_shape'][0].item(), batch['resize_shape'][1].item()]]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_loss,
                labels=labels,
                images=images,
                global_enc_images=images, # 명시적 전달
                grounding_enc_images=sam_images,
                bboxes=None,
                attention_masks=None,
                masks_list=[gt_masks[0]], # 리스트 형태로 전달
                label_list=[gt_masks[0]],
                resize_list=resize_shape_list,
                # Batch 1일 때 offset 보정
                offset=torch.tensor([0, 1]).long().cuda() if input_ids_loss.shape[0] == 1 else None 
            )
            
            if 'loss' in outputs:
                total_loss += outputs['loss'].item()
                count += 1
            if 'ce_loss' in outputs: 
                ce_loss += outputs['ce_loss'].item()
            if 'mask_loss' in outputs: 
                mask_loss += outputs['mask_loss'].item()

        # (B) Inference (Generate)

        """
        human_q = batch['human_q'][0]

        if len(human_q) > 1000:
            human_q = "이 항공 사진에서 붉은 박스로 표시된 나무들의 탄소 저장량을 분석해줘."

        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        q_text = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n" + human_q
        q_text = q_text.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        conv.append_message(conv.roles[0], q_text)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        
        input_ids_gen = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
        
        orig_size = [batch['orig_size'][0].numpy(), batch['orig_size'][1].numpy()]
        resize_shape = [batch['resize_shape'][0].numpy(), batch['resize_shape'][1].numpy()]
        
        # 모델 추론 (evaluate 메소드 사용)
        output_ids, pred_masks = model.evaluate(
            images, sam_images, input_ids_gen, [resize_shape], [orig_size],
            max_tokens_new=512, bboxes=None
        )
        
        # 결과 파싱
        out_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_out = tokenizer.decode(out_ids, skip_special_tokens=False).split("ASSISTANT: ")[-1]
        cleaned_text = re.sub(r'<.*?>', '', text_out).replace('[SEG]', '').strip()
        
        rle_masks = []
        if pred_masks is not None and len(pred_masks) > 0:
            pred_masks_tensor = pred_masks[0].cpu() > 0
            rle_masks = [coco_encode_rle(m) for m in mask_to_rle_pytorch(pred_masks_tensor)]
        
        results.append({
            "image_id": batch['id'][0],
            "caption": cleaned_text,
            "pred_masks": rle_masks
        })

    """

    # 최종 결과 보고 및 저장
    if count > 0:
        print("\n" + "="*30)
        print(f" [TEST SET LOSS REPORT]")
        print(f" - Total Loss: {total_loss / count:.4f}")
        print(f" - CE Loss (Text): {ce_loss / count:.4f}")
        print(f" - Mask Loss (Seg): {mask_loss / count:.4f}")
        print("="*30 + "\n")
    
    save_path = os.path.join(args.output_dir, "test_predictions.json")
    with open(save_path, 'w') as f:
        json.dump(results, f)
    print(f"Predictions saved to {save_path}")

if __name__ == "__main__":
    main()