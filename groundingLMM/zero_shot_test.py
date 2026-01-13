import sys, os, gc, torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava.conversation import conv_templates
from model.llava.mm_utils import tokenizer_image_token

# 1. 경로 및 환경 설정
model_path = os.path.expanduser("~/Winter-Project/groundingLMM/checkpoints/GLaMM-FullScope")
image_path = os.path.expanduser("~/Winter-Project/groundingLMM/test_image/test.png")
output_image_path = "final_carbon_analysis_result.png"

# GPU 메모리 초기화
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"[*] [1/5] GLaMM-FullScope 모델 및 토크나이저 로드 시작")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    use_fast=False,
    padding_side='right',
    model_max_length=2048
)

# GLaMM 전용 특수 토큰 등록
special_tokens = ["[SEG]", "<p>", "</p>", "<grounding>"]
tokenizer.add_tokens(special_tokens, special_tokens=True)

# SentencePiece 엔진이 인식 가능한 실제 단어장 크기 기록
base_vocab_size = tokenizer.sp_model.get_piece_size()

# 모델 로드 (5090 최적화: BF16 사용)
model = GLaMMForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    seg_token_idx=tokenizer.convert_tokens_to_ids("[SEG]")
)

# 모델 임베딩 크기 조정 및 토큰 설정
model.resize_token_embeddings(len(tokenizer))
seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
model.config.seg_token_idx = seg_token_idx

# 2. 모델 GPU 이동
print("[*] [2/5] 모델 CUDA(GPU) 이동 완료")
model.to("cuda")
model.eval()

# 3. 데이터 전처리 
print("[*] [3/5] 이미지 전처리 중")
raw_image = Image.open(image_path).convert("RGB")

# CLIP Vision Tower 전처리
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"]
image_tensor = image_tensor.to("cuda", dtype=torch.bfloat16)

# SAM Segmentation 전처리
sam_image_res = raw_image.resize((1024, 1024))
sam_image_tensor = torch.from_numpy(np.array(sam_image_res)).permute(2, 0, 1).float()
sam_image_tensor = ((sam_image_tensor - torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)) / 
                    torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)).unsqueeze(0)
sam_image_tensor = sam_image_tensor.to("cuda", dtype=torch.bfloat16)

# 4. 복합 환경 분석 추론
print("[*] [4/5] RTX 5090 추론 엔진 가동 중")
conv = conv_templates["vicuna_v1"].copy()

prompt = (
    "Please perform an expert-level environmental assessment of this image:\n"
    "1. Identify the level of drought or water stress in the soil and overall terrain.\n"
    "2. Categorize the specific types of trees and vegetation present.\n"
    "3. Use the [SEG] token to segment every tree and plant for area calculation.\n"
    "4. Finally, evaluate the carbon sequestration efficiency based on the density and species of the vegetation, "
    "and provide a structured ecological impact report."
)

conv.append_message(conv.roles[0], "<image>\n" + prompt)
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, -200, return_tensors='pt').unsqueeze(0).to("cuda")

with torch.inference_mode():
    output_ids, pred_masks = model.evaluate(
        global_enc_images=image_tensor,
        grounding_enc_images=sam_image_tensor,
        input_ids=input_ids,
        resize_list=[raw_image.size[::-1]],
        orig_sizes=[raw_image.size[::-1]],
        max_tokens_new=1024,
    )

# 5. 결과 텍스트 출력 및 이미지 시각화
print("[*] [5/5] 추론 결과 분석 및 결과물 저장 중...")

# sp_model이 아는 범위(base_vocab_size)의 ID만 남겨서 IndexError 원천 차단
safe_ids = [int(idx) for idx in output_ids[0] if int(idx) < base_vocab_size]

try:
    # 1. 일반 디코딩 시도
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
except IndexError:
    # 2. 실패 시 필터링된 safe_ids로 디코딩
    print("특수 토큰 해석을 위한 디코딩 실행")
    response = tokenizer.decode(safe_ids, skip_special_tokens=True).strip()
    # 추가된 특수 태그들이 텍스트에 남을 경우 제거
    for tag in special_tokens:
        response = response.replace(tag, "")

print("="*60)
print(response)
print("="*60 + "\n")

# 이미지 저장 로직
if pred_masks is not None and len(pred_masks) > 0:
    vis_image = np.array(raw_image).astype(np.float32)
    
    for i, mask in enumerate(pred_masks[0]):
        mask_np = mask.nan_to_num(0.0).cpu().numpy() > 0.0
        if not np.any(mask_np): continue
        
        color = np.random.randint(0, 255, 3)
        for c in range(3):
            vis_image[:, :, c] = np.where(mask_np, vis_image[:, :, c] * 0.5 + color[c] * 0.5, vis_image[:, :, c])
    
    final_image = Image.fromarray(vis_image.astype(np.uint8))
    final_image.save(output_image_path)
    print(f"분석 성공: 리포트 출력 및 '{output_image_path}' 저장 완료.")
else:
    print("경고: 리포트는 생성되었으나 마스크 이미지 생성에 실패했습니다.")