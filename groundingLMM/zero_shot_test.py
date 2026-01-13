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

# SentencePiece 엔진의 실제 한계치
sp_limit = tokenizer.sp_model.get_piece_size()

# 모델 로드
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
print("[*] [4/5] 추론 중")
conv = conv_templates["vicuna_v1"].copy()

prompt = (
    "As an environmental expert, please provide a very detailed report for this scene.\n"
    "Step 1: Write a long paragraph describing the terrain, soil moisture, and any visible drought indicators.\n"
    "Step 2: List the specific species of trees and vegetation you can identify.\n"
    "Step 3: Provide a detailed evaluation of carbon sequestration efficiency (Low/Medium/High) and why.\n"
    "Step 4: Finally, output the [SEG] token for every tree and plant to visualize your analysis."
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

# 5. 결과 분석 및 시각화 저장 (전수조사 버전)
print("[*] [5/5] 추론 결과 분석 및 결과물 저장 중...")

# 질문 길이를 제외한 순수 답변 ID 추출
input_token_len = input_ids.shape[1]
response_ids = output_ids[0][input_token_len:].cpu().tolist()

# [디버깅] 현재 시스템이 알고 있는 ID 번호 확인
print(f"   > [디버깅] 내비게이션 번호표:")
print(f"     - [SEG] ID: {tokenizer.convert_tokens_to_ids('[SEG]')}")
print(f"     - <p> ID: {tokenizer.convert_tokens_to_ids('<p>')}")
print(f"     - </p> ID: {tokenizer.convert_tokens_to_ids('</p>')}")
print(f"     - <grounding> ID: {tokenizer.convert_tokens_to_ids('<grounding>')}")

decoded_tokens = []
sp_limit = tokenizer.sp_model.get_piece_size()

for tid in response_ids:
    tid_val = int(tid)
    # 1. 일반 텍스트 토큰 (사전 내 범위)
    if tid_val < sp_limit:
        try:
            token_text = tokenizer.sp_model.IdToPiece(tid_val).replace('\u2581', ' ')
            decoded_tokens.append(token_text)
        except:
            continue
    # 2. 특수 토큰 처리 (사전 외 범위)
    else:
        # 현재 토크나이저에 등록된 이름을 가져옴
        token_name = tokenizer.convert_ids_to_tokens(tid_val)
        
        if token_name == "[SEG]":
            decoded_tokens.append(" [SEG] ")
        elif token_name == "<p>":
            decoded_tokens.append("\n<p> ")
        elif token_name == "</p>":
            decoded_tokens.append(" </p>\n")
        elif token_name == "<grounding>":
            decoded_tokens.append(" <grounding> ")
        else:
            # 예상치 못한 ID가 나왔을 경우 (예: <0x0A> 등)
            decoded_tokens.append(f" [{token_name if token_name else tid_val}] ")

# 최종 리포트 완성
response = "".join(decoded_tokens).replace("<s>", "").replace("</s>", "").strip()

print("\n" + "="*65)
print("  [GLaMM AI 정밀 분석 리포트 (태그 포함 버전)]")
print("-" * 65)
print(response)
print("="*65 + "\n")

# 시각화 및 이미지 저장
if pred_masks is not None and len(pred_masks) > 0:
    vis_image = np.array(raw_image).astype(np.float32)
    
    # 마스크 시각화: 식생 객체별로 다른 색상 부여
    for i, mask in enumerate(pred_masks[0]):
        # 마스크 이진화 (0.0보다 큰 지점)
        mask_np = mask.nan_to_num(0.0).cpu().numpy() > 0.0
        if not np.any(mask_np): continue
        
        # 랜덤 색상 생성
        color = np.random.randint(0, 255, 3)
        
        # 반투명 오버레이 적용 (원본 50% + 색상 50%)
        for c in range(3):
            vis_image[:, :, c] = np.where(mask_np, vis_image[:, :, c] * 0.5 + color[c] * 0.5, vis_image[:, :, c])
    
    final_image = Image.fromarray(vis_image.astype(np.uint8))
    final_image.save(output_image_path)
    print(f"분석 성공: 리포트 출력 및 '{output_image_path}' 저장 완료.")
else:
    print("경고: 리포트는 생성되었으나 마스크 이미지 생성에 실패했습니다.")