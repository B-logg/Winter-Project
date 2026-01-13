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
    "Please perform an expert level environmental assessment of this image:\n"
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

# 5090이 내뱉은 숫자들(ID)을 리스트로 변환
generated_ids = output_ids[0].cpu().tolist()
decoded_tokens = []

# SentencePiece 엔진의 실제 한계치
sp_limit = tokenizer.sp_model.get_piece_size()

for tid in generated_ids:
    # 1. 일반적인 단어 (사전 범위 내)
    if tid < sp_limit:
        try:
            token_text = tokenizer.sp_model.IdToPiece(int(tid))
            # SentencePiece 특유의 '_' 공백 문자를 실제 공백으로 복구
            decoded_tokens.append(token_text.replace(" ", " "))
        except:
            continue
    # 2. 특수 토큰 (사전 범위를 벗어난 GLaMM 전용 신호)
    else:
        # 모델이 학습한 ID 번호에 맞춰 직접 글자로 매핑
        if tid == tokenizer.convert_tokens_to_ids("<p>"):
            decoded_tokens.append("\n[분석 항목] ")
        elif tid == tokenizer.convert_tokens_to_ids("</p>"):
            decoded_tokens.append("\n")
        elif tid == tokenizer.convert_tokens_to_ids("[SEG]"):
            decoded_tokens.append(" [객체 세그멘테이션 실행] ")
        else:
            # 기타 알 수 없는 ID는 그냥 번호로 표시 (에러 방지)
            decoded_tokens.append(f" [Special_Token_{tid}] ")

# 최종 리포트 텍스트 완성
response = "".join(decoded_tokens).replace("  ", " ").strip()

print("\n" + "="*60)
print("  [RTX 5090 기반 GLaMM AI 환경 분석 리포트]")
print("="*60)
if response:
    print(response)
else:
    print("텍스트 추출에 실패했지만, 이미지 생성 단계로 넘어갑니다.")
print("="*60 + "\n")

# 이후 이미지 저장 코드는 동일...