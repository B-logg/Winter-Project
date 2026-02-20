import sys, os, gc, torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import PeftModel # 추가된 부분
from model.GLaMM import GLaMMForCausalLM
from model.llava.conversation import conv_templates
from model.llava.mm_utils import tokenizer_image_token

# 1. 경로 및 환경 설정
model_path = os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/checkpoints/GLaMM-GCG")
lora_path = os.path.expanduser("~/학부연구생/bosung/Winter-Project/checkpoints/GLaMM-Forest-A40-4GPU/checkpoint-epoch-3")
image_path = os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/test_image/test.png") # 테스트 이미지
output_image_path = "final_carbon_analysis_result.png"

# GPU 메모리 최적화
gc.collect()
torch.cuda.empty_cache()

print(f"[1/5] 모델 및 토크나이저 로드 및 파인튜닝 가중치 병합")

# 토크나이저 로드 및 특수 토큰 설정
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
special_tokens = ["[SEG]", "<p>", "</p>"]
tokenizer.add_tokens(special_tokens, special_tokens=True)
sp_limit = tokenizer.sp_model.get_piece_size()
seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")

# 1-1. 베이스 모델 로드
model = GLaMMForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    seg_token_idx=seg_token_id
)
model.resize_token_embeddings(len(tokenizer))
model.config.seg_token_idx = seg_token_id

# 1-2. LoRA 가중치 로드 및 병합 (Merge)
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload().bfloat16() # 베이스 모델과 완전히 병합

# 2. 모델 GPU 이동 및 몽키 패치
print("[2/5] 모델 CUDA(GPU) 이동 및 SAM 몽키 패치 적용")
model.to("cuda")
model.eval()

# SAM bfloat16 입력 충돌 방지용 Monkey Patch
base_glamm = model.get_model() if hasattr(model, "get_model") else model.base_model
if hasattr(base_glamm, "grounding_encoder"):
    mask_decoder = base_glamm.grounding_encoder.mask_decoder
    original_forward = mask_decoder.forward
    def mask_decoder_forward_wrapper(*args, **kwargs):
        new_args = [a.to(torch.bfloat16) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args]
        new_kwargs = {k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for k, v in kwargs.items()}
        return original_forward(*new_args, **new_kwargs)
    mask_decoder.forward = mask_decoder_forward_wrapper

# 3. 데이터 전처리
print("[3/5] 탄소 흡수원 이미지 전처리")
raw_image = Image.open(image_path).convert("RGB")

# CLIP용 전처리
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)

# SAM용 전처리
sam_image_res = raw_image.resize((1024, 1024))
sam_image_tensor = torch.from_numpy(np.array(sam_image_res)).permute(2, 0, 1).float()
sam_image_tensor = ((sam_image_tensor - torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)) / 
                    torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

# 4. 복합 환경 분석 추론
print("[4/5] 추론")
conv = conv_templates["vicuna_v1"].copy()

# 환경 분석 전문가용 프롬프트
prompt = (
    "당신은 산림 생태 및 탄소 순환 전문가입니다. 산림 이미지에서 나무들을 식별하고, 해당 구역의 탄소 저장량을 추론하세요. 다음 단계를 엄격히 따라 분석 보고서를 작성하세요.\n"
    "1단계: <p> 태그를 사용하여 전체적인 지형, 임분 밀도(나무의 빽빽한 정도) 상세히 묘사하세요.\n"
    "2단계: 화면에 보이는 수종(활엽수, 침엽수 등)을 분류하고, 잎의 색상과 수관(Canopy)의 크기를 바탕으로 수목의 건강 상태를 평가하세요.\n"
    "3단계: 관찰된 수관의 크기와 밀도를 기반으로 이 식생의 총 탄소 저장량을 논리적으로 추론하여 수치나 등급으로 제시하세요.\n"
    "4단계: 산림 이미지에서 식별 가능한 모든 나무에 대해, 나무의 특징을 짧게 묘사한 직후 반드시 [SEG] 토큰을 삽입하세요."
    "해당 내용을 보고서로 작성하세요.\n"
)

conv.append_message(conv.roles[0], "<image>\n" + prompt)

# Prefilling
forced_prefix = "Based on my expert ecological analysis of this scene, <p>"
conv.append_message(conv.roles[1], forced_prefix)

input_prompt = conv.get_prompt()
if input_prompt.endswith("</s>"):
    input_prompt = input_prompt[:-4]

input_ids = tokenizer_image_token(input_prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).to("cuda")

with torch.inference_mode():
    output_ids, pred_masks = model.evaluate(
        global_enc_images=image_tensor, 
        grounding_enc_images=sam_image_tensor,
        input_ids=input_ids, 
        resize_list=[raw_image.size[::-1]],
        orig_sizes=[raw_image.size[::-1]], 
        max_tokens_new=1024,
    )

# 5. 결과 분석 및 시각화 저장
print("[5/5] 결과 분석 및 이미지 시각화 중")

# 질문 길이를 제외한 순수 생성 부분 추출
input_token_len = input_ids.shape[1]
response_ids = output_ids[0][input_token_len:].cpu().tolist()

special_map = {32004: "[SEG]", 32005: "<p>", 32006: "</p>"}

raw_tokens = []
clean_tokens = []

for tid in response_ids:
    if tid < sp_limit:
        try:
            txt = tokenizer.sp_model.IdToPiece(int(tid)).replace('\u2581', ' ')
            raw_tokens.append(txt)
            clean_tokens.append(txt)
        except: continue
    else:
        tag = special_map.get(tid, f"[ID_{tid}]")
        raw_tokens.append(f" {tag} ")
        if tag == "<p>": clean_tokens.append("<p>")
        elif tag == "[SEG]": clean_tokens.append("[SEG]")
        elif tag == "</p>": clean_tokens.append("</p>")

final_raw = forced_prefix + "".join(raw_tokens).strip()
final_clean = forced_prefix.replace("<p>", "\n") + "".join(clean_tokens).replace("  ", " ").strip()

print(final_raw)
print(final_clean)

# 시각화 저장 로직
if pred_masks is not None and len(pred_masks) > 0:
    vis_image = np.array(raw_image).astype(np.float32)
    for i, mask in enumerate(pred_masks[0]):
        mask_np = mask.nan_to_num(0.0).cpu().numpy() > 0.0
        if not np.any(mask_np): continue
        color = np.random.randint(0, 255, 3)
        for c in range(3):
            vis_image[:, :, c] = np.where(mask_np, vis_image[:, :, c] * 0.5 + color[c] * 0.5, vis_image[:, :, c])
    
    Image.fromarray(vis_image.astype(np.uint8)).save(output_image_path)
    print(f"텍스트 출력 및 '{output_image_path}' 저장 완료.")
else:
    print("마스크 생성 실패")


# python inference_glamm.py --lora_model_path checkpoints/my-finetuned-lora --image_path sample.jpg --prompt "이 이미지에서 나무 영역을 분할해줘."