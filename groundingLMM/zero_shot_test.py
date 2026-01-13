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

# GPU 캐시 비우기
gc.collect()
torch.cuda.empty_cache()

print(f"[*] [1/5] 모델 및 토크나이저 로드 (RTX 5090 최적화)")

# 토크나이저 및 특수 토큰 설정
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
special_tokens = ["[SEG]", "<p>", "</p>", "<grounding>"]
tokenizer.add_tokens(special_tokens, special_tokens=True)
sp_limit = tokenizer.sp_model.get_piece_size()

# 모델 로드 (Bfloat16 정밀도 사용)
model = GLaMMForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    seg_token_idx=tokenizer.convert_tokens_to_ids("[SEG]")
)
model.resize_token_embeddings(len(tokenizer))
model.config.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

# 2. 모델 GPU 이동
print("[*] [2/5] 모델 CUDA(GPU) 이동 완료")
model.to("cuda")
model.eval()

# 3. 데이터 전처리
print("[*] [3/5] 탄소 흡수원 이미지 전처리")
raw_image = Image.open(image_path).convert("RGB")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)

sam_image_res = raw_image.resize((1024, 1024))
sam_image_tensor = torch.from_numpy(np.array(sam_image_res)).permute(2, 0, 1).float()
sam_image_tensor = ((sam_image_tensor - torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)) / 
                    torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

# 4. 복합 환경 분석 추론
print("[*] [4/5] RTX 5090 Blackwell 엔진 추론 가동")
conv = conv_templates["vicuna_v1"].copy()
prompt = (
    "As an environmental expert, please provide a comprehensive assessment for this scene.\n"
    "1. Describe the overall terrain and moisture levels in detail using <p> tags.\n"
    "2. Identify all tree species and vegetation types visible.\n"
    "3. Evaluate the carbon sequestration potential of this ecosystem.\n"
    "4. Crucially, insert the [SEG] token exactly where you identify a specific tree or plant."
)
conv.append_message(conv.roles[0], "<image>\n" + prompt)
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, -200, return_tensors='pt').unsqueeze(0).to("cuda")

with torch.inference_mode():
    output_ids, pred_masks = model.evaluate(
        global_enc_images=image_tensor, grounding_enc_images=sam_image_tensor,
        input_ids=input_ids, resize_list=[raw_image.size[::-1]],
        orig_sizes=[raw_image.size[::-1]], max_tokens_new=1024
    )

# 5. 결과 분석 및 시각화 저장 (완전 수동 매핑 방식)
print("[*] [5/5] 결과 분석 및 이미지 시각화 중")
input_token_len = input_ids.shape[1]
response_ids = output_ids[0][input_token_len:].cpu().tolist()

# 실제 검출된 ID를 바탕으로 한 매핑 테이블
# 보성님 서버의 실제 ID: [SEG]:32004, <p>:32005, </p>:32006
special_map = {
    32004: "[SEG]",
    32005: "<p>",
    32006: "</p>",
    32007: "<grounding>"
}

raw_tokens = []
clean_tokens = []

for tid in response_ids:
    if tid < sp_limit:
        try:
            # SentencePiece의 특수 공백 기호( ) 처리
            txt = tokenizer.sp_model.IdToPiece(int(tid)).replace('\u2581', ' ')
            raw_tokens.append(txt)
            clean_tokens.append(txt)
        except: continue
    else:
        # 특수 토큰 강제 치환
        tag = special_map.get(tid, f"[ID_{tid}]")
        raw_tokens.append(f" {tag} ")
        if tag == "<p>": clean_tokens.append(" <p> ")
        elif tag == "</p>": clean_tokens.append(" </p>")
        elif tag == "[SEG]": clean_tokens.append(" [SEG] ")

raw_report = "".join(raw_tokens).strip()
clean_report = "".join(clean_tokens).replace("  ", " ").strip()

print("="*70 + "\n")
print(raw_report)
print("="*70 + "\n")
print(clean_report)
print("="*70 + "\n")

# --- 시각화 저장 ---
if pred_masks is not None and len(pred_masks) > 0:
    vis_image = np.array(raw_image).astype(np.float32)
    for i, mask in enumerate(pred_masks[0]):
        mask_np = mask.nan_to_num(0.0).cpu().numpy() > 0.0
        if not np.any(mask_np): continue
        color = np.random.randint(0, 255, 3)
        for c in range(3):
            vis_image[:, :, c] = np.where(mask_np, vis_image[:, :, c] * 0.5 + color[c] * 0.5, vis_image[:, :, c])
    
    Image.fromarray(vis_image.astype(np.uint8)).save(output_image_path)
    print(f"분석 완료: '{output_image_path}' 저장됨.")