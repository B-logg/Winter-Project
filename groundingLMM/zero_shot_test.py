import sys, os, gc, torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava.conversation import conv_templates
from model.llava.mm_utils import tokenizer_image_token

# ==========================================================
# 1. 경로 및 환경 설정
# ==========================================================
model_path = os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/checkpoints/GLaMM-GCG")
image_path = os.path.expanduser("~/학부연구생/bosung/Winter-Project/groundingLMM/test.png")
output_image_path = "final_carbon_analysis_result.png"

# GPU 메모리 최적화
gc.collect()
torch.cuda.empty_cache()

print("[1/5] 모델 및 토크나이저 로드")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
special_tokens = ["[SEG]", "<p>", "</p>", "<grounding>"]
tokenizer.add_tokens(special_tokens, special_tokens=True)
sp_limit = tokenizer.sp_model.get_piece_size()

# 모델 로드
model = GLaMMForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    seg_token_idx=tokenizer.convert_tokens_to_ids("[SEG]")
)
model.resize_token_embeddings(len(tokenizer))
model.config.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

# ==========================================================
# 2. 모델 GPU 이동 및 몽키 패치 & 무한 추론 방지 설정
# ==========================================================
print("[2/5] 모델 CUDA(GPU) 이동 및 추론 설정 적용")
model.to("cuda") 
model.eval()

# RoPE(위치 정보) 손상 복구: NaN 에러 방지용
for name, buffer in model.named_buffers():
    if "inv_freq" in name:
        buffer.data = buffer.data.to(torch.float32)

# 💡 multinomial 에러 및 -200 인덱스 충돌 원천 차단
model.generation_config.do_sample = False  # 무작위 샘플링 끄기 (결정론적 생성)
model.generation_config.eos_token_id = tokenizer.eos_token_id 
model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

# 특정 모듈만 콕 집어서 bfloat16 캐스팅
base_glamm = model.get_model() if hasattr(model, "get_model") else model.base_model
modules_to_cast = ["grounding_encoder", "mm_projector", "text_hidden_fcs", "region_encoder"]
for mod_name in modules_to_cast:
    if hasattr(base_glamm, mod_name):
        getattr(base_glamm, mod_name).to(device="cuda", dtype=torch.bfloat16)

# SAM Mask Decoder 내부 Float32 충돌 방지용 Monkey Patch
if hasattr(base_glamm, "grounding_encoder"):
    mask_decoder = base_glamm.grounding_encoder.mask_decoder
    original_forward = mask_decoder.forward
    
    def mask_decoder_forward_wrapper(*args, **kwargs):
        new_args = [a.to(torch.bfloat16) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args]
        new_kwargs = {k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for k, v in kwargs.items()}
        return original_forward(*new_args, **new_kwargs)
        
    mask_decoder.forward = mask_decoder_forward_wrapper

# ==========================================================
# 3. 데이터 전처리
# ==========================================================
print("[3/5] 탄소 흡수원 이미지 전처리")
raw_image = Image.open(image_path).convert("RGB")

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)

sam_image_res = raw_image.resize((1024, 1024))
sam_image_tensor = torch.from_numpy(np.array(sam_image_res)).permute(2, 0, 1).float()
sam_image_tensor = ((sam_image_tensor - torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)) / 
                    torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

# ==========================================================
# 4. 복합 환경 분석 추론
# ==========================================================
print("[4/5] 추론")
conv = conv_templates["vicuna_v1"].copy()

prompt = (
    "You are an expert in forest ecology and the carbon cycle. Estimate the carbon storage of the area. Write an analysis report strictly following the steps below:\n"
    "Step 1: Use <p> tags to describe the overall terrain and stand density in detail.\n"
    "Step 2: Classify the visible tree species and evaluate their health condition.\n"
    "Step 3: Logically infer the total carbon storage of this vegetation.\n"
    "Step 4: Identify major tree clusters or canopy groups in the forest image rather than individual trees. Briefly describe each cluster's characteristics and immediately insert the [SEG] token right after the description.\n"
    "Compile this information into a structured report.\n"
)

conv.append_message(conv.roles[0], "<image>\n" + prompt)

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
        max_tokens_new=150, # 🚨 하드웨어 강제 컷: 무한 대기를 원천 차단합니다.
    )

# ==========================================================
# 5. 결과 분석 및 시각화 저장
# ==========================================================
print("[5/5] 결과 분석 및 이미지 시각화 중")

input_token_len = input_ids.shape[1]
response_ids = output_ids[0][input_token_len:].cpu().tolist()

special_map = {32004: "[SEG]", 32005: "<p>", 32006: "</p>", 32007: "<grounding>"}

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

print("="*70 + "\n")
print(final_raw)
print("="*70 + "\n")
print(final_clean)
print("="*70 + "\n")

if pred_masks is not None and len(pred_masks) > 0:
    vis_image = np.array(raw_image).astype(np.float32)
    for i, mask in enumerate(pred_masks[0]):
        mask_np = mask.nan_to_num(0.0).cpu().numpy() > 0.0
        if not np.any(mask_np): continue
        color = np.random.randint(0, 255, 3)
        for c in range(3):
            vis_image[:, :, c] = np.where(mask_np, vis_image[:, :, c] * 0.5 + color[c] * 0.5, vis_image[:, :, c])
    
    Image.fromarray(vis_image.astype(np.uint8)).save(output_image_path)
    print(f"✅ 텍스트 출력 및 '{output_image_path}' 시각화 저장 완료.")
else:
    print("⚠️ 마스크 생성 실패")