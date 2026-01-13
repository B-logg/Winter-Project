import sys, os, gc, torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava.conversation import conv_templates
from model.llava.mm_utils import tokenizer_image_token
from torchao.quantization import int8_weight_only, quantize_

# 1. 환경 설정 및 경로 (사용자님 경로 반영)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
model_path = "/Users/bosung/Desktop/GLaMM/groundingLMM/checkpoints/GLaMM-GCG"
image_path = "/Users/bosung/Desktop/GLaMM/groundingLMM/test_image/test.png"

# 메모리 청소
gc.collect()
torch.mps.empty_cache()

print(f"[*] [1/5] GLaMM-GCG 모델 로드 중 (M4 Pro 최적화)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 모델 로드 (low_cpu_mem_usage로 램 폭발 방지)
model = GLaMMForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    seg_token_idx=tokenizer.convert_tokens_to_ids("[SEG]")
)

# 2. 8비트 양자화 및 GPU 전체 이동
# LLM을 포함한 전체 모델을 8비트로 압축하여 16GB 램에 스왑 없이 올립니다.
print("[*] [2/5] 전체 모델 8비트 압축 및 MPS(GPU) 이동 중...")
quantize_(model, int8_weight_only()) # 모델 전체 압축
model.to("mps") # 3시간 30분을 1분으로 줄이는 핵심
gc.collect()

# 3. 이미지 및 텐서 준비
print("[*] [3/5] 비전 데이터 전처리...")
raw_image = Image.open(image_path).convert("RGB")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"].to("mps", dtype=torch.float16)

# SAM용 텐서 준비
sam_image = raw_image.resize((512, 512))
sam_image_tensor = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).float()
sam_image_tensor = ((sam_image_tensor - torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)) / 
                    torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)).unsqueeze(0).to("mps", dtype=torch.float16)
sam_image_tensor = F.interpolate(sam_image_tensor, size=(1024, 1024), mode='bilinear')

# 4. 고속 추론 (전체 GPU 연산)
print("[*] [4/5] MPS 고속 추론 시작...")
conv = conv_templates["vicuna_v1"].copy()
prompt = "Describe the image and segment objects with [SEG]."
conv.append_message(conv.roles[0], "<image>\n" + prompt)
conv.append_message(conv.roles[1], None)
input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, -200, return_tensors='pt').unsqueeze(0).to("mps")

with torch.inference_mode():
    output_ids, pred_masks = model.evaluate(
        global_enc_images=image_tensor,
        grounding_enc_images=sam_image_tensor,
        input_ids=input_ids,
        resize_list=[raw_image.size[::-1]],
        orig_sizes=[raw_image.size[::-1]],
        max_tokens_new=128
    )

# 5. 결과 분석 및 저장
print("[*] [5/5] 결과 분석 및 마스크 생성 확인")
response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
print(f"모델 응답: {response}")

if pred_masks is not None and len(pred_masks) > 0:
    vis_image = np.array(raw_image)
    for mask in pred_masks[0]:
        # 마스크를 CPU로 옮겨 시각화
        mask_np = mask.nan_to_num(0.0).cpu().numpy() > 0.0
        if not np.any(mask_np): continue
        vis_image[mask_np] = vis_image[mask_np] * 0.5 + np.random.randint(0,255,3) * 0.5
    Image.fromarray(vis_image.astype(np.uint8)).save("glamm_fast_success.png")
    print("성공! glamm_fast_success.png를 확인하세요.")
else:
    print("텍스트는 성공, 마스크는 비어있습니다. layers.py 수정 여부를 확인하세요.")