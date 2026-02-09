import sys
import os
import gc
import math
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava.conversation import conv_templates
from model.llava.mm_utils import tokenizer_image_token


# [Calculate Part]

# 탄소량을 계산하기 위한 수종별 계수 반환 함수
def get_carbon_coefficients():
    # 흉고직경(cm)
    dbh_coeffs = {
        "conifer": {"alpha": 1.004, "beta": 0.730},
        "broadleaf": {"alpha": 0.694, "beta": 0.730}
    }

    # 지상부 바이오매스(kg)
    agb_coeffs = {
        "" : {},
    }

    # 뿌리함량비(R), 탄소전환계수(CF)
    carbon_factors = {
        "" : {"R": 0.00, "CF": 0.00 },
    }
    
    return dbh_coeffs, agb_coeffs, carbon_factors

def fit_ellipse_and_get_cw(mask_np, pixel_scale):
    contours, _ = cv2.findCounters(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        return 0.0
    
    (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
    cw_pixels = (MA + ma) / 2
    cw_meters = cw_pixels * pixel_scale
    return cw_meters

def calculate_carbon_metrics(species_text, cw_m, height_m, dbh_coeffs, agb_coeffs, carbon_factors):
    species_lower = species_text.lower() # 수종
    
    if any(x in species_lower for x in ["pine ... (Conifer 종류 수종)"]):
        type_key = "conifers"
        species_key = species_lower
    elif any(x in species_lower for x in ["oak ... (broadleaf 종류 수종)"]):
        type_key = "broadleaf"
        species_key = species_lower
    else:
        # 기타 침엽수, 기타 활엽수 구분 로직
        type_key = "conifer" if "기타 침엽수에 해당될 조건" == True else "broadleaf"
        species_key = species_lower

    d_params = dbh_coeffs[type_key]
    sigma = 0.45 # 흉고직경 수식에서의 불확실성(잔차분산)

    # 흉고직경 계산
    try:
        ln_val = d_params["alpha"] + d_params["beta"] * math.log(height_m * cw_m) + (sigma**2)/2
        dbh_cm = math.exp(ln_val)
    except ValueError:
        return None
    
    # 지상부 바이오매스 계산
    bio_params = agb_coeffs.get(species_key, agb_coeffs[""])
    try:
        agb_kg = math.exp(bio_params["alpha"] + bio_params["beta"] * math.log((dbh_cm**2) * height_m))
    except:
        agb_kg = 0.0

    # 뿌리함량비(R), 탄소전환계수(CF) 불러와서 CO2 양 계산
    cf_params = carbon_factors.get(species_key, carbon_factors[""])
    total_biomass = agb_kg * (1 + cf_params["R"]) # 뿌리함량비(R) 불러오기
    carbon_stock = total_biomass * cf_params["CF"]
    co2_absorbed = carbon_stock * (44/12)

    return {
        "species_type": type_key,
        "species_name": species_key,
        "cw_m": cw_m,            # 수관폭
        "height_m": height_m,    # 수고
        "dbh_cm": dbh_cm,        # 흉고직경
        "agb_kg": agb_kg,        # 지상부 바이오매스
        "co2_kg": co2_absorbed   # 탄소 저장량
    }
    

# [Inference Part]
model_path = os.path.expanduser("~/Winter-Project/groundingLMM/checkpoints/GLaMM-GCG") # 가중치 경로 (Fine-Tuning 가중치 or Vanilla 가중치)
image_path = os.path.expanduser("~/Winter-Project/groundingLMM/test_image/test.png")
output_image_path = "final_carbon_analysis_result.png"

pixel_to_meter_scale = 0.00 # 이거 확인해보기

gc.collect()
torch.cuda.empty_cache() # GPU에 캐시된거 정리하기

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
special_tokens = ["[SEG]", "<p>", "</p>", "<grounding>"]
tokenizer.add_tokens(special_tokens, special_tokens=True)
sp_limit = tokenizer.sp_model.get_piece_size()

model = GLaMMForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    seg_token_idx=tokenizer.convert_tokens_to_ids("[SEG]")
)
model.resize_token_embeddings(len(tokenizer))
model.config.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

model.to("cuda")
model.eval()

raw_image = Image.open(image_path).convert("RGB")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)

sam_image_res = raw_image.resize((1024, 1024))
sam_image_tensor = torch.from_numpy(np.array(sam_image_res)).permute(2, 0, 1).float()
sam_image_tensor = ((sam_image_tensor - torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)) / 
                    torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

conv = conv_templates["vicuna_v1"].copy()
prompt = "Analyze the vegetation, calculate carbon sequestration, and identify trees with [SEG]."
conv.append_message(conv.roles[0], "<image>\n" + prompt)
conv.append_message(conv.roles[1], None)
input_prompt = conv.get_prompt()

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

input_token_len = input_ids.shape[1]
response_ids = output_ids[0][input_token_len:].cpu().tolist()

dbh_c, agb_c, cf_c = get_carbon_coefficients()

vis_image = np.array(raw_image).copy()
mask_idx = 0
final_text_output = ""
decoded_segment = ""

id_to_token = {v: k for k, v in tokenizer.get_vocab().items()} 

tokens = []
for tid in response_ids:
    if tid in [tokenizer.convert_tokens_to_ids(x) for x in special_tokens]:
        token_str = tokenizer.decode([tid])
    else:
        token_str = tokenizer.decode([tid])
    
    if "[SEG]" in token_str:
        if pred_masks is not None and mask_idx < len(pred_masks[0]):
            mask = pred_masks[0][mask_idx]
            mask_np = mask.nan_to_num(0.0).cpu().numpy() > 0.0
            
            cw_m = fit_ellipse_and_get_cw(mask_np, pixel_to_meter_scale)
            
            y_indices, x_indices = np.where(mask_np)
            if len(y_indices) > 0:
                pixel_height = np.max(y_indices) - np.min(y_indices)
                est_height_m = pixel_height * pixel_to_meter_scale
            else:
                est_height_m = 5.0 

            metrics = calculate_carbon_metrics(decoded_segment, cw_m, est_height_m, dbh_c, agb_c, cf_c)
            
            if metrics:
                info_str = f" [Tree ID:{mask_idx} {metrics['species_type'].upper()} | CW:{metrics['cw_m']:.2f}m | DBH:{metrics['dbh_cm']:.1f}cm | CO2:{metrics['co2_kg']:.2f}kg] "
                final_text_output += info_str
                
                color = np.random.randint(0, 255, 3)
                vis_img_layer = vis_image.copy()
                for c in range(3):
                    vis_img_layer[:, :, c] = np.where(mask_np, color[c], vis_img_layer[:, :, c])
                vis_image = cv2.addWeighted(vis_image, 0.6, vis_img_layer, 0.4, 0)
                
                center_y, center_x = np.mean(y_indices), np.mean(x_indices)
                cv2.putText(vis_image, f"ID:{mask_idx} {metrics['co2_kg']:.1f}kg", (int(center_x), int(center_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            mask_idx += 1
            decoded_segment = "" 
        else:
             final_text_output += token_str
    else:
        decoded_segment += token_str
        final_text_output += token_str

Image.fromarray(vis_image.astype(np.uint8)).save(output_image_path)




    
