import json
import os
import time
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from segment_anything_hq import sam_model_registry, SamPredictor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")

if not GEMINI_API_KEY: raise ValueError("API Key Missing")
genai.configure(api_key=GEMINI_API_KEY)

# 경로 설정
CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv") 
OUTPUT_PATH = os.path.join(CURRENT_DIR, "l1_dataset_neon")
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")

os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)


NEON_PRODUCT_ID = "DP3.30010.001"
TILE_SIZE = 1024
MIN_TREE_THRESHOLD = 3
SAM_BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

START_TILE_COUNT = 1000  # L3에서 1000개 만들었으므로 그 다음부터 시작
TARGET_TILE_COUNT = 5 # 추가로 만들 L1 데이터 개수 (총 1000개 생성 목표)

# Gemini 모델
gemini_model = genai.GenerativeModel('gemini-2.5-flash')


def download_neon_image(site, year, tile_id, save_dir):
    filename = f"{tile_id}.tif"
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024*1024: return save_path
    
    try:
        parts = tile_id.split('_')
        safe_year, safe_site = parts[0], parts[1]
        
        r = requests.get(f"https://data.neonscience.org/api/v0/products/{NEON_PRODUCT_ID}")
        if r.status_code != 200: return None
        data = r.json()
        site_info = next((s for s in data['data']['siteCodes'] if s['siteCode'] == safe_site), None)
        if not site_info: return None
        
        available_months = [m for m in site_info['availableMonths'] if m.startswith(str(safe_year))]
        file_url = None
        for month in sorted(available_months, reverse=True):
            r_files = requests.get(f"https://data.neonscience.org/api/v0/data/{NEON_PRODUCT_ID}/{safe_site}/{month}").json()
            if 'data' in r_files and 'files' in r_files['data']:
                for f in r_files['data']['files']:
                    if f['name'] == filename: file_url = f['url']; break
            if file_url: break
        
        if not file_url: return None
        
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return save_path
    except: return None

def normalize_image(img_array):
    img_min, img_max = img_array.min(), img_array.max()
    if img_max > img_min: img_norm = (img_array - img_min) / (img_max - img_min)
    else: img_norm = img_array
    return (img_norm * 255).astype(np.uint8)

def get_species_category(species_name):
    """수종 이름으로 침엽수/활엽수 분류"""
    species_name = species_name.lower()
    conifer_keywords = ['conifer']
    if any(k in species_name for k in conifer_keywords): return "Conifer"
    return "Broadleaf"

def filter_trees_in_tile(src, window, row_data):
    """타일 안에 들어오는 나무 박스와 수종 정보만 걸러냄"""
    filtered_boxes = []
    filtered_species = []

    def safe_parse(key):
        if key not in row_data: return []
        val = row_data[key]
        try: return eval(val) if isinstance(val, str) else val
        except: return []

    bboxes = safe_parse('individual_bboxes') # 컬럼명 확인 필요 (bboxes vs individual_bboxes)
    if not bboxes: bboxes = safe_parse('bboxes') # 호환성
    
    tree_types = safe_parse('individual_tree_types')

    if len(bboxes) == 0 or len(bboxes) != len(tree_types):
        return [], []

    win_col_off, win_row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    
    for i, utm_box in enumerate(bboxes):
        try:
            # UTM 좌표 -> 픽셀 좌표 변환
            row_tl, col_tl = src.index(utm_box[0], utm_box[3])
            row_br, col_br = src.index(utm_box[2], utm_box[1])
            
            # 박스 중심점 확인
            center_row, center_col = (row_tl + row_br) / 2, (col_tl + col_br) / 2
            
            if (win_row_off <= center_row < win_row_off + win_h) and \
               (win_col_off <= center_col < win_col_off + win_w):
                
                # 타일 내부 상대 좌표로 변환
                rel_x1 = max(0, col_tl - win_col_off)
                rel_y1 = max(0, row_tl - win_row_off)
                rel_x2 = min(win_w, col_br - win_col_off)
                rel_y2 = min(win_h, row_br - win_row_off)
                
                # 너무 작은 박스 제외
                if rel_x2 - rel_x1 > 2 and rel_y2 - rel_y1 > 2:
                    filtered_boxes.append([rel_x1, rel_y1, rel_x2, rel_y2])
                    filtered_species.append(get_species_category(tree_types[i]))
        except: continue
        
    return filtered_boxes, filtered_species


def generate_dynamic_qa(species_type, count):
    # 입력된 영문 수종을 한글로 변환
    korean_name = "침엽수" if species_type == "Conifer" else "활엽수"
    
    prompt = f"""
    역할: 시각 언어 모델(VLM) 학습용 데이터셋 생성기
    
    상황: 
    - 사용자가 숲 항공 사진을 보고 특정 수종({korean_name})을 찾아달라고 요청합니다.
    - AI는 이미지에서 해당 수종을 찾아 마스크(Segmentation)를 보여주며 답변합니다.
    
    임무: 
    다음 규칙을 엄격히 준수하여 JSON 형식의 질문(question)과 답변(answer) 쌍 1개를 생성하세요.
    
    [질문(Question) 생성 규칙]
    - "{korean_name}"라는 단어를 포함하여 다양하게 질문하세요.
    - 예시: 
      - "이 이미지에서 {korean_name}를 찾아줘."
      - "{korean_name}는 어디에 있어?"
      - "여기서 {korean_name}만 세그멘테이션 해봐."
      - "{korean_name}의 위치를 알려줘."
      
    [답변(Answer) 생성 규칙]
    - 반드시 "{korean_name}"라는 단어 바로 뒤에 [SEG] 토큰을 붙여야 합니다.
    - 예시 패턴:
      - "네, 이 {count}그루의 나무들은 {korean_name} [SEG]입니다."
      - "요청하신 {count}그루의 {korean_name} [SEG]를 찾았습니다."
      - "이미지에 보이는 {count}그루의 {korean_name} [SEG]입니다."
      - "여기 {count}그루의 {korean_name} [SEG]가 있습니다."
    
    [출력 포맷]
    {{
        "question": "생성된 질문",
        "answer": "생성된 답변"
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        # JSON 파싱을 위해 마크다운 코드 블록 제거
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text)
    except Exception as e:
        # 에러 발생 시 기본 템플릿 반환 (안전장치)
        return {
            "question": f"이 이미지에서 {korean_name}를 모두 찾아줘.",
            "answer": f"네, 이미지에 있는 {count}그루의 {korean_name} [SEG]를 표시했습니다."
        }


def process_l1_dataset():
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)
    
    print(f"Initializing SAM on {device}...")
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    temp_dir = os.path.join(OUTPUT_PATH, "temp_tif")
    os.makedirs(temp_dir, exist_ok=True)
    
    global_tile_count = 0 # 전체 누적 타일 수
    created_count = 0     # 이번 실행에서 만든 개수
    l1_results = []

    # CSV 행 단위 반복 (원본 지도)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if created_count >= TARGET_TILE_COUNT: break
        
        tile_id = row['tile_id']
        site, year = row['site'], row['year']
        
        # 1. 원본 이미지 다운로드
        tif_path = download_neon_image(site, year, tile_id, temp_dir)
        if not tif_path: continue
        
        try:
            with rasterio.open(tif_path) as src:
                h_img, w_img = src.shape
                
                # 2. 타일링 (Sliding Window)
                for row_off in range(0, h_img, TILE_SIZE):
                    for col_off in range(0, w_img, TILE_SIZE):
                        # 목표 개수 채웠으면 중단
                        if created_count >= TARGET_TILE_COUNT: break
                        
                        # 타일 윈도우 정의
                        width = min(TILE_SIZE, w_img - col_off)
                        height = min(TILE_SIZE, h_img - row_off)
                        window = Window(col_off, row_off, width, height)
                        
                        # 나무 필터링
                        boxes, species_list = filter_trees_in_tile(src, window, row)
                        
                        # 나무가 너무 적으면 패스
                        if len(boxes) < MIN_TREE_THRESHOLD: continue
                        
                        # 앞서 L3에서 만든 1000개는 건너뜀
                        global_tile_count += 1
                        if global_tile_count <= START_TILE_COUNT: continue

                        
                        print(f"Generating L1 Data... ({created_count + 1}/{TARGET_TILE_COUNT})")
                        
                        # 3. 이미지 저장
                        img_tile_raw = src.read([1, 2, 3], window=window)
                        img_tile_raw = np.moveaxis(img_tile_raw, 0, -1)
                        img_tile = normalize_image(img_tile_raw)
                        
                        # 패딩 (1024x1024 맞추기)
                        if img_tile.shape[0] != TILE_SIZE or img_tile.shape[1] != TILE_SIZE:
                            pad_h = TILE_SIZE - img_tile.shape[0]
                            pad_w = TILE_SIZE - img_tile.shape[1]
                            img_tile = np.pad(img_tile, ((0, pad_h), (0, pad_w), (0, 0)))
                        
                        tile_filename = f"{tile_id}_tile{global_tile_count}.jpg"
                        Image.fromarray(img_tile).save(os.path.join(OUTPUT_PATH, "images", tile_filename), quality=95)
                        
                        # 4. SAM 마스크 생성 (수종별 분리)
                        predictor.set_image(img_tile)
                        
                        # 수종별 그룹핑
                        species_groups = {"Conifer": [], "Broadleaf": []}
                        for box, sp in zip(boxes, species_list):
                            species_groups[sp].append(box)
                        
                        for sp_name, target_boxes in species_groups.items():
                            if not target_boxes: continue
                            
                            # SAM Batch
                            input_boxes = torch.tensor(target_boxes, device=device)
                            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img_tile.shape[:2])
                            
                            combined_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                            
                            for i in range(0, len(target_boxes), SAM_BATCH_SIZE):
                                batch = transformed_boxes[i:i+SAM_BATCH_SIZE]
                                if len(batch) > 0:
                                    masks, _, _ = predictor.predict_torch(
                                        point_coords=None, point_labels=None,
                                        boxes=batch, multimask_output=False
                                    )
                                    merged = torch.max(masks, dim=0)[0]
                                    mask_np = (merged[0].cpu().numpy() > 0).astype(np.uint8) * 255
                                    combined_mask = np.maximum(combined_mask, mask_np)
                                    del masks, merged
                                    torch.cuda.empty_cache()
                            
                            # 마스크 저장
                            mask_filename = f"mask_{tile_id}_tile{global_tile_count}_{sp_name}.png"
                            cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", mask_filename), combined_mask)
                            
                            # Q&A 생성
                            qa = generate_dynamic_qa(sp_name, len(target_boxes))
                            
                            l1_entry = {
                                "id": f"{tile_id}_tile{global_tile_count}_{sp_name}",
                                "image": tile_filename,
                                "mask_path": f"masks/{mask_filename}",
                                "conversations": [
                                    {"from": "human", "value": f"{qa['question']}\n<image>"},
                                    {"from": "gpt", "value": qa['answer']}
                                ]
                            }
                            l1_results.append(l1_entry)
                        
                        created_count += 1
                        
                        # 중간 저장
                        if len(l1_results) % 50 == 0:
                            with open(os.path.join(OUTPUT_PATH, "l1_dataset.json"), 'w', encoding='utf-8') as f:
                                json.dump(l1_results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing {tile_id}: {e}")
            continue
        finally:
            if os.path.exists(tif_path): os.remove(tif_path) # 용량 확보

    # 최종 저장
    with open(os.path.join(OUTPUT_PATH, "l1_dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(l1_results, f, indent=4, ensure_ascii=False)
    
    print(f"Created {len(l1_results)} L1 entries (Skipped first {START_TILE_COUNT} tiles)")

if __name__ == "__main__":
    process_l1_dataset()