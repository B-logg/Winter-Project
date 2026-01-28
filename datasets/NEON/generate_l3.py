import json
import os
import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import requests
from PIL import Image
import google.generativeai as genai
from segment_anything_hq import sam_model_registry, SamPredictor

from prepare_l3 import neon_l2_bridge

GEMINI_API_KEY = os.getenv("Gemini_API_KEY")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 경로(상대)
INPUT_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "l3_dataset")
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
NEON_PRODUCT_ID = "DP3.30010.001"

# Gemini API 설정 및 SAM 설정
def init_models():
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro') # 정교한 CoT를 위해 pro 모델 사용
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return gemini_model, predictor

def download_neon_image(site, year, tile_id, save_dir):
    """
    NEON API 구조 변경: Data API 대신 Product API로 가용 월(Month)을 먼저 조회합니다.
    """
    filename = f"{tile_id}.tif"
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        return save_path
    
    # tile_id 파싱 (안전장치)
    try:
        parts = tile_id.split('_')
        safe_year = parts[0]
        safe_site = parts[1]
    except:
        return None


    # 1. Product API를 통해 해당 사이트의 가용 월(Month) 조회
    # Data API는 월(Month) 없이 호출하면 400 에러가 뜸!
    product_url = f"https://data.neonscience.org/api/v0/products/{NEON_PRODUCT_ID}"
    
    try:
        r = requests.get(product_url)
        if r.status_code != 200:
            return None
        
        data = r.json()
        if 'data' not in data: return None

        # 해당 사이트(BART 등) 정보 찾기
        site_info = next((s for s in data['data']['siteCodes'] if s['siteCode'] == safe_site), None)
        
        if not site_info:
            return None
        
        # 해당 연도(year)가 포함된 월만 필터링 (예: "2022-06")
        available_months = [m for m in site_info['availableMonths'] if m.startswith(str(safe_year))]
        
        if not available_months:
            return None

        # 2. 각 월별 Data API를 조회하여 파일 URL 찾기
        file_url = None
        for month in sorted(available_months, reverse=True):
            # 이제 정확한 월(YYYY-MM)을 알았으니 Data API 호출 가능
            data_url = f"https://data.neonscience.org/api/v0/data/{NEON_PRODUCT_ID}/{safe_site}/{month}"
            r_files = requests.get(data_url).json()
            
            if 'data' not in r_files or 'files' not in r_files['data']:
                continue

            for file_info in r_files['data']['files']:
                if file_info['name'] == filename:
                    file_url = file_info['url']
                    break
            if file_url: break
        
        if not file_url:
            return None

        # 3. 다운로드 수행
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return save_path

    except Exception as e:
        print(f"Download Error: {e}")
        return None


def get_pixel_coords(tif_path, utm_bbox):
    """
    GeoTIFF의 Transform 정보를 이용해 UTM 좌표 -> Pixel 좌표 변환
    utm_bbox: [min_x, min_y, max_x, max_y] (지도 좌표)
    returns: [min_x, min_y, max_x, max_y] (픽셀 좌표)
    """
    with rasterio.open(tif_path) as src:
        # rasterio.index는 (row, col) = (y, x)를 반환함에 주의!
        # min_x, max_y (Top-Left)
        row_min, col_min = src.index(utm_bbox[0], utm_bbox[3]) 
        # max_x, min_y (Bottom-Right)
        row_max, col_max = src.index(utm_bbox[2], utm_bbox[1])
        
        h, w = src.shape
        
        # 음수 좌표 방지 및 이미지 크기 내로 클리핑
        x1 = max(0, min(col_min, w))
        y1 = max(0, min(row_min, h))
        x2 = max(0, min(col_max, w))
        y2 = max(0, min(row_max, h))
        
        # [min_x, min_y, max_x, max_y] 순서로 반환
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], w, h
    
def normalize_bbox(px_bbox, w, h):
    """ 픽셀 좌표 -> 0~1000 정규화 좌표"""
    return [
        int(px_bbox[1] / h * 1000), # y_min
        int(px_bbox[0] / w * 1000), # x_min
        int(px_bbox[3] / h * 1000), # y_max
        int(px_bbox[2] / w * 1000)  # x_max
    ]

def process_dataset(df, model, predictor):
    os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)
    temp_dir = os.path.join(OUTPUT_PATH, "temp_tif")
    os.makedirs(temp_dir, exist_ok=True)
    
    l3_results = []
    
    # 이미 Tile ID로 유니크하므로 groupby가 사실상 1개씩 처리함
    grouped = df.groupby('tile_id')

    for idx, (tile_id, group) in enumerate(grouped):
        # 그룹에는 row가 1개만 있다고 가정 (prepare_neon 구조상)
        row = group.iloc[0] 
        site = row['site']
        year = row['year']
        
        tif_path = download_neon_image(site, year, tile_id, temp_dir)
        if not tif_path: continue

        try:
            with rasterio.open(tif_path) as src:
                img_array = src.read([1, 2, 3]) 
                img_array = np.moveaxis(img_array, 0, -1)
            
            predictor.set_image(img_array)
            pil_img = Image.fromarray(img_array)
            
            jpg_filename = f"{tile_id}.jpg"
            pil_img.save(os.path.join(OUTPUT_PATH, "images", jpg_filename), quality=85)

            bboxes_list = eval(row['bboxes']) if isinstance(row['bboxes'], str) else row['bboxes']
            heights_list = eval(row['individual_heights']) if isinstance(row['individual_heights'], str) else row['individual_heights']
            
            areas_list = eval(row['individual_crown_areas']) if isinstance(row['individual_crown_areas'], str) else row['individual_crown_areas']
            dbhs_list = eval(row['individual_dbhs']) if isinstance(row['individual_dbhs'], str) else row['individual_dbhs']
            types_list = eval(row['individual_tree_types']) if isinstance(row['individual_tree_types'], str) else row['individual_tree_types']

            # 이미지 내의 나무 수만큼 반복
            for i, utm_box in enumerate(bboxes_list):
                try:
                    # utm_box는 이제 [x, y, x, y] 숫자 4개임!
                    px_box, w, h = get_pixel_coords(tif_path, utm_box)
                    
                    # 너무 작은 박스 스킵
                    if (px_box[2] - px_box[0]) < 3 or (px_box[3] - px_box[1]) < 3: continue

                    norm_box = normalize_bbox(px_box, w, h)

                    # Bridge용 데이터 포장 (단일 값을 리스트로 감싸서 전달)
                    wrapped_row = {
                        'site': site,
                        'tile_id': tile_id,
                        'bboxes': [utm_box], 
                        'individual_heights': [heights_list[i]],
                        'individual_crown_areas': [areas_list[i]], 
                        'individual_dbhs': [dbhs_list[i]], 
                        'individual_tree_types': [types_list[i]]
                    }
                    
                    # tree_idx는 무조건 0 (wrapped_row에 1개만 넣었으므로)
                    l3_data = neon_l2_bridge(wrapped_row, tree_idx=0, custom_bbox=norm_box)

                    l3_data['id'] = f"{tile_id}_{i}"
                    
                    # Gemini
                    response = model.generate_content([l3_data['prompt'], pil_img])
                    clean_text = response.text.replace('```json', '').replace('```', '').strip()
                    try: res_json = json.loads(clean_text)
                    except: res_json = {"dense_caption": clean_text}

                    # SAM-H
                    input_box = np.array(px_box)
                    masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
                    
                    mask_filename = f"mask_{l3_data['id']}.png"
                    cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", mask_filename), (masks[0] * 255).astype(np.uint8))

                    final_entry = {
                        "id": l3_data['id'],
                        "image": jpg_filename,
                        "conversations": [
                            {"from": "human", "value": l3_data['human_query']},
                            {"from": "gpt", "value": res_json.get('dense_caption', "")}
                        ],
                        "mask_path": f"masks/{mask_filename}",
                        "bbox_normalized": norm_box 
                    }
                    l3_results.append(final_entry)

                except Exception as e:
                    print(f"  Individual Tree Error: {e}")
                    continue
        
        except Exception as e:
            print(f"Tile Error: {e}")
        
        finally:
            if os.path.exists(tif_path):
                os.remove(tif_path)

    with open(os.path.join(OUTPUT_PATH, "l3_dataset.json"), "w", encoding='utf-8') as f:
        json.dump(l3_results, f, indent=4, ensure_ascii=False)
    print("All Done!")

if __name__ == "__main__":
    gemini, sam = init_models()
    df = pd.read_csv(INPUT_PATH)
    
    # [테스트용] 1개 타일만 실행
    unique_tiles = df['tile_id'].unique()
    sample_tiles = np.random.choice(unique_tiles, min(len(unique_tiles), 1), replace=False)
    df = df[df['tile_id'].isin(sample_tiles)]
    
    process_dataset(df, gemini, sam)