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
    NEON API를 통해 .tif 파일을 다운로드합니다.
    site, year를 CSV에서 가져오는 대신 tile_id에서 직접 추출하여 안전성 확보
    """
    filename = f"{tile_id}.tif"
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        return save_path

    # tile_id 예시: "2022_GRSM_6_270000_3937000_image"
    try:
        parts = tile_id.split('_')
        safe_year = parts[0]   # "2022"
        safe_site = parts[1]   # "GRSM"
    except IndexError:
        print(f"⚠️ Tile ID Parsing Failed: {tile_id}")
        return None

    # 디버깅용 URL 출력
    url_site = f"https://data.neonscience.org/api/v0/data/{NEON_PRODUCT_ID}/{safe_site}"
    # print(f"DEBUG: Checking API: {url_site}") 

    try:
        # 1. 해당 사이트의 데이터 가용성 확인
        r = requests.get(url_site)
        
        # API가 404나 400을 뱉으면 바로 중단
        if r.status_code != 200:
            print(f"API Error ({r.status_code}): {r.text} | URL: {url_site}")
            return None
            
        r_json = r.json()
        if 'error' in r_json: 
            print(f"API Error Response: {r_json['error']}")
            return None
        
        # 해당 연도의 데이터가 있는지 확인
        available_months = [m for m in r_json['data']['siteCodes'][0]['availableMonths'] if m.startswith(str(safe_year))]
        
        if not available_months:
            print(f"No data found for {safe_site} in {safe_year}")
            return None

        # 파일 URL 찾기
        file_url = None
        # 최신 데이터부터 찾기 위해 역순 정렬 (선택사항)
        for month in sorted(available_months, reverse=True):
            url_files = f"https://data.neonscience.org/api/v0/data/{NEON_PRODUCT_ID}/{safe_site}/{month}"
            r_files = requests.get(url_files).json()
            
            # 'files' 키가 없는 경우 방어
            if 'data' not in r_files or 'files' not in r_files['data']:
                continue

            for file_info in r_files['data']['files']:
                if file_info['name'] == filename:
                    file_url = file_info['url']
                    # print(f"Found URL in {month}")
                    break
            if file_url: break
        
        if not file_url:
            print(f"File not found in NEON database: {filename}")
            return None

        # 3. 다운로드 수행
        # print(f"Downloading {filename}...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # print("Download Complete.")
        return save_path

    except Exception as e:
        print(f"Download Exception: {e}")
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
    grouped = df.groupby('tile_id')
    
    print(f"Processing {len(grouped)} tiles...")

    for idx, (tile_id, group) in enumerate(grouped):
        site = group.iloc[0]['site']
        year = group.iloc[0]['year']
        
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

            for _, row in group.iterrows():
                try:
                    utm_box_str = row['bboxes']
                    utm_box = eval(utm_box_str) if isinstance(utm_box_str, str) else utm_box_str
                    
                    px_box, w, h = get_pixel_coords(tif_path, utm_box)
                    
                    if (px_box[2] - px_box[0]) < 3 or (px_box[3] - px_box[1]) < 3: continue

                    norm_box = normalize_bbox(px_box, w, h)

                    # Bridge용 데이터 포장
                    def safe_parse(val):
                        if isinstance(val, str) and val.startswith('['): return eval(val)
                        return val

                    wrapped_row = {
                        'site': row['site'],
                        'tile_id': row['tile_id'],
                        'bboxes': [utm_box], 
                        'individual_heights': [row['height']],
                        'individual_crown_areas': [row['crown_area']], 
                        'individual_dbhs': [safe_parse(row['est_dbh'])], 
                        'individual_tree_types': [row['est_type']]
                    }
                    
                    l3_data = neon_l2_bridge(wrapped_row, tree_idx=0, custom_bbox=norm_box)
                    
                    # Gemini (Pro)
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
                    print(f"  Saved: {l3_data['id']}")

                except Exception as e:
                    print(f"  Tree Error: {e}")
                    continue
        
        except Exception as e:
            print(f"Tile Error: {e}")
        
        finally:
            if os.path.exists(tif_path):
                os.remove(tif_path)
                print(f"Deleted temp file: {tif_path}")

    # JSON 저장
    with open(os.path.join(OUTPUT_PATH, "l3_dataset.json"), "w", encoding='utf-8') as f:
        json.dump(l3_results, f, indent=4, ensure_ascii=False)
    print("All Done!")

if __name__ == "__main__":
    gemini, sam = init_models()
    df = pd.read_csv(INPUT_PATH)
    
    # [테스트용] 5개 타일만 실행
    unique_tiles = df['tile_id'].unique()
    sample_tiles = np.random.choice(unique_tiles, min(len(unique_tiles), 5), replace=False)
    df = df[df['tile_id'].isin(sample_tiles)]
    
    process_dataset(df, gemini, sam)