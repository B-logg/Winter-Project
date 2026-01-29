import json
import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import requests
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from segment_anything_hq import sam_model_registry, SamPredictor

from prepare_l3 import neon_l2_bridge

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 경로(상대)
env_path = os.path.join(CURRENT_DIR, ".env")
load_dotenv(env_path)
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")

INPUT_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "l3_dataset")
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
NEON_PRODUCT_ID = "DP3.30010.001"

TILE_SIZE = 1024
SAM_BATCH_SIZE = 64

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

# src: 전체 큰 이미지(GeoTIFF 파일 객체) - 좌표 변환을 위해 필요
# window: 현재 잘라낸 타일의 위치와 크기 정보 - 전체 이미지에서 어디서 부터 어디까지 잘랐는지
# row_dat: CSV에서 읽어온 데이터, 이 큰 이미지 안에 있는 모든 나무의 정보가 들어있음
def filter_trees_in_tile(src, window, row_data):
    # 1024x1024 타일(Window) 안에 '중심점'이 포함되는 나무만 필터링

    # CSV 데이터 파싱(CSV 파일에 문자열 형태로 저장된 정보를 파이썬의 리스트 형태로 변환)
    bboxes = eval(row_data['bboxes']) if isinstance(row_data['bboxes'], str) else row_data['bboxes']
    heights = eval(row_data['individual_heights']) if isinstance(row_data['individual_heights'], str) else row_data['individual_heights']
    areas = eval(row_data['individual_crown_areas']) if isinstance(row_data['individual_crown_areas'], str) else row_data['individual_crown_areas']
    dbhs = eval(row_data['individual_dbhs']) if isinstance(row_data['individual_dbhs'], str) else row_data['individual_dbhs']
    types = eval(row_data['individual_tree_types']) if isinstance(row_data['individual_tree_types'], str) else row_data['individual_tree_types']
    
    # 윈도우 좌표 정보(현재 타일의 위치 정보 가져오기)
    win_col_off = window.col_off # 타일의 시작 X점(전체 이미지 기준)
    win_row_off = window.row_off # 타일의 시작 Y점(전체 이미지 기준)
    win_w = window.width # 타일 가로 길이(1024)
    win_h = window.height # 타일 세로 길이(1024)
    
    filtered_boxes = [] # SAM에 입력할 상대 좌표 박스들을 담을 리스트
    filtered_stats = {'heights': [], 'areas': [], 'dbhs': [], 'types': []}
    
    for i, utm_box in enumerate(bboxes): # 전체 이미지에 있는 모든 나무를 하나씩 꺼내서 utm 박스 검사
        # UTM -> 전체 이미지 픽셀 변환
        row_tl, col_tl = src.index(utm_box[0], utm_box[3])
        row_br, col_br = src.index(utm_box[2], utm_box[1])
        
        # 나무의 중심점(픽셀) 계산
        center_row = (row_tl + row_br) / 2
        center_col = (col_tl + col_br) / 2
        
        # 중심점이 현재 타일 안에 있는지 확인 (전체 이미지에 있는 나무들 중에 현재 계산하는 타일 안에 있는 나무인지 검사)
        if (win_row_off <= center_row < win_row_off + win_h) and \
           (win_col_off <= center_col < win_col_off + win_w):
            
            # 타일 내부 상대 좌표로 변환 (0~1024) - 타일 내부에 있는 나무라면 타일 내 상대 좌표로 변환
            rel_x1 = max(0, col_tl - win_col_off)
            rel_y1 = max(0, row_tl - win_row_off)
            rel_x2 = min(win_w, col_br - win_col_off)
            rel_y2 = min(win_h, row_br - win_row_off)
            
            # 유효한 박스인지 확인 - 경계에 걸려서 잘렸더니 박스가 1~2픽셀만 남은 경우 버림
            if rel_x2 - rel_x1 > 2 and rel_y2 - rel_y1 > 2:
                filtered_boxes.append([rel_x1, rel_y1, rel_x2, rel_y2])
                filtered_stats['heights'].append(heights[i])
                filtered_stats['areas'].append(areas[i])
                filtered_stats['dbhs'].append(dbhs[i])
                filtered_stats['types'].append(types[i])
                
    return np.array(filtered_boxes), filtered_stats


def process_dataset(df, model, predictor):
    os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)
    temp_dir = os.path.join(OUTPUT_PATH, "temp_tif")
    os.makedirs(temp_dir, exist_ok=True)
    
    l3_results = []
    grouped = df.groupby('tile_id')
    print(f"Processing {len(grouped)} large images...")

    for idx, (tile_id, group) in enumerate(grouped):
        row = group.iloc[0]
        site, year = row['site'], row['year']
        
        tif_path = download_neon_image(site, year, tile_id, temp_dir)
        if not tif_path: continue

        try:
            with rasterio.open(tif_path) as src:
                h_img, w_img = src.shape
                
                # Grid Tiling Loop (1024단위로 순회)
                tile_idx = 0
                for row_off in range(0, h_img, TILE_SIZE):
                    for col_off in range(0, w_img, TILE_SIZE):
                        
                        # 윈도우 생성 (이미지 끝부분 처리)
                        width = min(TILE_SIZE, w_img - col_off)
                        height = min(TILE_SIZE, h_img - row_off)
                        window = Window(col_off, row_off, width, height)
                        
                        # 1. 이 타일 안에 있는 나무들 필터링
                        boxes, stats = filter_trees_in_tile(src, window, row)
                        
                        tree_count = len(boxes)
                        if tree_count == 0: continue # 나무 없으면 패스
                        
                        # 2. 이미지 로드 (해당 타일만) -> 패딩
                        img_tile = src.read([1, 2, 3], window=window)
                        img_tile = np.moveaxis(img_tile, 0, -1)
                        
                        if img_tile.shape[0] != TILE_SIZE or img_tile.shape[1] != TILE_SIZE:
                            pad_h = TILE_SIZE - img_tile.shape[0]
                            pad_w = TILE_SIZE - img_tile.shape[1]
                            img_tile = np.pad(img_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                        
                        # 이미지 저장
                        tile_id_suffix = f"{tile_id}_tile{tile_idx}"
                        l3_data = neon_l2_bridge(stats, tile_id_suffix=tile_id_suffix)
                        
                        tile_filename = l3_data['image']
                        pil_tile = Image.fromarray(img_tile)
                        pil_tile.save(os.path.join(OUTPUT_PATH, "images", tile_filename), quality=85)
                        
                        # 3. SAM 배치 추론 (한 번에 N개 마스크 생성)
                        predictor.set_image(img_tile)
                        
                        all_masks = []
                        
                        # 박스를 하나씩 넣어서 안전하게 추론
                        for box in boxes:
                            # box: (4,) -> (1, 4) 형태로 차원 추가
                            mask, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=box[None, :], 
                                multimask_output=False
                            )
                            all_masks.append(mask)
                        
                        # 결과 합치기 (N, 1, H, W)
                        if all_masks:
                            masks = np.concatenate(all_masks, axis=0)
                        else:
                            masks = np.zeros((0, 1, TILE_SIZE, TILE_SIZE))
                        
                        # Segmentation Map 생성 (합치기)
                        combined_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                        for m in masks:
                            # m shape: (1, H, W)
                            combined_mask = np.maximum(combined_mask, m[0].astype(np.uint8) * 255)
                        
                        mask_filename = f"mask_{tile_id_suffix}.png"
                        cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", mask_filename), combined_mask)

                        # 4. Gemini 추론
                        response = model.generate_content([l3_data['prompt'], pil_tile])
                        try:
                            clean_text = response.text.replace('```json', '').replace('```', '').strip()
                            res_json = json.loads(clean_text)
                        except:
                            res_json = {"dense_caption": response.text}

                        # 5. 데이터 저장
                        l3_entry = {
                            "id": tile_id_suffix,
                            "image": tile_filename,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": l3_data['human_query']
                                },
                                {
                                    "from": "gpt",
                                    "value": res_json.get('dense_caption', "")
                                }
                            ],
                            "mask_path": f"masks/{mask_filename}",
                            "stats": {
                                "tree_count": tree_count,
                                "avg_height": np.mean(stats['heights']),
                                "avg_dbh": np.mean(stats['dbhs'])
                            }
                        }
                        l3_results.append(l3_entry)
                        print(f" {tile_id_suffix}: {tree_count} trees processed.")
                        
                        tile_idx += 1

        except Exception as e:
            print(f"Tile Error: {e}")
        
        finally:
            if os.path.exists(tif_path):
                os.remove(tif_path)
                print(f"Deleted temp TIF: {tif_path}")

    with open(os.path.join(OUTPUT_PATH, "l3_dataset.json"), "w", encoding='utf-8') as f:
        json.dump(l3_results, f, indent=4, ensure_ascii=False)
    print("All Done!")

if __name__ == "__main__":
    gemini, sam = init_models()
    df = pd.read_csv(INPUT_PATH)
    
    # [테스트] 3개의 대형 이미지만 샘플링
    unique_tiles = df['tile_id'].unique()
    sample_tiles = np.random.choice(unique_tiles, min(len(unique_tiles), 3), replace=False)
    df = df[df['tile_id'].isin(sample_tiles)]
    
    process_dataset(df, gemini, sam)