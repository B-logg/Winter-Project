import json
import os
import cv2
import time
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("API Key Missing")

ORIGINAL_CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv")
CARBON_CSV_PATH = os.path.join(CURRENT_DIR, "NEON_carbon_data.csv")

OUTPUT_PATH = os.path.join(CURRENT_DIR, "l3_dataset")
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
NEON_PRODUCT_ID = "DP3.30010.001"

TILE_SIZE = 1024
MIN_TREE_THRESHOLD = 3 
TEST_TILE_LIMIT = 5 # 테스트 타일 수

def init_models():
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
    
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return gemini_model, predictor

def download_neon_image(site, year, tile_id, save_dir):
    filename = f"{tile_id}.tif"
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path): return save_path
    
    try:
        parts = tile_id.split('_')
        safe_year, safe_site = parts[0], parts[1]
    except: return None

    try:
        r = requests.get(f"https://data.neonscience.org/api/v0/products/{NEON_PRODUCT_ID}")
        if r.status_code != 200: return None
        data = r.json()
        site_info = next((s for s in data['data']['siteCodes'] if s['siteCode'] == safe_site), None)
        if not site_info: return None
        available_months = [m for m in site_info['availableMonths'] if m.startswith(str(safe_year))]
        if not available_months: return None

        file_url = None
        for month in sorted(available_months, reverse=True):
            r_files = requests.get(f"https://data.neonscience.org/api/v0/data/{NEON_PRODUCT_ID}/{safe_site}/{month}").json()
            if 'data' not in r_files or 'files' not in r_files['data']: continue
            for file_info in r_files['data']['files']:
                if file_info['name'] == filename:
                    file_url = file_info['url']; break
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

def draw_bboxes_on_image(img_array, boxes):
    img_viz = img_array.copy()
    if len(img_viz.shape) == 2: img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    elif img_viz.shape[2] == 1: img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_viz

def save_single_entry(entry):
    json_path = os.path.join(OUTPUT_PATH, "l3_dataset.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: data = []
    else: data = []
    data.append(entry)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def filter_trees_in_tile(src, window, row_data):
    def safe_parse(key):
        val = row_data[key]
        return eval(val) if isinstance(val, str) else val

    bboxes = safe_parse('bboxes')
    heights = safe_parse('individual_heights')
    areas = safe_parse('individual_crown_areas')
    dbhs = safe_parse('individual_dbhs')
    carbon_annual = safe_parse('individual_carbon_annual')
    carbon_stored = safe_parse('individual_carbon_stored')

    win_col_off, win_row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    
    filtered_boxes = []
    filtered_stats = {'heights': [], 'areas': [], 'dbhs': [], 'carbon_annual': [], 'carbon_stored': []}
    
    for i, utm_box in enumerate(bboxes):
        try:
            row_tl, col_tl = src.index(utm_box[0], utm_box[3])
            row_br, col_br = src.index(utm_box[2], utm_box[1])
            center_row, center_col = (row_tl + row_br) / 2, (col_tl + col_br) / 2
            
            if (win_row_off <= center_row < win_row_off + win_h) and \
               (win_col_off <= center_col < win_col_off + win_w):
                
                rel_x1 = max(0, col_tl - win_col_off)
                rel_y1 = max(0, row_tl - win_row_off)
                rel_x2 = min(win_w, col_br - win_col_off)
                rel_y2 = min(win_h, row_br - win_row_off)
                
                if rel_x2 - rel_x1 > 2 and rel_y2 - rel_y1 > 2:
                    filtered_boxes.append([rel_x1, rel_y1, rel_x2, rel_y2])
                    filtered_stats['heights'].append(heights[i])
                    filtered_stats['areas'].append(areas[i])
                    filtered_stats['dbhs'].append(dbhs[i])
                    filtered_stats['carbon_annual'].append(carbon_annual[i])
                    filtered_stats['carbon_stored'].append(carbon_stored[i])
        except: continue
    return np.array(filtered_boxes), filtered_stats

def process_dataset(df, model, predictor):
    os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)
    temp_dir = os.path.join(OUTPUT_PATH, "temp_tif")
    os.makedirs(temp_dir, exist_ok=True)
    
    grouped = df.groupby('tile_id')
    total_processed_count = 0 # 처리된 타일 수 카운트
    print(f"Processing {len(grouped)} images (Batch SAM Mode)...")

    for idx, (tile_id, group) in enumerate(grouped):
        row = group.iloc[0]
        site, year = row['site'], row['year']
        
        # [Timer 1] 다운로드
        t_start = time.time() # 주석 처리
        tif_path = download_neon_image(site, year, tile_id, temp_dir)
        t_dl = time.time() - t_start # 주석 처리
        if not tif_path: continue
        print(f"[Time] Download: {t_dl:.2f}s | {tile_id}") # 주석 처리

        try:
            with rasterio.open(tif_path) as src:
                h_img, w_img = src.shape
                
                # [Timer 2] 타일링 및 필터링
                t_start = time.time() # 주석 처리
                valid_tiles = []
                for row_off in range(0, h_img, TILE_SIZE):
                    for col_off in range(0, w_img, TILE_SIZE):
                        width = min(TILE_SIZE, w_img - col_off)
                        height = min(TILE_SIZE, h_img - row_off)
                        window = Window(col_off, row_off, width, height)
                        boxes, stats = filter_trees_in_tile(src, window, row)
                        if len(boxes) >= MIN_TREE_THRESHOLD:
                            valid_tiles.append((window, boxes, stats))
                
                t_tiling = time.time() - t_start # 주석 처리
                print(f"[Time] Filtering: {t_tiling:.2f}s") # 주석 처리
                print(f"[{idx+1}/{len(grouped)}] {tile_id}: {len(valid_tiles)} tiles.")

                tile_idx = 0
                for window, boxes, stats in valid_tiles:
                    if total_processed_count >= TEST_TILE_LIMIT: 
                        print("Test Limit Reached. Stopping...") 
                        return
                    
                    # [Timer 3] 이미지 로드 및 전처리
                    t_start = time.time() # 주석 처리
                    img_tile_raw = src.read([1, 2, 3], window=window)
                    img_tile_raw = np.moveaxis(img_tile_raw, 0, -1)
                    img_tile = normalize_image(img_tile_raw)
                    
                    if img_tile.shape[0] != TILE_SIZE or img_tile.shape[1] != TILE_SIZE:
                        pad_h, pad_w = TILE_SIZE - img_tile.shape[0], TILE_SIZE - img_tile.shape[1]
                        img_tile = np.pad(img_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

                    tile_id_suffix = f"{tile_id}_tile{tile_idx}"
                    tile_filename = f"{tile_id_suffix}.jpg"
                    
                    stats_summary = {
                        'tree_count': len(boxes),
                        'avg_height': np.mean(stats['heights']),
                        'avg_area': np.mean(stats['areas']),
                        'avg_diameter': np.mean(stats['dbhs']),
                        'sum_carbon_annual': np.sum(stats['carbon_annual']),
                        'sum_carbon_stored': np.sum(stats['carbon_stored'])
                    }
                    l3_data = neon_l2_bridge(stats_summary, tile_id_suffix=tile_id_suffix)
                    if not l3_data: continue

                    img_gemini = draw_bboxes_on_image(img_tile, boxes)
                    pil_gemini = Image.fromarray(img_gemini)
                    Image.fromarray(img_tile).save(os.path.join(OUTPUT_PATH, "images", tile_filename), quality=95)
                    t_prep = time.time() - t_start # 주석 처리
                    print(f"[Time] Image Prep: {t_prep:.2f}s") # 주석 처리

                    # SAM Batch 추론
                    # [Timer 4] SAM Batch 추론
                    t_start = time.time() # 주석 처리
                    predictor.set_image(img_tile) # 이미지 임베딩
                    
                    # 1. 박스 좌표를 텐서로 변환
                    input_boxes = torch.tensor(boxes, device=device) # (N, 4)
                    
                    # 2. SAM에 맞는 좌표계로 변환
                    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img_tile.shape[:2])
                    
                    # 3. 한 번에 모든 박스 추론 (predict_torch 사용)
                    # output masks shape: (N, 1, H, W) -> N개의 나무 마스크가 한 번에 나옴
                    masks_tensor, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False
                    )
                    
                    # 4. GPU 상에서 마스크 병합 (Union)
                    # torch.max(dim=0)을 쓰면 N개 중 하나라도 1이면 1이 됨 -> 겹쳐짐
                    # merged_mask shape: (1, H, W)
                    merged_mask_tensor = torch.max(masks_tensor, dim=0)[0] 
                    
                    # 5. CPU로 가져와서 저장
                    final_mask = (merged_mask_tensor[0].cpu().numpy() > 0).astype(np.uint8) * 255
                    
                    mask_filename = f"mask_{tile_id_suffix}.png"
                    cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", mask_filename), final_mask)
                    t_sam = time.time() - t_start # 주석 처리
                    print(f"[Time] SAM (Batch): {t_sam:.2f}s ({len(boxes)} trees)") # 주석 처리


                    # [Timer 5] Gemini 추론
                    t_start = time.time() # 주석 처리
                    try:
                        response = model.generate_content([l3_data['prompt'], pil_gemini])
                        clean_text = response.text.replace('```json', '').replace('```', '').strip()
                        if clean_text.startswith('{'): res_json = json.loads(clean_text)
                        else: res_json = {"dense_caption": clean_text}
                    except: res_json = {"dense_caption": "Error"}

                    t_gemini = time.time() - t_start # 주석 처리
                    print(f"[Time] Gemini: {t_gemini:.2f}s") # 주석 처리

                    l3_entry = {
                        "id": tile_id_suffix,
                        "image": tile_filename,
                        "conversations": [
                            {"from": "human", "value": l3_data['human_query']},
                            {"from": "gpt", "value": res_json.get('dense_caption', "")}
                        ],
                        "mask_path": f"masks/{mask_filename}",
                        "stats": stats_summary 
                    }
                    save_single_entry(l3_entry)
                    print(f"Saved: {tile_id_suffix} (Trees: {len(boxes)})")
                    tile_idx += 1
                    total_processed_count += 1

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if os.path.exists(tif_path): os.remove(tif_path)

    print("All Done!")

if __name__ == "__main__":
    gemini, sam = init_models()
    
    if not os.path.exists(ORIGINAL_CSV_PATH) or not os.path.exists(CARBON_CSV_PATH):
        raise FileNotFoundError("CSV Missing")
    
    print("Merging...")
    df_org = pd.read_csv(ORIGINAL_CSV_PATH)
    df_carbon = pd.read_csv(CARBON_CSV_PATH)
    df_merged = pd.merge(df_org, df_carbon, on='tile_id', how='inner')
    
    unique_tiles = df_merged['tile_id'].unique()
    sample_tiles = unique_tiles[:3] # 상위 3개 이미지만 로드
    df_final = df_merged[df_merged['tile_id'].isin(sample_tiles)]
    
    process_dataset(df_final, gemini, sam)