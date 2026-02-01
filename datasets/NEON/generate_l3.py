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
CARBON_CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset_with_carbon.csv")

OUTPUT_PATH = os.path.join(CURRENT_DIR, "l3_dataset")
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
NEON_PRODUCT_ID = "DP3.30010.001"

TILE_SIZE = 1024
MIN_TREE_THRESHOLD = 3 
TEST_TILE_LIMIT = 5 
SAM_BATCH_SIZE = 32 

def init_models():
    print(f"Initializing Models on {device}...")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
    
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return gemini_model, predictor

def download_neon_image(site, year, tile_id, save_dir):
    filename = f"{tile_id}.tif"
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        if os.path.getsize(save_path) < 1024 * 1024:
            os.remove(save_path)
        else:
            return save_path
    
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
        
        if os.path.getsize(save_path) < 1000:
            os.remove(save_path)
            return None
        return save_path
    except: 
        if os.path.exists(save_path): os.remove(save_path)
        return None

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
    filtered_stats = {'heights': [], 'areas': [], 'dbhs': [], 'carbon_annual': [], 'carbon_stored': []}
    filtered_boxes = []

    def safe_parse(key):
        if key not in row_data: return []
        val = row_data[key]
        try: return eval(val) if isinstance(val, str) else val
        except: return []

    bboxes = safe_parse('bboxes')
    heights = safe_parse('individual_heights')
    areas = safe_parse('individual_crown_areas')
    dbhs = safe_parse('individual_dbhs')
    carbon_annual = safe_parse('individual_carbon_annual')
    carbon_stored = safe_parse('individual_carbon_stored')

    if len(bboxes) == 0:
        return np.array([]), filtered_stats

    win_col_off, win_row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    
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
    total_processed_count = 0
    
    print(f"Target: Process {TEST_TILE_LIMIT} valid tiles (Batch Size: {SAM_BATCH_SIZE}).")

    for idx, (tile_id, group) in enumerate(grouped):
        if total_processed_count >= TEST_TILE_LIMIT: break

        row = group.iloc[0]
        site, year = row['site'], row['year']
        
        # 1. Download
        t_start = time.time()
        tif_path = download_neon_image(site, year, tile_id, temp_dir)
        t_dl = time.time() - t_start
        if not tif_path: continue
        print(f"[Time] Download: {t_dl:.2f}s | {tile_id}")

        try:
            with rasterio.open(tif_path) as src:
                h_img, w_img = src.shape
                
                # 2. Filtering
                t_start = time.time()
                valid_tiles = []
                for row_off in range(0, h_img, TILE_SIZE):
                    for col_off in range(0, w_img, TILE_SIZE):
                        width = min(TILE_SIZE, w_img - col_off)
                        height = min(TILE_SIZE, h_img - row_off)
                        window = Window(col_off, row_off, width, height)
                        boxes, stats = filter_trees_in_tile(src, window, row)
                        if len(boxes) >= MIN_TREE_THRESHOLD:
                            valid_tiles.append((window, boxes, stats))
                
                t_tiling = time.time() - t_start
                print(f"[Time] Filtering: {t_tiling:.2f}s | Found {len(valid_tiles)} tiles")

                if len(valid_tiles) == 0: continue

                tile_idx = 0
                for window, boxes, stats in valid_tiles:
                    try:
                        if total_processed_count >= TEST_TILE_LIMIT:
                            print("Target Reached.")
                            return

                        if len(stats['heights']) == 0: continue

                        print(f"Processing Tile... [Progress: {total_processed_count + 1}/{TEST_TILE_LIMIT}]")

                        # 3. Image Prep
                        t_start = time.time()
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
                            'sum_carbon_stored': np.sum(stats['carbon_stored']),
                            'heights': stats['heights'],
                            'areas': stats['areas'],
                            'dbhs': stats['dbhs']
                        }
                        l3_data = neon_l2_bridge(stats_summary, tile_id_suffix=tile_id_suffix)
                        if not l3_data: continue

                        img_gemini = draw_bboxes_on_image(img_tile, boxes)
                        pil_gemini = Image.fromarray(img_gemini)
                        Image.fromarray(img_tile).save(os.path.join(OUTPUT_PATH, "images", tile_filename), quality=95)
                        t_prep = time.time() - t_start
                        print(f"[Time] Image Prep: {t_prep:.2f}s")

    
                        t_start = time.time()
                        predictor.set_image(img_tile) # 임베딩 1회

                        # 전체 박스 준비
                        input_boxes_all = torch.tensor(boxes, device=device)
                        transformed_boxes_all = predictor.transform.apply_boxes_torch(input_boxes_all, img_tile.shape[:2])
                        
                        combined_mask_cpu = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

                        # 배치 단위로 끊어서 실행
                        for i in range(0, len(boxes), SAM_BATCH_SIZE):
                            batch_boxes = transformed_boxes_all[i : i + SAM_BATCH_SIZE]
                            
                            if len(batch_boxes) > 0:
                                # 추론 (최대 32개씩)
                                masks_tensor, _, _ = predictor.predict_torch(
                                    point_coords=None,
                                    point_labels=None,
                                    boxes=batch_boxes,
                                    multimask_output=False
                                )
                                # 병합 (GPU)
                                batch_merged = torch.max(masks_tensor, dim=0)[0]
                                
                                # CPU로 가져와서 누적 (Union)
                                batch_mask_np = (batch_merged[0].cpu().numpy() > 0).astype(np.uint8) * 255
                                combined_mask_cpu = np.maximum(combined_mask_cpu, batch_mask_np)
                                
                                # 메모리 즉시 해제
                                del masks_tensor
                                del batch_merged
                                torch.cuda.empty_cache() 
                        
                        mask_filename = f"mask_{tile_id_suffix}.png"
                        cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", mask_filename), combined_mask_cpu)
                        t_sam = time.time() - t_start
                        print(f"[Time] SAM (Mini-Batch): {t_sam:.2f}s ({len(boxes)} trees)")

                        # 5. Gemini
                        t_start = time.time()
                        try:
                            response = model.generate_content([l3_data['prompt'], pil_gemini])
                            clean_text = response.text.replace('```json', '').replace('```', '').strip()
                            if clean_text.startswith('{'): res_json = json.loads(clean_text)
                            else: res_json = {"dense_caption": clean_text}
                        except Exception as e: 
                            res_json = {"dense_caption": f"Error: {e}"}

                        t_gemini = time.time() - t_start
                        print(f"[Time] Gemini: {t_gemini:.2f}s")

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
                        total_processed_count += 1
                        
                    except Exception as e:
                        print(f"Skipping Tile due to error: {e}")
                        torch.cuda.empty_cache() # 에러 시에도 메모리 정리
                        continue
                    finally:
                         tile_idx += 1

        except Exception as e:
            print(f"Image Error: {e}")
            continue
        finally:
            if os.path.exists(tif_path): os.remove(tif_path)

    print("Test Finished!")

if __name__ == "__main__":
    gemini, sam = init_models()
    
    if not os.path.exists(ORIGINAL_CSV_PATH) or not os.path.exists(CARBON_CSV_PATH):
        raise FileNotFoundError("CSV Missing")
    
    print("Merging...")
    df_org = pd.read_csv(ORIGINAL_CSV_PATH)
    df_carbon = pd.read_csv(CARBON_CSV_PATH)
    
    df_merged = pd.merge(df_org, df_carbon, on='tile_id', how='inner')
    print(f"Merged Data Shape: {df_merged.shape}")
    
    unique_tiles = df_merged['tile_id'].unique()
    sample_tiles = unique_tiles[:20] 
    df_final = df_merged[df_merged['tile_id'].isin(sample_tiles)]
    
    process_dataset(df_final, gemini, sam)