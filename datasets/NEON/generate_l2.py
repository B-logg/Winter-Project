import json
import os
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import cv2
from PIL import Image
from tqdm import tqdm # 진행률 표시
from dotenv import load_dotenv
from google import genai
from segment_anything_hq import sam_model_registry, SamPredictor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")

if not GEMINI_API_KEY: raise ValueError("API Key Missing")

client = genai.Client(api_key=GEMINI_API_KEY)

CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "l2_dataset")
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")

# 출력 폴더 생성
os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)

NEON_PRODUCT_ID = "DP3.30010.001"
TILE_SIZE = 1024
MIN_TREE_THRESHOLD = 3 # 타일에 나무 개수가 3개 이하면 패스
SAM_BATCH_SIZE = 64 
device = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_TILE_COUNT = 5 # 데이터 샘플 개수

# NEON API로 GeoTIFF 다운로드
def download_neon_image(site, year, tile_id, save_dir):
    # 저장할 파일명 (tile_id.tif)
    filename = f"{tile_id}.tif"
    save_path = os.path.join(save_dir, filename)
    # 파일이 이미 존재하고, 1MB 이상이면 유효한 이미지라고 판단하고 다운로드 패스
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024*1024: return save_path
    try:
        # tile_id 형식: "2020_GRSM_..." -> year, site 정보 추출
        parts = tile_id.split('_')
        safe_year, safe_site = parts[0], parts[1]
        # 제품 메타데이터 요청
        r = requests.get(f"https://data.neonscience.org/api/v0/products/{NEON_PRODUCT_ID}")
        if r.status_code != 200: return None # HTTP status 200 아니면 None 반환
        data = r.json()

        # 해당 사이트 코드 정보 찾기
        site_info = next((s for s in data['data']['siteCodes'] if s['siteCode'] == safe_site), None)
        if not site_info: return None

        # 해당 연도 월 목록
        available_months = [m for m in site_info['availableMonths'] if m.startswith(str(safe_year))]
        file_url = None

        # 최신 월부터 순회하며 파일 URL 탐색
        for month in sorted(available_months, reverse=True):
            r_files = requests.get(f"https://data.neonscience.org/api/v0/data/{NEON_PRODUCT_ID}/{safe_site}/{month}").json()
            if 'data' in r_files and 'files' in r_files['data']:
                for f in r_files['data']['files']:
                    if f['name'] == filename: file_url = f['url']; break
            if file_url: break
        if not file_url: return None
        print(f"Downloading: {filename}...")

        # 스트리밍 다운로드
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return save_path
    except: return None

# 배열 이미지를 0~255 uint8로 min-max 정규화
def normalize_image(img_array):
    img_min, img_max = img_array.min(), img_array.max()
    if img_max > img_min: img_norm = (img_array - img_min) / (img_max - img_min)
    else: img_norm = img_array
    return (img_norm * 255).astype(np.uint8)

# "Conifer" / "Broadleaf" 분류
def get_species_category(species_name):
    species_name = species_name.lower()
    conifer_keywords = ['conifer']
    if any(k in species_name for k in conifer_keywords): return "Conifer"
    return "Broadleaf"

# 래스터 src의 window 영역 안에 중심이 있는 나무만 bbox(픽셀), 수종 리스트로 반환
def filter_trees_in_tile(src, window, row_data):
    filtered_boxes = []
    filtered_species = []
    # CSV 셀 문자열을 리스트로 안전 파싱 (eval)
    def safe_parse(key):
        if key not in row_data: return []
        val = row_data[key]
        try: return eval(val) if isinstance(val, str) else val
        except: return []
    bboxes = safe_parse('individual_bboxes')
    if not bboxes: bboxes = safe_parse('bboxes')
    tree_types = safe_parse('individual_tree_types')
    # 개수 불일치 시 빈 결과
    if len(bboxes) == 0 or len(bboxes) != len(tree_types): return [], []
    win_col_off, win_row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    for i, utm_box in enumerate(bboxes):
        try:
            # UTM 좌표 → 픽셀 (tl, br)
            row_tl, col_tl = src.index(utm_box[0], utm_box[3])
            row_br, col_br = src.index(utm_box[2], utm_box[1])
            center_row, center_col = (row_tl + row_br) / 2, (col_tl + col_br) / 2
            # 중심이 윈도우 안에 있을 때만
            if (win_row_off <= center_row < win_row_off + win_h) and (win_col_off <= center_col < win_col_off + win_w):
                # 윈도우 기준 상대 bbox (클리핑)
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


# 수종별 개수로 상황 판단 후, Gemini에 질문·답변(문장 내 [SEG]) 생성 요청
def generate_l2_qa(species_counts):
    # species_counts: {'Conifer': 5, 'Broadleaf': 3} 형태의 딕셔너리

    has_conifer = species_counts.get('Conifer', 0) > 0
    has_broadleaf = species_counts.get('Broadleaf', 0) > 0

    # 침엽+활엽 혼효림
    if has_conifer and has_broadleaf:
        situation = "침엽수와 활엽수가 섞여 있는 혼효림(Mixed Forest)"
        instruction = "두 수종의 공존 관계나 위치를 설명하는 문장을 만드세요."
        example_ans = "이 숲에는 침엽수 [SEG]와 활엽수 [SEG]가 함께 자라고 있습니다."
    # 침엽만
    elif has_conifer:
        situation = "침엽수(Conifer)만 있는 단순림"
        instruction = "침엽수 군락만 존재함을 강조하세요."
        example_ans = "이 구역은 주로 침엽수 [SEG] 군락으로 이루어져 있습니다."
    # 활엽만
    elif has_broadleaf:
        situation = "활엽수(Broadleaf)만 있는 단순림"
        instruction = "활엽수 군락만 존재함을 강조하세요."
        example_ans = "여기 보이는 나무들은 대부분 활엽수 [SEG]입니다."
    else:
        return None

    # 프롬프트 문자열 (역할·상황·규칙·예시·출력 형식)
    prompt = f"""
    역할: 산림 생태 전문가로서 산림을 이루고 있는 나무들의 관계를 서술해주는 역할
    상황: {situation}
    
    임무: 
    사용자가 숲의 식생 구성이나 관계를 물어볼 때, 이미지 속 객체를 지칭하며 답변하는 JSON을 생성하세요.
    
    [규칙]
    1. 답변에서 언급된 수종 뒤에는 반드시 [SEG] 토큰을 붙여야 합니다.
    2. 언급 순서는 무조건 칩엽수를 먼저 언급하고, 그 이후에 활엽수를 언급하고, 실제 존재하는 수종만 언급하세요.
    3. 침엽수와 활엽수 둘 다 언급해야하는 경우 무조건 순서를 침엽수 먼저 업급하고, 활엽수를 이후에 언급하세요.
    3. 질문은 "이 숲의 구성은 어떻게 이루어져있어?", "이 숲에는 어떤 나무들이 있어?", "이 숲 나무들의 관계를 설명해줘" 등 다양하게 생성하세요.
    
    [예시 답변 패턴]
    - "{example_ans}"
    
    Format: {
                {
                    "question": "...(생략)",
                    "answer": "...(생략)"
                }
            }
    """

    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(text)
    except:
        return {
            "question": "이 숲을 이루고 있는 나무들에는 어떤 것들이 있으며, 이 둘은 어떤 관계를 보여?",
            "answer": "이 숲을 이루고 있는 나무로는 칩엽수 [SEG]와 활엽수 [SEG]가 있습니다."
        }


# CSV 기반으로 원본 이미지(GeoTiff) 다운로드 -> 타일 잘라서 이미지/마스크/Q&A 생성 -> l2_dataset.json 저장
def process_l2_dataset():
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)

    # GRSM 사이트만 사용
    df_grsm = df[df['site'] == 'GRSM']
    print(f"GRSM Maps Found: {len(df_grsm)}")

    if len(df_grsm) == 0: return

    print(f"Initializing SAM on {device}...")
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 다운로드한 .tif 임시 저장 폴더
    temp_dir = os.path.join(OUTPUT_PATH, "temp_tif")
    os.makedirs(temp_dir, exist_ok=True)

    # 전체 타일 인덱스
    global_tile_count = 0
    # 목표 개수까지 만든 샘플 수
    created_count = 0
    # JSON에 넣을 대화 목록
    l2_results = []

    print(f"Target: Generate {TARGET_TILE_COUNT} Level-2 Samples.")

    # CSV 각 행(타일) 순회
    for idx, row in tqdm(df_grsm.iterrows(), total=len(df_grsm), desc="Processing GRSM"):
        if created_count >= TARGET_TILE_COUNT: break

        tile_id = row['tile_id']
        site, year = row['site'], row['year']

        tif_path = download_neon_image(site, year, tile_id, temp_dir)
        if not tif_path: continue

        try:
            with rasterio.open(tif_path) as src:
                h_img, w_img = src.shape

                # 1024 스텝으로 타일 잘라서 순회
                for row_off in range(0, h_img, TILE_SIZE):
                    for col_off in range(0, w_img, TILE_SIZE):
                        if created_count >= TARGET_TILE_COUNT: break

                        width = min(TILE_SIZE, w_img - col_off)
                        height = min(TILE_SIZE, h_img - row_off)
                        window = Window(col_off, row_off, width, height)

                        boxes, species_list = filter_trees_in_tile(src, window, row)

                        # 나무가 최소 개수 이상일 때만 L2 샘플 생성
                        if len(boxes) >= MIN_TREE_THRESHOLD:
                            
                            # 1. 이미지 저장
                            img_tile_raw = src.read([1, 2, 3], window=window)
                            img_tile_raw = np.moveaxis(img_tile_raw, 0, -1)
                            img_tile = normalize_image(img_tile_raw)
                            
                            if img_tile.shape[0] != TILE_SIZE or img_tile.shape[1] != TILE_SIZE:
                                img_tile = np.pad(img_tile, ((0, TILE_SIZE - img_tile.shape[0]), (0, TILE_SIZE - img_tile.shape[1]), (0, 0)))
                            
                            tile_filename = f"{tile_id}_tile{global_tile_count}.jpg"
                            Image.fromarray(img_tile).save(os.path.join(OUTPUT_PATH, "images", tile_filename), quality=95)
                            
                            # 2. SAM 마스크 생성 (복수 마스크 준비)
                            predictor.set_image(img_tile)
                            
                            species_groups = {"Conifer": [], "Broadleaf": []}
                            for box, sp in zip(boxes, species_list): species_groups[sp].append(box)
                            
                            # 이번 타일에 존재하는 수종 확인
                            active_species = [sp for sp, boxes in species_groups.items() if len(boxes) > 0]
                            species_counts = {sp: len(boxes) for sp, boxes in species_groups.items()}
                            
                            if not active_species: continue # 나무가 없으면 스킵

                            # 마스크 경로 리스트 (순서 중요, Gemini 답변 순서와 맞춰야 함)
                            # 편의상 항상 [Conifer, Broadleaf] 순서로 처리하거나, 존재하는 것만 순서대로 넣음
                            # 여기서는 "존재하는 수종"에 대해서만 마스크 생성 및 경로 저장
                            mask_paths = []
                            
                            # 답변에서 [SEG]가 나오는 순서와 mask_paths 리스트 순서가 일치해야 학습이 됨.
                            # emini가 (침엽수 -> 활엽수) 순서로 말하게 유도하거나, 
                            # 우리가 코드로 먼저 마스크를 다 만들어놓고, Gemini 답변을 파싱해서 순서를 맞추는 게 정석.
                            # 하지만 간단하게 하기 위해:
                            # 항상 "침엽수" 마스크 먼저 처리, "활엽수" 마스크 나중 처리.
                            # Gemini에게도 "침엽수 먼저 언급하고 활엽수 언급해라"고 지시
                            
                            # 마스크 생성 루프 (Conifer -> Broadleaf 순서 고정)
                            ordered_species = ["Conifer", "Broadleaf"]
                            final_mask_list = []
                            
                            for sp_name in ordered_species:
                                target_boxes = species_groups[sp_name]
                                if not target_boxes: continue # 해당 수종 없으면 패스
                                
                                input_boxes = torch.tensor(target_boxes, device=device)
                                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img_tile.shape[:2])
                                combined_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                                
                                for i in range(0, len(target_boxes), SAM_BATCH_SIZE):
                                    batch = transformed_boxes[i:i+SAM_BATCH_SIZE]
                                    if len(batch) > 0:
                                        masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=batch, multimask_output=False)
                                        merged = torch.max(masks, dim=0)[0]
                                        combined_mask = np.maximum(combined_mask, (merged[0].cpu().numpy() > 0).astype(np.uint8) * 255)
                                        del masks, merged; torch.cuda.empty_cache()
                                
                                mask_filename = f"mask_{tile_id}_tile{global_tile_count}_{sp_name}.png"
                                cv2.imwrite(os.path.join(OUTPUT_PATH, "masks", mask_filename), combined_mask)
                                final_mask_list.append(f"masks/{mask_filename}")

                            # 3. Gemini Q&A (Level-2)
                            qa = generate_l2_qa(species_counts)
                            
                            # 만약 Gemini 답변 순서가 틀리면 학습 꼬임.
                            # 안전장치: Gemini가 답변에서 [SEG]를 몇 개 썼는지 확인하고, 마스크 개수와 비교
                            ans_text = qa['answer']
                            seg_count = ans_text.count("[SEG]")
                            
                            if seg_count != len(final_mask_list):
                                print(f": Text has {seg_count} SEGs, but created {len(final_mask_list)} masks. Skipping.")
                                # 복잡한 매칭 로직 대신 스킵 (데이터 품질 위해)
                                continue

                            # JSON 저장
                            l2_entry = {
                                "id": f"{tile_id}_tile{global_tile_count}_L2",
                                "image": tile_filename,
                                "conversations": [
                                    {"from": "human", "value": f"{qa['question']}\n<image>"},
                                    {"from": "gpt", "value": qa['answer']}
                                ],
                                "mask_path": final_mask_list  # 리스트 형태 (Level-1 데이터와의 차이점)
                            }
                            l2_results.append(l2_entry)
                            
                            created_count += 1
                            global_tile_count += 1
                            print(f"Created L2 Tile {created_count}/{TARGET_TILE_COUNT}")

                            if len(l2_results) % 5 == 0:
                                with open(os.path.join(OUTPUT_PATH, "l2_dataset.json"), 'w', encoding='utf-8') as f:
                                    json.dump(l2_results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing {tile_id}: {e}")
            continue
        finally:
            if os.path.exists(tif_path): os.remove(tif_path)

    with open(os.path.join(OUTPUT_PATH, "l2_dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(l2_results, f, indent=4, ensure_ascii=False)
    
    print(f"Level-2 Generation Complete! {len(l2_results)} entries created.")

if __name__ == "__main__":
    process_l2_dataset()