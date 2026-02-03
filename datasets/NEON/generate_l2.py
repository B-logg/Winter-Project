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
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai 
from segment_anything_hq import sam_model_registry, SamPredictor


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")

if not GEMINI_API_KEY: raise ValueError("API Key Missing")

client = genai.Client(api_key=GEMINI_API_KEY)

# 경로 설정
CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "l2_dataset")
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")

os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)

# 파라미터
NEON_PRODUCT_ID = "DP3.30010.001"
TILE_SIZE = 1024
MIN_TREE_THRESHOLD = 3
SAM_BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# 테스트: 타일 5장/실전: 타일 1000장
TARGET_TILE_COUNT = 5 

# site: 지역 코드, year: 관찰년도, save_dir: 저장할 폴더 경로
def download_neon_image(site, year, tile_id, save_dir):
    filename = f"{tile_id}.tif" # 저장할 파일 이름
    save_path = os.path.join(save_dir, filename) # 저장할 경로
    # 저장할 경로에 이미 파일이 있는지(이미 타일을 다운받았는지), 다운받은 파일 크기가 1MB 보다 크다면, 저장경로를 반환하고 함수 종료
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024*1024: return save_path
    try:
        parts = tile_id.split('_') # 언더바(_) 기준으로 분리해서 site, year을 추출
        safe_year, safe_site = parts[0], parts[1]
        # NEON API 서버에 이 데이터 제품(NEON_PRODUCT_ID)에 대한 정보를 달라고 요청
        r = requests.get(f"https://data.neonscience.org/api/v0/products/{NEON_PRODUCT_ID}")
        if r.status_code != 200: return None # 성공 코드(200)이 아니면 None 반환하고 함수 종료
        data = r.json() # API가 보내준 응답(json)을 추출 
        # 받아온 데이터 중에서 우리가 찾는 사이트(예: GRSM, safe_site)에 대한 정보만 찾아서 site_info에 저장
        site_info = next((s for s in data['data']['siteCodes'] if s['siteCode'] == safe_site), None)
        if not site_info: return None # 못 찾으면 None 반환

        # 찾아온 사이트에서 데이터가 존재하는 모든 월(Month) 중에서, 우리가 원하는 연도로 시작하는 달만 골라냄
        available_months = [m for m in site_info['availableMonths'] if m.startswith(str(safe_year))]
        file_url = None
        for month in sorted(available_months, reverse=True): # 가능한 달 목록을 최신순으로 정렬해서 하나씩 확인
            # 해당 연도-월의 구체적인 파일 목록을 API에게 요청
            url = f"https://data.neonscience.org/api/v0/data/{NEON_PRODUCT_ID}/{safe_site}/{month}"
            r_files = requests.get(url).json()

            if not isinstance(r_files.get('data'), dict): continue
            if 'files' not in r_files['data']: continue

            for f in r_files['data']['files']:
                if f['name'] == filename:
                    file_url = f['url']
                    break
            if file_url: break
        if not file_url: return None
            
        print(f"Downloading GRSM Map: {filename}...")
        with requests.get(file_url, stream=True) as r: # streaming으로 다운로드 요청
            r.raise_for_status() # 에러 잡기
            with open(save_path, 'wb') as f: # 저장할 경로에 쓰기 모드로 열기
                # 데이터를 청크 단위(8192byte)씩 잘라서 받아오고, 그걸 파일에 쓰기
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk) 
        return save_path
    except Exception as e: 
        return None

def normalize_image(img_array):
    img_min, img_max = img_array.min(), img_array.max()
    if img_max > img_min: img_norm = (img_array - img_min) / (img_max - img_min)
    else: img_norm = img_array
    return (img_norm * 255).astype(np.uint8)

def get_species_category(species_name):
    species_name = species_name.lower()
    conifer_keywords = ['conifer']
    if any(k in species_name for k in conifer_keywords): return "Conifer"
    return "Broadleaf"

def filter_trees_in_tile(src, window, row_data):
    filtered_boxes = []
    filtered_species = []
    def safe_parse(key):
        if key not in row_data: return []
        val = row_data[key]
        try: return eval(val) if isinstance(val, str) else val
        except: return []
    bboxes = safe_parse('individual_bboxes')
    if not bboxes: bboxes = safe_parse('bboxes')
    tree_types = safe_parse('individual_tree_types')
    if len(bboxes) == 0 or len(bboxes) != len(tree_types): return [], []
    win_col_off, win_row_off = window.col_off, window.row_off
    win_w, win_h = window.width, window.height
    for i, utm_box in enumerate(bboxes):
        try:
            row_tl, col_tl = src.index(utm_box[0], utm_box[3])
            row_br, col_br = src.index(utm_box[2], utm_box[1])
            center_row, center_col = (row_tl + row_br) / 2, (col_tl + col_br) / 2
            if (win_row_off <= center_row < win_row_off + win_h) and (win_col_off <= center_col < win_col_off + win_w):
                rel_x1 = max(0, col_tl - win_col_off)
                rel_y1 = max(0, row_tl - win_row_off)
                rel_x2 = min(win_w, col_br - win_col_off)
                rel_y2 = min(win_h, row_br - win_row_off)
                if rel_x2 - rel_x1 > 2 and rel_y2 - rel_y1 > 2:
                    filtered_boxes.append([rel_x1, rel_y1, rel_x2, rel_y2])
                    filtered_species.append(get_species_category(tree_types[i]))
        except: continue
    return filtered_boxes, filtered_species

 # species_counts: {'Conifer': 5, 'Broadleaf': 3} 형태의 딕셔너리
def generate_l2_qa(species_counts):
    has_conifer = species_counts.get('Conifer', 0) > 0
    has_broadleaf = species_counts.get('Broadleaf', 0) > 0
    
    # 상황별 프롬프트 구성
    if has_conifer and has_broadleaf:
        situation = "침엽수와 활엽수가 섞여 있는 혼효림(Mixed Forest)"
        instruction = "두 수종의 공존 관계나 위치를 설명하고, 두 수종이 공존함으로써 산림에 어떤 영향을 주는지 설명하세요."
        example_ans = "이 숲에는 침엽수 [SEG]와 활엽수 [SEG]가 함께 자라고 있고, 이러한 혼효림은 숲 생태계에서 침엽수 [SEG]와 활엽수 [SEG]의 비율이 3대 7일 때 가장 생태적으로 안정적이고, 생물 다양성도 높게 유지된다."
    elif has_conifer:
        situation = "침엽수(Conifer)만 있는 단순림"
        instruction = "침엽수 군락만 존재함을 강조하고, 침엽수 군락이 산림에게 주는 영향을 설명하세요"
        example_ans = "이 구역은 주로 침엽수 [SEG] 군락으로 이루어져있고, 자연적인 기후 안정화와 강우량 조절 역할을 하고, 다양한 새와 작은 동물들에게 쉼터와 먹이(열매)를 제공하여 생태계의 균형을 유지한다.."
    elif has_broadleaf:
        situation = "활엽수(Broadleaf)만 있는 단순림"
        instruction = "활엽수 군락만 존재함을 강조하세요, 활엽수 군락이 산림에게 주는 영향을 설명하세요."
        example_ans = "여기 보이는 나무들은 대부분 활엽수 [SEG] 군락으로 이루어져있고, 활엽수는 [SEG] 수분 함유량이 높고 잎이 넓어, 산불 발생 시 불길이 번지는 것을 막아주는 자연 방화림 역할을 한다."
    else:
        return None

    prompt = f"""
    역할: 산림 생태 전문가 (VLM 학습 데이터 생성기)
    상황: {situation}
    
    임무: 
    사용자가 숲의 식생 구성이나 관계를 물어볼 때, 이미지 속 객체를 지칭하며 답변하는 JSON을 생성하세요.
    
    [규칙]
    1. 답변에서 언급된 수종(침엽수 or 활엽수) 뒤에는 반드시 [SEG] 토큰을 붙여야 합니다.
    2. 침엽수, 활엽수 언급 순서는 무조건 침엽수를 먼저 언급하고, 이후에 활엽수를 언급하고, 실제 존재하는 수종만 언급하세요.
    3. 질문은 "이 숲의 구성은?", "어떤 나무들이 있어?", "나무들의 관계를 설명해줘" 등 다양하게.
    4. 질문과 답변은 한국어로 1~2 문장으로 해주고, 최대한 예시 답변 패턴, 예시 정보 처럼 산림에 대한 다양한 정보를 추가해줘.
    5. 답변은 처음에 이미지에서 보이는 침엽수, 활엽수의 구성이나 관계, 위치를 설명하고, 이후에 산림 전체와 침엽수, 활엽수의 관계를 설명하고, 추가적인 설명이 들어가도록 해줘.
    
    [예시 답변 패턴]
    - "{example_ans}"

    [언급 정보 예시]
    - 활엽수 군락: 토양의 비옥화, 높은 생물 다양성, 탄소 흡수 능력, 수분 함량 등 ...
    - 침엽수 군락: 산림의 기본 구조(한국 산림), 피톤치드 및 산림 치유, 녹색댐 기능, 고유 생태계 유지 등 ...
    - 취약성: 침엽수(소나무 등)는 고온, 가뭄, 해충 피해에 취약, 기후 변화로 집단 고사하는 등 생존 위협을 받고 있음, 뿌리가 얕은 침엽수는 깊게 내리는 활엽수에 비해 산사태에 상대적으로 취약하다 등

    Format: {{"question": "...", "answer": "..."}}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt,
            config={'response_mime_type': 'application/json'})
        
        text = response.text.strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            text = text[start_idx : end_idx + 1]
        
        return json.loads(text)
    
    except Exception as e:
        print(f"Error: {e}")
        return {
            "question": "두 수종의 공존 관계나 위치를 설명하고, 두 수종이 공존함으로써 산림에 어떤 영향을 주는지 설명하세요.",
            "answer": "이 숲에는 침엽수 [SEG]와 활엽수 [SEG]가 함께 자라고 있고, 이러한 혼효림은 숲 생태계에서 침엽수 [SEG]와 활엽수 [SEG]의 비율이 3대 7일 때 가장 생태적으로 안정적이고, 생물 다양성도 높게 유지된다."
        }
    
def align_masks_with_text(text, mask_dict):
    """
    텍스트 내의 [SEG] 토큰을 찾고, 그 앞 단어를 분석해 올바른 마스크 경로를 리스트로 반환
    text: "여기에 침엽수 [SEG]가 있고, 활엽수 [SEG]가 있다."
    mask_dict: {'Conifer': 'path/to/c.png', 'Broadleaf': 'path/to/b.png'}
    return: ['path/to/c.png', 'path/to/b.png']
    """
    ordered_masks = []
    
    # [SEG] 토큰 기준으로 문장을 쪼개서 앞부분을 분석
    # 예: "여기에 침엽수 [SEG]" -> "여기에 침엽수 " 분석 -> "침엽수" 있음 -> Conifer 마스크 추가
    
    segments = text.split('[SEG]')
    # 마지막 조각은 [SEG] 뒤에 오는 말이므로 제외 (토큰 개수 = split 개수 - 1)
    
    if len(segments) <= 1:
        return [] # [SEG]가 없음

    for i in range(len(segments) - 1):
        chunk = segments[i] # [SEG] 앞부분 텍스트
        
        # 간단한 키워드 매칭 (뒤에서부터 검색하는게 더 정확할 수 있지만 여기선 단순 포함 여부)
        # 텍스트 덩어리의 뒤쪽 10글자만 봐도 충분함
        recent_chunk = chunk[-15:] 
        
        if "침엽수" in recent_chunk or "Conifer" in recent_chunk:
            if 'Conifer' in mask_dict:
                ordered_masks.append(mask_dict['Conifer'])
            else:
                # 텍스트엔 있는데 마스크가 없으면? 패스
                pass
        elif "활엽수" in recent_chunk or "Broadleaf" in recent_chunk:
            if 'Broadleaf' in mask_dict:
                ordered_masks.append(mask_dict['Broadleaf'])
            else:
                pass
        else:
            # 침엽수/활엽수란 말 없이 [SEG]만 쓴 경우 패스
            pass         
    return ordered_masks

def process_l2_dataset():
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)
    
    # GRSM 필터링 (NEON_dataset.csv에서 GRSM만 추출)
    df_grsm = df[df['site'] == 'GRSM']
    print(f"GRSM Maps: {len(df_grsm)}")
    
    if len(df_grsm) == 0: return

    # SAM 설정
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    temp_dir = os.path.join(OUTPUT_PATH, "temp_tif")
    os.makedirs(temp_dir, exist_ok=True)
    
    global_tile_count = 0 
    created_count = 0     
    l2_results = []

    print(f"Target: Generate {TARGET_TILE_COUNT} Level-2 Samples.")

    for idx, row in tqdm(df_grsm.iterrows(), total=len(df_grsm), desc="Processing GRSM"):
        if created_count >= TARGET_TILE_COUNT: break
        
        tile_id = row['tile_id']
        site, year = row['site'], row['year']
        
        tif_path = download_neon_image(site, year, tile_id, temp_dir)
        if not tif_path: continue
        
        try:
            with rasterio.open(tif_path) as src:
                h_img, w_img = src.shape
                
                # 타일링
                for row_off in range(0, h_img, TILE_SIZE):
                    for col_off in range(0, w_img, TILE_SIZE):
                        if created_count >= TARGET_TILE_COUNT: break
                        
                        width = min(TILE_SIZE, w_img - col_off)
                        height = min(TILE_SIZE, h_img - row_off)
                        window = Window(col_off, row_off, width, height)
                        
                        boxes, species_list = filter_trees_in_tile(src, window, row)
                        
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

                            available_mask_paths = {}
                                                 
                            for sp_name, target_boxes in species_groups.items():
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
                                
                                available_mask_paths[sp_name] = f"masks/{mask_filename}"
                            
                            if not available_mask_paths: continue

                            # 3. Gemini Q&A (Level-2)
                            species_counts = {sp: len(boxes) for sp, boxes in species_groups.items()}
                            qa = generate_l2_qa(species_counts)

                            ordered_mask_list = align_masks_with_text(qa['answer'], available_mask_paths)

                            seg_count_in_text = qa['answer'].count("[SEG]")

                            if len(ordered_mask_list) == 0:
                                print(f"Alignment Failed: {qa['answer']}")
                                continue
                            
                            if len(ordered_mask_list) != seg_count_in_text:
                                continue

                            # JSON 저장
                            l2_entry = {
                                "id": f"{tile_id}_tile{global_tile_count}_L2",
                                "image": tile_filename,
                                "conversations": [
                                    {"from": "human", "value": f"{qa['question']}\n<image>"},
                                    {"from": "gpt", "value": qa['answer']}
                                ],
                                "mask_path": ordered_mask_list
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