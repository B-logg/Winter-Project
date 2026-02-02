import json
import os
import ast
import time
import requests
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from segment_anything_hq import sam_model_registry, SamPredictor


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

GEMINI_API_KEY = os.getenv("Gemini_API_KEY")
if not GEMINI_API_KEY: raise ValueError("API Key Missing")

genai.configure(api_key=GEMINI_API_KEY)
CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv")

# 결과 저장 경로
OUTPUT_L1_PATH = os.path.join(CURRENT_DIR, "l1_dataset_neon")
os.makedirs(os.path.join(OUTPUT_L1_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_L1_PATH, "masks"), exist_ok=True)

# 모델 설정
SAM_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "sam_hq_vit_l.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
SAM_BATCH_SIZE = 64

# Gemini 설정
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except:
        return []

def download_image(url, save_path):
    if os.path.exists(save_path): return True
    try:
        response = requests.get(url, stream=True, timeout=20)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        return False
    return False


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
            "answer": f"네, 이미지에 있는 {korean_name} [SEG]입니다."
        }

def generate_l1_neon_final():
    print("Loading NEON CSV Data...")
    if not os.path.exists(CSV_PATH):
        print("CSV file not found.")
        return
        
    df = pd.read_csv(CSV_PATH)

    # L3에서 사용하지 않은 데이터 구간 설정 (1000 ~ 2000)
    start_idx = 1000
    end_idx = 1005
    df_target = df.iloc[start_idx:end_idx]
    
    print(f"Target Dataset: {len(df_target)} tiles (Index {start_idx}~{end_idx})")

    print(f"Initializing SAM on {device}...")
    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    l1_results = []
    
    for idx, row in tqdm(df_target.iterrows(), total=len(df_target)):
        tile_id = row['tile_id']
        img_url = row['rgb_url']
        
        # 1. 이미지 다운로드
        img_filename = f"{tile_id}.jpg"
        img_path = os.path.join(OUTPUT_L1_PATH, "images", img_filename)
        
        if not download_image(img_url, img_path):
            continue

        try:
            # 2. 이미지 로드
            image_cv = cv2.imread(img_path)
            if image_cv is None: continue
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            h, w = image_cv.shape[:2]
            
            # 3. 데이터 파싱
            bboxes = safe_eval(row['individual_bboxes'])
            tree_types = safe_eval(row.get('individual_tree_types', '[]'))
            
            if not bboxes or not tree_types or len(bboxes) != len(tree_types):
                continue

            # 4. 수종별 박스 분류
            # CSV에 이미 'Conifer', 'Broadleaf'로 적혀 있으므로 그대로 키로 사용
            species_groups = {"Conifer": [], "Broadleaf": []}
            
            for bbox, t_type in zip(bboxes, tree_types):
                # 공백 제거 및 혹시 모를 대소문자 통일
                clean_type = t_type.strip() # "Conifer" or "Broadleaf"
                
                if clean_type in species_groups:
                    species_groups[clean_type].append(bbox)
            
            # 5. SAM 이미지 인코딩
            predictor.set_image(image_cv)
            
            # 수종별 데이터 생성
            for species, target_boxes in species_groups.items():
                if not target_boxes: continue 
                
                # SAM 추론
                input_boxes = torch.tensor(target_boxes, device=device)
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, (h, w))
                
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                
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
                mask_filename = f"mask_{tile_id}_{species}.png"
                cv2.imwrite(os.path.join(OUTPUT_L1_PATH, "masks", mask_filename), combined_mask)
                
                # Gemini 질문 생성
                qa_pair = generate_dynamic_qa(species, len(target_boxes))
                
                # JSON 추가
                l1_entry = {
                    "id": f"{tile_id}_{species}",
                    "image": img_filename,
                    "mask_path": f"masks/{mask_filename}",
                    "conversations": [
                        {"from": "human", "value": f"{qa_pair['question']}\n<image>"},
                        {"from": "gpt", "value": qa_pair['answer']}
                    ],
                    "metadata": {
                        "species": species,
                        "count": len(target_boxes),
                        "source": "NEON_CSV"
                    }
                }
                l1_results.append(l1_entry)
                time.sleep(0.5) 
            
            # 중간 저장
            if len(l1_results) % 50 == 0:
                with open(os.path.join(OUTPUT_L1_PATH, "l1_dataset.json"), 'w', encoding='utf-8') as f:
                    json.dump(l1_results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error {tile_id}: {e}")
            continue

    # 최종 저장
    with open(os.path.join(OUTPUT_L1_PATH, "l1_dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(l1_results, f, indent=4, ensure_ascii=False)
    
    print(f"NEON L1 Dataset Created! ({len(l1_results)} items)")

if __name__ == "__main__":
    generate_l1_neon_final()