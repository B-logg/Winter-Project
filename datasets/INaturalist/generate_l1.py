import torch
import cv2
import numpy as np
import json
import os
import random
import requests
import google.generativeai as genai
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict, annotate
from tqdm import tqdm



GEMINI_API_KEY = os.getenv("Gemini_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 경로(상대)

# 입력 파일 경로
INPUT_JSON_PATH = os.path.join(CURRENT_DIR, "l1_prepare_data.json")

# 출력 파일 경로
OUTPUT_DIR = os.path.join(CURRENT_DIR, "l1_dataset")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
JSON_DIR = os.path.join(OUTPUT_DIR, "l1_train.json")

# SAM ViT-H 가중치 경로
CHECKPOINT_ROOT = os.path.dirname(CURRENT_DIR)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, "checkpoints", "sam_vit_h_4b8939.pth")

CONFIG_PATH = os.path.join(CHECKPOINT_ROOT, "checkpoints" , "GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(CHECKPOINT_ROOT, "checkpoints", "groundingdino_swint_ogc.pth")

grounding_dino_model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# 결과 저장 디렉토리 생성
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# SAM-H 설정
print("SAM ViT-H 불러오는 중")
try:
    sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    predictor = SamPredictor(sam)
except Exception as e:
    print(f"모델 로드 중 에러 발생: {e}")
    exit()

TREE_TYPE = {
    "Conifer (침엽수)" : "침엽수",
    "Broad-leaf (활엽수)" : "활엽수",
    "Ginkgo (은행나무)" : "은행나무"
}

LEVEL1_QUESTIONS = [
    "이 이미지는 어떤 나무인가요?",
    "사진 속 나무의 종류를 알려주세요.",
    "이 사진에 있는 식물은 무엇입니까?",
    "이미지 속 대상이 무엇인지 식별해 주세요.",
    "이 나무가 침엽수인지 활엽수인지 알려주세요.",
    "이것은 무슨 나무인가요?"
]

# 데이터 생성에 사용할 이미지 다운로드 함수
def download_image(url, save_path):
    if os.path.exists(save_path):
        return True
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            # 이미지 깨짐 확인
            img = cv2.imread(save_path)
            if img is None:
                os.remove(save_path)
                return False
            return True
    except Exception:
        return False
    return False

def get_grounding_box(image_path, text_prompt="tree"):
    
    # Grounding DINO를 사용하여 나무에 해당하는 Bounding Box를 찾습니다.

    IMAGE_SOURCE, image = load_image(image_path)
    
    # 박스 예측 (Box Threshold: 0.35, Text Threshold: 0.25)
    boxes, logits, phrases = predict(
        model=grounding_dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
        device=device
    )
    
    # 박스가 없으면 None 반환
    if len(boxes) == 0:
        return None

    # 가장 신뢰도(Logit)가 높은 박스 하나만 선택
    max_idx = torch.argmax(logits)
    best_box = boxes[max_idx] # (cx, cy, w, h) - normalized 0~1

    # SAM 입력을 위해 좌표 변환 (Normalized -> Absolute XYXY)
    h, w, _ = IMAGE_SOURCE.shape
    best_box = best_box * torch.Tensor([w, h, w, h])
    best_box[:2] -= best_box[2:] / 2
    best_box[2:] += best_box[:2]
    
    return best_box.numpy() # [x_min, y_min, x_max, y_max]



def get_gemini_answer(question, tree_label):
    # 이미지와 질문을 받아 Gemini가 답변을 생성합니다.
    # [SEG] 토큰은 지칭하는 단어 '뒤'에 붙도록 유도합니다.

    try:
        # 프롬프트: 답변만 생성하도록 지시
        prompt = f"""
        User Question: "{question}"
        Image Context: This is a '{tree_label}'.

        Examples:
        - "사진 속 나무 [SEG]는 침엽수입니다."
        - "이것 [SEG]은 활엽수입니다."
        - "이 이미지는 은행나무 [SEG]를 보여줍니다."
        - "이 나무 [SEG]는 활엽수입니다."
        - "이미지에 있는 나무 [SEG]는 침엽수 입니다."
        
        Task: Generate an answer as in the example, referring to the 'Image Context'.
        
        [CONSTRAINT - IMPORTANT]
        When you refer to the tree object
        (using words like '나무', '이것', '이 나무', '이미지에 있는 나무' or the specific name '{tree_label}'), 
        you MUST place the token '[SEG]' immediately AFTER that word like 'Examples'.
        
        Output: Only one sentence in Korean. No JSON, no explanations.
        """
        
        response = gemini_model.generate_content(prompt)
        answer_text = response.text.strip()
        
        # 만약 Gemini가 [SEG]를 안 붙였으면 강제로 답변 하드 코딩(안전 장치)
        if "[SEG]" not in answer_text:
            answer_text = f"이미지 속 나무 [SEG]는 {tree_label}입니다."
            
        return answer_text

    except Exception as e:
        print(f"에러 발생: {e}")
        return f"사진 속 나무 [SEG]는 {tree_label}입니다."

def process_data():
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"파일 없음: {INPUT_JSON_PATH}")
        return

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    l1_dataset = []
    global_idx = 0

    print(f"처리 시작: 총 {len(raw_data)}개 그룹")

    for entry in tqdm(raw_data):
        tree_type_raw = entry.get('tree_type', 'None')
        if tree_type_raw not in TREE_TYPE: continue
            
        tree_label = TREE_TYPE[tree_type_raw] # 침엽수/활엽수/은행나무
        image_urls = entry.get('image_urls', [])

        for url in image_urls:
            img_id = url.split('/')[-2]
            file_name = f"{img_id}_{global_idx}.jpg"
            img_save_path = os.path.join(IMAGE_DIR, file_name)

            # 이미지 다운로드
            if not download_image(url, img_save_path): continue

            # Grounding DINO로 Bounding Box 찾기 -> SAM이 Mask를 정확하게 그릴 수 있도록 힌트 제공
            box = get_grounding_box(img_save_path, text_prompt="tree")

            if box is None:
                continue
            
            # 이미지 로드 (SAM이 마스크 그릴 준비)
            image = cv2.imread(img_save_path)
            if image is None: continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                # SAM 마스크 생성
                predictor.set_image(image_rgb)

                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True,
                )
                
                best_mask = masks[np.argmax(scores)]
                
                # 마스크 저장
                mask_filename = f"{img_id}_{global_idx}_mask.png"
                cv2.imwrite(os.path.join(MASK_DIR, mask_filename), (best_mask * 255).astype(np.uint8))

                # 질문 선택 및 Gemini 답변 생성
                selected_question = random.choice(LEVEL1_QUESTIONS)
                
                # Gemini 호출
                gemini_answer = get_gemini_answer(selected_question, tree_label)

                # JSON 데이터 구성
                l1_entry = {
                    "id": f"{img_id}_{global_idx}",
                    "image": file_name,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{selected_question}" # 이미지 토큰 + 하드코딩 질문
                        },
                        {
                            "from": "gpt",
                            "value": gemini_answer # Gemini가 생성한 [SEG] 포함 답변
                        }
                    ],
                    "mask_path": f"masks/{mask_filename}", 
                    "category": tree_label
                }
                l1_dataset.append(l1_entry)
                global_idx += 1
            
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            # 중간 저장 (50개 단위)
            if global_idx % 50 == 0:
                with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
                    json.dump(l1_dataset, f, ensure_ascii=False, indent=2)

    # 최종 저장
    
    with open(f"{OUTPUT_DIR}/result.json", "w", encoding="utf-8") as f:
        json.dump(l1_dataset, f, ensure_ascii=False, indent=2)

    print(f"완료! 총 {len(l1_dataset)}개의 데이터 생성.")
    print(f"저장 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_data()
