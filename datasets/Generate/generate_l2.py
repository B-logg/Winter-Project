import google.generativeai as genai
import torch
from mobile_sam import sam_model_registry, SamPredictor
import cv2
import numpy as np
import json
import os
from PIL import Image

# Gemini API 설정
genai.configure(api_key="GEMINI_API_KEY")
model = model.GenerativeModel('gemini-1.5-flash')

# MobileSAM
model_type = "vit_t"
checkpoint = "checkpoints/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
predictor = SamPredictor(sam)

# Gemini 모델을 사용하여 설명 생성 + SAM 마스크 생성 + Level-2 데이터 생성
def generate_l2(image_dir, output_dir, plant_metadata):

    os.makedirs(os.path.join(output_dir, "masks/l2")), exist_ok=True
    l2_dataset = []

    for info in plant_metadata:
        img_path = os.path.join(image_dir, info['file_name'])
        pil_img = Image.open(img_path)
        img_cv = cv2.cvtColor(np.arryay(pil_img), cv2.COLOR_RGB2BGR)
        predictor.set_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        h, w = img_cv.shape[:2]

        # Gemini 프롬프트
        prompt = f"""
        이 이미지의 전체 장면을 분석해서 한 문장을 생성해줘.
        식물 종 이름은 '{info['label']}'이야.

        Task:
        1. 이미지 내 객체들 간의 '관계'를 설명하는 짧은 캡션를 생성해줘.
            예: "A는 B 위에 있다."  
        2. 각 캡션에서 핵심 물리 객체를 추출하고, 그 객체들의 중심좌표 [x, y] (0~1000 상대좌표)를 찾아줘.
        3. 이 장면의 랜드마크 카테고리(Primary, Sub-category)를 정해줘.
        
        출력 형식(JSON):
        {{
            "caption": [
                {{"text": "마른 토양 [SEG] 위에 {info['label']} [SEG]가 서 있습니다.", "objects": [[x1, y1], [x2, y2]]}},
                {{"text": "푸른 하늘 [SEG] 아래로 {info['label']} [SEG]의 수관이 펼쳐져 있습니다.", "objects": [[x3, y3], [x4, y4]]}}
            ],
            "landmark": {{"primary": "산림", "sub_category": "침엽수림"}}
        }}
        """
        try:
            response = model.generate_content([prompt, pil_img])
            # JSON 응답 정제
            res_text = response.text.replace('```json', '').replace('```', '').strip()
            res_json = json.load(res_text)

            total_masks = []
            combined_caption = " ".join([c['text'] for c in res_json['captions']])

            # 캡션의 모든 객체들에 대해 새로운 마스크 생성(SAM)
            seg_count = 1
            for cap in res_json['captions']:
                for coords in cap['objects']:
                    # 좌표 변환 및 SAM 실행
                    px, py = int(coords[0] * w / 1000), int(coords[1] * h / 1000)
                    masks, _, _ = predictor.predict(np.array([[px, py]]), np.array([1]), multimask_output=False)

                    m_filename = f"l2_{info['image_id']}_seg{seg_count}.png"
                    cv2.imwrite(os.path.join(output_dir, "masks/l2", m_filename), (masks[0]*255).astype(np.uint8))
                    total_masks.append(os.path.join("masks/l2", m_filename))
                    seg_count += 1
            
            # GLaMM Level-2 표준 포맷
            l2_entry = {
                "id": f"L2_{info['image_id']}",
                "image": info['file_name'],
                "conversations": [
                    {
                        "from": "human",
                        "value": "이 장면의 구성 요소들과 그들 사이의 관계를 자세히 설명하고, 이곳이 어디를 가리키는지 알려줘."
                    },
                    {
                        "from": "gpt",
                        "value": f"{combined_caption} 이곳의 주요 카테고리는 {res_json['landmark']['primary']}이며, 구체적으로는 {res_json['landmark']['sub_category']}입니다."
                    }
                ],
                "masks": total_masks, # 캡션 순서에 따른 모든 마스크 리스트
                "landmark": res_json['landmark']
            }
            l2_dataset.append(l2_entry)
        except Exception as e:
            print(f"Error processing {info['image_id']}: {e}")
    
    with open(os.path.join(output_dir, "l2_train.json"), "w", encoding="utf-8") as f:
        json.dump(l2_dataset, f, ensure_ascii=False, indent=2)

    