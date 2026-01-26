import google.generativeai as genai
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor

# Gemini API 설정 및 Mobile SAM 설정
def init_models(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') # 정교한 CoT를 위해 pro 모델 사용

    # MobileSAM
    model_type = "vit_t"
    checkpoint = "/Users/bosung/Desktop/GLaMM/datasets/checkpoints/model_sam.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SamPredictor(sam_model_registry[model_type](checkpoint=checkpoint).to(device))
    return model, predictor

def generate_l3(image_dir, output_dir, l2_data_sample, prompt, model, predictor):
    img_path = os.path.join(image_dir, l2_data_sample['image'])
    pil_img = Image.open(img_path)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    predictor.set_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    h, w = img_cv.shape[:2]

    # l2_data_sample에서 프롬프트 추출
    prompt = l2_data_sample['prompt']

    try:
        response = model.generate_content([prompt, pil_img])
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        res_json = json.loads(clean_text)

        # 마스크 생성
        l3_masks = []
        for i, (seg_tag, coords) in enumerate(res_json['objects'].items()):
            # 좌표 변환(0~1000 -> 이미지 픽셀)
            px, py = int(coords[0] * w / 1000), int(coords[1] * h / 1000)
            masks, _, _ = predictor.predict(np.array([[px, py]]), np.array([1]), multimask_output=False)

            m_filename = f"l3_{l2_data_sample['id']}_seg{i+1}.png"
            m_path = os.path.join(output_dir, "masks/l3", m_filename)
            cv2.imwrite(m_path, (masks[0] * 255).astype(np.uint8))
            l3_masks.append(m_path)
        
        # Level-3 JSON 구성
        l3_entry = {
            "id": f"L3_{l2_data_sample['id']}",
            "image": l2_data_sample['image'],
            "conversations": [
                {
                    "from": "human",
                    "value": l2_data_sample['human_query']
                },
                {
                    "from": "gpt",
                    "value": res_json['dense_caption']
                }
            ],
            "masks": l3_masks,
            "scene_graph_query": l2_data_sample['landmark'] # 어떤 정보를 넣을지 고민 중(수정 필요)
        }
        return l3_entry
    
    except Exception as e:
        print(f"L3 생성 실패: {e}")
        return None
