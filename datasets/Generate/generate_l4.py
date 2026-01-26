import google.generativeai as genai
import torch
from mobile_sam import sam_model_registry, SamPredictor
import cv2
import numpy as np
import json
import os
from PIL import Image

# Gemini API 설정
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-pro')

# MobileSAM
model_type = "vit_t"
checkpoint = "checkpoints/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
predictor = SamPredictor(sam)

def generate_l4(image_path, l3_data_sample, tree_metadata):
    img_pil = Image.open(image_path)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    predictor.set_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    h, w = img_cv.shape[:2]

    # 전문가 지식 및 3종 공식 주입 프롬프트
    prompt = f"""
    당신은 산림 탄소 전문가입니다. 제공된 이미지와 데이터를 바탕으로 기본적인 객체 식별 및 관계를 넘어,
    랜드마크에 대한 세부 정보, 역사적 문맥, 장면과의 상호작용 지침, 그리고 미래 사건에 대한 예측요소까지 
    포함하는 확장된 문맥 통찰 데이터를 생성하세요.

    {l3_data_sample['dense_caption']}을 참고하세요.
    
    [필수 포함 공식 3종]
    1. 면적 기반 간편 공식: 
       - 연간 탄소 흡수량 = 수관 면적(m2) * 0.28 kg C / m2 / 년
       - 총 탄소 저장량 = 수관 면적(m2) * 7.69 kg C / m2
       - CO2 환산 = 탄소량(C) * 3.67
    2. 개체별 정밀 공식 (생장 기반):
       - 지상부 바이오매스(AGB) = 0.25 * D^2 * H (D:직경, H:높이)
       - 총 바이오매스(TB) = AGB * 1.2
       - 건조 중량 = TB * 0.725, 탄소 함량 = 건조 중량 * 0.5
    3. 국립산림과학원 표준 공식:
       - CO2 흡수량 = Vol * WD * BEF * (1 + R) * CF * (44/12)
       - 계수: VoI(재적 생장량, m3), WD(목재 기본 밀도), BEF(바이오매스 확장계수), R(뿌리 함량비), CF(탄소 전환계수 0.5)

    [입력 데이터]
    - 수종: {tree_metadata['label']}
    - 측정치: DBH {tree_metadata.get('diameter', 0)}cm, 수고 {tree_metadata.get('height', 0)}m, 수관 면적 {tree_metadata.get('area', 0)}m2
    - 관련 계수: {tree_metadata.get('params', {{}})}

    [태스크]
    1. 위 공식을 사용하여 탄소 흡수량을 논리적으로 추론하는 답변을 작성하세요.
    2. Chain-of-Thought 방식을 사용하여 추론 과정을 단계별로 설명한 후에 최종 답변(추론한 탄소 흡수량)을 제시하세요.
    3. 답변의 근거가 되는 '시각적 지점'을 선정하고, 시각적 지점에 대한 단어 뒤에 [SEG] 토큰을 붙이세요. 
        (예: 마른 토양 [SEG]으로 인해 탄소흡수율이 15% 감소할 것으로 예상됩니다.
        (예: 굵은 줄기 [SEG]는 넓은 수관면적으로 이어지므로 해당 나무 [SEG]의 탄소흡수율이 연간 100kg CO2에 달할 것으로 예상됩니다.)
    4. 각 SEG 토큰이 가리키는 지점의 [x, y] 상대좌표(0~1000)를 제공하세요.

    출력 형식(JSON):
    {{
      "reasoning_answer": "Chain-of-Thought 답변 내용...",
      "evidence_points": {{"evidence_points_1": [x1, y1], "evidenve_points_2": [x2, y2]}}
    }}
    """

    try:
        response = model.generate_content([prompt, img_pil])
        res_json = json.loads(response.text.replace('```json', '').replace('```', '').strip())

        # 마스크 생성
        l4_masks = []
        for i, (seg_tag, coords) in enumerate(res_json['evidence_points'].items()):
            px, py = int(coords[0] * w / 1000), int(coords[1] * h / 1000)
            masks, _, _ = predictor.predict(np.array([[px, py]]), np.array([1]), multimask_output=False)

            m_filename = f"l4_evidence_{tree_metadata['image_id']}_seg{i+1}.png"
            cv2.imwrite(os.path.join("masks/l4", m_filename), (masks[0] * 255).astype(np.uint8))
            l4_masks.append(m_filename)

        return res_json['reasoning_answer'], l4_masks
    
    except Exception as e:
        print(f"Error: {e}")
        return None, []
    
def build_l4_dataset(image_dir, output_dir, l3_dataset, external_db):
    os.makedirs(os.path.join(output_dir, "masks/l4"), exist_ok=True)
    l4_final_json = []

    for l3_entry in l3_dataset:
        image_id = l3_entry['id'].split('_')[-1]
        tree_info = external_db.get(image_id)

        if tree_info:
            answer, l4_masks = generate_l4(
                os.path.join(image_dir, l3_entry['image']), l3_entry, tree_info
            )

            l4_final_json.append({
                "id": f"L4_{image_id}",
                "image": l3_entry['image'],
                "conversations": [
                    {
                        "from": "human",
                        "value": "이미지 속 나무, 식물, 토양 등 환경요소를 고려하여 상태를 분석하고, 탄소 흡수량을 추론해줘."
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
                "masks": l4_masks # L4 전용 마스크(추론 근거 마스킹)
            })
    with open(os.path.join(output_dir, "l4_final.json"), "w", encoding="utf-8") as f:
        json.dump(l4_final_json, f, ensure_ascii=False, indent=2)