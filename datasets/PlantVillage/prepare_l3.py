import os
import shutil
import random

# src_root: 전처리될 데이터 루트 폴더 경로
# dst_root: 전처리한 데이터 루트 폴더 경로
def prepare_l3_samplings(src_root, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # 데이터 폴더만 추출
    all_folders = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    total_sampled = 0

    for folder in all_folders:
        is_healthy = "healthy" in folder.lower()
        sample_count = 200 if is_healthy else 50

        src_path = os.path.join(src_root, folder)
        dst_path = os.path.join(dst_root, folder)
        os.makedirs(dst_path, exist_ok=True)

        # 전체 이미지 데이터 목록 뽑기
        images = [img for img in os.listdir(src_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 추출할 데이터 개수
        actual_sample_size = min(len(images), sample_count)

        # 랜덤으로 추출할 데이터 개수만큼 추출
        sampled_images = random.sample(images, actual_sample_size)

        # 추출할 데이터 전처리 후 폴더에 삽입
        for img in sampled_images:
            shutil.copy(os.path.join(src_path, img), os.path.join(dst_path, img))

        total_sampled += actual_sample_size
        status_label = "Healthy (200)" if is_healthy else "Unhealthy (50)"
        print(f" {folder}: {actual_sample_size}장 복사완료 [{status_label}]")
    
    print(f"\n총 {total_sampled}장의 이미지가 샘플링됨")


def l2_bridge_binary(image_path, folder_name):
    # 폴더 명에서 식물 종과 병명 추출
    parts = folder_name.replace("___", " ").split()
    species = parts[0] # 종
    disease_name = " ".join(parts[1:]) # 질병명

    # Healthy/Unhealthy 판단
    is_healthy = "healthy" in folder_name.lower()
    label = "Healthy" if is_healthy else "Unhealthy"

    condition_detail = "특별한 병징이 없는 건강한 잎의 상태" if is_healthy else f"잎에 {disease_name} 증상이 나타난 건강하지 않은 상태"

    # 사용자가 던질 질문
    human_query = f"이 {species} 잎의 외형을 관찰하고 현재 건강 상태가 어떤지 시각적 근거를 들어 설명해줘."

    # PlantVillage 데이터셋 전용 Prompt
    plant_village_prompt = f"""
        당신은 식물 병리 학자입니다. 제공된 이미지와 정보를 바탕으로 잎의 상태를 정밀 진단하세요.
        {species} 잎 ({label} - {condition_detail})
    
        Task:
            1. 잎의 변색, 반점, 질감 등 시각적 특징을 논리적으로 분석하세요.
            2. 모든 특징적 부위(이상 징후나 건강한 조직) 뒤에 [SEG] 토큰을 붙이세요.
            3. 결론적으로 이 잎은 왜 '{label}' 상태인지 설명하는 긴 글(Dense Caption)을 작성하세요.
            4. 각 [SEG]의 정확한 [x, y] 상대좌표(0~1000)를 JSON에 포함하세요.

        출력 형식(JSON):
            {{
                "analysis": "이미지 분석 내용...",
                "dense_caption": "최종 캡션...",
                "objects": {{"symptom_1": [x, y], ...}}
            }}
            """

    return { # level-3 데이터를 제작하기 위해서는 level-2 데이터가 필요하므로 임의로 제작
        "id": f"{species}_{os.path.basename(image_path).split('.')[0]}",
        "image": image_path,
        "landmark": f"{label} leaf",
        "original_disease": disease_name, 
        "human_query": human_query,
        "prompt": plant_village_prompt
    }





