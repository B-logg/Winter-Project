import json
import os
import random


# 1. 환경 설정 (Configuration)
# 데이터셋들이 모여있는 루트 폴더 (절대 경로로 변경)
BASE_DIR = os.path.expanduser("~/Winter-Project/datasets/datasets")

# 결과 파일 저장 경로
OUTPUT_DIR = BASE_DIR

# 랜덤 시드 고정 (재현성 확보)
random.seed(42)

# 검증 데이터 비율(0.3: 훈련 70% / 검증 30%)
VAL_RARIO = 0.3

# 각 데이터셋별 설정 (경로 매핑)
dataset_configs = [
    {
        "name": "Level-1",
        "json_path": os.path.join(BASE_DIR, "l1_dataset/l1_dataset.json"),
        "img_prefix": "l1_dataset/images",  # JSON의 'image' 필드 앞에 붙일 경로
        "mask_prefix": "l1_dataset/masks"   # JSON의 'mask_path' 앞에 붙일 경로
    },
    {
        "name": "Level-2",
        "json_path": os.path.join(BASE_DIR, "l2_dataset/l2_dataset.json"),
        "img_prefix": "l2_dataset/images",
        "mask_prefix": "l2_dataset/masks"
    },
    {
        "name": "Level-3 ",
        "json_path": os.path.join(BASE_DIR, "l3_dataset/l3_dataset.json"),
        "img_prefix": "l3_dataset/images",
        "mask_prefix": "l3_dataset/masks"
    },
    {
        "name": "Level-4",
        "json_path": os.path.join(BASE_DIR, "l4_dataset/l4_dataset.json"),
        "img_prefix": "l3_dataset/images", # L4는 L3 이미지를 공유함
        "mask_prefix": None ,
    },
    # iNaturalist도 똑같은 양식으로 추가 
    {
        "name": "Level-1-iNaturalist",
        "json_path": os.path.join(BASE_DIR, "l1_dataset_inaturalist/l1_dataset_inaturalist.json"),
        "img_prefix": "l1_dataset_inaturalist/images",
        "mask_prefix": "l1_dataset_inaturalist/masks"
    }
]

# 2. 경로 정규화 함수
def normalize_path(path, prefix):
    # 파일명만 있는 경우 prefix를 붙여주고, 이미 경로가 포함된 경우 중복되지 않게 처리합니다.
    if not path:
        return None
    
    # 이미 prefix가 포함되어 있으면 그대로 반환 (중복 방지)
    if prefix in path:
        return path
    
    # 파일명만 있는 경우 (예: tile_0.jpg) -> prefix 결합
    return os.path.join(prefix, os.path.basename(path))

# 3. 층화 분할 및 병합

train_set = []
val_set = []
print(f"[Start] Stratified Splitting (Val Ratio: {VAL_RARIO})")

for config in dataset_configs:
    json_file = config['json_path']
    mask_prefix = config.get('mask_prefix')
    
    if not os.path.exists(json_file):
        print(f"[Skip] File not found: {json_file}")
        continue

    # 해당 레벨 데이터 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        level_data = json.load(f)

    # 경로 정규화
    processed_data = []
    for entry in level_data:
        if 'image' in entry:
            entry['image'] = normalize_path(entry['image'], config['img_prefix'])
        
        if mask_prefix and 'mask_path' in entry and entry['mask_path']:
            mask_p = entry['mask_path']
            if isinstance(mask_p, list):
                entry['mask_path'] = [normalize_path(m, mask_prefix) for m in mask_p]
            else:
                entry['mask_path'] = normalize_path(mask_p, mask_prefix)
        processed_data.append(entry)

    # 해당 레벨 내에서 셔플 및 분할
    random.shuffle(processed_data)

    total_len = len(processed_data)
    n_val = int(total_len * VAL_RARIO)
    n_train = total_len - n_val

    level_train = processed_data[:n_train]
    level_val = processed_data[n_train:]

    # 글로벌 리스트에 추가
    train_set.extend(level_train)
    val_set.extend(level_val)

    print(f"[{config['name']}] Total: {total_len} -> Train: {len(level_train)} | Val: {len(level_val)}")
        
# 최종 섞기
# 학습 시 레벨 1, 2, 3, 4가 섞여서 나오도록 전체를 한 번 더 섞음

random.shuffle(train_set)
random.shuffle(val_set)

print("Merge Complete")
print(f"Final Train Set: {len(train_set)} samples")
print(f"Final Val Set  : {len(val_set)} samples")

# 저장
train_path = os.path.join(OUTPUT_DIR, "train.json")
val_path = os.path.join(OUTPUT_DIR, "val.json")

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_set, f, indent=4, ensure_ascii=False)

with open(val_path, 'w', encoding='utf-8') as f:
    json.dump(val_set, f, indent=4, ensure_ascii=False)

print(f"Saved: {train_path}")
print(f"Saved: {val_path}")