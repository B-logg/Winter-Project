import os
import json

# 1. 파일 경로 설정
# 현재 테스트 서버에 있는 glamm_test.json의 실제 경로를 지정합니다.
json_path = os.path.expanduser("~/Winter-Project/groundingLMM/dataset/datasets/glamm_test.json")

# 백업 파일명 (혹시 모를 에러에 대비)
backup_path = json_path + ".bak"

# 2. 새로운 베이스 경로 설정
new_image_base = "~/Winter-Project/groundingLMM/dataset/datasets/glamm_images_train"
new_mask_base = "~/Winter-Project/groundingLMM/dataset/datasets/glamm_masks_train"

def main():
    # JSON 파일 읽기
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 원본 백업 저장
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Backup created at: {backup_path}")

    # 데이터 순회하며 경로 변경
    for item in data:
        # 이미지 경로 변경 (파일명만 추출 후 새 경로와 결합)
        if 'image' in item:
            img_filename = os.path.basename(item['image'])
            item['image'] = os.path.join(new_image_base, img_filename)
            
        # 마스크 경로 변경 (리스트 내부의 각 경로 변경)
        if 'mask_path' in item:
            updated_masks = []
            for mask in item['mask_path']:
                mask_filename = os.path.basename(mask)
                updated_masks.append(os.path.join(new_mask_base, mask_filename))
            item['mask_path'] = updated_masks

    # 변경된 데이터를 다시 JSON 파일로 덮어쓰기
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully updated {len(data)} items in {json_path}")

if __name__ == "__main__":
    main()