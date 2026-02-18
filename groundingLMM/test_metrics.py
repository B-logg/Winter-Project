import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocoevalcap.eval import COCOEvalCap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="Path to test_predictions.json")
    parser.add_argument("--gt_path", required=True, help="Path to test.json")
    parser.add_argument("--image_folder", required=True, help="Image folder")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 데이터 로드
    with open(args.gt_path, 'r') as f: gt_data = json.load(f)
    with open(args.pred_path, 'r') as f: preds_list = json.load(f)
    
    # 검색 편의를 위해 dict로 변환
    preds = {str(item['image_id']): item for item in preds_list}
    
    ious = []
    recalls = [] # Pixel Recall
    ap50s = []   # Instance Success Rate (IoU > 0.5)

    print(">>> Calculating Segmentation Metrics (mIoU, AP50, Recall)...")
    
    for item in tqdm(gt_data):
        img_id = str(item['id'])
        if img_id not in preds: continue
        
        # 1. GT Mask 준비
        mask_paths = item.get('mask_path', [])
        if isinstance(mask_paths, str): mask_paths = [mask_paths]
        
        gt_mask = None
        valid = False
        for mp in mask_paths:
            full_mp = os.path.join(args.image_folder, mp)
            if not os.path.exists(full_mp): continue
            m = cv2.imread(full_mp, 0)
            if m is None: continue
            
            # 모델 출력 크기에 맞춤 (또는 원본 크기) -> 여기선 통일
            # 예측값이 RLE로 저장될 때 원본 크기로 리사이즈 되었는지 확인 필요.
            # infer_forest.py에서는 output size를 맞춰서 RLE로 저장함.
            # 따라서 여기서는 GT를 그냥 읽으면 됨. (단, 크기 매칭 필요)
            
            if gt_mask is None: gt_mask = np.zeros_like(m)
            gt_mask = np.maximum(gt_mask, m)
            valid = True
            
        if not valid or gt_mask is None: continue
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # 2. Pred Mask 준비
        pred_item = preds[img_id]
        pred_rles = pred_item.get('pred_masks', [])
        
        pred_mask_final = np.zeros_like(gt_mask)
        for rle in pred_rles:
            m = maskUtils.decode(rle)
            if m.shape != gt_mask.shape:
                m = cv2.resize(m, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask_final = np.maximum(pred_mask_final, m)
            
        # 3. Metric 계산
        intersection = np.logical_and(pred_mask_final, gt_mask).sum()
        union = np.logical_or(pred_mask_final, gt_mask).sum()
        gt_area = gt_mask.sum()
        
        # IoU
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
        
        # AP50 (Binary: IoU > 0.5 이면 성공)
        ap50s.append(1.0 if iou >= 0.5 else 0.0)
        
        # Recall (Pixel Level: GT 중 맞춘 비율)
        recall = intersection / gt_area if gt_area > 0 else 0.0
        recalls.append(recall)

    print("\n" + "="*30)
    print(" [SEGMENTATION PERFORMANCE]")
    print(f" - mIoU:   {np.mean(ious)*100:.2f}")
    print(f" - AP50:   {np.mean(ap50s)*100:.2f}")
    print(f" - Recall: {np.mean(recalls)*100:.2f}")
    print("="*30 + "\n")

    # ----------------------------------------------
    print(">>> Calculating Captioning Metrics (CIDEr, METEOR)...")
    
    # COCO 포맷 생성
    coco_gt = {"images": [], "annotations": []}
    coco_res = []
    
    ann_id = 0
    for item in gt_data:
        img_id = str(item['id'])
        int_id = int(hash(img_id) % 1e8) # 정수 ID 생성
        
        coco_gt["images"].append({"id": int_id})
        coco_gt["annotations"].append({
            "image_id": int_id,
            "id": ann_id,
            "caption": item['conversations'][1]['value']
        })
        ann_id += 1
        
        if img_id in preds:
            coco_res.append({
                "image_id": int_id,
                "caption": preds[img_id]['caption']
            })
            
    # 임시 파일 저장
    with open("temp_gt.json", "w") as f: json.dump(coco_gt, f)
    with open("temp_res.json", "w") as f: json.dump(coco_res, f)
    
    # 평가
    coco = COCO("temp_gt.json")
    coco_result = coco.loadRes("temp_res.json")
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    
    print("\n" + "="*30)
    print(" [CAPTIONING PERFORMANCE]")
    for metric, score in coco_eval.eval.items():
        print(f" - {metric}: {score*100:.2f}")
    print("="*30 + "\n")
    
    # 정리
    os.remove("temp_gt.json")
    os.remove("temp_res.json")

if __name__ == "__main__":
    main()