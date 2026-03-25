import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score  # 🚀 R-squared(결정계수) 추가

from config.param_parser import TrainParser
from config_mf import CONFIGURE, BATCH_SIZE
from data.build import build_data_loader
from model.build import build_model
from model.metrics import CarbonLoss

def calculate_metrics(pred_cls, label_cls, pred_reg, label_reg):
    """분류 정확도(Pixel Acc), 회귀 상관계수(Pearson R), 결정계수(R-squared) 계산"""
    # 1. Classification Accuracy 
    pred_cls_labels = torch.argmax(pred_cls, dim=1)
    correct = (pred_cls_labels == label_cls).sum().item()
    total = label_cls.numel()
    acc_corr = correct / total if total > 0 else 0.0

    # 2. Regression Metrics (Pearson R & R2)
    pred_reg_flat = pred_reg.detach().cpu().numpy().flatten()
    label_reg_flat = label_reg.detach().cpu().numpy().flatten()
    
    # 0으로만 이루어진 배경 마스크 등 분산이 0인 경우를 방지
    if np.std(pred_reg_flat) == 0 or np.std(label_reg_flat) == 0:
        acc_r = 0.0
        r2 = 0.0
    else:
        acc_r, _ = pearsonr(pred_reg_flat, label_reg_flat) # 피어슨 상관계수
        r2 = r2_score(label_reg_flat, pred_reg_flat)       # 결정계수 (R2)
        
    return acc_corr, acc_r, r2

def get_color_palette():
    colors = np.array([
        [0, 0, 0],         # 인덱스 0 (레이블 0: 판독불가) -> 투명 처리
        [255, 0, 0],       # 인덱스 1 (레이블 110: 소나무) -> Red
        [0, 0, 255],       # 인덱스 2 (레이블 120: 낙엽송) -> Blue
        [255, 255, 0],     # 인덱스 3 (레이블 130: 기타 침엽수) -> Yellow
        [0, 255, 0],       # 인덱스 4 (레이블 140: 활엽수) -> Green
        [255, 0, 255],     # 인덱스 5 (레이블 150: 침엽수) -> Magenta
        [0, 0, 0]          # 인덱스 6 (레이블 190: 비산림) -> 투명 처리
    ], dtype=np.uint8)
    return colors

def tensor_to_cv2_image(tensor_img):
    """Dataloader에서 나온 텐서를 원본 BGR 이미지로 복원"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    
    orig_img = tensor_img * std + mean
    orig_img = orig_img.cpu().numpy().transpose(1, 2, 0)
    orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

def main():

    # 2. 환경 및 인자 설정
    parser = TrainParser()
    args = parser.parse_args()
    cfg = CONFIGURE(args.image_type)
    
    cfg.VAL_CSV = "test_forest_AP_10_25.csv" 
    
    OUTPUT_DIR = "./inference_results"
    OUTPUT_CARBON_DIR = os.path.join(OUTPUT_DIR, "carbon_maps")
    OUTPUT_MASK_DIR = os.path.join(OUTPUT_DIR, "seg_masks")
    os.makedirs(OUTPUT_CARBON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    # 3. 데이터로더 및 모델 빌드
    print(f"Loading Test Dataset from {cfg.VAL_CSV}...")
    # 시각화 매핑을 위해 batch_size를 1로 고정
    test_loader = build_data_loader("datalists/", cfg.VAL_CSV, 1, args.num_workers, args.local_rank, cfg, shuffle=False)

    print("Building Model...")
    model = build_model(args.net, num_class=cfg.NUM_CLASSES, dropout=args.enc_dropout)
    model.cuda()

    checkpoint_path = "./src/outputs/forest_AP_10_25/pths/checkpoint_000199.pth"
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    criterion = CarbonLoss()
    palette = get_color_palette()

    # 평가지표 누적용
    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    total_acc_corr, total_acc_r, total_r2 = 0.0, 0.0, 0.0
    num_batches = 0

    print("\nStarting Inference & Visualization...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            images = batch_data[0].cuda()
            labels_cls = batch_data[1].cuda()
            labels_reg = batch_data[2].cuda()
            
            # [추론 및 Metric 계산]
            preds_cls, preds_reg = model(images)
            
            losses = criterion(preds_cls, preds_reg, labels_cls, labels_reg)
            if isinstance(losses, tuple):
                loss, cls_loss, reg_loss = losses
            else:
                loss = losses; cls_loss = losses; reg_loss = losses
                
            acc_corr, acc_r, r2 = calculate_metrics(preds_cls, labels_cls, preds_reg, labels_reg)
            
            total_loss += loss.item() if hasattr(loss, 'item') else loss
            total_cls_loss += cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss
            total_reg_loss += reg_loss.item() if hasattr(reg_loss, 'item') else reg_loss
            total_acc_corr += acc_corr
            total_acc_r += acc_r
            total_r2 += r2
            num_batches += 1


            # 시각화 1: 탄소량 히트맵 (Regression)
           
            pred_carbon_np = preds_reg[0].squeeze().cpu().numpy()
            carbon_save_path = os.path.join(OUTPUT_CARBON_DIR, f"test_img_{batch_idx:04d}_carbon.png")
            plt.imsave(carbon_save_path, pred_carbon_np, cmap='magma')

        
            # 시각화 2: 수종 분할 결과 원본 오버레이
            
            orig_cv2_img = tensor_to_cv2_image(images[0])
            orig_h, orig_w = orig_cv2_img.shape[:2]

            pred_cls_idx = torch.argmax(preds_cls[0], dim=0).cpu().numpy()
            pred_cls_resized = cv2.resize(pred_cls_idx, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            color_mask = palette[pred_cls_resized]

            # 투명 오버레이 적용 (0.5 비율)
            alpha = 0.5
            blended_img = cv2.addWeighted(orig_cv2_img, 1 - alpha, color_mask, alpha, 0)
            
            # '판독불가(0)'와 '비산림(6)' 영역은 마스킹을 지워 원본이 100% 보이게 처리
            bg_mask = (pred_cls_resized == 0) | (pred_cls_resized == 6)
            blended_img[bg_mask] = orig_cv2_img[bg_mask]

            mask_save_path = os.path.join(OUTPUT_MASK_DIR, f"test_img_{batch_idx:04d}_overlay.jpg")
            cv2.imwrite(mask_save_path, blended_img)


    # 4. 최종 테스트셋 평가지표 출력
    if num_batches > 0:
        print("\n" + "="*60)
        print("Test Dataset Evaluation Results (Epoch 199) 🎯")
        print("="*60)
        print(f"Total Loss                     : {total_loss/num_batches:.4f}")
        print(f"Classification Loss (cls_loss) : {total_cls_loss/num_batches:.4f}")
        print(f"Regression Loss     (reg_loss) : {total_reg_loss/num_batches:.4f}")
        print("-" * 60)
        print(f"Classification Accuracy (Corr) : {(total_acc_corr/num_batches)*100:.2f}%")
        print(f"Pearson Correlation (R)        : {total_acc_r/num_batches:.4f}")
        print(f"R-squared (R2 Score)           : {total_r2/num_batches:.4f}")
        print("="*60)
        print(f"Visualizations successfully saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()