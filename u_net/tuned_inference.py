import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score 

from config.param_parser import TrainParser
from config_mf import CONFIGURE, BATCH_SIZE
from data.build import build_data_loader
from model.build import build_model
from model.metrics import CarbonLoss


def get_color_palette():
    """Forest_ap_nir 데이터셋 5개 클래스에 맞춘 완벽한 GLaMM 스타일 색상 팔레트"""
    colors = np.array([
        [0, 0, 0],         # 인덱스 0: 비산림 (원본 190) -> 투명 처리될 예정
        [255, 0, 0],       # 인덱스 1: 소나무 (원본 110) -> Red
        [0, 0, 255],       # 인덱스 2: 낙엽송 (원본 120) -> Blue
        [255, 255, 0],     # 인덱스 3: 기타 침엽수 (원본 130) -> Yellow
        [0, 255, 0]        # 인덱스 4: 활엽수 (원본 140) -> Green
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
    cfg.NUM_CLASSES = 5
    
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
    model = build_model(args.net, num_class=5, dropout=args.enc_dropout)
    model.cuda()

    checkpoint_path = "./src/outputs/forest_AP_10_25/pths/best_checkpoints_loss.pth"
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    state_dict = checkpoint['model']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    criterion = CarbonLoss()
    palette = get_color_palette()

    # 평가지표 누적용
    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    total_acc_corr, total_acc_r, total_pixel_acc = 0.0, 0.0, 0.0
    num_batches = 0

    print("\nStarting Inference & Visualization...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            images = batch_data["image"].cuda()
            labels_cls = batch_data["label_cls"].cuda()
            labels_reg = batch_data["label_reg"].cuda()

            labels_cls[(labels_cls < 0) | (labels_cls >= 5)] = 255
            
            # [추론 및 Metric 계산]
            preds_cls, preds_reg = model(images)
            
            losses = criterion(preds_cls, preds_reg, labels_cls, labels_reg)
            if isinstance(losses, tuple):
                loss, cls_loss, reg_loss, batch_corr, batch_r2 = losses
            else:
                continue # 정상적인 튜플이 아니면 패스!
                
            if not (math.isnan(loss.item()) or math.isnan(cls_loss.item())):
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()
                
                # CarbonLoss가 계산한 완벽한 회귀 지표 가져오기
                total_acc_corr += batch_corr 
                total_acc_r += batch_r2      
                
                pred_cls_labels = torch.argmax(preds_cls, dim=1)
                valid_mask = (labels_cls != 255)
                if valid_mask.sum() > 0: 
                    correct = (pred_cls_labels[valid_mask] == labels_cls[valid_mask]).sum().item()
                    total_pixel_acc += correct / valid_mask.sum().item()
                
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
            
            bg_mask = (pred_cls_resized == 0)
            blended_img[bg_mask] = orig_cv2_img[bg_mask]

            mask_save_path = os.path.join(OUTPUT_MASK_DIR, f"test_img_{batch_idx:04d}_overlay.jpg")
            cv2.imwrite(mask_save_path, blended_img)


    # 4. 최종 테스트셋 평가지표 출력
    if num_batches > 0:
        print("\n" + "="*60)
        print("Test Dataset Evaluation Results")
        print("="*60)
        print(f"Total Loss                     : {total_loss/num_batches:.4f}")
        print(f"Classification Loss (cls_loss) : {total_cls_loss/num_batches:.4f}")
        print(f"Regression Loss     (reg_loss) : {total_reg_loss/num_batches:.4f}")
        print("-" * 60)
        print(f"Classification Accuracy (Corr) : {(total_acc_corr/num_batches)*100:.2f}%")
        print(f"Pearson Correlation (R)        : {total_acc_r/num_batches:.4f}")
        print(f"R-squared (R2 Score)           : {total_acc_r/num_batches:.4f}")
        print("="*60)
        print(f"Visualizations successfully saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()