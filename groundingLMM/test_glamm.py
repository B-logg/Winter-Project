import os
import re
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import PeftModel

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN

# 1. 평가 지표 라이브러리 (NLG, Regression, Classification)
from sklearn.metrics import mean_absolute_percentage_error, r2_score, accuracy_score, f1_score
from scipy.stats import pearsonr
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider

# 추론 결과 저장하기 -> 추론 결과 직접 살펴봐야겠음.

# NLTK Wordnet 다운로드 (METEOR용)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Full Evaluation (Text, Mask, Carbon, Species)")
    parser.add_argument("--hf_model_path", required=True, help="Path to checkpoint")
    parser.add_argument("--test_json_path", required=True, help="Path to test.json")
    parser.add_argument("--image_folder", required=True, help="Image folder")
    parser.add_argument("--output_dir", required=True, help="Result save dir")
    parser.add_argument("--batch_size", default=1, type=int)
    return parser.parse_args()

class ForestEvalDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, image_processor, transform):
        with open(json_path, 'r') as f: self.data = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform
        
    def __len__(self): 
        return len(self.data)

    def preprocess_image(self, image_path):
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # SAM Image
        image_sam = self.transform.apply_image(image_np)
        h, w = image_sam.shape[:2]
        image_sam_padded = cv2.copyMakeBorder(image_sam, 0, 1024-h, 0, 1024-w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image_sam_tensor = torch.from_numpy(image_sam_padded).permute(2, 0, 1).float()
        pixel_mean, pixel_std = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1), torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        image_sam_tensor = (image_sam_tensor - pixel_mean) / pixel_std
        
        # CLIP Image
        clip_img = self.image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        return clip_img, image_sam_tensor, (h, w)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'].replace('~', '/home/sbosung1789'))
        clip_img, sam_img, orig_shape = self.preprocess_image(image_path)
        
        human_q = item['conversations'][0]['value']
        gt_text = item['conversations'][1]['value']
        clean_q = human_q.replace(DEFAULT_IMAGE_TOKEN, "").replace("<image>", "").strip()

        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + clean_q)
        conv.append_message(conv.roles[1], None) 
        input_ids_gen = tokenizer_image_token(conv.get_prompt(), self.tokenizer, return_tensors='pt')
        
        # GT Masks Load
        gt_masks = []
        mask_paths = item.get('mask_path', [])
        for mp in mask_paths:
            m = cv2.imread(mp.replace('~', '/home/sbosung1789'), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                gt_masks.append(torch.from_numpy(m > 0).float())
        
        gt_masks_tensor = torch.stack(gt_masks) if gt_masks else torch.zeros((0, 1024, 1024)).float()

        return {
            "id": item['id'],
            "image_path": image_path,
            "clip_img": clip_img,
            "sam_img": sam_img,
            "input_ids_gen": input_ids_gen,
            "gt_text": gt_text,
            "gt_masks": gt_masks_tensor,
            "orig_shape": orig_shape
        }

# 2. 딕셔너리 파싱 헬퍼 함수 (순서대로 수종-탄소량 쌍 추출)
def parse_forest_info(text):
    results = []
    blocks = text.split('<p>')
    for block in blocks[1:]: 
        # 수종 추출
        species_match = re.search(r'\s*(.*?)\s*</p>', block)
        species = species_match.group(1).strip() if species_match else "unknown"
        
        # 탄소량 추출 (t C 앞의 숫자)
        tc_match = re.search(r'([\d\.]+)\s*t\s*[Cc]', block)
        if tc_match:
            carbon = float(tc_match.group(1))
        else:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", block)
            carbon = float(nums[-1]) if nums else 0.0
            
        results.append({"species": species, "carbon": carbon})
    return results

# 3. 마스크 지표 (mIoU, Recall, AP50) 계산 함수
def calculate_mask_metrics(pred_masks, gt_masks):
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0, 0.0, 0.0, []
    
    pred_masks = pred_masks.cpu().numpy() > 0
    gt_masks = gt_masks.cpu().numpy() > 0
    
    M, N = len(pred_masks), len(gt_masks)
    iou_matrix = np.zeros((M, N))
    
    for i in range(M): # i: pred_idx
        for j in range(N): # j: gt_idx
            intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
            iou_matrix[i, j] = intersection / (union + 1e-6)
            
    # 매칭 (Greedy)
    tp = 0
    matched_gt = set()
    iou_sum = 0.0
    matched_pairs = [] # (gt_idx, pred_idx) 짝꿍 저장
    
    for i in range(M):
        best_iou, best_gt = 0, -1
        for j in range(N):
            if j not in matched_gt and iou_matrix[i, j] > 0.5:
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt = j
        if best_gt != -1:
            tp += 1
            matched_gt.add(best_gt)
            iou_sum += best_iou
            matched_pairs.append((best_gt, i)) # 정답 인덱스와 예측 인덱스를 묶음!

    recall = tp / N if N > 0 else 0.0
    ap50 = tp / M if M > 0 else 0.0 # Precision @ IoU 0.5
    miou = iou_sum / M if M > 0 else 0.0
    
    return miou, recall, ap50, matched_pairs

def main():
    args = parse_args()
    BASE_MODEL_PATH = "/home/sbosung1789/Winter-Project/groundingLMM/checkpoints/GLaMM-GCG"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    print("Loading Model")
    model = GLaMMForCausalLM.from_pretrained(BASE_MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, seg_token_idx=seg_token_idx)
    model = PeftModel.from_pretrained(model, args.hf_model_path)
    model = model.merge_and_unload().cuda().bfloat16()
    
    non_lora_path = os.path.join(args.hf_model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print("Loading non-LoRA weights")
        non_lora_state_dict = torch.load(non_lora_path, map_location="cpu")
        
        # PEFT로 학습하면서 앞부분에 'base_model.model.' 꼬리표가 붙은 경우 깔끔하게 떼어줍니다.
        clean_state_dict = {}
        for k, v in non_lora_state_dict.items():
            if k.startswith("base_model.model."):
                clean_state_dict[k.replace("base_model.model.", "")] = v
            else:
                clean_state_dict[k] = v
                
        # 껍데기가 벗겨진 본체(model)에 직접 가중치를 주입합니다.
        model.load_state_dict(clean_state_dict, strict=False)

    base_glamm = model.get_model()
    if hasattr(base_glamm, "grounding_encoder"):
        mask_decoder = base_glamm.grounding_encoder.mask_decoder
        original_forward = mask_decoder.forward
        def mask_decoder_forward_wrapper(*a, **k):
            new_a = [x.to(torch.bfloat16) if isinstance(x, torch.Tensor) and torch.is_floating_point(x) else x for x in a]
            new_k = {key: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for key, v in k.items()}
            return original_forward(*new_a, **new_k)
        mask_decoder.forward = mask_decoder_forward_wrapper

    dataset = ForestEvalDataset(args.test_json_path, args.image_folder, tokenizer, CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336"), ResizeLongestSide(1024))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 지표 저장소
    gt_carbon_all, pred_carbon_all = [], []
    gt_species_all, pred_species_all = [], []
    
    cider_refs, cider_hyps = {}, {}
    meteor_scores = []
    
    miou_list, recall_list, ap50_list = [], [], []

    all_predictions = []

    model.eval()
    print("Starting Inference & Metric Evaluation")
    for step, batch in enumerate(tqdm(dataloader)):
        images = batch['clip_img'].cuda().bfloat16()
        sam_images = batch['sam_img'].cuda().bfloat16()
        input_ids = batch['input_ids_gen'].cuda()
        gt_masks = batch['gt_masks'][0].cuda() # (N, H, W)
        
        gt_text = batch['gt_text'][0]
        data_id = batch['id'][0]
        image_path = batch['image_path'][0]
        
        with torch.no_grad():
            # 1. 텍스트 생성
            output_ids = model.generate(inputs=input_ids, images=images, max_new_tokens=256, use_cache=True)
            raw_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)

            pred_text = raw_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

            all_predictions.append({
                "image_id": data_id,
                "gt_text": gt_text,
                "pred_text": pred_text,
                "prompt_used": tokenizer.decode(input_ids[0])
            })

            num_seg_tokens = (output_ids == seg_token_idx).sum().item()

            if num_seg_tokens == 0:
                pred_masks = []
            else:
                offset = [0, num_seg_tokens]

                batch_size = images.shape[0]
                dummy_sizes = [(1024, 1024)] * batch_size
                
                # 2. 생성된 텍스트 기반으로 마스크 추출 (Forward Pass)
                outputs = model(
                    input_ids=output_ids,
                    images=images,
                    global_enc_images=images,
                    grounding_enc_images=sam_images,
                    bboxes=None,          
                    labels=None,          
                    attention_masks=torch.ones_like(output_ids),
                    offset=offset,          
                    masks_list=None,      
                    label_list=dummy_sizes,      
                    resize_list=dummy_sizes,     
                    inference=True        
                )
                if 'pred_masks' in outputs and outputs['pred_masks'] is not None and len(outputs['pred_masks']) > 0:
                    pred_masks = outputs['pred_masks'][0]
                else:
                    pred_masks = []

                if len(pred_masks) > 0:
                    vis_dir = os.path.join(args.output_dir, "vis_results") # 저장할 폴더 이름
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # 1. 원본 이미지 불러오기
                    orig_img = cv2.imread(image_path)
                    if orig_img is not None:
                        orig_h, orig_w = orig_img.shape[:2]
                        overlay = orig_img.copy()
                        
                        # 2. 각각의 나무 마스크마다 색상 칠하기
                        for mask in pred_masks:
                            # 1024x1024 크기의 모델 예측 마스크를 가져와서 numpy로 변환
                            m_np = mask.cpu().numpy().astype(np.uint8)
                            
                            # 마스크를 원본 이미지 크기로 늘리기/줄이기
                            m_resized = cv2.resize(m_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                            
                            # 랜덤한 형광펜 색상 만들기 (BGR)
                            color = np.random.randint(100, 255, (3,)).tolist()
                            
                            # 마스크가 있는 영역(값이 1인 곳)에 색깔 덧칠하기
                            overlay[m_resized > 0] = color
                        
                        # 3. 투명도(알파 블렌딩) 50% 섞어서 예쁘게 합성하기
                        vis_img = cv2.addWeighted(overlay, 0.5, orig_img, 0.5, 0)
                        
                        # 4. 파일로 저장 (예: 1번이미지.jpg)
                        safe_id = str(data_id).split("/")[-1] # 혹시 모를 경로 오류 방지
                        save_file = os.path.join(vis_dir, f"{safe_id}_pred.jpg")
                        cv2.imwrite(save_file, vis_img)


        # 3. 텍스트 기반 딕셔너리 추출
        gt_list = parse_forest_info(gt_text)
        pred_list = parse_forest_info(pred_text)

        # 4. 세그멘테이션 마스크 지표 및 위치 기반 매칭(IoU)
        matched_pairs = []
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            miou, recall, ap50, matched_pairs = calculate_mask_metrics(pred_masks, gt_masks)
            miou_list.append(miou)
            recall_list.append(recall)
            ap50_list.append(ap50)

        # 5. 위치(Mask IoU) 기반 수종/탄소량 정확도 측정!
        for gt_idx, pred_idx in matched_pairs:
            # 텍스트 내에서 파싱된 정보 개수와 마스크 개수가 일치하는지 안전장치
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                
                # 같은 물리적 위치를 짚어냈으므로, 이제 수종과 탄소량을 비교합니다.
                gt_species_all.append(gt_list[gt_idx]['species'])
                pred_species_all.append(pred_list[pred_idx]['species'])
                
                gt_carbon_all.append(gt_list[gt_idx]['carbon'])
                pred_carbon_all.append(pred_list[pred_idx]['carbon'])

        # 6. 텍스트 품질 지표 (METEOR, CIDEr)
        cider_refs[data_id] = [gt_text]
        cider_hyps[data_id] = [pred_text]
        
        gt_words = nltk.word_tokenize(gt_text)
        pred_words = nltk.word_tokenize(pred_text)
        meteor_scores.append(meteor_score([gt_words], pred_words))

        if step >= 9:
            break

    # --- 최종 지표 계산 ---
    # 1. Regression (탄소량: MAPE, R2, Pearson)
    if len(gt_carbon_all) > 1:
        gt_arr, pred_arr = np.array(gt_carbon_all), np.array(pred_carbon_all)
        valid_idx = gt_arr > 0
        mape = mean_absolute_percentage_error(gt_arr[valid_idx], pred_arr[valid_idx]) * 100 if np.sum(valid_idx) > 0 else 0.0
        r2 = r2_score(gt_carbon_all, pred_carbon_all)
        correlation, _ = pearsonr(gt_carbon_all, pred_carbon_all) if np.std(gt_carbon_all) > 0 and np.std(pred_carbon_all) > 0 else (0.0, 0.0)
    else:
        mape, r2, correlation = 0.0, 0.0, 0.0
        
    # 2. Classification (수종: Accuracy, F1-Score)
    if len(gt_species_all) > 0:
        cls_acc = accuracy_score(gt_species_all, pred_species_all) * 100
        cls_f1 = f1_score(gt_species_all, pred_species_all, average='macro') * 100
    else:
        cls_acc, cls_f1 = 0.0, 0.0
        
    # 3. NLG (텍스트: CIDEr, METEOR)
    cider_score, _ = Cider().compute_score(cider_refs, cider_hyps)
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0.0
    
    # 4. Mask (Seg: mIoU, Recall, AP50)
    avg_miou = np.mean(miou_list) if miou_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_ap50 = np.mean(ap50_list) if ap50_list else 0.0
    
    print("\n" + "="*50)
    print(f" [1. Text Generation]")
    print(f"  - CIDEr Score:       {cider_score:.4f}")
    print(f"  - METEOR Score:      {avg_meteor:.4f}")
    print(f" \n[2. Species Classification]")
    print(f"  - Accuracy:          {cls_acc:.2f} %")
    print(f"  - F1-Score (Macro):  {cls_f1:.2f} %")
    print(f" \n[3. Carbon Regression]")
    print(f"  - MAPE:              {mape:.2f} %")
    print(f"  - R2 Score:          {r2:.4f}")
    print(f"  - Pearson Correlation:       {correlation:.4f}")
    print(f" \n[4. Segmentation Mask]")
    print(f"  - mIoU:              {avg_miou:.4f}")
    print(f"  - Recall:            {avg_recall:.4f}")
    print(f"  - AP50:              {avg_ap50:.4f}")
    print("="*50)

    import json
    save_path = os.path.join(args.output_dir, "glamm_predictions.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)
    print(f"\n추론 텍스트 결과가 '{save_path}'에 저장되었습니다!")

if __name__ == "__main__": 
    main()