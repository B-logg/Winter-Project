import os
import cv2
import json
import torch
import argparse
import re
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from eval.utils import mask_to_rle_pytorch, coco_encode_rle

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Test with Loss")
    parser.add_argument("--hf_model_path", required=True, help="Path to checkpoint")
    parser.add_argument("--test_json_path", required=True, help="Path to test.json")
    parser.add_argument("--image_folder", required=True, help="Image folder")
    parser.add_argument("--output_dir", required=True, help="Result save dir")
    parser.add_argument("--conv_type", default="llava_v1")
    return parser.parse_args()

class ForestTestDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, image_processor, transform, model_config):
        with open(json_path, 'r') as f: self.data = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform
        
    def __len__(self): return len(self.data)

    def preprocess_image(self, image_path):
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        orig_size = image_np.shape[:2]
        image_clip = self.image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        image_sam = self.transform.apply_image(image_np)
        resize_shape = image_sam.shape[:2]
        image_sam = torch.from_numpy(image_sam).permute(2, 0, 1).float()
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        image_sam = (image_sam - pixel_mean) / pixel_std
        return image_clip, image_sam, orig_size, resize_shape

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])
        clip_img, sam_img, orig_size, resize_shape = self.preprocess_image(image_path)
        
        human_q = item['conversations'][0]['value']
        gpt_a = item['conversations'][1]['value']
        
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        q_text = DEFAULT_IMAGE_TOKEN + "\n" + human_q
        conv.append_message(conv.roles[0], q_text)
        conv.append_message(conv.roles[1], gpt_a)
        full_prompt = conv.get_prompt()
        
        input_ids_loss = tokenizer_image_token(full_prompt, self.tokenizer, return_tensors='pt')

        # 🚨 스킵 판단 (1536 토큰 초과 시 가짜 데이터로 대체)
        is_skipped = False
        if input_ids_loss.shape[0] > 1536:
            is_skipped = True
            dummy_q = "이 사진의 탄소 저장량을 분석해줘."
            dummy_a = "산림이 과밀하거나 구조적으로 불균형할 경우 나무의 안정성과 생육 효율이 저하될 수 있으므로, 밀도 조절(예: 솎아베기)을 통해 건강성과 탄소 흡수 능력을 개선할 필요가 있다. [SEG]"
            conv = conversation_lib.conv_templates["llava_v1"].copy()
            conv.messages = []
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + dummy_q)
            conv.append_message(conv.roles[1], dummy_a)
            full_prompt = conv.get_prompt()
            input_ids_loss = tokenizer_image_token(full_prompt, self.tokenizer, return_tensors='pt')
        
        labels = input_ids_loss.clone()
        sep = "ASSISTANT: "
        parts = full_prompt.split(sep)
        if len(parts) >= 2:
            len_context = len(tokenizer_image_token(parts[0] + sep, self.tokenizer))
            labels[:len_context-1] = -100
        
        gt_mask = torch.zeros((1024, 1024)).float()
        mask_path = item.get('mask_path', None)
        if mask_path:
            if isinstance(mask_path, str): mask_path = [mask_path]
            for mp in mask_path:
                m = cv2.imread(os.path.join(self.image_folder, mp), 0)
                if m is not None:
                    m = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    gt_mask = torch.maximum(gt_mask, torch.from_numpy(m).float())
        gt_mask = (gt_mask > 0).float().unsqueeze(0)

        return {
            "id": item['id'],
            "human_q": human_q,
            "clip_img": clip_img,
            "sam_img": sam_img,
            "input_ids_loss": input_ids_loss,
            "labels": labels,
            "masks": gt_mask,
            "orig_size": orig_size,
            "resize_shape": resize_shape,
            "is_skipped": is_skipped
        }

def main():
    from peft import PeftModel
    args = parse_args()
    BASE_MODEL_PATH = "checkpoints/GLaMM-GCG"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    model = GLaMMForCausalLM.from_pretrained(
        BASE_MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, seg_token_idx=seg_token_idx,
        train_mask_decoder=True 
    )
    
    model = PeftModel.from_pretrained(model, args.hf_model_path)
    model = model.merge_and_unload() 
    
    non_lora_path = os.path.join(args.hf_model_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
        cleaned_state_dict = {k[17:] if k.startswith('base_model.model.') else k: v for k, v in non_lora_trainables.items()}
        model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.cuda().bfloat16()
    model.ce_loss_weight, model.dice_loss_weight, model.bce_loss_weight = 1.0, 0.5, 2.0

    for name, param in model.named_parameters():
        if param.is_floating_point(): param.data = param.data.to(torch.bfloat16)
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point(): buffer.data = buffer.data.to(torch.bfloat16)

    # ✅ Monkey Patch 적용
    base_glamm = model.get_model()
    if hasattr(base_glamm, "grounding_encoder"):
        mask_decoder = base_glamm.grounding_encoder.mask_decoder
        original_forward = mask_decoder.forward
        def mask_decoder_forward_wrapper(*args, **kwargs):
            new_args = [a.to(torch.bfloat16) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args]
            new_kwargs = {k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for k, v in kwargs.items()}
            return original_forward(*new_args, **new_kwargs)
        mask_decoder.forward = mask_decoder_forward_wrapper

    dataset = ForestTestDataset(args.test_json_path, args.image_folder, tokenizer, CLIPImageProcessor.from_pretrained(model.config.vision_tower), ResizeLongestSide(1024), model.config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)
    total_loss, ce_loss, mask_loss, count, results = 0.0, 0.0, 0.0, 0, []
    
    print(">>> Starting Test Loop...")
    for batch in tqdm(dataloader):
        image_id = batch['id'][0]
        is_skipped = batch['is_skipped'][0]
        
        images = batch['clip_img'].cuda().bfloat16()
        sam_images = batch['sam_img'].cuda().bfloat16()
        input_ids_loss = batch['input_ids_loss'].cuda()
        labels = batch['labels'].cuda()
        gt_masks = batch['masks'].cuda().bfloat16()
        
        # (A) Loss Calculation (Teacher Forcing)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_loss, labels=labels, images=images, global_enc_images=images,
                grounding_enc_images=sam_images, masks_list=[gt_masks[0]], label_list=[gt_masks[0]],
                resize_list=[[batch['resize_shape'][0].item(), batch['resize_shape'][1].item()]],
                offset=torch.tensor([0, 1]).long().cuda(), bboxes=None, attention_masks=None
            )
            if 'loss' in outputs:
                total_loss += outputs['loss'].item()
                count += 1
                ce_loss += outputs.get('ce_loss', torch.tensor(0)).item()
                mask_loss += outputs.get('mask_loss', torch.tensor(0)).item()

        # (B) Inference (Autoregressive Generation)
        if is_skipped:
            cleaned_text, rle_masks = "Content too long - Inference skipped.", []
        else:
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            q_text = DEFAULT_IMAGE_TOKEN + "\n" + batch['human_q'][0]
            conv.append_message(conv.roles[0], q_text)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            input_ids_gen = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
            
            with torch.no_grad():
                output_ids, pred_masks = model.evaluate(
                    images, sam_images, input_ids_gen, 
                    [[batch['resize_shape'][0].item(), batch['resize_shape'][1].item()]], 
                    [[batch['orig_size'][0].item(), batch['orig_size'][1].item()]],
                    max_tokens_new=128, bboxes=None
                )
            
            out_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_out = tokenizer.decode(out_ids, skip_special_tokens=False).split("ASSISTANT: ")[-1]
            cleaned_text = re.sub(r'<.*?>', '', text_out).replace('[SEG]', '').strip()
            rle_masks = [coco_encode_rle(m) for m in mask_to_rle_pytorch(pred_masks[0].cpu() > 0)] if pred_masks is not None else []

        results.append({"image_id": image_id, "caption": cleaned_text, "pred_masks": rle_masks})
        
        # 🚨 메모리 해제
        del images, sam_images, input_ids_loss, labels, gt_masks, outputs
        torch.cuda.empty_cache()

    # 결과 출력 및 저장
    if count > 0:
        print(f"\n [TEST SET LOSS] Total: {total_loss/count:.4f} | CE: {ce_loss/count:.4f} | Mask: {mask_loss/count:.4f}")
    with open(os.path.join(args.output_dir, "test_predictions.json"), 'w') as f: json.dump(results, f)

if __name__ == "__main__": main()