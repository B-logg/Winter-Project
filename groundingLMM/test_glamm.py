import os
import cv2
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Test Loss Only")
    parser.add_argument("--hf_model_path", required=True, help="Path to checkpoint")
    parser.add_argument("--test_json_path", required=True, help="Path to test.json")
    parser.add_argument("--image_folder", required=True, help="Image folder")
    parser.add_argument("--output_dir", required=True, help="Result save dir")
    return parser.parse_args()

class ForestLossDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, image_processor, transform):
        with open(json_path, 'r') as f: self.data = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform
        
    def __len__(self): return len(self.data)

    def preprocess_image(self, image_path):
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_clip = self.image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        image_sam = self.transform.apply_image(image_np)
        resize_shape = image_sam.shape[:2]
        image_sam = torch.from_numpy(image_sam).permute(2, 0, 1).float()
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        image_sam = (image_sam - pixel_mean) / pixel_std
        return image_clip, image_sam, resize_shape

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])
        clip_img, sam_img, resize_shape = self.preprocess_image(image_path)
        
        human_q = item['conversations'][0]['value']
        gpt_a = item['conversations'][1]['value']
        
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        q_text = DEFAULT_IMAGE_TOKEN + "\n" + human_q
        conv.append_message(conv.roles[0], q_text)
        conv.append_message(conv.roles[1], gpt_a)
        full_prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, return_tensors='pt')

        # 🚨 스킵 로직: 너무 긴 데이터는 가짜 데이터로 대체하여 OOM 방지
        if input_ids.shape[0] > 1536:
            conv.messages = []
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + "이 사진을 분석해줘.")
            conv.append_message(conv.roles[1], "데이터 초과로 스킵합니다. [SEG]")
            input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, return_tensors='pt')
        
        labels = input_ids.clone()
        sep = "ASSISTANT: "
        parts = full_prompt.split(sep)
        if len(parts) >= 2:
            len_context = len(tokenizer_image_token(parts[0] + sep, self.tokenizer))
            labels[:len_context-1] = -100
        
        # GT Mask 로드
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
            "clip_img": clip_img,
            "sam_img": sam_img,
            "input_ids": input_ids,
            "labels": labels,
            "masks": gt_mask,
            "resize_shape": resize_shape
        }

def main():
    from peft import PeftModel
    args = parse_args()
    BASE_MODEL_PATH = "checkpoints/GLaMM-GCG"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    model = GLaMMForCausalLM.from_pretrained(BASE_MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, seg_token_idx=seg_token_idx)
    model = PeftModel.from_pretrained(model, args.hf_model_path)
    model = model.merge_and_unload().cuda().bfloat16()
    
    model.ce_loss_weight, model.dice_loss_weight, model.bce_loss_weight = 1.0, 0.5, 2.0

    # Monkey Patch 적용 (SAM 입력 BF16 보장)
    base_glamm = model.get_model()
    if hasattr(base_glamm, "grounding_encoder"):
        mask_decoder = base_glamm.grounding_encoder.mask_decoder
        original_forward = mask_decoder.forward
        def mask_decoder_forward_wrapper(*args, **kwargs):
            new_args = [a.to(torch.bfloat16) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args]
            new_kwargs = {k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for k, v in kwargs.items()}
            return original_forward(*new_args, **new_kwargs)
        mask_decoder.forward = mask_decoder_forward_wrapper

    dataset = ForestLossDataset(args.test_json_path, args.image_folder, tokenizer, CLIPImageProcessor.from_pretrained(model.config.vision_tower), ResizeLongestSide(1024))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    total_loss, ce_loss, mask_loss, count = 0.0, 0.0, 0.0, 0
    
    print(">>> Starting Test Loss Calculation...")
    for batch in tqdm(dataloader):
        images = batch['clip_img'].cuda().bfloat16()
        sam_images = batch['sam_img'].cuda().bfloat16()
        input_ids = batch['input_ids'].cuda()
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        labels = batch['labels'].cuda()
        if labels.dim() == 1: labels = labels.unsqueeze(0)
        gt_masks = batch['masks'].cuda().bfloat16()
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, labels=labels, images=images, global_enc_images=images,
                grounding_enc_images=sam_images, masks_list=[gt_masks[0]], label_list=[gt_masks[0]],
                resize_list=[[batch['resize_shape'][0].item(), batch['resize_shape'][1].item()]],
                offset=torch.tensor([0, 1]).long().cuda(), bboxes=None, attention_masks=None
            )
            if 'loss' in outputs:
                total_loss += outputs['loss'].item()
                ce_loss += outputs.get('ce_loss', torch.tensor(0)).item()
                mask_loss += outputs.get('mask_loss', torch.tensor(0)).item()
                count += 1
        
        del images, sam_images, input_ids, labels, gt_masks, outputs
        torch.cuda.empty_cache()

    if count > 0:
        print("\n" + "="*40)
        print(f" [FINAL TEST RESULTS]")
        print(f" - Average Loss: {total_loss/count:.4f}")
        print(f" - Text (CE) Loss: {ce_loss/count:.4f}")
        print(f" - Mask (Seg) Loss: {mask_loss/count:.4f}")
        print("="*40)

if __name__ == "__main__": main()