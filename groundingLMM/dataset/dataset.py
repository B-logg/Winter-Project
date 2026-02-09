import torchvision.transforms.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
import json
import copy

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
from dataset.caption_datasets.LLavaInstruct_vqa_ds import LLaVAInstructDataset
from dataset.region_datasets.Flickr_Region_ds import Flickr30kRegDataset
from dataset.segm_datasets.Semantic_Segm_ds import SemanticSegmDataset
from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from dataset.gcg_datasets.GranDf_gcg_ds import GranDfDataset, OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
from dataset.region_datasets.RefCOCO_VG_Region_ds import (RefCocoRegDataset, RefCocoGRegDataset, RefCocoPRegDataset,
                                                          VisualGenomeRegDataset)
from dataset.caption_datasets.GranD_ShortCaption_ds import GrandShortCaptionDataset
from dataset.region_datasets.GranD_ReferringRegion_ds import GrandReferRegDataset
from dataset.segm_datasets.GranD_ReferringSegm_ds import GrandReferSegmDataset
from tools.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def _convert_conv_to_string(conv):
    if isinstance(conv, str): return conv
    if isinstance(conv, list):
        parts = []
        for t in conv:
            if isinstance(t, dict) and "value" in t:
                parts.append(t["value"])
            else:
                parts.append(str(t))
        return "\n".join(parts)
    return str(conv)

class HybridDatasetBase(torch.utils.data.Dataset):
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config,
                 epoch_samples=500 * 8 * 2 * 10, batch_size=2, precision="fp32", image_size=224,
                 num_classes_per_sample=3, sample_rate=None):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.global_image_encoder = global_image_encoder
        self.dataset = dataset
        self.datasets_config = datasets_config
        self.epoch_samples = epoch_samples
        self.batch_size = batch_size
        self.precision = precision
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_list = dataset.split("||")
        self.sample_rate = np.array(sample_rate or [1] * len(self.dataset_list))
        self.sample_rate /= self.sample_rate.sum()
        self.all_datasets = self.create_datasets()

    def create_datasets(self):
        datasets = []
        for ds in self.dataset_list:
            dataset_cls = self.datasets_config.get(ds)
            if dataset_cls:
                if ds == 'Semantic_Segm':
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir, self.tokenizer, self.global_image_encoder, self.epoch_samples,
                            self.precision, self.image_size, self.num_classes_per_sample, self.semantic_segm_data, )
                        )
                elif ds == 'Refer_Segm':
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir, self.tokenizer, self.global_image_encoder, self.epoch_samples,
                            self.precision, self.image_size, self.num_classes_per_sample, self.refer_segm_data, )
                        )
                else:
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir, self.tokenizer, self.global_image_encoder, self.epoch_samples,
                            self.precision, self.image_size, self.num_classes_per_sample, )
                        )
        return datasets

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        data = selected_dataset[0]
        return (*data,)


class HybridCapDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10, batch_size=2,
                 precision="fp32", image_size=224, num_classes_per_sample=3,
                 dataset="CocoCap||LLaVaInstruct", sample_rate=[1, 1]):
        datasets_config = {"CocoCap": CocoCapDataset,
                           "LLaVaInstruct": LLaVAInstructDataset,
                           "GrandCaptionDataset": GrandShortCaptionDataset,
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples, batch_size,
            precision, image_size, num_classes_per_sample, sample_rate
        )


class HybridRegDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10, batch_size=2,
                 precision="fp32", image_size=224, num_classes_per_sample=3,
                 dataset="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg||Flickr_Reg", sample_rate=[1, 1, 1, 1, 1]):
        datasets_config = {"RefCoco_Reg": RefCocoRegDataset,
                           "RefCocoG_Reg": RefCocoGRegDataset,
                           "RefCocoP_Reg": RefCocoPRegDataset,
                           "VisGen_Reg": VisualGenomeRegDataset,
                           "Flickr_Reg": Flickr30kRegDataset,
                           "GrandRefer_Reg": GrandReferRegDataset,
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples, batch_size,
            precision, image_size, num_classes_per_sample, sample_rate
        )


class HybridSegDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10, batch_size=2,
                 precision="fp32", image_size=224, num_classes_per_sample=3,
                 dataset="Semantic_Segm||Refer_Segm||PSG_GCG||RefCoco_GCG||GranDf_GCG||Flickr_GCG",
                 sample_rate=[5,4,1,1,1,1],
                 semantic_segm_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
                 refer_segm_data="refcoco||refcocog||refcoco+||refclef"):
        self.semantic_segm_data = semantic_segm_data
        self.refer_segm_data = refer_segm_data
        datasets_config = {"Semantic_Segm": SemanticSegmDataset,
                           "Refer_Segm": ReferSegmDataset,
                           "PSG_GCG": OpenPsgGCGDataset,
                           "RefCoco_GCG": RefCOCOgGCGDataset,
                           "GranDf_GCG": GranDfDataset,
                           "Flickr_GCG": Flickr30kGCGDataset,
                           "GrandRefer_Segm": GrandReferSegmDataset,
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples, batch_size,
            precision, image_size, num_classes_per_sample, sample_rate
        )


def _lazy_load_image(data, target_size=336):
    """
    이미지 로드 및 전처리 헬퍼 함수
    - Tensor인 경우 리사이즈
    - Path인 경우 로드 후 전처리
    - None인 경우 더미 텐서 반환
    """
    if isinstance(data, torch.Tensor):
        if data.shape[-1] != target_size:
            # 배치 차원이 없는 [C, H, W] 가정
            return F.resize(data, (target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC)
        return data
        
    img = None
    if isinstance(data, str):
        # [Fix] 절대 경로 처리를 위해 base_root는 상황에 맞게 수정 혹은 제거
        if os.path.exists(data):
            try: img = Image.open(data).convert('RGB')
            except: pass
    elif isinstance(data, Image.Image):
        img = data.convert('RGB')
        
    if img is None: 
        return torch.zeros((3, target_size, target_size))
        
    try:
        # GLaMM/LLaVA 표준 전처리
        img = img.resize((target_size, target_size), resample=Image.BICUBIC)
        tensor = F.to_tensor(img)
        # CLIP Normalize Mean/Std
        tensor = F.normalize(tensor, mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        return tensor
    except: 
        return torch.zeros((3, target_size, target_size))


def custom_collate_fn(batch, tokenizer=None, use_mm_start_end=True, inference=False, local_rank=-1):
    # Initializing lists
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    bboxes_list, conversation_list, masks_list = [], [], []
    # label_list는 모델이 마스크 정보를 얻기 위해 사용하므로, 마스크를 담을 리스트로 활용합니다.
    resize_list, questions_list = [], [] 
    selected_labels_list, offset_list, inferences = [], [0], []
    cnt = 0

    for item in batch:
        if isinstance(item, dict):
            image_path = item.get("image_path", item.get("file_name", None))
            global_enc_image = item.get("global_enc_images", item.get("global_enc_image", item.get("image", None)))
            grounding_enc_image = item.get("grounding_enc_images", item.get("grounding_enc_image", global_enc_image))
            
            bboxes = item.get("bboxes", None)
            conversations = item.get("conversations", None)
            masks = item.get("masks", None)
            # label = item.get("label", None) # <-- [삭제] 모델은 label_list를 마스크 shape 확인용으로 씁니다.
            resize = item.get("resize_list", item.get("resize", None))
            questions = item.get("questions", None)
            sampled_classes = item.get("sampled_classes", None)
        # (Tuple unpacking 로직 생략 - Dict 기반이므로)
        else:
             # Fallback logic if needed
             pass

        # 이미지 로드
        global_enc_image = _lazy_load_image(global_enc_image, 336)
        grounding_enc_image = _lazy_load_image(grounding_enc_image, 1024)

        image_path_list.append(image_path)
        global_enc_image_list.append(global_enc_image)
        grounding_enc_image_list.append(grounding_enc_image)
        
        bboxes_list.append(bboxes)
        
        if isinstance(conversations, list):
            conversation_list.extend(conversations)
        else:
            conversation_list.append(conversations)

        # [마스크 처리]
        # ForestDataset에서 이미 빈 텐서(0, 1024, 1024) 처리를 했으므로 그대로 담습니다.
        # None 체크는 방어적으로 수행
        if masks is not None:
             masks_list.append(masks.float())
        else:
             masks_list.append(torch.zeros((0, 1024, 1024)).float())

        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        
        conv_len = len(conversations) if isinstance(conversations, list) else 1
        cnt += conv_len
        offset_list.append(cnt)
        inferences.append(inference)

    # 대화 처리 및 토큰화 로직 (기존과 동일)
    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        new_conv_list = []
        for conv in conversation_list:
            if isinstance(conv, str):
                new_conv_list.append(conv.replace(DEFAULT_IMAGE_TOKEN, replace_token))
            elif isinstance(conv, list):
                new_turn_list = []
                for turn in conv:
                    if isinstance(turn, dict) and "value" in turn:
                        turn_copy = copy.deepcopy(turn)
                        turn_copy["value"] = turn["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
                        new_turn_list.append(turn_copy)
                    else:
                        new_turn_list.append(turn)
                new_conv_list.append(new_turn_list)
            else:
                new_conv_list.append(conv)
        conversation_list = new_conv_list

    input_ids_list = []
    for prompt in conversation_list:
        prompt_str = _convert_conv_to_string(prompt)
        tokenized = tokenizer_image_token(prompt_str, tokenizer, return_tensors="pt")
        input_ids_list.append(tokenized)
        
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()
    conv_template = conversation_lib.default_conversation.copy()
    sep = conv_template.sep + conv_template.roles[1] + ": "
    sep2 = conv_template.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 576
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    # [최종 반환] GLaMM.py의 model_forward 인자와 1:1 매핑
    return {
        "image_paths": image_path_list,
        "global_enc_images": torch.stack(global_enc_image_list, dim=0),
        "grounding_enc_images": torch.stack(grounding_enc_image_list, dim=0),
        "bboxes": bboxes_list if any(x is not None for x in bboxes_list) else None,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        
        # [핵심] masks_list와 label_list에 동일한 마스크 리스트를 전달합니다.
        "masks_list": masks_list, 
        "label_list": masks_list, # 모델 내부에서 label_list[i].shape를 호출하므로 필수
        
        "resize_list": resize_list if any(x is not None for x in resize_list) else None,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        # 호환성을 위해 추가
        "images": torch.stack(global_enc_image_list, dim=0) 
    }

def _process_conversation(conversation, target, tokenizer, sep, sep2):
    """
    대화 내용을 기반으로 Masking된 Label(Target)을 생성하는 함수
    - Human 발화: IGNORE_INDEX (-100)
    - GPT 발화: Loss 계산
    """
    # 컨텍스트 길이 안전장치
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < 2048:
        tokenizer.model_max_length = 2048

    roles = {"human": "human", "gpt": "gpt"}
    if isinstance(conversation, list) and len(conversation) > 0:
        if "from" in conversation[0]:
             roles = {"human": conversation[0]["from"], "gpt": conversation[1]["from"]}

    source = conversation
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_ids_list = []
    targets_list = []
    
    for i, sentence in enumerate(source):
        role = roles[sentence["from"]]
        val = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        
        if role == roles["human"]:
            input_ids_list.append(tokenizer.bos_token_id)
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids_list.extend(ids)
            
            # [안전장치] sep 토큰화
            if sep is not None:
                sep_ids = tokenizer(sep, add_special_tokens=False).input_ids
                input_ids_list.extend(sep_ids)
                current_len = 1 + len(ids) + len(sep_ids) 
            else:
                current_len = 1 + len(ids)
            
            targets_list.extend([IGNORE_INDEX] * current_len)
            
        elif role == roles["gpt"]:
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids_list.extend(ids)
            
            # [핵심 수정] sep2가 None인지 확인 후 처리
            if sep2 is not None:
                sep2_ids = tokenizer(sep2, add_special_tokens=False).input_ids
                input_ids_list.extend(sep2_ids)
                
                targets_list.extend(ids)
                targets_list.extend(sep2_ids)
            else:
                # sep2가 없으면 그냥 문장만 추가
                targets_list.extend(ids)

    # Tensor 할당 (길이 검증)
    limit = min(len(targets_list), target.shape[0])
    target[:limit] = torch.tensor(targets_list[:limit], dtype=torch.long)
    
    return target