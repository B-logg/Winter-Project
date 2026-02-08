import torchvision.transforms.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch

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
                           # Add other dataset mappings here
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
                           # Add other dataset mappings here
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
                           # Add other dataset mappings here
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples, batch_size,
            precision, image_size, num_classes_per_sample, sample_rate
        )



def _lazy_load_image(data, target_size=336):
    import torch
    from PIL import Image
    import os
    import torchvision.transforms.functional as F
    if isinstance(data, torch.Tensor):
        if data.shape[-1] != target_size:
            return F.resize(data, (target_size, target_size))
        return data
    img = None
    if isinstance(data, str):
        base_root = '/shared/home/naislab/학부연구생/bosung/Winter-Project/datasets/datasets'
        full_path = data if data.startswith('/') else os.path.join(base_root, data)
        if os.path.exists(full_path):
            try: img = Image.open(full_path).convert('RGB')
            except: pass
    elif isinstance(data, Image.Image):
        img = data.convert('RGB')
    if img is None: return torch.zeros((3, target_size, target_size))
    try:
        img = img.resize((target_size, target_size))
        tensor = F.to_tensor(img)
        tensor = F.normalize(tensor, mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        return tensor
    except: return torch.zeros((3, target_size, target_size))

def custom_collate_fn(batch, tokenizer=None, use_mm_start_end=True, inference=False, local_rank=-1):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    bboxes_list, conversation_list, masks_list = [], [], []
    label_list, resize_list, questions_list = [], [], []
    selected_labels_list, offset_list, inferences = [], [0], []
    cnt = 0

    # Iterating through the batch
    for item in batch:
        # [Fix] 딕셔너리 형태 데이터 처리 (Key로 접근)
        if isinstance(item, dict):
            image_path = item.get("image_path", item.get("file_name", None))
            # 이미지가 들어올 곳을 찾음 (global_enc_image or image)
            global_enc_image = item.get("global_enc_images", item.get("global_enc_image", item.get("image", None)))
            grounding_enc_image = item.get("grounding_enc_images", item.get("grounding_enc_image", global_enc_image))
            bboxes = item.get("bboxes", None)
            conversations = item.get("conversations", None)
            masks = item.get("masks", None)
            label = item.get("label", None)
            resize = item.get("resize", None)
            questions = item.get("questions", None)
            sampled_classes = item.get("sampled_classes", None)
        # 기존 튜플 형태 데이터 처리
        elif len(item) == 5:
            image_path, global_enc_image, grounding_enc_image, bboxes, conversations = item
            masks = None; label = None; resize = None; questions = None; sampled_classes = None
        else:
            image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, resize, questions, sampled_classes = item

        # [Fix] 이미지 로딩 및 텐서 변환 보장
        # global_enc_image가 문자열(경로)이면 로딩 시도
        if True: # 무조건 변환 시도 (str/Image 모두)
            global_enc_image = _lazy_load_image(global_enc_image, 336)
        # 로딩 실패하거나 None인 경우 더미 이미지
        if global_enc_image is None:
            global_enc_image = torch.zeros((3, 336, 336))
        # grounding 이미지도 없으면 global 이미지 복사
        if grounding_enc_image is None or isinstance(grounding_enc_image, str):
            grounding_enc_image = global_enc_image

        image_path_list.append(image_path)
        global_enc_image_list.append(global_enc_image)
        # [Fix] Grounding 이미지 텐서 변환 (리스트 추가 직전)
        grounding_enc_image = _lazy_load_image(grounding_enc_image, 1024)
        grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        masks_list.append([] if masks is None else masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        offset_list.append(cnt := cnt + len(conversations))
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # [Fix] 대화 데이터가 리스트일 경우 내부 value 수정
        new_conv_list = []
        for conv in conversation_list:
            if isinstance(conv, str):
                new_conv_list.append(conv.replace(DEFAULT_IMAGE_TOKEN, replace_token))
            elif isinstance(conv, list):
                new_turn_list = []
                for turn in conv:
                    if isinstance(turn, dict) and "value" in turn:
                        turn["value"] = turn["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
                    new_turn_list.append(turn)
                new_conv_list.append(new_turn_list)
            else:
                new_conv_list.append(conv)
        conversation_list = new_conv_list

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [tokenizer_image_token(_convert_conv_to_string(prompt), tokenizer, return_tensors="pt") for prompt in conversation_list],
        batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 575
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                )

    return {
        "image_paths": image_path_list,
        "global_enc_images": torch.stack(global_enc_image_list, dim=0),
        "grounding_enc_images": None if grounding_enc_image_list[0] is None else torch.stack(grounding_enc_image_list, dim=0),
        "bboxes": None if bboxes_list[0] is None else bboxes_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": None if masks_list[0] is None else masks_list,
        "label_list": None if label_list[0] is None else label_list,
        "resize_list": None if resize_list[0] is None else resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    # [Config] 컨텍스트 길이 확장 (User Request: 2048 -> 8192)
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < 8192:
        tokenizer.model_max_length = 8192
    import torch
    import json
    try:
        roles = {'human': conversation[0]['from'], 'gpt': conversation[1]['from']}
    except:
        roles = {'human': 'human', 'gpt': 'gpt'}
    source = conversation
    if roles[source[0]['from']] != roles['human']:
        source = source[1:]

    input_ids = []
    targets = []
    IGNORE_INDEX = -100

    for i, sentence in enumerate(source):
        role = roles[sentence['from']]
        val = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if role == roles['human']:
            input_ids.append(tokenizer.bos_token_id)
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids.extend(ids)
            input_ids.append(sep)
            targets.extend([IGNORE_INDEX] * (len(ids) + 2))
        elif role == roles['gpt']:
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids.extend(ids)
            input_ids.append(sep2)
            targets.extend(ids)
            targets.append(sep2)

    # [Fix] 오직 정수(int)만 남기기 (문자열 제거)
    input_ids = [x for x in input_ids if isinstance(x, int)]
    targets = [x for x in targets if isinstance(x, int)]

    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[:tokenizer.model_max_length]
        targets = targets[:tokenizer.model_max_length]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    import torch
    import json
    try:
        roles = {'human': conversation[0]['from'], 'gpt': conversation[1]['from']}
    except:
        roles = {'human': 'human', 'gpt': 'gpt'}
    source = conversation
    if roles[source[0]['from']] != roles['human']:
        source = source[1:]

    input_ids = []
    targets = []
    IGNORE_INDEX = -100

    for i, sentence in enumerate(source):
        role = roles[sentence['from']]
        val = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if role == roles['human']:
            input_ids.append(tokenizer.bos_token_id)
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids.extend(ids)
            input_ids.append(sep)
            targets.extend([IGNORE_INDEX] * (len(ids) + 2))
        elif role == roles['gpt']:
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids.extend(ids)
            input_ids.append(sep2)
            targets.extend(ids)
            targets.append(sep2)

    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[:tokenizer.model_max_length]
        targets = targets[:tokenizer.model_max_length]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    try:
        roles = {'human': conversation[0]['from'], 'gpt': conversation[1]['from']}
    except:
        roles = {'human': 'human', 'gpt': 'gpt'}
    source = conversation
    if roles[source[0]['from']] != roles['human']:
        source = source[1:]

    input_ids = []
    targets = []
    # IGNORE_INDEX는 보통 -100
    IGNORE_INDEX = -100

    for i, sentence in enumerate(source):
        role = roles[sentence['from']]
        val = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if role == roles['human']:
            input_ids.append(tokenizer.bos_token_id)
            # 텍스트 토큰화 결과는 리스트로 변환
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids.extend(ids)
            input_ids.append(sep)
            targets.extend([IGNORE_INDEX] * (len(ids) + 2)) # bos + sep 포함
        elif role == roles['gpt']:
            ids = tokenizer(val, add_special_tokens=False).input_ids
            input_ids.extend(ids)
            input_ids.append(sep2)
            targets.extend(ids)
            targets.append(sep2)

    # 길이 제한 (모델 max_length 초과 방지)
    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[:tokenizer.model_max_length]
        targets = targets[:tokenizer.model_max_length]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    import json
    import torch
    try:
        # 1. 데이터 위생 처리 (문자열이면 딕셔너리로 변환)
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        if isinstance(conversation, list) and len(conversation) > 0 and isinstance(conversation[0], str):
            conversation = [json.loads(c) for c in conversation]

        # 2. 데이터 유효성 검사 (이상하면 기본값 사용)
        if not isinstance(conversation, list) or len(conversation) < 2:
            raise ValueError("Invalid conversation format")

        # 3. 정상 처리 로직
        roles = {"human": conversation[0]["from"], "gpt": conversation[1]["from"]}
        source = conversation
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_ids = []
        targets = []
        for i, sentence in enumerate(source):
            role = roles[sentence["from"]]
            val = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if role == roles["human"]:
                input_ids.append(tokenizer.bos_token_id)
                input_ids.extend(tokenizer(val, add_special_tokens=False).input_ids)
                input_ids.append(sep)
                targets.extend([IGNORE_INDEX] * (len(input_ids) - len(targets)))
            elif role == roles["gpt"]:
                tokenized = tokenizer(val, add_special_tokens=False).input_ids
                input_ids.extend(tokenized)
                input_ids.append(sep2)
                targets.extend(tokenized)
                targets.append(sep2)
    except Exception as e:
        # 4. 최후의 수단: 에러 시 더미 데이터 반환 (학습 중단 방지)
        # print(f"Data Skip: {e}") # 로그 너무 많을까봐 주석 처리
        return torch.tensor([1, 2]), torch.tensor([1, 2])

    return torch.tensor(input_ids), torch.tensor(targets)
