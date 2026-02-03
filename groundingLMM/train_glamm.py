import os
import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from model.GLaMM import GLaMMForCausalLM 
from model.llava.model.language_model.llava_llama import LlamaConfig

def find_target_linear_modules(model, exclude_keywords=[]):
    # LoRA를 적용할 선형 레이어를 찾기
    # Full Fine-Tuning할 모듈은 제외
    cls = torch.nn.Linear
    lora_module_names = set()

    for name, module in model.named_modules():
        if any(keyword in name for keyword in exclude_keywords):
            continue

        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

def print_trainable_parameters(model):
    # 학습 가능한 파라미터 수를 출력
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

def train():
    # Argument Parsing
    parser = transformers.HfArgumentParser(
        (transformers.TrainingArguments)
    )

    # 실제 실행 시에는 command line args로 받으므로 여기서는 빈 괄호 or sys.argv 사용
    # 편의 상 스크립트 실행 시 전달된 인자를 받도록 설정
    training_args, = parser.parse_args_into_dataclasses()

    # 모델 경로
    model_path = os.path.expanduser("~/Winter-Project/groundingLMM/checkpoints/GLaMM-GCG")

    # 모델 로드 및 양자화 설정
    # Vision Tower: CLIP
    # grounding_encoder: SAM 전체
    # mm_projector: LLaVA Projector
    # text_hidden_fcs: GLaMM Projector
    # region_encoder: Region Feature Extractor

    skip_modules = ["vision_tower", "grounding_encoder", "mm_projector", 
                    "text_hidden_fcs", "region_encoder", "lm_head"]
    
    print(f"Loading Model with NF4 Quantization (Skipping: {skip_modules})")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # A100 가속
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules
    )

    model = GLaMMForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16, # 스킵된 모듈을 BF16으로 로드
        trust_remote_code=True
    )

    # Q-LoRA 전처리 (Gradient Checkpointing 등)
    model = prepare_model_for_kbit_training(model)

    # 데이터 타입 보정 (BF16으로 맞추기)
    print("Ensuring critical modules are in BF16")

    # model.model은 GLaMMModel 인스턴스
    glamm_model = model.model

    # Vision Tower Casting
    if hasattr(glamm_model, "vision_tower"):
        vt = glamm_model.vision_tower
        if isinstance(vt, list): vt = vt[0]
        for param in vt.parameters():
            param.data = param.data.to(torch.bfloat16)
    
    # SAM Casting
    # GLaMMBaseModel에 정의됨
    if hasattr(glamm_model, "grounding_encoder"):
        for param in glamm_model.grounding_encoder.parameters():
            param.data = param.data.to(torch.bfloat16)

    # Projectors Casting
    # mm_projector (LlavaMetaModel)
    if hasattr(glamm_model, "mm_projector"):
        for param in glamm_model.mm_projector.parameters():
            param.data = param.data.to(torch.bfloat16)

    # text_hidden_fcs (GLaMMBaseModel)
    if hasattr(glamm_model, "text_hidden_fcs"):
        for param in glamm_model.text_hidden_fcs.parameters():
            param.data = param.data.to(torch.bfloat16)

    # region_encoder(LlavaMetaModel)
    if hasattr(glamm_model, "region_encoder"):
        for param in glamm_model.region_encoder.parameters():
            param.data = param.data.to(torch.bfloat16)
    
    # LoRA 설정 (LLM, Vision Tower)
    # LoRA 적용: LLM(Q-LoRA), Vision Tower(Global Image Encoder)
    # Full Fine-Tuning: Mask Decoder(Pixel Decoder), All projectors, Region Encoder

    # LoRA 타겟에서 제외할 키워드
    exclude_keywords = [
        "grounding_encoder", # SAM 전체
        "mm_projector", # projection layer
        "text_hidden_fcs", # projection layer
        "region_encoder" # Region Encoder
    ]

    target_modules = find_target_linear_modules(model, exclude_keywords)
    print(f"LoRA Target Modules: {target_modules}")

    lora_config = LoraConfig(
        r=128, # High rank
        lora_alpha=256,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Full Fine-Tuning 모듈
    # model.base_model.model로 접근해야 원본 모델 접근 가능
    base_glamm = model.base_model.model.model

    # Mask Decoder require_grad=True
    # SAM 구조: grounding_encoder -> mask_decoder
    if hasattr(base_glamm, "grounding_encoder"):
        mask_decoder = base_glamm.grounding_encoder.mask_decoder
        for name, param in mask_decoder.named_parameters():
            param.requires_grad=True

        # Image Encoder & Prompt Encoder는 유지
        for param in base_glamm.grounding_encoder.image_encoder.parameters():
            param.requires_grad = False
        for param in base_glamm.grounding_encoder.prompt_encoder.parameters():
            param.requires_grad = False

    # Projectors & Region Encoder requires_grad=True
    modules_to_unfreeze = ["mm_projector", "text_hidden_fcs", "region_encoder"]

    for mod_name in modules_to_unfreeze:
        if hasattr(base_glamm, mod_name):
            module = getattr(base_glamm, mod_name)
            for param in module.parameters():
                param.requires_grad = True
    
    # 통계 출력
    print_trainable_parameters(model)

    # Trainer 설정 (데이터셋 로더 필요)

    # 기존 GLaMM의 train.py에 있는 DataCollator와 Dataset 클래스를 사용 필요
    # from dataset import GLaMMDataset, DataCollatorForGLaMM
    
    # dataset = GLaMMDataset(...)
    
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     ...
    # )
    
    # trainer.train()
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()