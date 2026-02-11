import torch
import transformers
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import argparse
import bitsandbytes as bnb

# 사용자님의 모델 파일 import
from model.GLaMM import GLaMMForCausalLM 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    # 경로 관련 인자는 사용자님 환경에 맞게 수정 필요하면 수정해주세요
    parser.add_argument("--local_rank", default=0, type=int)
    return parser.parse_args()

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # 4비트 레이어 클래스도 확인
    import bitsandbytes as bnb
    
    print("\n[Diagnostic] Searching for Linear layers...")
    for name, module in model.named_modules():
        # 모듈의 진짜 타입을 확인
        if isinstance(module, cls) or isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[-1])
            
            # 샘플로 몇 개만 자세히 출력 (너무 많으니까)
            if "grounding_encoder" in name or "layers.0.self_attn" in name:
                print(f"  Found: {name} | Type: {type(module)} | Dtype: {module.weight.dtype}")
                
    if 'lm_head' in lora_module_names: 
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    
    print(f"Loading GLaMM from {args.version}...")
    
    # 1. 4-bit 로드 설정 (사용자님 코드와 동일)
    skip_modules = ["vision_tower", "grounding_encoder", "mm_projector", 
                    "text_hidden_fcs", "region_encoder", "lm_head", "embed_tokens"]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules
    )
    
    # 2. 모델 로드
    model = GLaMMForCausalLM.from_pretrained(
        args.version,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map = {"": args.local_rank},
        # 필수 인자들 더미로 채움
        train_mask_decoder=True, out_dim=256,
        ce_loss_weight=1.0, dice_loss_weight=0.5, bce_loss_weight=2.0,
        seg_token_idx=123, vision_pretrained="./checkpoints/sam_vit_h_4b8939.pth",
        vision_tower="openai/clip-vit-large-patch14-336",
        use_mm_start_end=True, mm_vision_select_layer=-2, with_region=True
    )
    
    # 3. 모델 구조 출력
    print("\n" + "="*50)
    print("🔍 [1] 전체 모델 구조 요약 (Top-level modules)")
    print("="*50)
    for name, module in model.named_children():
        print(f"[{name}]: {type(module)}")
        
    print("\n" + "="*50)
    print("🔍 [2] SAM (Grounding Encoder) 내부 구조 확인")
    print("="*50)
    if hasattr(model.model, "grounding_encoder"):
        sam = model.model.grounding_encoder
        print(f"SAM Type: {type(sam)}")
        # SAM 내부의 첫 번째 블록만 찍어서 구조 확인
        for name, mod in sam.named_modules():
            if "layers.0" in name: 
                print(f" - {name} : {type(mod)}")
    else:
        print("❌ SAM not found in model.model.grounding_encoder")

    print("\n" + "="*50)
    print("🔍 [3] find_target_linear_modules 결과 확인")
    print("="*50)
    targets = find_all_linear_names(model)
    print(f"\n👉 Detected Target Modules: {targets}")
    
    print("\n" + "="*50)
    print("✅ 진단 완료. 위 내용을 보여주세요.")
    print("="*50)

if __name__ == "__main__":
    main()