#!/bin/bash


# 1. 기본 경로 설정
PROJ_ROOT=/shared/home/naislab/학부연구생/bosung/Winter-Project
DATA_PATH="$PROJ_ROOT/datasets/datasets/train.json"
IMAGE_FOLDER="$PROJ_ROOT/datasets/datasets"
MODEL_PATH="$PROJ_ROOT/groundingLMM/checkpoints/GLaMM-GCG"
VISION_PRETRAINED="$PROJ_ROOT/groundingLMM/checkpoints/sam_vit_h_4b8939.pth"
OUTPUT_DIR="$PROJ_ROOT/checkpoints/GLaMM-Forest-A40-4GPU"
mkdir -p $OUTPUT_DIR

# 2. GPU 설정
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="./:$PYTHONPATH"

# 3. [핵심] 라이브러리 경로 강제 지정 (찾으신 경로 적용!)
# -----------------------------------------------------------------------
# 찾으신 cusparse 경로의 상위 폴더(lib)를 지정합니다.
export LD_LIBRARY_PATH=/shared/home/naislab/학부연구생/bosung/my_envs/glamm/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH

# PyTorch 및 기타 라이브러리 경로도 절대 경로로 지정
export LD_LIBRARY_PATH=/shared/home/naislab/학부연구생/bosung/my_envs/glamm/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/shared/home/naislab/학부연구생/bosung/my_envs/glamm/lib:$LD_LIBRARY_PATH
# -----------------------------------------------------------------------

# 4. 학습 시작
deepspeed --num_gpus=2 train_glamm.py \
    --version $MODEL_PATH \
    --dataset_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_pretrained $VISION_PRETRAINED \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --grad_accumulation_steps 6 \
    --epochs 3 \
    --lr 2e-4 \
    --workers 8 \
    --print_freq 1 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --vision_tower "openai/clip-vit-large-patch14-336" \
    --use_mm_start_end \
    --train_mask_decoder

# nohup ./run_a100_forest.sh > log.txt 2>&1 &
# tail -f log.txt