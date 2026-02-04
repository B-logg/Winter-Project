#!/bin/bash

# [A100 산림 특화 학습 설정]

# 프로젝트 루트
PROJ_ROOT=~/Winter-Project

# 데이터셋 경로
DATA_PATH="$PROJ_ROOT/datasets/datasets/train.json"
IMAGE_FOLDER="$PROJ_ROOT/datasets/datasets"

# 모델 체크포인트
MODEL_PATH="$PROJ_ROOT/groundingLMM/checkpoints/GLaMM-GCG"
# SAM 체크포인트 (GLaMM 모델 안에 있으면 자동 로드되지만 명시)
VISION_PRETRAINED="$PROJ_ROOT/groundingLMM/checkpoints/sam_vit_h_4b8939.pth"

# 결과 저장
OUTPUT_DIR="$PROJ_ROOT/checkpoints/GLaMM-Forest-Weight"

# GPU 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

# DeepSpeed 실행
deepspeed --num_gpus=8 train_glamm.py \
    --version $MODEL_PATH \
    --dataset_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_pretrained $VISION_PRETRAINED \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --epochs 5 \
    --lr 2e-4 \
    --lora_r 128 \
    --lora_alpha 256 \
    --use_mm_start_end
