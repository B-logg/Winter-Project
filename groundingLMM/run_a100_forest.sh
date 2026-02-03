#!/bin/bash

# ==========================================
# [A100 8대 산림 특화 학습 설정]
# ==========================================

PROJ_ROOT=~/Winter-Project
# 병합된 데이터셋 경로
DATA_PATH="$PROJ_ROOT/datasets/datasets/final_train.json"
# 이미지 최상위 폴더
IMAGE_FOLDER="$PROJ_ROOT/datasets/datasets"
# 체크포인트 경로
MODEL_PATH="$PROJ_ROOT/groundingLMM/checkpoints/GLaMM-GCG"
# 결과 저장
OUTPUT_DIR="$PROJ_ROOT/checkpoints/GLaMM-Forest-A100-v1"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

# DeepSpeed 실행
deepspeed --num_gpus=8 train_glamm_forest.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version v1 \
    --dataset_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --use_qlora True \
    --qlora_r 128 \
    --qlora_alpha 256 \
    --use_grounding True