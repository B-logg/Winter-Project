#!/bin/bash

export OMP_NUM_THREADS=8 

echo "Starting DeepSpeed Training on 3 GPUs"

# 절대 경로 변수 세팅
BASE_DIR="$HOME/학부연구생/bosung/Winter-Project/groundingLMM"
DATA_DIR="$BASE_DIR/dataset"
CHK_DIR="$BASE_DIR/checkpoints"

BASE_MODEL_PATH="$CHK_DIR/GLaMM-GCG"
OUTPUT_DIR="$CHK_DIR/GLaMM-GCG_tuned"

deepspeed --num_gpus=3 train_glamm.py \
    --deepspeed deepspeed_config.json \
    --version "$BASE_MODEL_PATH" \
    --dataset_path "$DATA_DIR/glamm_train.json" \
    --image_folder "$DATA_DIR/glamm_images_train" \
    --vision_pretrained "$CHK_DIR/sam_vit_h_4b8939.pth" \
    --vision_tower "openai/clip-vit-large-patch14-336" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 2 \
    --grad_accumulation_steps 6 \
    --workers 8 \
    --lr 2e-4 \
    --epochs 4 \
    --val_ratio 0.05 \
    --lora_r 128 \
    --lora_alpha 256

echo "Training finished successfully!"


# nohup ./run_train.sh > train_log.txt 2>&1 &
# tail -f train_log.txt
# tmux attach-session -t <session number>
