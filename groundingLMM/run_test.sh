#!/bin/bash

CHECKPOINT_PATH="/home/sbosung1789/Winter-Project/groundingLMM/checkpoints/checkpoint-epoch-3"

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH가 비어있습니다."
    exit 1
fi

echo "Starting Evaluation with checkpoint: $CHECKPOINT_PATH"

python test_glamm.py \
    --hf_model_path "$CHECKPOINT_PATH" \
    --test_json_path "/home/sbosung1789/Winter-Project/groundingLMM/dataset/datasets/glamm_test.json" \
    --image_folder "/home/sbosung1789/Winter-Project/groundingLMM/dataset/datasets/glamm_images_train" \
    --output_dir "./test_results" \
    --batch_size 1

echo "Evaluation Finished!"