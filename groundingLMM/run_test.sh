#!/bin/bash

CHECKPOINT_PATH=""

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH가 비어있습니다."
    exit 1
fi

echo "Starting Evaluation with checkpoint: $CHECKPOINT_PATH"

python test_glamm.py \
    --hf_model_path "$CHECKPOINT_PATH" \
    --test_json_path "/shared/home/naislab/학부연구생/bosung/Winter-Project/groundingLMM/dataset/glamm_test.json" \
    --image_folder "/shared/home/naislab/학부연구생/bosung/Winter-Project/groundingLMM/dataset/glamm_images_train" \
    --output_dir "./test_results" \
    --batch_size 1

echo "Evaluation Finished!"