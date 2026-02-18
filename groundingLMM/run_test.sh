#!/bin/bash

# ==================== [경로 설정] ====================
PROJ_ROOT=/shared/home/naislab/학부연구생/bosung/Winter-Project

# 학습된 모델 경로
CKPT_PATH="$PROJ_ROOT/checkpoints/GLaMM-Forest-A40-4GPU/checkpoint-epoch-3"

# 테스트 데이터셋 (val.json 대신 test.json 사용)
TEST_JSON="$PROJ_ROOT/datasets/datasets/val.json"
IMAGE_FOLDER="$PROJ_ROOT/datasets/datasets"

# 결과 저장 폴더
RESULT_DIR="./final_test_results"
# ====================================================

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"

echo "##################################################"
echo " PHASE 1: Calculating Test Loss & Generating Output"
echo "##################################################"

python eval/gcg/test_forest.py \
    --hf_model_path $CKPT_PATH \
    --test_json_path $TEST_JSON \
    --image_folder $IMAGE_FOLDER \
    --output_dir $RESULT_DIR

echo ""
echo "##################################################"
echo " PHASE 2: Calculating mIoU, AP50, Recall, CIDEr.."
echo "##################################################"

python eval/gcg/calc_metrics.py \
    --pred_path "$RESULT_DIR/test_predictions.json" \
    --gt_path $TEST_JSON \
    --image_folder $IMAGE_FOLDER