#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

SINGLE_LABEL_CLASSIFICATION_DIR="../../../Single-label_Classification"
MODEL_PATH="./output_models/model"

python3 -u "${SINGLE_LABEL_CLASSIFICATION_DIR}/test_multi.py" \
--model-path ${MODEL_PATH} \