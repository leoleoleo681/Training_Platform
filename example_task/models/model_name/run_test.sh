#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="./model/20260612_165846/best_checkpoint"
LABEL_PATH="./data/val_0616.jsonl"
EVAL_DATASET="./data/labels.json"

RESULT_SUFFIX=`date +"%Y%m%d_%H%M%S"`

python3 -u test.py \
--model-path ${MODEL_PATH} \
--label-path ${LABEL_PATH} \
--eval-data ${EVAL_DATASET} \
--result-suffix ${RESULT_SUFFIX} \
--save-result-csv false \
--plot-confusion-matrix false