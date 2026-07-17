#!/usr/bin/env bash
set -euo pipefail

# ==================== 可修改配置 ====================
# 必须改为容器内 Single-label_Classification 目录的绝对路径。
SINGLE_LABEL_CLASSIFICATION_DIR="/user_home/du.jing/国家安全/Single-label_Classification"
TRAIN_FILE_NAME="train_80-train_human_label_all_until_0716_top10labels.jsonl"
LABEL_FILE_NAME="labels_0716_10type.json"
TEST_FILE_NAME="test_20-train_human_label_all_until_0716_top10labels.jsonl"
# ================== 可修改配置结束 ==================

# run_train.sh 固定放在 tasks_id_x/models/model_name/ 下。
# model_name 可以变化，任务目录内部的其他相对路径保持不变。
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
MODEL_DIR="$(dirname -- "${SCRIPT_PATH}")"
TASK_DIR="$(cd -- "${MODEL_DIR}/../.." && pwd)"

# 训练开始前，模型目录中只允许存在.sh脚本，防止混入上一次训练产物。
shopt -s nullglob dotglob
for entry in "${MODEL_DIR}"/*; do
    if [[ ! -f "${entry}" || "${entry}" != *.sh ]]; then
        echo "模型目录中存在非sh文件或目录，拒绝开始训练: ${entry}" >&2
        exit 1
    fi
done
shopt -u nullglob dotglob

[[ -f "${SINGLE_LABEL_CLASSIFICATION_DIR}/train.py" ]] || { echo "训练脚本不存在: ${SINGLE_LABEL_CLASSIFICATION_DIR}/train.py" >&2; exit 1; }
[[ -d "${SINGLE_LABEL_CLASSIFICATION_DIR}/pretrained_model/TinyBert" ]] || { echo "预训练模型目录不存在: ${SINGLE_LABEL_CLASSIFICATION_DIR}/pretrained_model/TinyBert" >&2; exit 1; }
[[ -f "${TASK_DIR}/datasets/training/${TRAIN_FILE_NAME}" ]] || { echo "训练数据不存在: ${TASK_DIR}/datasets/training/${TRAIN_FILE_NAME}" >&2; exit 1; }
[[ -f "${TASK_DIR}/datasets/labels/${LABEL_FILE_NAME}" ]] || { echo "标签文件不存在: ${TASK_DIR}/datasets/labels/${LABEL_FILE_NAME}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONDONTWRITEBYTECODE=1
LOG_FILE="${MODEL_DIR}/train_$(date +'%Y%m%d_%H%M%S').log"

echo "脚本位置: ${SCRIPT_PATH}"
echo "模型输出: ${MODEL_DIR}/output_models/model"
echo "训练日志: ${LOG_FILE}"

python3 -u "${SINGLE_LABEL_CLASSIFICATION_DIR}/train.py" \
    --model_name_or_path "${SINGLE_LABEL_CLASSIFICATION_DIR}/pretrained_model/TinyBert" \
    --train_file "${TASK_DIR}/datasets/training/${TRAIN_FILE_NAME}" \
    --label_file "${TASK_DIR}/datasets/labels/${LABEL_FILE_NAME}" \
    --output_dir "${MODEL_DIR}/output_models" \
    --loss_type "Focal" \
    --distribution_gamma 0.25 \
    --focal_gamma 1.0 \
    --non_security_keep_ratio 1.0 \
    --save_each_epoch \
    --evaluate_file "${TASK_DIR}/datasets/test/${TEST_FILE_NAME}" \
    2>&1 | tee "${LOG_FILE}"
