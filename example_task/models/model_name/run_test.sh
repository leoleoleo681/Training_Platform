#!/usr/bin/env bash
set -euo pipefail

# ==================== 可修改配置 ====================
# 必须改为容器内 Single-label_Classification 目录的绝对路径。
SINGLE_LABEL_CLASSIFICATION_DIR="/user_home/du.jing/国家安全/Single-label_Classification"
LABEL_FILE_NAME="labels_0716_10type.json"
TEST_FILE_NAME="test_20-train_human_label_all_until_0716_top10labels.jsonl"
# ================== 可修改配置结束 ==================

# run_test.sh 固定放在 tasks_id_x/models/model_name/ 下。
# 无论从哪个工作目录启动，模型、数据和测试输出都按脚本位置解析。
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
MODEL_DIR="$(dirname -- "${SCRIPT_PATH}")"
TASK_DIR="$(cd -- "${MODEL_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_DIR}/output_models/model"
LABEL_PATH="${TASK_DIR}/datasets/labels/${LABEL_FILE_NAME}"
EVAL_DATASET="${TASK_DIR}/datasets/test/${TEST_FILE_NAME}"
RESULT_SUFFIX="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${MODEL_DIR}/test_${RESULT_SUFFIX}.log"

[[ -f "${SINGLE_LABEL_CLASSIFICATION_DIR}/test.py" ]] || { echo "测试代码不存在: ${SINGLE_LABEL_CLASSIFICATION_DIR}/test.py" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "模型不存在或不完整: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${LABEL_PATH}" ]] || { echo "标签文件不存在: ${LABEL_PATH}" >&2; exit 1; }
[[ -f "${EVAL_DATASET}" ]] || { echo "测试数据不存在: ${EVAL_DATASET}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONDONTWRITEBYTECODE=1

echo "脚本位置: ${SCRIPT_PATH}"
echo "测试模型: ${MODEL_PATH}"
echo "测试数据: ${EVAL_DATASET}"
echo "测试输出根目录: ${MODEL_DIR}"
echo "测试日志: ${LOG_FILE}"

python3 -u "${SINGLE_LABEL_CLASSIFICATION_DIR}/test.py" \
    --model-path "${MODEL_PATH}" \
    --label-path "${LABEL_PATH}" \
    --eval-data "${EVAL_DATASET}" \
    --output-dir "${MODEL_DIR}" \
    --result-suffix "${RESULT_SUFFIX}" \
    --save-result-csv false \
    --plot-confusion-matrix false \
    2>&1 | tee "${LOG_FILE}"
