#!/usr/bin/env bash
set -Eeuo pipefail

TASK_ROOT="${TASK_ROOT:-/mnt/sdb/user_home/lu.boyu/Training_Platform/docker_Training_Platform/Training_Platform/task_test}"
TASK_ID="${TASK_ID:-task_test}"
MODEL_NAME="${MODEL_NAME:-first_test_docker}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
IMAGE="${IMAGE:-training-nlp:1.0.0}"

if [[ ! "${TASK_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Invalid TASK_ID: ${TASK_ID}" >&2
    exit 2
fi
if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Invalid MODEL_NAME: ${MODEL_NAME}" >&2
    exit 2
fi
if [[ ! "${GPU_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_DEVICES must be a comma-separated list of numeric GPU ids" >&2
    exit 2
fi

TASK_ROOT="$(realpath -- "${TASK_ROOT}")"
DATASETS_DIR="${TASK_ROOT}/datasets"
MODEL_DIR="${TASK_ROOT}/models/${MODEL_NAME}"
CONFIG_FILE="${MODEL_DIR}/run_train.json"

[[ -d "${DATASETS_DIR}" ]] || { echo "Missing datasets directory: ${DATASETS_DIR}" >&2; exit 3; }
[[ -d "${MODEL_DIR}" ]] || { echo "Missing model directory: ${MODEL_DIR}" >&2; exit 3; }
[[ -f "${CONFIG_FILE}" ]] || { echo "Missing training config: ${CONFIG_FILE}" >&2; exit 3; }

docker run \
    --name "train-${TASK_ID}-${MODEL_NAME}" \
    --init \
    --gpus "\"device=${GPU_DEVICES}\"" \
    --shm-size=8g \
    --user 1000:1000 \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --network none \
    --env "TRAINING_TASK_ID=${TASK_ID}" \
    --mount "type=bind,src=${DATASETS_DIR},dst=/mnt/task/datasets,readonly" \
    --mount "type=bind,src=${MODEL_DIR},dst=/mnt/task/models/${MODEL_NAME}" \
    --mount "type=bind,src=${CONFIG_FILE},dst=/mnt/task/models/${MODEL_NAME}/run_train.json,readonly" \
    --tmpfs /tmp:rw,nosuid,nodev,size=2g \
    --tmpfs /mnt/task/dataset_cache:rw,nosuid,nodev,size=4g \
    --cpus=8 \
    --memory=16g \
    --memory-swap=16g \
    --pids-limit=1024 \
    --stop-timeout=30 \
    --log-driver=json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    "${IMAGE}" \
    train --task-type text_classification_single \
          --config "/mnt/task/models/${MODEL_NAME}/run_train.json"
