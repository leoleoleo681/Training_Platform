FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG IMAGE_VERSION=1.0.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/training \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    TRAINING_TASK_ROOT=/mnt/task \
    TRAINING_IMAGE_FAMILY=training-text-single \
    TRAINING_PROTOCOL_VERSION=1 \
    TRAINING_ALGORITHM_ROOT=/opt/training/algorithm \
    TRAINING_PRETRAINED_MODEL=/opt/training/algorithm/pretrained_model/TinyBert \
    TRAINING_IMAGE_VERSION=${IMAGE_VERSION}

WORKDIR /opt/training

COPY requirements.txt /opt/training/requirements.txt
RUN python -m pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" \
        -r /opt/training/requirements.txt

COPY algorithm/ /opt/training/algorithm/
COPY entrypoint.py /opt/training/entrypoint.py

RUN test -f "${TRAINING_PRETRAINED_MODEL}/config.json" \
    && test -f "${TRAINING_PRETRAINED_MODEL}/vocab.txt" \
    && find "${TRAINING_PRETRAINED_MODEL}" -maxdepth 1 -type f \
         \( -name 'pytorch_model.bin' -o -name 'model.safetensors' \) | grep -q . \
    && python -c "import torch, transformers, sklearn, tensorboard, pandas, openpyxl, matplotlib, seaborn, onnx, onnxruntime; assert torch.__version__.split('+')[0] == '2.2.2'; assert torch.version.cuda == '11.8'; assert transformers.__version__ == '4.40.2'; assert sklearn.__version__ == '1.4.2'; assert tensorboard.__version__ == '2.16.2'; assert pandas.__version__ == '2.2.2'" \
    && cd /opt/training/algorithm \
    && python -c "import train, test, test_multi, convert_to_onnx"

RUN if ! getent group 1000 >/dev/null; then groupadd --gid 1000 trainer; fi \
    && if ! getent passwd 1000 >/dev/null; then useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin trainer; fi \
    && mkdir -p /tmp/huggingface /tmp/matplotlib /tmp/cache \
    && chown -R 1000:1000 /tmp/huggingface /tmp/matplotlib /tmp/cache

LABEL org.opencontainers.image.title="Training Text Single-label" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      io.training-platform.image-family="training-text-single" \
      io.training-platform.protocol-version="1" \
      io.training-platform.task-types="text_classification_single"

USER 1000:1000
ENTRYPOINT ["python", "/opt/training/entrypoint.py"]
