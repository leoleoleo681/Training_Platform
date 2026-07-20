# 构建单标签文本分类镜像

本文只说明`training-text-single`镜像的准备、构建和检查。训练数据挂载及容器启动见[运行说明](../RUN.md)。

## 1. 构建目录

`docker/text-classification-single/`是完整且独立的Docker构建上下文：

```text
text-classification-single/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── entrypoint.py
└── algorithm/
    ├── train.py
    ├── test.py
    ├── classifier_model.py
    ├── data_processor.py
    ├── loss_function.py
    └── pretrained_model/TinyBert/        # 构建前由使用者放入
```

Dockerfile只复制`requirements.txt`、`entrypoint.py`和`algorithm/`。Dockerfile、`.dockerignore`及本说明不会进入镜像。

## 2. 构建前准备

### 2.1 Docker环境

构建机器需要能够执行`docker build`并拉取以下基础镜像：

```dockerfile
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
```

构建阶段不需要物理GPU。实际GPU可用性只能在安装了NVIDIA驱动和NVIDIA Container Toolkit的运行机器上验证。

### 2.2 TinyBert模型

将完整预训练模型放到：

```text
docker/text-classification-single/algorithm/pretrained_model/TinyBert/
```

至少包含：

```text
config.json
vocab.txt
pytorch_model.bin    # 或 model.safetensors
```

`pretrained_model/`被Git忽略，是本地构建输入。缺少上述任一配置、词表或权重时，构建会失败。

### 2.3 Python依赖

[requirements.txt](requirements.txt)是当前镜像依赖的唯一版本清单。主要版本包括：

| 依赖 | 版本 | 作用 |
| --- | --- | --- |
| scikit-learn | 1.4.2 | 训练和评估指标 |
| transformers | 4.40.2 | BERT模型和Tokenizer |
| tensorboard | 2.16.2 | 训练可视化日志 |
| pandas / openpyxl | 2.2.2 / 3.1.5 | 评估表格输出 |
| matplotlib / seaborn | 3.8.4 / 0.13.2 | 混淆矩阵图片 |
| onnx / onnxruntime | 1.16.1 / 1.18.1 | 现有ONNX相关代码导入 |

Dockerfile通过构建参数指定PyPI源，不执行永久性的`pip config set`。

## 3. 构建镜像

在项目根目录执行：

```bash
docker build \
  --build-arg IMAGE_VERSION=1.0.0 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t training-text-single:1.0.0 \
  docker/text-classification-single
```

| 构建参数 | 默认值 | 说明 |
| --- | --- | --- |
| `IMAGE_VERSION` | `1.0.0` | 写入镜像环境变量和OCI标签 |
| `PIP_INDEX_URL` | 清华PyPI镜像 | 当前构建使用的Python包索引 |

镜像标签中的版本应与`IMAGE_VERSION`保持一致。

## 4. Dockerfile执行内容

构建过程依次执行：

1. 继承PyTorch 2.2.2、CUDA 11.8、cuDNN 8开发镜像；
2. 安装固定版本Python依赖；
3. 将算法复制到`/opt/training/algorithm/`；
4. 将通用入口复制到`/opt/training/entrypoint.py`；
5. 检查TinyBert的配置、词表和至少一种权重文件；
6. 导入全部运行依赖，并断言关键依赖版本；
7. 导入现有训练、评估和ONNX模块，提前发现语法或依赖错误；
8. 创建或复用UID/GID 1000的非特权用户；
9. 将容器入口设置为`python /opt/training/entrypoint.py`。

其中`torch.version.cuda == "11.8"`只验证PyTorch编译使用的CUDA版本，不代表构建机器或运行机器已经有可见GPU。

通用入口不读取JSON配置，也不维护训练参数列表。它只把`train`或`validate`转换为对应算法脚本调用，并传递`--config`。训练与评估脚本自行读取配置，因此增加算法参数时无需同步修改入口。

## 5. 构建后检查

确认镜像存在并查看标签：

```bash
docker image inspect training-text-single:1.0.0
```

确认容器入口可启动：

```bash
docker run --rm training-text-single:1.0.0 --help
```

在GPU运行机器上检查PyTorch能否看到容器分配的GPU：

```bash
docker run --rm \
  --gpus 'device=0' \
  --entrypoint python \
  training-text-single:1.0.0 \
  -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

预期至少满足：

```text
PyTorch基础版本：2.2.2
PyTorch CUDA版本：11.8
cuda_available：True
gpu_count：不小于1
```

## 6. 发布要求

- 使用明确版本号或镜像摘要，不要让生产任务只依赖`latest`；
- 更新镜像版本时同步更新[docker/image-map.json](../image-map.json)；
- 推送前至少完成镜像构建、入口帮助命令和目标GPU机器检查；
- 依赖、算法或入口发生变化时生成新镜像版本，不在运行中的容器内安装包或修改代码。
