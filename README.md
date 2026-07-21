# 训练平台 Docker 镜像

本项目按任务族维护训练镜像。文本分类和信息抽取任务共用 NLP 镜像；当前已接入单标签文本分类：

```text
task_type: text_classification_single
image:     training-nlp:1.0.0
```

## 项目结构

```text
Training_Platform/
├── docker/
│   ├── image-map.json                   # 平台使用的 task_type → 镜像映射
│   ├── RUN.md                           # 容器启动、挂载、状态和结果说明
│   └── nlp/                             # NLP任务族的完整构建上下文
│       ├── Dockerfile
│       ├── .dockerignore
│       ├── requirements.txt
│       ├── entrypoint.py
│       ├── platform_runtime/
│       ├── BUILD.md
│       └── algorithms/
│           └── text_classification_single/
└── examples/
    └── text-classification-single-task/ # 宿主机任务目录示例
```

## 文件职责

| 位置 | 作用 | 是否进入镜像 | 是否运行时挂载 |
| --- | --- | --- | --- |
| `docker/nlp/Dockerfile` | 定义 NLP 镜像构建步骤 | 否 | 否 |
| `docker/nlp/.dockerignore` | 过滤构建上下文 | 否 | 否 |
| `requirements.txt` | 固定当前镜像的 Python 依赖 | 是 | 否 |
| `entrypoint.py` | 通用入口；按任务类型和操作启动对应算法脚本 | 是 | 否 |
| `platform_runtime/` | 统一写入状态、事件、指标和运行日志 | 是 | 否 |
| `algorithms/` | NLP任务族内各任务类型的训练和评估代码 | 是 | 否 |
| `examples/text-classification-single-task/` | 展示宿主机任务目录格式 | 否 | 否 |
| `/data/tasks/{user_id}/{task_id}` | 生产环境真实任务目录 | 否 | 是，挂载到 `/mnt/task` |

Docker构建上下文不等于运行时挂载。只有Dockerfile中的`COPY`内容会进入镜像，只有`docker run --mount`指定的宿主机目录会出现在运行中的容器内。

## 使用顺序

1. 按[镜像构建说明](docker/nlp/BUILD.md)准备TinyBert并构建`training-nlp:1.0.0`。
2. 平台通过[镜像映射](docker/image-map.json)根据`task_type`选择镜像。
3. 参考[任务目录示例](examples/text-classification-single-task)生成宿主机任务数据和JSON配置。
4. 按[容器运行说明](docker/RUN.md)启动训练或评估。
5. 平台读取`runtime/status.json`、事件、指标、标准结果文件和容器退出码。

## 当前范围

当前实现只接入了单标签文本分类的训练、评估、状态和结果协议。数据上传、合并、划分、自动提取标签、Docker资源调度、超时控制和前端API由外部平台后端负责。

新增同属 NLP 任务族的多标签分类或 UIE 时，在`docker/nlp/algorithms/`下增加任务目录，并在`entrypoint.py`注册任务类型；只有依赖体系明显不兼容的任务族才创建新的平级镜像目录。随后在`image-map.json`中将对应任务类型映射到`training-nlp`。
