# 训练平台 Docker 镜像

本项目按任务类型维护独立的训练镜像。当前只实现单标签文本分类：

```text
task_type: text_classification_single
image:     training-text-single:1.0.0
```

## 项目结构

```text
Training_Platform/
├── docker/
│   ├── image-map.json                   # 平台使用的 task_type → 镜像映射
│   ├── RUN.md                           # 容器启动、挂载、状态和结果说明
│   └── text-classification-single/      # 单标签镜像的完整构建上下文
│       ├── Dockerfile
│       ├── .dockerignore
│       ├── requirements.txt
│       ├── entrypoint.py
│       ├── BUILD.md
│       └── algorithm/
└── examples/
    └── text-classification-single-task/ # 宿主机任务目录示例
```

## 文件职责

| 位置 | 作用 | 是否进入镜像 | 是否运行时挂载 |
| --- | --- | --- | --- |
| `docker/text-classification-single/Dockerfile` | 定义镜像构建步骤 | 否 | 否 |
| `docker/text-classification-single/.dockerignore` | 过滤构建上下文 | 否 | 否 |
| `requirements.txt` | 固定当前镜像的 Python 依赖 | 是 | 否 |
| `entrypoint.py` | 校验配置并启动训练或评估 | 是 | 否 |
| `algorithm/` | 单标签训练和评估算法 | 是 | 否 |
| `examples/text-classification-single-task/` | 展示宿主机任务目录格式 | 否 | 否 |
| `/data/tasks/{user_id}/{task_id}` | 生产环境真实任务目录 | 否 | 是，挂载到 `/mnt/task` |

Docker构建上下文不等于运行时挂载。只有Dockerfile中的`COPY`内容会进入镜像，只有`docker run --mount`指定的宿主机目录会出现在运行中的容器内。

## 使用顺序

1. 按[镜像构建说明](docker/text-classification-single/BUILD.md)准备TinyBert并构建`training-text-single:1.0.0`。
2. 平台通过[镜像映射](docker/image-map.json)根据`task_type`选择镜像。
3. 参考[任务目录示例](examples/text-classification-single-task)生成宿主机任务数据和JSON配置。
4. 按[容器运行说明](docker/RUN.md)启动训练或评估。
5. 平台读取`status.json`、标准结果文件和容器退出码。

## 当前范围

本项目只负责单标签文本分类镜像内部的训练、评估、状态和结果协议。数据上传、合并、划分、自动提取标签、Docker资源调度、超时控制和前端API由外部平台后端负责。

新增任务时，在`docker/`下创建平级的独立镜像目录，并在`image-map.json`增加映射；不要向现有单标签镜像混入其他任务的依赖和入口。
