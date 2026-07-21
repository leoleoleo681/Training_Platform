# 使用镜像训练和评估

本文说明镜像构建完成后，外部平台如何准备宿主机任务目录并启动训练或评估容器。镜像构建见[BUILD.md](nlp/BUILD.md)。

## 1. 运行边界

```text
平台后端
  ├─ 生成任务目录、数据和JSON配置
  ├─ 根据task_type选择镜像
  ├─ 分配CPU、内存和GPU
  └─ docker run
       ├─ 镜像内：entrypoint.py + algorithms/
       └─ 挂载：宿主机当前任务目录 → /mnt/task
```

运行时不会挂载项目源码或Dockerfile。容器只能访问镜像内代码以及显式挂载到`/mnt/task`的当前任务目录。

## 2. 选择镜像

平台读取[image-map.json](image-map.json)：

```json
{
  "schema_version": 1,
  "task_images": {
    "text_classification_single": "training-nlp:1.0.0"
  }
}
```

前端只提交受控业务参数，不能直接指定镜像名、宿主机路径或Docker命令。平台应使用参数数组调用Docker，不将前端字符串拼接为Shell命令。

## 3. 宿主机任务目录

完整示例见[examples/text-classification-single-task](../examples/text-classification-single-task)。生产任务建议保存在：

```text
/data/tasks/{user_id}/{task_id}/
├── datasets/
│   ├── training/{train_file}.jsonl
│   ├── test/{test_file}.jsonl
│   └── labels/{label_file}.json
├── models/{model_name}/
│   └── run_train.json
└── evaluation/{model_name}/tests/{test_id}/
    └── run_test.json
```

平台只从当前`task_id`目录中挂载运行模板需要的子目录，不把整个任务目录作为可写目录暴露给容器：

```text
datasets/                         -> /mnt/task/datasets（只读）
models/{model_name}/             -> /mnt/task/models/{model_name}（训练可写、验证只读）
evaluation/.../tests/{test_id}/  -> /mnt/task/evaluation/.../tests/{test_id}（验证可写）
```

任务目录必须允许UID/GID 1000创建目录和原子替换JSON文件。

## 4. 数据要求

训练和测试数据必须是UTF-8 JSONL，每个非空行是一个JSON对象：

```json
{"content":"今天股票上涨","label":"财经"}
```

要求：

- `content`是非空字符串；
- `label`是非空字符串；
- 不接受`labels`字段；
- 训练集至少实际包含两个类别；
- 测试集中的标签必须存在于标签映射中。

标签文件必须是UTF-8 JSON对象：

```json
{"财经":0,"体育":1}
```

标签名必须是非空字符串，编号必须是从0开始的连续唯一整数。文件上传、合并、划分和自动生成标签映射由平台后端完成，不在训练容器内处理。

## 5. 训练配置

训练配置必须命名为`run_train.json`并位于：

```text
/mnt/task/models/{model_name}/run_train.json
```

完整示例见[run_train.json](../examples/text-classification-single-task/models/demo_model/run_train.json)。配置由训练脚本直接读取；与训练脚本参数同名的字段会自动应用，平台元数据字段由训练脚本按需使用。

### 5.1 基础字段

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `schema_version` | 推荐 | 配置协议版本，当前示例为整数`1` |
| `model_name` | 推荐 | 用于训练摘要；模型目录由配置文件所在目录确定 |
| `label_file` | 是 | `datasets/labels/`下的文件名，只允许basename |
| `train_file` | 是 | `datasets/training/`下的文件名，只允许basename |
| `evaluate_file` | 否 | `datasets/test/`下的验证集文件名 |
| `choose_device` | 否 | `cpu`或`gpu`，默认`gpu` |
| `training_mode` | 否 | 仅作为训练摘要中的平台元数据 |

`output_dir`、状态文件、结果文件和缓存目录均根据`run_train.json`的位置自动生成，不需要写入配置。

### 5.2 常用可选字段

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `max_length` | `512` | Token最大长度，范围1至512 |
| `gradient_accumulation_steps` | `1` | 梯度累积步数 |
| `weight_decay` | `0.01` | 权重衰减 |
| `warmup_steps` | `0.05` | 当前按比例解释，范围`[0,1)` |
| `logging_steps` | `500` | 日志间隔 |
| `save_steps` | `0` | checkpoint步数配置 |
| `save_each_epoch` | `false` | 是否每个epoch保存checkpoint |
| `seed` | `42` | 随机种子 |
| `loss_type` | `Focal` | `CE`或`Focal` |
| `dataloader_num_workers` | `12` | DataLoader worker数量 |
| `overwrite_cache` | `false` | 是否忽略已有共享缓存并重新生成 |
| `fp16` | `false` | 当前镜像未安装Apex，必须保持`false` |

其他损失函数参数以示例配置和入口代码为准。`label_distribution`当前只能是`auto`。

## 6. 评估配置

评估配置必须命名为`run_test.json`并位于：

```text
/mnt/task/evaluation/{model_name}/tests/{test_id}/run_test.json
```

完整示例见[run_test.json](../examples/text-classification-single-task/evaluation/demo_model/tests/706e9a79-7865-4255-b93c-59a5be694fda/run_test.json)。

| 字段 | 约束 |
| --- | --- |
| `schema_version` | 当前只能是整数`1` |
| `test_id` | 标准小写UUID |
| `model_name` | 必须对应已有训练模型目录 |
| `test_file` | `datasets/test/`下的`.jsonl`文件名，只允许basename |
| `label_file` | `datasets/labels/`下的`.json`文件名，只允许basename |
| `choose_device` | `cpu`或`gpu` |
| `per_gpu_eval_batch_size` | 不小于1的整数 |
| `save_result_csv` | 可选，默认`false` |
| `plot_confusion_matrix` | 可选，默认`false` |

评估优先读取`models/{model_name}/best_checkpoint/`，不存在时回退到`output_models/model/`。

## 7. 启动训练

### 7.1 GPU训练

```bash
docker run \
  --name train-{task_id}-{model_name} \
  --init \
  --gpus '"device=0,1"' \
  --shm-size=8g \
  --user 1000:1000 \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id}/datasets,dst=/mnt/task/datasets,readonly \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id}/models/{model_name},dst=/mnt/task/models/{model_name} \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id}/models/{model_name}/run_train.json,dst=/mnt/task/models/{model_name}/run_train.json,readonly \
  --cpus=8 \
  --memory=16g \
  --memory-swap=16g \
  --pids-limit=1024 \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --network none \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --tmpfs /mnt/task/dataset_cache:rw,nosuid,nodev,size=4g \
  --stop-timeout=30 \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  training-nlp:1.0.0 \
  train --task-type text_classification_single \
        --config /mnt/task/models/{model_name}/run_train.json
```

配置中的`choose_device`必须为`gpu`。容器内只使用Docker暴露的GPU，不写宿主机GPU编号。

多GPU训练只修改Docker参数，例如：

```bash
--gpus '"device=0,1"'
```

### 7.2 CPU训练

CPU训练删除`--gpus`参数，并将配置中的`choose_device`设置为`cpu`。其余命令和目录协议不变。

## 8. 启动评估

```bash
docker run \
  --name validate-{task_id}-{model_name}-{test_id} \
  --init \
  --gpus 'device=0' \
  --shm-size=8g \
  --user 1000:1000 \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id}/datasets,dst=/mnt/task/datasets,readonly \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id}/models/{model_name},dst=/mnt/task/models/{model_name},readonly \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id}/evaluation/{model_name}/tests/{test_id},dst=/mnt/task/evaluation/{model_name}/tests/{test_id} \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id}/evaluation/{model_name}/tests/{test_id}/run_test.json,dst=/mnt/task/evaluation/{model_name}/tests/{test_id}/run_test.json,readonly \
  --cpus=4 \
  --memory=8g \
  --memory-swap=8g \
  --pids-limit=512 \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --network none \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --tmpfs /mnt/task/dataset_cache:rw,nosuid,nodev,size=1g \
  --stop-timeout=30 \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  training-nlp:1.0.0 \
  validate --task-type text_classification_single \
           --config /mnt/task/evaluation/{model_name}/tests/{test_id}/run_test.json
```

CPU评估同样删除`--gpus`并使用`"choose_device":"cpu"`。

## 9. 状态和产物

`runtime/status.json`通过临时文件和原子替换更新。`state`表示生命周期，`phase`表示当前执行阶段：

```text
state: RUNNING -> SUCCEEDED | FAILED | CANCELLED
phase: INITIALIZING | TRAINING | VALIDATING | SAVING | FINALIZING
```

训练结束后的主要目录：

```text
models/{model_name}/
├── run_train.json
├── runtime/
│   ├── status.json
│   ├── events.jsonl
│   ├── metrics.jsonl
│   └── train_result.json
├── logs/
│   └── run.log
├── output_models/
│   ├── model/
│   └── tensorboard/
├── checkpoints/
├── best_checkpoint/
└── training_summary.json
```

预处理特征缓存写入容器内`/mnt/task/dataset_cache`，该路径使用 tmpfs，不会写回只读数据集，并在容器销毁时清理。若以后需要跨任务复用缓存，应设计独立的受控缓存挂载模板，不能把缓存重新写入`datasets/`。

评估目录中的标准产物：

```text
evaluation/{model_name}/tests/{test_id}/
├── run_test.json
├── runtime/
│   ├── status.json
│   ├── events.jsonl
│   └── metrics.jsonl
├── logs/
│   └── run.log
└── outputs/
    ├── report.json
    ├── predictions.jsonl
    └── eval_result_*/
```

`report.json`顶层`precision`、`recall`和`f1`使用macro算法。表格、TXT、预测缓存和混淆矩阵保存在当前测试目录的`outputs/eval_result_*`子目录中。

每个独立验证任务必须使用新的`test_id`，避免覆盖已有测试状态和产物。训练任务的重试、恢复和输出冲突策略由平台后端决定。

## 10. 停止任务

平台应使用`docker stop`发送SIGTERM，并留出退出时间：

```bash
docker stop --time 30 train-{task_id}-{model_name}
```

入口会转发终止信号并尝试写入`CANCELLED`。SIGKILL、宿主机故障或磁盘故障可能导致最后一次状态来不及落盘，因此平台必须同时检查Docker容器退出状态。

## 11. 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功 |
| `2` | 命令、任务类型或配置错误 |
| `3` | 输入文件、标签或数据错误 |
| `4` | 输出目录已有状态或产物 |
| `10` | 训练/评估进程或执行层失败 |
| `11` | CUDA显存不足 |
| `12` | 算法结束但缺少标准产物 |
| `143` | SIGTERM取消 |

## 12. 安全和运行要求

- 禁止使用`--privileged`；
- 禁止挂载Docker Socket、宿主机系统目录或其他任务目录；
- 保持`--read-only`，只让`/mnt/task`和`/tmp`可写；
- 使用非特权用户`1000:1000`；
- 限制CPU、内存、共享内存和进程数；
- GPU由平台调度，前端不得直接控制宿主机设备编号；
- 平台负责运行超时、容器停止和异常容器清理；
- 状态判断必须同时参考`status.json`、标准结果文件和容器退出码。
