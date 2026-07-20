# 使用镜像训练和评估

本文说明镜像构建完成后，外部平台如何准备宿主机任务目录并启动训练或评估容器。镜像构建见[BUILD.md](text-classification-single/BUILD.md)。

## 1. 运行边界

```text
平台后端
  ├─ 生成任务目录、数据和JSON配置
  ├─ 根据task_type选择镜像
  ├─ 分配CPU、内存和GPU
  └─ docker run
       ├─ 镜像内：entrypoint.py + algorithm/
       └─ 挂载：宿主机当前任务目录 → /mnt/task
```

运行时不会挂载项目源码或Dockerfile。容器只能访问镜像内代码以及显式挂载到`/mnt/task`的当前任务目录。

## 2. 选择镜像

平台读取[image-map.json](image-map.json)：

```json
{
  "schema_version": 1,
  "task_images": {
    "text_classification_single": "training-text-single:1.0.0"
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

平台每次只挂载当前`task_id`目录：

```text
宿主机：/data/tasks/{user_id}/{task_id}
容器内：/mnt/task
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

完整示例见[run_train.json](../examples/text-classification-single-task/models/demo_model/run_train.json)。入口严格拒绝未知字段。

### 5.1 必填字段

| 字段 | 约束 |
| --- | --- |
| `schema_version` | 当前只能是整数`1` |
| `model_name` | 1至64位，只允许英文字母、数字、`_`和`-`，首位为字母或数字 |
| `output_dir` | 必须严格等于`/mnt/task/models/{model_name}` |
| `label_file` | `datasets/labels/`下的`.json`文件名，只允许basename |
| `train_file` | `datasets/training/`下的`.jsonl`文件名，只允许basename |
| `choose_device` | `cpu`或`gpu` |
| `training_mode` | `quick`、`balance`或`quality`，仅记录模式；平台仍需写入明确超参数 |
| `per_gpu_train_batch_size` | 不小于1的整数 |
| `per_gpu_eval_batch_size` | 不小于1的整数 |
| `learning_rate` | 大于0 |
| `num_train_epochs` | 大于0 |

### 5.2 常用可选字段

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `evaluate_file` | `null` | `datasets/test/`下的验证集文件名 |
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
| `overwrite_cache` | `false` | 是否重新生成数据缓存 |
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
docker run --rm \
  --name train-{task_id}-{model_name} \
  --gpus 'device=0' \
  --shm-size=8g \
  --user 1000:1000 \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id},dst=/mnt/task \
  --cpus=8 \
  --memory=16g \
  --memory-swap=16g \
  --pids-limit=1024 \
  --security-opt=no-new-privileges \
  --read-only \
  --tmpfs /tmp \
  training-text-single:1.0.0 \
  train --task-type text_classification_single \
        --config /mnt/task/models/{model_name}/run_train.json
```

配置中的`choose_device`必须为`gpu`。容器内只使用Docker暴露的GPU，不写宿主机GPU编号。

多GPU训练只修改Docker参数，例如：

```bash
--gpus 'device=0,1'
```

### 7.2 CPU训练

CPU训练删除`--gpus`参数，并将配置中的`choose_device`设置为`cpu`。其余命令和目录协议不变。

## 8. 启动评估

```bash
docker run --rm \
  --name validate-{task_id}-{model_name}-{test_id} \
  --gpus 'device=0' \
  --shm-size=8g \
  --user 1000:1000 \
  --mount type=bind,src=/data/tasks/{user_id}/{task_id},dst=/mnt/task \
  --cpus=4 \
  --memory=8g \
  --memory-swap=8g \
  --pids-limit=512 \
  --security-opt=no-new-privileges \
  --read-only \
  --tmpfs /tmp \
  training-text-single:1.0.0 \
  validate --task-type text_classification_single \
           --config /mnt/task/evaluation/{model_name}/tests/{test_id}/run_test.json
```

CPU评估同样删除`--gpus`并使用`"choose_device":"cpu"`。

## 9. 状态和产物

`status.json`通过临时文件和原子替换更新：

```text
STARTING -> VALIDATING -> RUNNING -> SUCCEEDED
                                     FAILED
                                     CANCELLED
```

训练结束后的主要目录：

```text
models/{model_name}/
├── run_train.json
├── status.json
├── logs/train.log
├── train_{timestamp}.log
├── output_models/
│   ├── model/
│   └── tensorboard/
├── checkpoints/
├── best_checkpoint/
├── runtime/
│   ├── cache/
│   └── train_result.json
└── training_summary.json
```

评估目录中的标准产物：

```text
evaluation/{model_name}/tests/{test_id}/
├── run_test.json
├── status.json
├── val.log
├── report.json
└── predictions.json
```

`report.json`顶层`precision`、`recall`和`f1`使用macro算法。原评估表格、TXT、预测缓存和混淆矩阵仍保存在模型目录的`eval_result_*`子目录中。

同一个模型目录或评估目录只允许首次执行。存在状态或产物时入口返回`OUTPUT_CONFLICT`，不会覆盖、清理或自动续训。

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
