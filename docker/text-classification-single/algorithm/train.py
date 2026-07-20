import argparse
from collections import Counter
import logging
import os
import random
import shutil
from time import sleep
import time
import timeit
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from data_processor import (
    CustomDatasetForBERT,
    DataProcessorForBERT,
)
from classifier_model import BertForMultiLabel

logger = logging.getLogger(__name__)
_ACTIVE_STATUS_FILE = None

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMultiLabel, BertTokenizer),
    # "roberta": (BertConfig, BertForMultiLabel, BertTokenizer)
}


def update_platform_status(status_file, progress, force=False, state=None):
    """Atomically update optional platform progress without changing legacy logs."""
    if not status_file:
        return
    now_monotonic = time.monotonic()
    last_update = getattr(update_platform_status, "_last_update", 0.0)
    if not force and now_monotonic - last_update < 1.0:
        return
    update_platform_status._last_update = now_monotonic
    try:
        payload = {}
        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        if state:
            payload["status"] = state
        payload["progress"] = progress
        now = datetime.now(timezone.utc).isoformat()
        payload["updated_at"] = now
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            payload["finished_at"] = now
        if state == "SUCCEEDED":
            payload["error"] = None
        temp_file = "{}.tmp.{}".format(status_file, os.getpid())
        with open(temp_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_file, status_file)
    except Exception as exc:
        logger.warning("更新平台状态失败: %s", exc)


def atomic_save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp_path), str(path))


def resolve_task_file(task_root, directory, filename, field):
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("{}必须是文件名，不能包含路径".format(field))
    path = (task_root / "datasets" / directory / filename).resolve(strict=True)
    try:
        path.relative_to(task_root)
    except ValueError as exc:
        raise ValueError("{}越出任务目录".format(field)) from exc
    if not path.is_file():
        raise ValueError("{}不是普通文件".format(field))
    return str(path)


def apply_json_config(parser, args):
    """Let train.py own its JSON schema and platform path conventions."""
    global _ACTIVE_STATUS_FILE
    if not args.config:
        return args

    config_path = Path(args.config).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("训练配置根节点必须是JSON对象")

    parser_fields = {action.dest: action for action in parser._actions}
    for field, value in config.items():
        action = parser_fields.get(field)
        if action is None or field == "config":
            continue
        if action.nargs == 0 and isinstance(action.const, bool) and type(value) is not bool:
            raise ValueError("训练参数{}必须是布尔值".format(field))
        try:
            converted = action.type(value) if action.type and value is not None else value
        except (TypeError, ValueError) as exc:
            raise ValueError("训练参数{}类型错误".format(field)) from exc
        if action.choices is not None and converted not in action.choices:
            raise ValueError("训练参数{}不支持值{}".format(field, converted))
        setattr(args, field, converted)

    task_root = Path(os.environ.get("TRAINING_TASK_ROOT", "/mnt/task")).resolve(strict=True)
    try:
        config_path.relative_to(task_root)
    except ValueError as exc:
        raise ValueError("训练配置越出任务目录") from exc

    model_root = config_path.parent
    algorithm_root = Path(os.environ.get("TRAINING_ALGORITHM_ROOT", Path(__file__).resolve().parent))
    args.model_name_or_path = os.environ.get(
        "TRAINING_PRETRAINED_MODEL",
        str(algorithm_root / "pretrained_model" / "TinyBert"),
    )
    args.output_dir = str(model_root / "output_models")
    args.label_file = resolve_task_file(task_root, "labels", config.get("label_file"), "label_file")
    args.train_file = resolve_task_file(task_root, "training", config.get("train_file"), "train_file")
    evaluate_file = config.get("evaluate_file")
    args.evaluate_file = (
        resolve_task_file(task_root, "test", evaluate_file, "evaluate_file")
        if evaluate_file
        else None
    )
    args.dataset_cache_dir = str(model_root / "runtime" / "cache")
    args.status_file = str(model_root / "status.json")
    args.result_file = str(model_root / "runtime" / "train_result.json")
    args.platform_model_root = str(model_root)
    args.platform_config = config
    args.platform_started_at = datetime.now(timezone.utc).isoformat()
    _ACTIVE_STATUS_FILE = args.status_file
    atomic_save_json(
        args.status_file,
        {
            "schema_version": 1,
            "operation": "train",
            "status": "STARTING",
            "started_at": args.platform_started_at,
            "updated_at": args.platform_started_at,
            "finished_at": None,
            "progress": {
                "current_epoch": 0,
                "total_epochs": None,
                "current_step": 0,
                "total_steps": None,
                "percent": 0.0,
            },
            "error": None,
        },
    )
    return args


def finalize_platform_training(args):
    if not getattr(args, "config", None) or args.local_rank not in [-1, 0]:
        return
    model_root = Path(args.platform_model_root)
    model_source = Path(args.model_output_dir)
    if not (model_source / "config.json").is_file():
        raise RuntimeError("训练完成但模型目录不完整: {}".format(model_source))
    best_checkpoint = model_root / "best_checkpoint"
    shutil.copytree(str(model_source), str(best_checkpoint))

    result = {}
    result_path = Path(args.result_file)
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    finished_at = datetime.now(timezone.utc).isoformat()
    config = args.platform_config
    atomic_save_json(
        model_root / "training_summary.json",
        {
            "schema_version": 1,
            "status": "SUCCEEDED",
            "started_at": args.platform_started_at,
            "finished_at": finished_at,
            "model_name": config.get("model_name"),
            "training_mode": config.get("training_mode"),
            "global_step": result.get("global_step"),
            "average_loss": result.get("average_loss"),
            "best_epoch": result.get("best_epoch"),
            "selection_score": result.get("selection_score"),
            "best_checkpoint": "best_checkpoint",
            "legacy_model_dir": "output_models/model",
            "checkpoint_dir": "checkpoints",
        },
    )
    update_platform_status(
        args.status_file,
        {
            "current_epoch": int(args.num_train_epochs),
            "total_epochs": int(args.num_train_epochs),
            "percent": 100.0,
        },
        force=True,
        state="SUCCEEDED",
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def iter_json_or_jsonl(input_file):
    """
    兼容读取 .json 和 .jsonl。

    .jsonl：逐行读取，每行一个样本，避免大训练集 json.load 爆内存。
    .json ：按 JSON 列表读取，适合较小文件。
    """
    if input_file.lower().endswith(".jsonl"):
        with open(input_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as e:
                    logger.warning("JSONL解析失败，文件: %s，行号: %s，错误: %s", input_file, line_no, repr(e))
                    continue
                if not isinstance(row, dict):
                    logger.warning("JSONL格式错误，文件: %s，行号: %s，该行不是dict，而是: %s", input_file, line_no, type(row))
                    continue
                yield row
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("{} 根对象不是JSON列表，而是 {}".format(input_file, type(data)))
        for row in data:
            if isinstance(row, dict):
                yield row


def get_first_label(labels):
    """
    兼容 labels 是字符串或列表两种情况。
    单标签任务里 labels 通常是字符串。
    """
    if isinstance(labels, list):
        return labels[0] if labels else None
    return labels


def save_training_checkpoint(
    args,
    model,
    tokenizer,
    output_dir,
    optimizer=None,
    scheduler=None,
):
    """Save model/tokenizer/training args, and optionally optimizer/scheduler states."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    if optimizer is not None or scheduler is not None:
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

class DynamicNonSecuritySampler(torch.utils.data.Sampler):
    """
    每个epoch：
    1. 保留所有其他类别样本；
    2. 按keep_ratio随机保留部分“非国家安全”样本；
    3. keep_ratio表示相对于原始“非国家安全”数量的保留比例；
    4. 不对其他类别进行加权或重复采样。
    """

    def __init__(
        self,
        dataset,
        non_security_id,
        keep_ratio=0.30,
    ):
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio必须在(0, 1]范围内，当前值: {}".format(keep_ratio))

        self.non_security_indices = []
        self.other_indices = []

        for index, feature in enumerate(dataset.features):
            label = feature.label
            if isinstance(label, int):
                is_non_security = label == non_security_id
            elif isinstance(label, (list, tuple)):
                is_non_security = (
                    len(label) > non_security_id
                    and label[non_security_id] == 1
                    and sum(label) == 1
                )
            else:
                is_non_security = False

            if is_non_security:
                self.non_security_indices.append(index)
            else:
                self.other_indices.append(index)

        self.keep_non_security_count = round(
            keep_ratio * len(self.non_security_indices)
        )

        # 防止要求保留的数量超过实际数量
        self.keep_non_security_count = min(
            self.keep_non_security_count,
            len(self.non_security_indices),
        )

        self.epoch_size = (
            len(self.other_indices)
            + self.keep_non_security_count
        )

        final_ratio = (
            self.keep_non_security_count / self.epoch_size
            if self.epoch_size > 0 else 0.0
        )

        logger.info(
            "动态采样：其他类别=%d，非国家安全总数=%d，"
            "每轮保留非国家安全=%d（原数量的%.2f%%），每轮总样本=%d，"
            "采样后非国家安全占比=%.2f%%",
            len(self.other_indices),
            len(self.non_security_indices),
            self.keep_non_security_count,
            keep_ratio * 100,
            self.epoch_size,
            final_ratio * 100,
        )

    def __iter__(self):
        # 每次DataLoader开始一个新epoch，都会重新随机抽取
        selected_non_security = random.sample(
            self.non_security_indices,
            self.keep_non_security_count,
        )

        epoch_indices = (
            self.other_indices
            + selected_non_security
        )

        random.shuffle(epoch_indices)
        return iter(epoch_indices)

    def __len__(self):
        return self.epoch_size

def train(args, train_dataset, model, tokenizer, evalute_dataset=None):
    """ Train the model """
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_dir = os.path.join(args.output_dir, "tensorboard")
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        tb_writer = SummaryWriter(tb_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.local_rank == -1:
        if "非国家安全" in args.label2int:
            train_sampler = DynamicNonSecuritySampler(
                dataset=train_dataset,
                non_security_id=args.label2int["非国家安全"],
                keep_ratio=args.non_security_keep_ratio,
            )
        else:
            logger.info("标签表中没有‘非国家安全’，使用标准随机采样器。")
            train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    steps_per_epoch = int(np.ceil(len(train_dataloader) / args.gradient_accumulation_steps))  # use np.ceil() to include all batches

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = int(np.ceil(args.max_steps / steps_per_epoch))
    else:
        t_total = steps_per_epoch * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("bert.") and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.startswith("bert.") and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if not n.startswith("bert.") and not any(nd in n for nd in no_decay)],
            "lr": 1e-3,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if not n.startswith("bert.") and any(nd in n for nd in no_decay)],
            "lr": 1e-3,
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_warmup_steps = int(args.warmup_steps * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_to_skip = 0
    batches_to_skip = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split(os.sep)[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // steps_per_epoch
            steps_to_skip = global_step % steps_per_epoch
            batches_to_skip = steps_to_skip * args.gradient_accumulation_steps

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_to_skip)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    total_epochs_for_status = int(args.num_train_epochs)
    update_platform_status(
        args.status_file,
        {
            "current_epoch": epochs_trained,
            "total_epochs": total_epochs_for_status,
            "current_step": global_step,
            "total_steps": int(t_total),
            "percent": round(min(100.0, 100.0 * global_step / max(float(t_total), 1.0)), 2),
        },
        force=True,
        state="RUNNING",
    )

    model.zero_grad()
    
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    
    tr_loss, logging_loss = 0.0, 0.0
    total_loss_batches = 0
    logging_step_start = global_step
    best_score = None
    best_epoch = None

    for epoch_idx in train_iterator:
        epoch_loss_sum = 0.0
        epoch_loss_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for batch_no, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            model.train()
            
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "labels": batch[2].to(args.device)
            }
            
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            # 单卡返回shape [1]，DataParallel多卡返回shape [n_gpu]，统一取平均得到标量。
            loss = outputs[0].mean()

            batch_loss = loss.item()
            tr_loss += batch_loss
            total_loss_batches += 1
            epoch_loss_sum += batch_loss
            epoch_loss_batches += 1

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (batch_no + 1) % args.gradient_accumulation_steps == 0 or batch_no == len(train_dataloader) - 1:
                if args.max_grad_norm > 0.0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                update_platform_status(
                    args.status_file,
                    {
                        "current_epoch": epoch_idx + 1,
                        "total_epochs": total_epochs_for_status,
                        "current_step": global_step,
                        "total_steps": int(t_total),
                        "percent": round(min(100.0, 100.0 * global_step / max(float(t_total), 1.0)), 2),
                    },
                )

                # Log train loss / learning rate only.
                # 注意：这里不再使用 logging_steps 触发验证集评估。
                # logging_steps 现在只负责写入 TensorBoard 的训练 loss 和 lr。
                # 验证集评估改为跟随 save_steps：每保存一次 checkpoint，就验证一次。
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / (global_step - logging_step_start), global_step)

                    logging_loss = tr_loss
                    logging_step_start = global_step

                # 按步保存只负责断点续训，不参与最优模型选择。
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.checkpoint_dir, "checkpoint-step-{}".format(global_step))
                    save_training_checkpoint(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=output_dir,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

            if args.max_steps > 0 and global_step >= args.max_steps:
                epoch_iterator.close()
                break

        epoch_number = epoch_idx + 1

        # 每个epoch只做一次选优：有验证集时最大化macro_f1，否则最小化平均训练loss。
        if args.local_rank in [-1, 0] and epoch_loss_batches > 0:
            epoch_avg_loss = epoch_loss_sum / epoch_loss_batches
            tb_writer.add_scalar("epoch/train_loss", epoch_avg_loss, epoch_number)

            if evalute_dataset is not None:
                metrics = evaluate(args, model, evalute_dataset, prefix="epoch-{}".format(epoch_number))
                for key, value in metrics.items():
                    tb_writer.add_scalar("eval/{}".format(key), value, epoch_number)
                current_score = metrics["macro_f1"]
                is_better = best_score is None or current_score > best_score
                logger.info(
                    "Epoch %s selection metric: validation macro_f1=%s",
                    epoch_number,
                    current_score,
                )
            else:
                current_score = epoch_avg_loss
                is_better = best_score is None or current_score < best_score
                logger.info(
                    "Epoch %s selection metric: training loss=%s",
                    epoch_number,
                    current_score,
                )

            if is_better:
                best_score = current_score
                best_epoch = epoch_number
                save_training_checkpoint(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=args.model_output_dir,
                )
                logger.info(
                    "Updated selected model at epoch %s: %s",
                    best_epoch,
                    args.model_output_dir,
                )

        # checkpoint是可选的训练状态快照，与稳定模型选优互不影响。
        if args.local_rank in [-1, 0] and args.save_each_epoch:
            output_dir = os.path.join(args.checkpoint_dir, "checkpoint-epoch-{}".format(epoch_number))
            save_training_checkpoint(
                args=args,
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                optimizer=optimizer,
                scheduler=scheduler,
            )

        update_platform_status(
            args.status_file,
            {
                "current_epoch": epoch_number,
                "total_epochs": total_epochs_for_status,
                "current_step": global_step,
                "total_steps": int(t_total),
                "percent": round(min(100.0, 100.0 * global_step / max(float(t_total), 1.0)), 2),
            },
            force=True,
        )

        if args.max_steps > 0 and global_step >= args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        if best_score is None:
            save_training_checkpoint(
                args=args,
                model=model,
                tokenizer=tokenizer,
                output_dir=args.model_output_dir,
            )
            logger.warning("No epoch selection score was produced; saved the current model as fallback.")
        else:
            logger.info("Selected epoch=%s, score=%s", best_epoch, best_score)
        tb_writer.close()
        
    avg_loss = tr_loss / max(total_loss_batches, 1)
    
    return global_step, avg_loss, best_epoch, best_score


def evaluate(args, model, evalute_dataset, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(evalute_dataset)
    eval_dataloader = DataLoader(
        evalute_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(evalute_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_pred = []
    all_truth = []

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
            }
            truth = to_list(batch[2].to(torch.int))
            all_truth.extend(truth)
            
            outputs = model(**inputs)
            logits = outputs[0]  # logits.shape=(eval_batch_size, num_labels)
            if args.loss_type in ['BCE','ASL']:
                pred = to_list((torch.sigmoid(logits) > args.threshold).to(torch.int))
            elif args.loss_type in ['CE','Focal']: 
                # logits to one-hot matrix
                pred = to_list(torch.softmax(logits,dim=-1).max(1).indices)
                # print(pred)
            all_pred.extend(pred)
            
    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(evalute_dataset))

    # Compute metrics of classification, e.g., accuracy, f1, precision, recall, AUC
    # metrics is a dict of (metric, value)
    metrics = eval_classification_metrics(args, all_pred, all_truth)

    return metrics


# you can write your own func of evaluating metrics
def eval_classification_metrics(args, y_pred: list, y_true: list) -> dict:
    """
    
    Args:
        y_pred: 
        y_true: 
    Returns:
        metrics: dict
    """

    # 分类别计算
    macro_prec = sklearn.metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    # 整体计算
    micro_prec = sklearn.metrics.precision_score(y_true, y_pred, average="micro", zero_division=0) 
    micro_recall = sklearn.metrics.recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)


    if args.loss_type in ['CE', 'Focal']:
        """
        single-label shape: 1D-array
        """
        acc = sklearn.metrics.accuracy_score(y_true, y_pred) # 对于one-hot非常严格

        metrics = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "macro_prec": macro_prec,
            "micro_prec": micro_prec,
            "macro_recall": macro_recall,
            "micro_recall": micro_recall,
        }

    else:
        """
        multi-label shape: 2D-array
        """
        # 对每个样本计算
        samples_prec = sklearn.metrics.precision_score(y_true, y_pred, average="samples", zero_division=0)
        samples_recall = sklearn.metrics.recall_score(y_true, y_pred, average="samples", zero_division=0)
        samples_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="samples", zero_division=0) 
        
        metrics = {
            "samples_f1": samples_f1,
            "samples_prec": samples_prec,
            "samples_recall": samples_recall,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "macro_prec": macro_prec,
            "micro_prec": micro_prec,
            "macro_recall": macro_recall,
            "micro_recall": micro_recall,

        }
    return metrics


def load_and_cache_dataset(
    args,
    tokenizer,
    data_file_name,
    skip_empty_input
    ):
    if args.local_rank not in [-1, 0]:
        # 非主进程等待主进程生成缓存。
        torch.distributed.barrier()

    input_dir = args.data_dir if args.data_dir else "."
    data_file = (
        data_file_name
        if os.path.isabs(data_file_name)
        else os.path.join(input_dir, data_file_name)
    )
    cache_directory = args.dataset_cache_dir or os.path.dirname(data_file)
    if not os.path.exists(cache_directory) and args.local_rank in [-1, 0]:
        os.makedirs(cache_directory)
    cached_file = os.path.join(
        cache_directory,
        "cached_{}_for_{}_seqA{}_loss-{}.pkl".format(
            os.path.basename(data_file),
            args.model_type,
            args.max_length,
            args.loss_type,
        ),
    )

    if os.path.exists(cached_file) and not args.overwrite_cache:
        logger.info("Loading Dataset from cached file %s", cached_file)
        dataset = CustomDatasetForBERT.load(cached_file)
    else:
        logger.info("Creating Dataset from file -- %s", data_file)
        dprc = DataProcessorForBERT(label_map=args.label2int)
        dprc.add_examples_from_json(input_file=data_file)
        dataset = dprc.get_feature(
            tokenizer=tokenizer,
            max_length=args.max_length,
            skip_empty_input=skip_empty_input,
            loss_type=args.loss_type,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving Dataset into cached file %s", cached_file)
            dataset.save(cached_file)

    if args.local_rank == 0:
        # 缓存写完后再放行其他进程。
        torch.distributed.barrier()

    return dataset


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None, help="平台JSON训练配置")

    # 命令行默认值与参数约束由训练脚本维护；--config 中的同名字段会覆盖默认值。
    parser.add_argument("--model_type",default="bert",type=str,help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path",default=None,type=str,help="Path to pre-trained model",)
    parser.add_argument("--output_dir",default=None,type=str,help="Root directory for the selected model and TensorBoard output.",)
    parser.add_argument("--label_file",default=None,type=str,help="label 2 int map")
    parser.add_argument("--max_length",default=512,type=int,help="The maximum sequence A and B total length after tokenization, no more than 512.")
    parser.add_argument("--threshold",default=0.5,type=float,help="the threshold of label probability")

    # Other parameters

    parser.add_argument("--data_dir",default=None,type=str,help="The input data dir. Should contain the .csv files for the task.",)
    parser.add_argument("--train_file",default=None,type=str,help="The input training file. If a data dir is specified, will look for the file there",)
    parser.add_argument("--evaluate_file",default=None,type=str,help="The input evaluation file. If a data dir is specified, will look for the file there",)
    
    parser.add_argument("--do_train",action="store_true",default=True,help="Whether to run training.")

    parser.add_argument("--config_name",default="",type=str,help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",default="",type=str,help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--do_lower_case",action="store_true",help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3",)

    parser.add_argument("--per_gpu_train_batch_size",default=32,type=int,help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",default=32,type=int,help="Batch size per GPU/CPU for evaluation or prediction.")
    parser.add_argument("--learning_rate",default=3e-5,type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",default=20.0,type=float,help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",default=-1,type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps",default=0.05,type=float,help="Warmup占总优化步数的比例，范围[0, 1)。",)

    parser.add_argument("--verbose_logging",action="store_true",help="If true, all of the warnings related to data processing will be printed. ",)
    parser.add_argument("--lang_id",default=0,type=int,help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",)

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps. Set <= 0 to disable step-based saving.")
    parser.add_argument("--save_each_epoch", action="store_true", help="Save checkpoint at the end of each epoch.")
    parser.add_argument("--choose_device",type=str,choices=["cpu", "gpu"],default="gpu",help="选择运行设备：cpu 或 gpu，默认 gpu",)
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True, help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", help="Rebuild the cached training and evaluation sets")
    parser.add_argument(
        "--dataset_cache_dir",
        default=None,
        type=str,
        help="可选的数据集缓存目录；未设置时保持原行为，写在数据文件旁。",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        default=12,
        type=int,
        help="DataLoader worker 数；默认值保持原行为。",
    )
    parser.add_argument("--status_file", default=None, type=str, help="可选的平台状态文件")
    parser.add_argument("--result_file", default=None, type=str, help="可选的平台训练结果文件")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--fp16",action="store_true",help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level",type=str,default="O1",help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument(
        "--loss_type",
        type=str,
        default="Focal",
        choices=["BCE", "ASL", "CE", "Focal"],
        help="What loss function to use",
    )
    parser.add_argument("--label_distribution", type=str, default="auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--distribution_gamma", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument(
        "--non_security_keep_ratio",
        type=float,
        default=1.0,
        help="每个epoch保留的非国家安全样本比例，按其原始数量计算，范围(0, 1]。",
    )

    args = apply_json_config(parser, parser.parse_args())

    for field in ("model_name_or_path", "output_dir", "label_file", "train_file"):
        if not getattr(args, field):
            parser.error("--{}不能为空".format(field))

    # 稳定模型固定写入output_dir/model；checkpoint单独写入相邻的checkpoints目录。
    args.output_dir = os.path.abspath(args.output_dir)
    args.model_output_dir = os.path.join(args.output_dir, "model")
    args.checkpoint_dir = os.path.join(os.path.dirname(args.output_dir), "checkpoints")

    if args.focal_gamma < 0:
        raise ValueError("focal_gamma不能小于0")
    if args.distribution_gamma < 0:
        raise ValueError("distribution_gamma不能小于0")
    if not 0 <= args.warmup_steps < 1:
        raise ValueError("warmup_steps当前按比例解释，必须在[0, 1)范围内")

    label_file = (
        args.label_file
        if os.path.isabs(args.label_file)
        else os.path.join(args.data_dir or ".", args.label_file)
    )
    with open(label_file, "r", encoding="utf-8") as f:
        args.label2int = json.load(f)
            
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    device = None
    if args.local_rank == -1 or args.choose_device=="cpu":
        device = torch.device("cuda" if torch.cuda.is_available() and not args.choose_device=="cpu" else "cpu")
        args.n_gpu = 0 if args.choose_device=="cpu" else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # -+--- 模型加载 -----------------------------------------------------------------------+- 

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.do_train:
        config.update({
            "num_labels": len(args.label2int),
            "loss_type":args.loss_type,
        })

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.label_distribution is not None:
        data_file = (
            args.train_file
            if os.path.isabs(args.train_file)
            else os.path.join(args.data_dir or ".", args.train_file)
        )
        c = Counter()

        for doc_idx, doc in enumerate(iter_json_or_jsonl(data_file)):
            if "label" not in doc:
                logger.warning("缺少label字段，文件: %s，样本序号: %s", data_file, doc_idx)
                continue

            labels = doc["label"]
            if args.loss_type in ["CE", "Focal"]:
                lb = get_first_label(labels)
                if lb is not None:
                    c[lb] += 1
            else:
                if not isinstance(labels, list):
                    labels = [labels]
                for lb in labels:
                    if lb is not None:
                        c[lb] += 1

        args.label_distribution = torch.tensor([c[k] for k, _ in args.label2int.items()], dtype=torch.float).to(args.device)
        # args.label_distribution = torch.tensor([int(i) for i in args.label_distribution.split(", ")], dtype=torch.float).to(args.device)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        # # label_distribution=torch.tensor([3909, 2500, 3961, 757, 960, 657, 225, 232, 46, 349, 217, 57],dtype=torch.float).to(args.device)
        label_distribution=args.label_distribution,
        distribution_gamma = args.distribution_gamma,
        focal_gamma = args.focal_gamma,
        alpha = args.alpha,
    )

    # -+--- end -----------------------------------------------------------------------+- 

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation/prediction parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # -+--- 数据加载\训练 -----------------------------------------------------------------------+- 

    evalute_dataset = (
        load_and_cache_dataset(args, tokenizer, args.evaluate_file, skip_empty_input=False)
        if args.evaluate_file
        else None
    )
    if evalute_dataset is not None and len(evalute_dataset) == 0:
        raise ValueError("验证集在预处理后没有有效样本")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_dataset(args, tokenizer, args.train_file, skip_empty_input=True)
        if len(train_dataset) == 0:
            raise ValueError("训练集在预处理后没有有效样本")
        global_step, tr_loss, best_epoch, best_score = train(args, train_dataset, model, tokenizer, evalute_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        if args.result_file and args.local_rank in [-1, 0]:
            result_dir = os.path.dirname(os.path.abspath(args.result_file))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            result_payload = {
                "global_step": global_step,
                "average_loss": tr_loss,
                "best_epoch": best_epoch,
                "selection_score": best_score,
            }
            temp_result = "{}.tmp.{}".format(args.result_file, os.getpid())
            with open(temp_result, "w", encoding="utf-8") as handle:
                json.dump(result_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_result, args.result_file)

    # 最优模型已在训练过程中直接写入固定目录，不再在训练结束后重复验证checkpoint。

    logger.info("script finishes")

    finalize_platform_training(args)

    return 0


def main():
    try:
        return run()
    except Exception as exc:
        if _ACTIVE_STATUS_FILE:
            update_platform_status(
                _ACTIVE_STATUS_FILE,
                {"percent": 0.0},
                force=True,
                state="FAILED",
            )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # print("delay10000")
    # sleep(10000)
    main()
