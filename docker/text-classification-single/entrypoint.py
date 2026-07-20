"""Strict, config-driven entrypoint for the single-label text image."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


TASK_TYPE = "text_classification_single"
TRAIN_CONFIG_NAME = "run_train.json"
TEST_CONFIG_NAME = "run_test.json"
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
UNSAFE_FILENAME_RE = re.compile(r"[\\/;&|`$<>\r\n]")
TRAINING_MODES = {"quick", "balance", "quality"}
DEVICE_CHOICES = {"cpu", "gpu"}
LOSS_CHOICES = {"CE", "Focal"}

EXIT_CONFIG = 2
EXIT_INPUT = 3
EXIT_CONFLICT = 4
EXIT_RUNTIME = 10
EXIT_CUDA_OOM = 11
EXIT_ARTIFACT = 12
EXIT_CANCELLED = 143

TRAIN_REQUIRED_FIELDS = {
    "schema_version",
    "model_name",
    "output_dir",
    "label_file",
    "train_file",
    "choose_device",
    "training_mode",
    "per_gpu_train_batch_size",
    "per_gpu_eval_batch_size",
    "learning_rate",
    "num_train_epochs",
}
TRAIN_OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "evaluate_file": None,
    "max_length": 512,
    "threshold": 0.5,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "warmup_steps": 0.05,
    "logging_steps": 500,
    "save_steps": 0,
    "save_each_epoch": False,
    "seed": 42,
    "fp16": False,
    "fp16_opt_level": "O1",
    "loss_type": "Focal",
    "label_distribution": "auto",
    "alpha": 1.0,
    "distribution_gamma": 0.0,
    "focal_gamma": 0.0,
    "non_security_keep_ratio": 1.0,
    "dataloader_num_workers": 12,
    "overwrite_cache": False,
}

TEST_REQUIRED_FIELDS = {
    "schema_version",
    "test_id",
    "model_name",
    "test_file",
    "label_file",
    "choose_device",
    "per_gpu_eval_batch_size",
}
TEST_OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "save_result_csv": False,
    "plot_confusion_matrix": False,
}


class PlatformError(Exception):
    def __init__(self, code: str, message: str, exit_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def image_metadata() -> Dict[str, Any]:
    return {
        "image_family": os.environ.get("TRAINING_IMAGE_FAMILY", "unknown"),
        "image_version": os.environ.get("TRAINING_IMAGE_VERSION", "unknown"),
        "protocol_version": int(os.environ.get("TRAINING_PROTOCOL_VERSION", "1")),
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp_path), str(path))


def load_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except UnicodeDecodeError as exc:
        raise PlatformError("CONFIG_INVALID", "配置文件必须使用 UTF-8 编码", EXIT_CONFIG) from exc
    except json.JSONDecodeError as exc:
        raise PlatformError(
            "CONFIG_INVALID",
            "配置文件不是合法 JSON: 第 {} 行第 {} 列".format(exc.lineno, exc.colno),
            EXIT_CONFIG,
        ) from exc
    if not isinstance(value, dict):
        raise PlatformError("CONFIG_INVALID", "配置根节点必须是 JSON 对象", EXIT_CONFIG)
    return value


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_existing_path(path: Path, root: Path, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PlatformError("INPUT_INVALID", "{}不存在: {}".format(description, path), EXIT_INPUT) from exc
    if not is_relative_to(resolved, root):
        raise PlatformError("PATH_OUTSIDE_TASK", "{}越出任务目录: {}".format(description, path), EXIT_CONFIG)
    return resolved


def get_task_root() -> Path:
    raw = os.environ.get("TRAINING_TASK_ROOT", "/mnt/task")
    try:
        root = Path(raw).resolve(strict=True)
    except FileNotFoundError as exc:
        raise PlatformError("INPUT_INVALID", "任务挂载目录不存在: {}".format(raw), EXIT_INPUT) from exc
    if not root.is_dir():
        raise PlatformError("INPUT_INVALID", "任务挂载路径不是目录: {}".format(root), EXIT_INPUT)
    return root


def validate_config_path(raw_path: str, task_root: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        raise PlatformError("CONFIG_INVALID", "--config 必须是容器内绝对路径", EXIT_CONFIG)
    resolved = resolve_existing_path(path, task_root, "配置文件")
    if not resolved.is_file():
        raise PlatformError("CONFIG_INVALID", "配置路径不是文件: {}".format(path), EXIT_CONFIG)
    return resolved


def validate_schema_fields(
    config: Dict[str, Any],
    required: Iterable[str],
    optional_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    required_set = set(required)
    allowed = required_set | set(optional_defaults)
    missing = sorted(required_set - set(config))
    unknown = sorted(set(config) - allowed)
    if missing:
        raise PlatformError("CONFIG_INVALID", "配置缺少字段: {}".format(", ".join(missing)), EXIT_CONFIG)
    if unknown:
        raise PlatformError("CONFIG_INVALID", "配置包含未知字段: {}".format(", ".join(unknown)), EXIT_CONFIG)
    merged = dict(optional_defaults)
    merged.update(config)
    if type(merged["schema_version"]) is not int or merged["schema_version"] != 1:
        raise PlatformError("CONFIG_INVALID", "schema_version 目前只支持整数 1", EXIT_CONFIG)
    return merged


def require_string(config: Dict[str, Any], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PlatformError("CONFIG_INVALID", "{} 必须是非空字符串".format(field), EXIT_CONFIG)
    return value


def require_bool(config: Dict[str, Any], field: str) -> bool:
    value = config.get(field)
    if type(value) is not bool:
        raise PlatformError("CONFIG_INVALID", "{} 必须是布尔值".format(field), EXIT_CONFIG)
    return value


def require_int(config: Dict[str, Any], field: str, minimum: Optional[int] = None) -> int:
    value = config.get(field)
    if type(value) is not int:
        raise PlatformError("CONFIG_INVALID", "{} 必须是整数".format(field), EXIT_CONFIG)
    if minimum is not None and value < minimum:
        raise PlatformError("CONFIG_INVALID", "{} 不能小于 {}".format(field, minimum), EXIT_CONFIG)
    return value


def require_number(
    config: Dict[str, Any],
    field: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    minimum_exclusive: bool = False,
) -> float:
    value = config.get(field)
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise PlatformError("CONFIG_INVALID", "{} 必须是有限数值".format(field), EXIT_CONFIG)
    number = float(value)
    if minimum is not None:
        invalid = number <= minimum if minimum_exclusive else number < minimum
        if invalid:
            operator = "大于" if minimum_exclusive else "不小于"
            raise PlatformError("CONFIG_INVALID", "{} 必须{} {}".format(field, operator, minimum), EXIT_CONFIG)
    if maximum is not None and number > maximum:
        raise PlatformError("CONFIG_INVALID", "{} 不能大于 {}".format(field, maximum), EXIT_CONFIG)
    return number


def validate_model_name(value: Any) -> str:
    if not isinstance(value, str) or not MODEL_NAME_RE.fullmatch(value):
        raise PlatformError(
            "CONFIG_INVALID",
            "model_name 必须匹配 {}".format(MODEL_NAME_RE.pattern),
            EXIT_CONFIG,
        )
    return value


def validate_basename(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or UNSAFE_FILENAME_RE.search(value)
    ):
        raise PlatformError("CONFIG_INVALID", "{} 只允许安全文件名，不能包含路径".format(field), EXIT_CONFIG)
    if "\x00" in value:
        raise PlatformError("CONFIG_INVALID", "{} 包含非法字符".format(field), EXIT_CONFIG)
    return value


def resolve_input_file(task_root: Path, relative_dir: Sequence[str], filename: str, field: str) -> Path:
    directory = task_root.joinpath(*relative_dir)
    directory_resolved = resolve_existing_path(directory, task_root, "{}目录".format(field))
    candidate = resolve_existing_path(directory / filename, task_root, field)
    if candidate.parent != directory_resolved or not candidate.is_file():
        raise PlatformError("PATH_OUTSIDE_TASK", "{}不是指定目录内的普通文件".format(field), EXIT_CONFIG)
    return candidate


def validate_label_map(path: Path) -> Dict[str, int]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlatformError("INPUT_INVALID", "标签映射不是合法 UTF-8 JSON: {}".format(path), EXIT_INPUT) from exc
    if not isinstance(value, dict) or not value:
        raise PlatformError("INPUT_INVALID", "标签映射必须是非空 JSON 对象", EXIT_INPUT)
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise PlatformError("INPUT_INVALID", "标签映射的键必须是非空字符串", EXIT_INPUT)
    ids = list(value.values())
    if any(type(item) is not int for item in ids):
        raise PlatformError("INPUT_INVALID", "标签映射值必须是整数", EXIT_INPUT)
    if sorted(ids) != list(range(len(ids))):
        raise PlatformError("INPUT_INVALID", "标签映射值必须是从 0 开始的连续唯一整数", EXIT_INPUT)
    return value


def validate_jsonl_dataset(path: Path, label_map: Dict[str, int], require_two_labels: bool) -> Tuple[int, set]:
    sample_count = 0
    observed = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PlatformError(
                        "INPUT_INVALID",
                        "{} 第 {} 行不是合法 JSON".format(path.name, line_number),
                        EXIT_INPUT,
                    ) from exc
                if not isinstance(row, dict):
                    raise PlatformError("INPUT_INVALID", "{} 第 {} 行必须是 JSON 对象".format(path.name, line_number), EXIT_INPUT)
                if "labels" in row:
                    raise PlatformError(
                        "INPUT_INVALID",
                        "{} 第 {} 行使用了不支持的 labels 字段，请改为 label".format(path.name, line_number),
                        EXIT_INPUT,
                    )
                content = row.get("content")
                label = row.get("label")
                if not isinstance(content, str) or not content.strip():
                    raise PlatformError("INPUT_INVALID", "{} 第 {} 行 content 必须是非空字符串".format(path.name, line_number), EXIT_INPUT)
                if not isinstance(label, str) or not label.strip():
                    raise PlatformError("INPUT_INVALID", "{} 第 {} 行 label 必须是非空字符串".format(path.name, line_number), EXIT_INPUT)
                if label not in label_map:
                    raise PlatformError("INPUT_INVALID", "{} 第 {} 行出现未知标签: {}".format(path.name, line_number, label), EXIT_INPUT)
                observed.add(label)
                sample_count += 1
    except UnicodeDecodeError as exc:
        raise PlatformError("INPUT_INVALID", "数据文件必须使用 UTF-8 编码: {}".format(path), EXIT_INPUT) from exc
    if sample_count == 0:
        raise PlatformError("INPUT_INVALID", "数据文件没有有效样本: {}".format(path), EXIT_INPUT)
    if require_two_labels and len(observed) < 2:
        raise PlatformError("INPUT_INVALID", "训练集必须至少包含两个类别", EXIT_INPUT)
    return sample_count, observed


def validate_gpu_available() -> None:
    try:
        import torch
    except ImportError as exc:
        raise PlatformError("GPU_UNAVAILABLE", "无法导入 PyTorch", EXIT_RUNTIME) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PlatformError("GPU_UNAVAILABLE", "配置要求 GPU，但容器没有可见 CUDA 设备", EXIT_RUNTIME)


class StatusWriter:
    def __init__(self, path: Path, operation: str):
        self.path = path
        self.payload: Dict[str, Any] = {
            "schema_version": 1,
            "operation": operation,
            "status": "STARTING",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "finished_at": None,
            "progress": {
                "current_epoch": 0,
                "total_epochs": None,
                "current_step": 0,
                "total_steps": None,
                "percent": 0.0,
            },
            "error": None,
        }

    def update(self, status: str, **values: Any) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    current = json.load(handle)
                if isinstance(current, dict):
                    self.payload = current
            except (OSError, json.JSONDecodeError):
                pass
        progress_update = values.pop("progress", None)
        if progress_update is not None:
            current_progress = dict(self.payload.get("progress") or {})
            current_progress.update(progress_update)
            values["progress"] = current_progress
        self.payload.update(values)
        self.payload["status"] = status
        self.payload["updated_at"] = utc_now()
        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            self.payload["finished_at"] = utc_now()
        atomic_write_json(self.path, self.payload)


def output_conflicts(directory: Path, operation: str) -> bool:
    if not directory.exists():
        return False
    allowed_names = {TRAIN_CONFIG_NAME} if operation == "train" else {TEST_CONFIG_NAME}
    for entry in directory.iterdir():
        if entry.name in allowed_names:
            continue
        if operation == "train" and entry.is_file() and entry.suffix == ".sh":
            continue
        return True
    return False


def stream_subprocess(argv: Sequence[str], log_paths: Sequence[Path]) -> Tuple[int, bool]:
    handles = []
    process: Optional[subprocess.Popen] = None
    cancelled = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
        if process is not None and process.poll() is None:
            process.terminate()

    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        for path in log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(path.open("w", encoding="utf-8", buffering=1))
        process = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            for handle in handles:
                handle.write(line)
        return process.wait(), cancelled
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        for handle in handles:
            handle.close()


def read_log_tail(path: Path, max_chars: int = 16000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            value = handle.read()
        return value[-max_chars:]
    except OSError:
        return ""


def classify_process_failure(log_path: Path, operation: str) -> PlatformError:
    tail = read_log_tail(log_path).lower()
    if "cuda out of memory" in tail or "cuda error: out of memory" in tail:
        return PlatformError("CUDA_OUT_OF_MEMORY", "CUDA 显存不足", EXIT_CUDA_OOM)
    code = "TRAINING_FAILED" if operation == "train" else "EVALUATION_FAILED"
    message = "训练进程执行失败" if operation == "train" else "评估进程执行失败"
    return PlatformError(code, message, EXIT_RUNTIME)


def write_failure_summary(
    path: Path,
    status: StatusWriter,
    error: PlatformError,
    started_at: str,
    log_path: Optional[Path],
) -> None:
    summary = {
        "schema_version": 1,
        "status": "CANCELLED" if error.code == "CANCELLED" else "FAILED",
        "started_at": started_at,
        "finished_at": utc_now(),
        "train_log": str(log_path) if log_path else None,
        **image_metadata(),
        "error_code": error.code,
        "error_message": error.message,
    }
    atomic_write_json(path, summary)
    status.update(summary["status"], exit_code=error.exit_code, error={"code": error.code, "message": error.message})


def prepare_train_config(config: Dict[str, Any], config_path: Path, task_root: Path) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    if config_path.name != TRAIN_CONFIG_NAME:
        raise PlatformError("CONFIG_INVALID", "训练配置文件名必须是 run_train.json", EXIT_CONFIG)
    cfg = validate_schema_fields(config, TRAIN_REQUIRED_FIELDS, TRAIN_OPTIONAL_DEFAULTS)
    model_name = validate_model_name(cfg["model_name"])
    model_root = task_root / "models" / model_name
    if config_path.parent.resolve() != model_root.resolve():
        raise PlatformError("CONFIG_INVALID", "训练配置文件必须位于对应模型目录", EXIT_CONFIG)
    expected_output = str(model_root)
    if require_string(cfg, "output_dir") != expected_output:
        raise PlatformError("CONFIG_INVALID", "output_dir 必须严格等于 {}".format(expected_output), EXIT_CONFIG)
    if require_string(cfg, "choose_device") not in DEVICE_CHOICES:
        raise PlatformError("CONFIG_INVALID", "choose_device 只支持 cpu 或 gpu", EXIT_CONFIG)
    if require_string(cfg, "training_mode") not in TRAINING_MODES:
        raise PlatformError("CONFIG_INVALID", "training_mode 只支持 quick/balance/quality", EXIT_CONFIG)
    if require_string(cfg, "loss_type") not in LOSS_CHOICES:
        raise PlatformError("CONFIG_INVALID", "单标签任务 loss_type 只支持 CE 或 Focal", EXIT_CONFIG)
    if cfg["label_distribution"] != "auto":
        raise PlatformError("CONFIG_INVALID", "label_distribution 当前只支持 auto", EXIT_CONFIG)
    if cfg["fp16_opt_level"] not in {"O0", "O1", "O2", "O3"}:
        raise PlatformError("CONFIG_INVALID", "fp16_opt_level 非法", EXIT_CONFIG)

    for field, minimum in {
        "per_gpu_train_batch_size": 1,
        "per_gpu_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "dataloader_num_workers": 0,
    }.items():
        require_int(cfg, field, minimum)
    require_int(cfg, "max_length", 1)
    if cfg["max_length"] > 512:
        raise PlatformError("CONFIG_INVALID", "max_length 不能大于 512", EXIT_CONFIG)
    require_int(cfg, "seed", 0)
    for field in {"max_steps", "save_steps"}:
        require_int(cfg, field)
    require_number(cfg, "learning_rate", 0, minimum_exclusive=True)
    require_number(cfg, "num_train_epochs", 0, minimum_exclusive=True)
    require_number(cfg, "weight_decay", 0)
    require_number(cfg, "adam_epsilon", 0, minimum_exclusive=True)
    require_number(cfg, "max_grad_norm", 0)
    require_number(cfg, "warmup_steps", 0, 0.999999)
    require_number(cfg, "threshold", 0, 1)
    require_number(cfg, "alpha", 0)
    require_number(cfg, "distribution_gamma", 0)
    require_number(cfg, "focal_gamma", 0)
    require_number(cfg, "non_security_keep_ratio", 0, 1, minimum_exclusive=True)
    for field in {"save_each_epoch", "fp16", "overwrite_cache"}:
        require_bool(cfg, field)
    if cfg["fp16"]:
        raise PlatformError("CONFIG_INVALID", "当前镜像未包含 Apex，fp16 必须为 false", EXIT_CONFIG)

    label_name = validate_basename(cfg["label_file"], "label_file")
    train_name = validate_basename(cfg["train_file"], "train_file")
    if not label_name.lower().endswith(".json") or not train_name.lower().endswith(".jsonl"):
        raise PlatformError("CONFIG_INVALID", "label_file 必须是 .json，train_file 必须是 .jsonl", EXIT_CONFIG)
    evaluate_name = cfg["evaluate_file"]
    if evaluate_name is not None:
        evaluate_name = validate_basename(evaluate_name, "evaluate_file")
        if not evaluate_name.lower().endswith(".jsonl"):
            raise PlatformError("CONFIG_INVALID", "evaluate_file 必须是 .jsonl", EXIT_CONFIG)

    label_path = resolve_input_file(task_root, ("datasets", "labels"), label_name, "label_file")
    train_path = resolve_input_file(task_root, ("datasets", "training"), train_name, "train_file")
    evaluate_path = (
        resolve_input_file(task_root, ("datasets", "test"), evaluate_name, "evaluate_file")
        if evaluate_name is not None
        else None
    )
    label_map = validate_label_map(label_path)
    validate_jsonl_dataset(train_path, label_map, require_two_labels=True)
    if evaluate_path is not None:
        validate_jsonl_dataset(evaluate_path, label_map, require_two_labels=False)
    paths = {"model_root": model_root, "label": label_path, "train": train_path}
    if evaluate_path is not None:
        paths["evaluate"] = evaluate_path
    return cfg, paths


def build_train_argv(cfg: Dict[str, Any], paths: Dict[str, Path], status_path: Path, result_path: Path) -> Sequence[str]:
    algorithm_root = Path(os.environ.get(
        "TRAINING_ALGORITHM_ROOT",
        "/opt/training/algorithm",
    ))
    pretrained = Path(os.environ.get(
        "TRAINING_PRETRAINED_MODEL",
        str(algorithm_root / "pretrained_model" / "TinyBert"),
    ))
    train_script = algorithm_root / "train.py"
    for path, label in ((train_script, "训练脚本"), (pretrained / "config.json", "预训练模型配置")):
        if not path.is_file():
            raise PlatformError("INPUT_INVALID", "{}不存在: {}".format(label, path), EXIT_INPUT)

    model_root = paths["model_root"]
    argv = [
        sys.executable,
        "-u",
        str(train_script),
        "--model_name_or_path", str(pretrained),
        "--train_file", str(paths["train"]),
        "--label_file", str(paths["label"]),
        "--output_dir", str(model_root / "output_models"),
        "--choose_device", cfg["choose_device"],
        "--status_file", str(status_path),
        "--result_file", str(result_path),
        "--dataset_cache_dir", str(model_root / "runtime" / "cache"),
    ]
    if "evaluate" in paths:
        argv.extend(["--evaluate_file", str(paths["evaluate"])])

    value_fields = [
        "max_length", "threshold", "per_gpu_train_batch_size", "per_gpu_eval_batch_size",
        "learning_rate", "gradient_accumulation_steps", "weight_decay", "adam_epsilon",
        "max_grad_norm", "num_train_epochs", "max_steps", "warmup_steps", "logging_steps",
        "save_steps", "seed", "fp16_opt_level", "loss_type", "label_distribution", "alpha",
        "distribution_gamma", "focal_gamma", "non_security_keep_ratio", "dataloader_num_workers",
    ]
    for field in value_fields:
        argv.extend(["--{}".format(field), str(cfg[field])])
    for field in ("save_each_epoch", "fp16", "overwrite_cache"):
        if cfg[field]:
            argv.append("--{}".format(field))
    return argv


def run_train(config_path: Path, task_root: Path) -> int:
    model_root = config_path.parent
    if output_conflicts(model_root, "train"):
        raise PlatformError("OUTPUT_CONFLICT", "模型目录已存在训练状态或产物，拒绝覆盖", EXIT_CONFLICT)

    status = StatusWriter(model_root / "status.json", "train")
    status.update("STARTING", **image_metadata())
    started_at = status.payload["started_at"]
    summary_path = model_root / "training_summary.json"
    fixed_log = model_root / "logs" / "train.log"
    legacy_log: Optional[Path] = None
    try:
        status.update("VALIDATING")
        cfg, paths = prepare_train_config(load_json_object(config_path), config_path, task_root)
        if cfg["choose_device"] == "gpu":
            validate_gpu_available()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        legacy_log = model_root / "train_{}.log".format(timestamp)
        result_path = model_root / "runtime" / "train_result.json"
        argv = build_train_argv(cfg, paths, status.path, result_path)
        status.update("RUNNING", model_name=cfg["model_name"], training_mode=cfg["training_mode"])
        return_code, cancelled = stream_subprocess(argv, (fixed_log, legacy_log))
        if cancelled:
            raise PlatformError("CANCELLED", "训练任务被终止", EXIT_CANCELLED)
        if return_code != 0:
            raise classify_process_failure(fixed_log, "train")
        if not result_path.is_file():
            raise PlatformError("ARTIFACT_ERROR", "训练完成但缺少内部结果文件", EXIT_ARTIFACT)
        legacy_model = model_root / "output_models" / "model"
        if not (legacy_model / "config.json").is_file():
            raise PlatformError("ARTIFACT_ERROR", "训练完成但旧模型目录不完整", EXIT_ARTIFACT)
        best_checkpoint = model_root / "best_checkpoint"
        shutil.copytree(legacy_model, best_checkpoint)
        result = load_json_object(result_path)
        summary = {
            "schema_version": 1,
            "status": "SUCCEEDED",
            "started_at": started_at,
            "finished_at": utc_now(),
            "model_name": cfg["model_name"],
            "training_mode": cfg["training_mode"],
            "global_step": result.get("global_step"),
            "average_loss": result.get("average_loss"),
            "best_epoch": result.get("best_epoch"),
            "selection_score": result.get("selection_score"),
            "best_checkpoint": "best_checkpoint",
            "legacy_model_dir": "output_models/model",
            "checkpoint_dir": "checkpoints",
            "train_log": "logs/train.log",
            "legacy_train_log": legacy_log.name,
            **image_metadata(),
        }
        atomic_write_json(summary_path, summary)
        status.update("SUCCEEDED", exit_code=0, error=None, progress={"percent": 100.0})
        return 0
    except PlatformError as exc:
        write_failure_summary(summary_path, status, exc, started_at, fixed_log if fixed_log.exists() else legacy_log)
        raise
    except Exception as exc:
        error = PlatformError("TRAINING_FAILED", "训练执行层异常: {}".format(exc), EXIT_RUNTIME)
        traceback.print_exc()
        write_failure_summary(summary_path, status, error, started_at, fixed_log if fixed_log.exists() else legacy_log)
        raise error from exc


def validate_test_id(value: Any) -> str:
    import uuid
    if not isinstance(value, str):
        raise PlatformError("CONFIG_INVALID", "test_id 必须是 UUID 字符串", EXIT_CONFIG)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise PlatformError("CONFIG_INVALID", "test_id 必须是合法 UUID", EXIT_CONFIG) from exc
    if str(parsed) != value.lower():
        raise PlatformError("CONFIG_INVALID", "test_id 必须使用标准小写 UUID 格式", EXIT_CONFIG)
    return value


def prepare_test_config(config: Dict[str, Any], config_path: Path, task_root: Path) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    if config_path.name != TEST_CONFIG_NAME:
        raise PlatformError("CONFIG_INVALID", "评估配置文件名必须是 run_test.json", EXIT_CONFIG)
    cfg = validate_schema_fields(config, TEST_REQUIRED_FIELDS, TEST_OPTIONAL_DEFAULTS)
    model_name = validate_model_name(cfg["model_name"])
    test_id = validate_test_id(cfg["test_id"])
    expected_root = task_root / "evaluation" / model_name / "tests" / test_id
    if config_path.parent.resolve() != expected_root.resolve():
        raise PlatformError("CONFIG_INVALID", "评估配置文件必须位于对应 test_id 目录", EXIT_CONFIG)
    if require_string(cfg, "choose_device") not in DEVICE_CHOICES:
        raise PlatformError("CONFIG_INVALID", "choose_device 只支持 cpu 或 gpu", EXIT_CONFIG)
    require_int(cfg, "per_gpu_eval_batch_size", 1)
    require_bool(cfg, "save_result_csv")
    require_bool(cfg, "plot_confusion_matrix")
    label_name = validate_basename(cfg["label_file"], "label_file")
    test_name = validate_basename(cfg["test_file"], "test_file")
    if not label_name.lower().endswith(".json") or not test_name.lower().endswith(".jsonl"):
        raise PlatformError("CONFIG_INVALID", "label_file 必须是 .json，test_file 必须是 .jsonl", EXIT_CONFIG)
    label_path = resolve_input_file(task_root, ("datasets", "labels"), label_name, "label_file")
    test_path = resolve_input_file(task_root, ("datasets", "test"), test_name, "test_file")
    label_map = validate_label_map(label_path)
    validate_jsonl_dataset(test_path, label_map, require_two_labels=False)
    model_root = resolve_existing_path(task_root / "models" / model_name, task_root, "模型目录")
    model_path = model_root / "best_checkpoint"
    model_config = model_path / "config.json"
    if not model_config.is_file():
        model_path = model_root / "output_models" / "model"
        model_config = model_path / "config.json"
    if not model_config.is_file():
        raise PlatformError("INPUT_INVALID", "模型不存在或不完整: {}".format(model_root), EXIT_INPUT)
    model_config = resolve_existing_path(model_config, task_root, "模型配置")
    model_path = model_config.parent
    return cfg, {
        "test_root": expected_root,
        "model_root": model_root,
        "model": model_path,
        "label": label_path,
        "test": test_path,
    }


def build_test_argv(cfg: Dict[str, Any], paths: Dict[str, Path]) -> Sequence[str]:
    algorithm_root = Path(os.environ.get(
        "TRAINING_ALGORITHM_ROOT",
        "/opt/training/algorithm",
    ))
    test_script = algorithm_root / "test.py"
    if not test_script.is_file():
        raise PlatformError("INPUT_INVALID", "评估脚本不存在: {}".format(test_script), EXIT_INPUT)
    test_root = paths["test_root"]
    return [
        sys.executable,
        "-u",
        str(test_script),
        "--model-path", str(paths["model"]),
        "--eval-data", str(paths["test"]),
        "--label-path", str(paths["label"]),
        "--output-dir", str(paths["model_root"]),
        "--result-suffix", cfg["test_id"],
        "--choose-device", cfg["choose_device"],
        "--per-gpu-eval-batch-size", str(cfg["per_gpu_eval_batch_size"]),
        "--save-result-csv", str(cfg["save_result_csv"]).lower(),
        "--plot-confusion-matrix", str(cfg["plot_confusion_matrix"]).lower(),
        "--report-path", str(test_root / "report.json"),
        "--predictions-path", str(test_root / "predictions.json"),
        "--model-id", cfg["model_name"],
        "--report-id", cfg["test_id"],
        "--dataset-id", paths["test"].name,
    ]


def run_validate(config_path: Path, task_root: Path) -> int:
    test_root = config_path.parent
    if output_conflicts(test_root, "validate"):
        raise PlatformError("OUTPUT_CONFLICT", "评估目录已存在状态或产物，拒绝覆盖", EXIT_CONFLICT)
    status = StatusWriter(test_root / "status.json", "validate")
    status.update("STARTING", **image_metadata())
    fixed_log = test_root / "val.log"
    legacy_log: Optional[Path] = None
    try:
        status.update("VALIDATING")
        cfg, paths = prepare_test_config(load_json_object(config_path), config_path, task_root)
        if cfg["choose_device"] == "gpu":
            validate_gpu_available()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        legacy_log = paths["model_root"] / "test_{}.log".format(timestamp)
        argv = build_test_argv(cfg, paths)
        status.update("RUNNING", model_name=cfg["model_name"], test_id=cfg["test_id"])
        return_code, cancelled = stream_subprocess(argv, (fixed_log, legacy_log))
        if cancelled:
            raise PlatformError("CANCELLED", "评估任务被终止", EXIT_CANCELLED)
        if return_code != 0:
            raise classify_process_failure(fixed_log, "validate")
        for required in (test_root / "report.json", test_root / "predictions.json"):
            if not required.is_file() or required.stat().st_size == 0:
                raise PlatformError("ARTIFACT_ERROR", "评估完成但缺少标准产物: {}".format(required.name), EXIT_ARTIFACT)
        status.update(
            "SUCCEEDED",
            exit_code=0,
            error=None,
            progress={"current_step": 1, "total_steps": 1, "percent": 100.0},
        )
        return 0
    except PlatformError as exc:
        terminal = "CANCELLED" if exc.code == "CANCELLED" else "FAILED"
        status.update(terminal, exit_code=exc.exit_code, error={"code": exc.code, "message": exc.message})
        raise
    except Exception as exc:
        traceback.print_exc()
        error = PlatformError("EVALUATION_FAILED", "评估执行层异常: {}".format(exc), EXIT_RUNTIME)
        status.update("FAILED", exit_code=error.exit_code, error={"code": error.code, "message": error.message})
        raise error from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training platform container entrypoint")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("train", "validate"):
        child = subparsers.add_parser(name)
        child.add_argument("--task-type", required=True)
        child.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        if args.task_type != TASK_TYPE:
            raise PlatformError("TASK_TYPE_UNSUPPORTED", "当前镜像不支持任务类型: {}".format(args.task_type), EXIT_CONFIG)
        task_root = get_task_root()
        config_path = validate_config_path(args.config, task_root)
        if args.operation == "train":
            return run_train(config_path, task_root)
        return run_validate(config_path, task_root)
    except PlatformError as exc:
        print("[{}] {}".format(exc.code, exc.message), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
