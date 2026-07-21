"""Generic container entrypoint for training images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn, Optional, Sequence


DEFAULT_ALGORITHMS_ROOT = "/opt/training/algorithms"
TASK_OPERATION_SCRIPTS = {
    "text_classification_single": {
        "train": "text_classification_single/train.py",
        "validate": "text_classification_single/test.py",
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training platform container entrypoint")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    operations = sorted(
        {
            operation
            for handlers in TASK_OPERATION_SCRIPTS.values()
            for operation in handlers
        }
    )
    for operation in operations:
        child = subparsers.add_parser(operation)
        child.add_argument(
            "--task-type",
            required=True,
            choices=sorted(TASK_OPERATION_SCRIPTS),
            help="镜像内算法类型",
        )
        child.add_argument("--config", required=True)
    return parser.parse_args(argv)


def resolve_script(task_type: str, operation: str) -> Path:
    algorithms_root = Path(
        os.environ.get("TRAINING_ALGORITHMS_ROOT", DEFAULT_ALGORITHMS_ROOT)
    ).resolve()
    relative_script = TASK_OPERATION_SCRIPTS[task_type].get(operation)
    if relative_script is None:
        raise RuntimeError(
            "任务类型{}不支持操作{}".format(task_type, operation)
        )
    script = (algorithms_root / relative_script).resolve()
    try:
        script.relative_to(algorithms_root)
    except ValueError as exc:
        raise RuntimeError("算法脚本必须位于统一算法目录内") from exc
    if not script.is_file():
        raise RuntimeError("算法脚本不存在: {}".format(script))
    return script


def main(argv: Optional[Sequence[str]] = None) -> NoReturn:
    args = parse_args(argv)
    script = resolve_script(args.task_type, args.operation)
    os.environ["TRAINING_ALGORITHM_ROOT"] = str(script.parent)
    os.execv(
        sys.executable,
        [sys.executable, "-u", str(script), "--config", args.config],
    )
    raise AssertionError("os.execv returned unexpectedly")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print("[ENTRYPOINT_ERROR] {}".format(exc), file=sys.stderr)
        sys.exit(10)
