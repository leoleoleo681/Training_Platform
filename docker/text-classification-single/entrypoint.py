"""Generic container entrypoint for training images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn, Optional, Sequence


DEFAULT_ALGORITHM_ROOT = "/opt/training/algorithm"
OPERATION_SCRIPTS = {
    "train": "train.py",
    "validate": "test.py",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training platform container entrypoint")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in OPERATION_SCRIPTS:
        child = subparsers.add_parser(operation)
        child.add_argument("--task-type", help="由平台用于选择镜像，镜像内不再解析")
        child.add_argument("--config", required=True)
    return parser.parse_args(argv)


def resolve_script(operation: str) -> Path:
    algorithm_root = Path(
        os.environ.get("TRAINING_ALGORITHM_ROOT", DEFAULT_ALGORITHM_ROOT)
    ).resolve()
    script = (algorithm_root / OPERATION_SCRIPTS[operation]).resolve()
    try:
        script.relative_to(algorithm_root)
    except ValueError as exc:
        raise RuntimeError("算法脚本必须位于算法目录内") from exc
    if not script.is_file():
        raise RuntimeError("算法脚本不存在: {}".format(script))
    return script


def main(argv: Optional[Sequence[str]] = None) -> NoReturn:
    args = parse_args(argv)
    script = resolve_script(args.operation)
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
