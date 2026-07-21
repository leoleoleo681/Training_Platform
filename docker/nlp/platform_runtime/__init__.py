"""Shared runtime protocol for training-platform jobs."""

from .arguments import training_argument_snapshot
from .runtime import JobRuntime

__all__ = ["JobRuntime", "training_argument_snapshot"]
