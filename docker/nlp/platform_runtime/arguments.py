"""Helpers that keep process-only platform state out of saved algorithm arguments."""

import argparse


def training_argument_snapshot(args):
    """Return portable effective arguments without process-only platform fields."""
    if not isinstance(args, argparse.Namespace):
        raise TypeError("args must be an argparse.Namespace")

    values = {}
    for name, value in vars(args).items():
        if name.startswith("platform_"):
            continue
        if name == "device" and value is not None:
            value = str(value)
        elif name == "label_distribution":
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach()
            to_cpu = getattr(value, "cpu", None)
            if callable(to_cpu):
                value = to_cpu()
        values[name] = value

    return argparse.Namespace(**values)
