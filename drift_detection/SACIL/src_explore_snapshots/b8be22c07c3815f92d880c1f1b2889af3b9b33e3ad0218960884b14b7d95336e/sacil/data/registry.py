from __future__ import annotations

from pathlib import Path

from .cifar100 import CIFAR100DataModule
from .sessions import ClassOrderProtocol


def build_data_module(
    dataset_name: str,
    root: str | Path,
    protocol: ClassOrderProtocol,
    *,
    download: bool = False,
    color_jitter: bool = False,
):
    normalized = dataset_name.lower().replace("-", "").replace("_", "")
    if normalized == "cifar100":
        return CIFAR100DataModule(
            root,
            protocol,
            download=download,
            color_jitter=color_jitter,
        )
    raise ValueError(f"unsupported dataset: {dataset_name}")
