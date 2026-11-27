from __future__ import annotations

from pathlib import Path

from .cifar100 import CIFAR100DataModule
from .imagenet100 import ImageNet100DataModule
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
    if normalized == "imagenet100":
        if download:
            raise ValueError("ImageNet-100 cannot be downloaded automatically")
        return ImageNet100DataModule(root, protocol)
    raise ValueError(f"unsupported dataset: {dataset_name}")
