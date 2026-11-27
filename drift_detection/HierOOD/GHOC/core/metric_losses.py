from __future__ import annotations

from typing import Any, Callable

from loss.HiMSmin import HiMS_min_loss
from loss.MS import MS_loss
from loss.WeiHiMS import HiMS_min_wei_loss


def build_metric_loss(name: str, **kwargs) -> Callable:
    loss_name = name.lower()

    if loss_name == "ms":
        return lambda features, path_labels: MS_loss(features, path_labels, **kwargs)
    if loss_name == "himsmin":
        return lambda features, path_labels: HiMS_min_loss(features, path_labels, **kwargs)
    if loss_name == "weihims":
        return lambda features, path_labels: HiMS_min_wei_loss(features, path_labels, **kwargs)
    raise ValueError(f"Unsupported loss name: {name}")
