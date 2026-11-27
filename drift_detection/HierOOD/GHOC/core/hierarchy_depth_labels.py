from __future__ import annotations

from pathlib import Path

import torch

from core.hierarchy_labels import build_leaf_path_matrix


def build_depth_label_matrix(
    hierarchy,
    leaf_class_names: list[str],
    use_pruned: bool = True,
    hierarchy_path: str | Path | None = None,
):
    matrix, meta = build_leaf_path_matrix(
        hierarchy,
        leaf_class_names,
        use_pruned=use_pruned,
        hierarchy_path=hierarchy_path,
    )
    return matrix.long(), meta


def targets_to_depth_labels(targets: torch.Tensor, depth_label_matrix: torch.Tensor, device=None) -> torch.Tensor:
    labels = depth_label_matrix[targets.long().cpu()]
    if device is not None:
        labels = labels.to(device)
    return labels


def depth_labels_to_dict(depth_labels: torch.Tensor) -> dict[int, torch.Tensor]:
    if depth_labels.ndim != 2:
        raise ValueError(f"depth_labels must be rank-2, got shape {tuple(depth_labels.shape)}")
    return {depth + 1: depth_labels[:, depth].long() for depth in range(depth_labels.shape[1])}
