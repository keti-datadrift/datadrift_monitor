from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _read_full_hierarchy(hierarchy_path: str | Path):
    node_list: list[str] = []
    child2parent: dict[str, str] = {}

    def _walk(node: dict[str, Any], parent: str | None = None):
        name = node["name"]
        node_list.append(name)
        if parent is not None:
            child2parent[name] = parent
        for child in node.get("children", []):
            _walk(child, name)

    with Path(hierarchy_path).open("r", encoding="utf-8") as handle:
        tree = json.load(handle)
    _walk(tree)
    return sorted(node_list), child2parent


def _build_raw_paths(leaf_class_names: list[str], node_list: list[str], child2parent: dict[str, str]):
    node_to_index = {name: idx for idx, name in enumerate(node_list)}
    root_name = "root"
    paths = []
    max_len = 0
    for leaf in leaf_class_names:
        path_names = []
        cursor = leaf
        while True:
            path_names.append(cursor)
            if cursor not in child2parent:
                break
            cursor = child2parent[cursor]
        path_names.reverse()
        path_names = [name for name in path_names if name != root_name]
        path = [node_to_index[name] for name in path_names]
        max_len = max(max_len, len(path))
        paths.append(path)
    padded = []
    for path in paths:
        leaf_token = path[-1]
        padded.append(path + [leaf_token] * (max_len - len(path)))
    return torch.tensor(padded, dtype=torch.long), node_to_index


def build_leaf_path_matrix(
    hierarchy,
    leaf_class_names: list[str],
    use_pruned: bool = True,
    hierarchy_path: str | Path | None = None,
):
    if use_pruned:
        max_depth = hierarchy.max_depth
        leaf_paths = []
        for leaf_name in leaf_class_names:
            path = hierarchy.node_ancestors[leaf_name].copy()
            if path:
                path = path[1:]
            path.append(hierarchy.id_node_list.index(leaf_name))
            leaf_token = path[-1]
            if len(path) < max_depth:
                path = path + [leaf_token] * (max_depth - len(path))
            leaf_paths.append(path)
        return torch.tensor(leaf_paths, dtype=torch.long), {
            "mode": "pruned",
            "node_names": hierarchy.id_node_list,
        }

    if hierarchy_path is None:
        raise ValueError("hierarchy_path is required when use_pruned is False")

    node_list, child2parent = _read_full_hierarchy(hierarchy_path)
    path_matrix, node_to_index = _build_raw_paths(leaf_class_names, node_list, child2parent)
    return path_matrix, {
        "mode": "unpruned",
        "node_names": node_list,
        "node_to_index": node_to_index,
    }


def targets_to_path_labels(targets: torch.Tensor, leaf_path_matrix: torch.Tensor, device=None) -> torch.Tensor:
    labels = leaf_path_matrix[targets.long().cpu()]
    if device is not None:
        labels = labels.to(device)
    return labels
