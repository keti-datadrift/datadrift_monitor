from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.feature_io import load_artifact


def make_generated_split_payload(
    split_artifact: dict[str, Any],
    depth_features: dict[int, torch.Tensor],
    generator_type: str,
    generator_config: dict[str, Any],
    base_feature_path: str,
):
    payload = dict(split_artifact)
    payload["depth_features"] = {int(depth): tensor.cpu() for depth, tensor in depth_features.items()}
    payload["generator_type"] = generator_type
    payload["generator_config"] = generator_config
    payload["base_feature_path"] = base_feature_path
    return payload


def load_generated_feature_artifact(path: str | Path) -> dict[str, Any]:
    artifact = load_artifact(path)
    artifact["depth_features"] = {int(depth): tensor for depth, tensor in artifact["depth_features"].items()}
    return artifact


def resolve_feature_tensor(config: dict[str, Any], experiment_dir: Path, split: str) -> tuple[torch.Tensor, dict[str, Any]]:
    fg_cfg = config.get("feature_generation", {})
    enabled = bool(fg_cfg.get("enable", False))
    if enabled:
        generated_path = experiment_dir / f"generated_features_{split}.pt"
        artifact = load_generated_feature_artifact(generated_path)
        selected_depth = int(fg_cfg.get("selected_depth", 1))
        if selected_depth not in artifact["depth_features"]:
            raise KeyError(f"Selected depth {selected_depth} missing from {generated_path}")
        return artifact["depth_features"][selected_depth].float(), {
            "source_type": "generated",
            "artifact_path": str(generated_path),
            "selected_depth": selected_depth,
            "generator_type": artifact.get("generator_type"),
        }
    base_path = experiment_dir / f"features_{split}.pt"
    artifact = load_artifact(base_path)
    return artifact["features"].float(), {
        "source_type": "base",
        "artifact_path": str(base_path),
        "selected_depth": None,
        "generator_type": None,
    }
