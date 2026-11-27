from __future__ import annotations

from typing import Any

import torch

from feature_extractor import CLIPImageEncoder, ResNet50


def build_backbone(config: dict[str, Any]):
    backbone_cfg = config["backbone"]
    model_type = backbone_cfg["type"].lower()
    feat_dim = backbone_cfg["feat_dim"]
    normalize = backbone_cfg.get("normalize", False)
    freeze_backbone = backbone_cfg.get("freeze_backbone", False)

    if model_type == "resnet50":
        return ResNet50(
            feat_dim=feat_dim,
            weights=backbone_cfg.get("weights"),
            freeze_backbone=freeze_backbone,
            normalize=normalize,
        )
    if model_type == "clip":
        return CLIPImageEncoder(
            feat_dim=feat_dim,
            model_name=backbone_cfg.get("model_name", "openai/clip-vit-base-patch32"),
            freeze_backbone=freeze_backbone,
            normalize=normalize,
        )
    raise ValueError(f"Unsupported backbone type: {model_type}")


def backbone_summary(config: dict[str, Any]) -> dict[str, Any]:
    backbone_cfg = dict(config["backbone"])
    return {
        "type": backbone_cfg.get("type"),
        "feat_dim": backbone_cfg.get("feat_dim"),
        "normalize": backbone_cfg.get("normalize"),
        "freeze_backbone": backbone_cfg.get("freeze_backbone"),
        "weights": backbone_cfg.get("weights"),
        "model_name": backbone_cfg.get("model_name"),
    }


def maybe_wrap_dataparallel(model, config: dict[str, Any]):
    runtime_cfg = config.get("runtime", {})
    use_multi_gpu = runtime_cfg.get("multi_gpu", False)
    device_ids = runtime_cfg.get("device_ids")

    if not use_multi_gpu:
        return model
    if not torch.cuda.is_available():
        return model
    if torch.cuda.device_count() < 2:
        return model

    if device_ids:
        return torch.nn.DataParallel(model, device_ids=device_ids)
    return torch.nn.DataParallel(model)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_backbone_checkpoint(model, checkpoint_path, map_location="cpu"):
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    try:
        model.load_state_dict(state_dict)
        return model
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            stripped[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(stripped)
        return model
