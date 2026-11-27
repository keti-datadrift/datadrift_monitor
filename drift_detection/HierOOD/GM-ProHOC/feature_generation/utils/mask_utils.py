from __future__ import annotations

from typing import Any

import torch


def normalize_scores(scores: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return scores
    max_val = scores.max().clamp_min(1e-12)
    return scores / max_val


def build_mask_from_scores(scores: torch.Tensor, mask_cfg: dict[str, Any]) -> torch.Tensor:
    mask_type = str(mask_cfg.get("type", "binary")).lower()
    topk = mask_cfg.get("topk")
    threshold = mask_cfg.get("threshold")
    normalized = normalize_scores(scores, bool(mask_cfg.get("normalize_scores", True)))
    if threshold is not None:
        mask = (normalized >= float(threshold)).to(scores.dtype)
    elif topk is not None:
        k = max(1, min(int(topk), normalized.numel()))
        top_indices = torch.topk(normalized, k=k, largest=True).indices
        mask = torch.zeros_like(normalized)
        mask[top_indices] = 1.0
    else:
        mask = torch.ones_like(normalized)
    if mask_type == "soft":
        return normalized * mask
    return mask
