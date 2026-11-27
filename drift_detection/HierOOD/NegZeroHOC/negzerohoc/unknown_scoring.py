from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def as_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim != 2:
        raise ValueError(
            f"Expected a feature vector or matrix, got shape {tuple(features.shape)}"
        )
    return features


def grouped_unknown_logits(
    image_features: torch.Tensor,
    known_features: torch.Tensor,
    unknown_features: torch.Tensor,
    *,
    logit_scale: float,
    aggregation: str = "logmeanexp",
) -> torch.Tensor:
    """Return known-child logits plus one aggregated unknown logit.

    ``logmeanexp`` treats prototypes as a marginalized single class and is the
    legacy Idea 4/5 behavior. ``logsumexp`` preserves total negative mass as in
    the original NegPrompt softmax denominator.
    """
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be positive")

    images = F.normalize(as_feature_matrix(image_features).float(), dim=-1)
    known = F.normalize(as_feature_matrix(known_features).float(), dim=-1)
    unknown = F.normalize(as_feature_matrix(unknown_features).float(), dim=-1)
    if known.shape[1] != images.shape[1] or unknown.shape[1] != images.shape[1]:
        raise ValueError("Image, known, and unknown feature dimensions must match")

    known_logits = float(logit_scale) * (images @ known.t())
    prototype_logits = float(logit_scale) * (images @ unknown.t())
    if aggregation == "logmeanexp":
        unknown_logit = torch.logsumexp(prototype_logits, dim=1) - math.log(
            prototype_logits.shape[1]
        )
    elif aggregation == "logsumexp":
        unknown_logit = torch.logsumexp(prototype_logits, dim=1)
    else:
        raise ValueError(f"Unsupported unknown aggregation: {aggregation!r}")
    return torch.cat([known_logits, unknown_logit.unsqueeze(1)], dim=1)
