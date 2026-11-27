from __future__ import annotations

import torch


def compute_class_means(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(labels.long(), sorted=True)
    means = []
    counts = []
    for cls in classes:
        cls_features = features[labels == cls]
        means.append(cls_features.mean(dim=0))
        counts.append(torch.tensor(cls_features.shape[0], dtype=torch.long))
    return torch.stack(means, dim=0), torch.stack(counts, dim=0)


def variance_scores(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    class_means, _ = compute_class_means(features, labels)
    if class_means.shape[0] <= 1:
        return torch.zeros(features.shape[1], dtype=features.dtype)
    return class_means.var(dim=0, unbiased=False)


def fisher_scores(features: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    class_means, counts = compute_class_means(features, labels)
    if class_means.shape[0] <= 1:
        return torch.zeros(features.shape[1], dtype=features.dtype)
    global_mean = features.mean(dim=0)
    between = torch.zeros(features.shape[1], dtype=features.dtype)
    within = torch.zeros(features.shape[1], dtype=features.dtype)
    classes = torch.unique(labels.long(), sorted=True)
    for idx, cls in enumerate(classes):
        cls_features = features[labels == cls]
        diff_mean = class_means[idx] - global_mean
        between += counts[idx].to(features.dtype) * diff_mean.pow(2)
        if cls_features.shape[0] > 1:
            within += (cls_features - class_means[idx]).pow(2).mean(dim=0)
    between = between / max(int(counts.sum().item()), 1)
    within = within / max(len(classes), 1)
    return between / within.clamp_min(eps)
