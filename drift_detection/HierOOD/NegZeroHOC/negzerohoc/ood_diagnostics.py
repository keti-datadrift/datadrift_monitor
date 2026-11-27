from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve


def max_cosine_scores(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return maximum cosine similarity and prototype index per image."""
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("image_features and text_features must both be matrices")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("image and text feature dimensions must match")
    images = F.normalize(image_features.float(), dim=-1)
    texts = F.normalize(text_features.float(), dim=-1)
    return (images @ texts.t()).max(dim=1)


def score_summary(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def binary_ood_metrics(id_ood_scores, ood_scores) -> dict:
    """Measure OOD separation when larger scores mean more OOD-like."""
    id_array = np.asarray(id_ood_scores, dtype=np.float64)
    ood_array = np.asarray(ood_scores, dtype=np.float64)
    if id_array.size == 0 or ood_array.size == 0:
        raise ValueError("Both ID and OOD score arrays must be non-empty")

    labels = np.concatenate([
        np.zeros(id_array.size, dtype=np.int64),
        np.ones(ood_array.size, dtype=np.int64),
    ])
    scores = np.concatenate([id_array, ood_array])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    balanced = 0.5 * (tpr + (1.0 - fpr))
    best_index = int(np.argmax(balanced))
    reaches_95 = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[reaches_95].min()) if reaches_95.size else 1.0

    id_labels = 1 - labels
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr_out": float(average_precision_score(labels, scores)),
        "aupr_in": float(average_precision_score(id_labels, -scores)),
        "fpr95": fpr95,
        "best_balanced_acc_diagnostic_only": float(balanced[best_index]),
        "best_ood_score_threshold_diagnostic_only": float(thresholds[best_index]),
        "id_ood_score": score_summary(id_array),
        "ood_score": score_summary(ood_array),
    }


def balanced_label_accuracy(targets, predictions) -> float:
    return float(balanced_accuracy_score(list(targets), list(predictions)))


def macro_class_ood_metrics(
    id_ood_scores,
    ood_scores,
    ood_labels: list[str],
) -> dict:
    ood_array = np.asarray(ood_scores, dtype=np.float64)
    if len(ood_labels) != int(ood_array.size):
        raise ValueError("ood_labels and ood_scores must have matching lengths")
    by_class = {}
    for label in sorted(set(ood_labels)):
        mask = np.asarray([item == label for item in ood_labels], dtype=bool)
        by_class[label] = binary_ood_metrics(id_ood_scores, ood_array[mask])
    macro_keys = ("auroc", "aupr_out", "aupr_in", "fpr95", "best_balanced_acc_diagnostic_only")
    macro = {
        key: float(np.mean([metrics[key] for metrics in by_class.values()]))
        for key in macro_keys
    }
    return {"macro": macro, "by_class": by_class}

