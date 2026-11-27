from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


CLASSIFIER_METRIC_COLUMNS = ["auroc", "ap", "fpr95", "ece", "ace"]


def evaluate_classifier(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def compute_fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    if y_true.size == 0 or not np.any(y_true == 1) or not np.any(y_true == 0):
        return float("nan")
    fpr, tpr, _thresholds = roc_curve(y_true, y_score, pos_label=1)
    mask = tpr >= float(target_tpr)
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(np.float64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.0
    scores = np.clip(y_score, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    ece = 0.0
    n = float(y_true.size)
    for i in range(int(n_bins)):
        lo = edges[i]
        hi = edges[i + 1]
        if i == int(n_bins) - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(scores[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def compute_ace(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(np.float64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.0
    scores = np.clip(y_score, 0.0, 1.0)
    order = np.argsort(scores)
    splits = np.array_split(order, int(max(1, n_bins)))
    gaps: list[float] = []
    for idx in splits:
        if idx.size == 0:
            continue
        conf = float(scores[idx].mean())
        acc = float(y_true[idx].mean())
        gaps.append(abs(acc - conf))
    if not gaps:
        return 0.0
    return float(np.mean(np.asarray(gaps, dtype=np.float64)))
