import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def _best_threshold(y_true: np.ndarray, scores: np.ndarray):
    thresholds = np.unique(scores)
    best_acc = -1.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int64)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    y_true = np.concatenate(
        [np.zeros(len(id_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64)]
    )
    scores = np.concatenate([id_scores, ood_scores])
    threshold = _best_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(np.int64)
    return {
        "AUROC": float(roc_auc_score(y_true, scores)),
        "FPR@95": fpr_at_95_tpr(y_true, scores),
        "Detection Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "threshold": threshold,
        "threshold_rule": "best_test_threshold",
    }
