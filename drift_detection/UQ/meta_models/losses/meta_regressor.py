from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64).reshape(-1), 0.0, 1.0)
    return {
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else 0.0,
    }
