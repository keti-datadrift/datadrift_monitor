from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


def build_estimator(model_name: str, device: str, random_seed: int = 42) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]

    if model_name == "logistic":
        clf = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=5000,
            n_jobs=None,
            random_state=int(random_seed),
        )
    elif model_name == "gb_classifier":
        if xgb is None:
            raise ImportError("xgboost is required for model='gb_classifier'.")
        xgb_kwargs = {}
        if str(device).lower().startswith("cuda"):
            xgb_kwargs.update({"tree_method": "gpu_hist", "gpu_id": 0})
        clf = xgb.XGBClassifier(eval_metric="logloss", random_state=int(random_seed), **xgb_kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    steps.append(("clf", clf))
    return Pipeline(steps)


def param_grid(model_name: str) -> dict[str, list[Any]]:
    if model_name == "logistic":
        return {"clf__C": [0.5, 0.3, 0.1, 0.05, 0.01]}
    if model_name == "gb_classifier":
        return {
            "clf__n_estimators": list(range(10, 31, 5)),
            "clf__max_depth": [2, 3, 4, 5, 6],
            "clf__learning_rate": [0.3],
            "clf__reg_alpha": [0.5, 1.0, 1.5],
            "clf__reg_lambda": [0.0],
        }
    raise ValueError(f"Unsupported model: {model_name}")
