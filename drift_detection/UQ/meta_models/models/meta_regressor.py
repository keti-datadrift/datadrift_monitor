from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


def build_estimator(model_name: str, device: str, random_seed: int = 42) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]

    if model_name == "ridge":
        reg = Ridge(random_state=int(random_seed))
    elif model_name == "rf_regressor":
        reg = RandomForestRegressor(
            n_estimators=200,
            random_state=int(random_seed),
            n_jobs=-1,
        )
    elif model_name == "gb_regressor":
        reg = GradientBoostingRegressor(random_state=int(random_seed))
    elif model_name == "xgb_regressor":
        if xgb is None:
            raise ImportError("xgboost is required for model='xgb_regressor'.")
        xgb_kwargs = {}
        if str(device).lower().startswith("cuda"):
            xgb_kwargs.update({"tree_method": "gpu_hist", "gpu_id": 0})
        reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=int(random_seed),
            **xgb_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    steps.append(("reg", reg))
    return Pipeline(steps)


def param_grid(model_name: str) -> dict[str, list[Any]]:
    if model_name == "ridge":
        return {"reg__alpha": [0.1, 1.0, 10.0, 100.0]}
    if model_name == "rf_regressor":
        return {
            "reg__n_estimators": [100, 200],
            "reg__max_depth": [None, 5, 10],
            "reg__min_samples_leaf": [1, 3, 5],
        }
    if model_name == "gb_regressor":
        return {
            "reg__n_estimators": [50, 100, 200],
            "reg__max_depth": [2, 3, 4],
            "reg__learning_rate": [0.03, 0.1, 0.3],
        }
    if model_name == "xgb_regressor":
        return {
            "reg__n_estimators": [50, 100, 200],
            "reg__max_depth": [2, 3, 4],
            "reg__learning_rate": [0.03, 0.1, 0.3],
            "reg__reg_alpha": [0.0, 0.5, 1.0],
            "reg__reg_lambda": [0.0, 1.0],
        }
    raise ValueError(f"Unsupported model: {model_name}")
