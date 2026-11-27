from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from meta_models.commands.common import (
    FeatureSpec,
    build_feature_matrix,
    infer_feature_spec,
    load_object,
    load_training_dataframe,
    sanitize_feature_matrix,
    save_object,
)
from meta_models.losses.meta_regressor import evaluate_regressor


def _parse_model_index(path: Path) -> int:
    match = re.match(r"^model_(\d+)\.joblib$", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _resolve_model_paths(model_cfg: dict[str, Any]) -> list[Path]:
    model_root_raw = str(model_cfg.get("model_root", "")).strip()
    if not model_root_raw:
        raise ValueError("For mode='test', set model.model_root in config.")
    model_root = Path(model_root_raw).resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"model.model_root not found: {model_root}")

    paths = [*model_root.glob("model_*.joblib")]
    if not paths and model_root.name != "models":
        nested_model_dir = model_root / "models"
        paths = [*nested_model_dir.glob("model_*.joblib")]
    paths = sorted({p.resolve() for p in paths}, key=_parse_model_index)
    if not paths:
        raise FileNotFoundError(f"No model_*.joblib files found in {model_root}")
    return paths


def _resolve_train_run_root(model_paths: list[Path]) -> Path:
    if not model_paths:
        raise ValueError("model_paths is empty.")
    first_parent = model_paths[0].parent
    return first_parent.parent if first_parent.name == "models" else first_parent


def _load_feature_spec_for_test(train_run_root: Path, df: pd.DataFrame, current_features: list[str]) -> FeatureSpec:
    metadata_path = train_run_root / "metadata.json"
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        input_features = metadata.get("input_features")
        dim_by_feature = metadata.get("dim_by_feature")
        if isinstance(input_features, list) and isinstance(dim_by_feature, dict):
            missing = [c for c in input_features if c not in df.columns]
            if missing:
                raise ValueError(
                    "Missing required feature columns from training metadata: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
                )
            dim_dict = {str(k): int(v) for k, v in dim_by_feature.items() if str(k) in input_features}
            return FeatureSpec(grad_columns=[str(c) for c in input_features], dim_by_column=dim_dict)
    return infer_feature_spec(df, current_features)

def run_test(config: dict[str, Any], run_dir: Path) -> Path:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    out_dir = run_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    models_dir = out_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_paths = _resolve_model_paths(model_cfg)
    train_run_root = _resolve_train_run_root(model_paths)

    df, label_col, input_features, root_info = load_training_dataframe(dataset_cfg, task="regressor")
    y = df[label_col].astype(float).to_numpy()
    spec = _load_feature_spec_for_test(train_run_root, df, input_features)
    x = build_feature_matrix(df, spec)
    x, nonfinite_stats = sanitize_feature_matrix(x)
    had_nonfinite = any(v > 0 for v in nonfinite_stats.values())

    eval_rows: list[dict[str, float | int | str]] = []
    for model_path in model_paths:
        estimator = load_object(model_path)
        y_pred = np.clip(estimator.predict(x), 0.0, 1.0)
        metrics = evaluate_regressor(y, y_pred)
        model_name = model_path.stem
        eval_rows.append(
            {
                "model_file": model_path.name,
                "model_index": int(_parse_model_index(model_path)) if _parse_model_index(model_path) < 10**9 else -1,
                **metrics,
            }
        )
        pd.DataFrame({"y_test": y, "y_pred": y_pred}).to_csv(
            results_dir / f"eval_data_test_{model_name}.csv",
            index=False,
        )
        save_object(estimator, models_dir / model_name)

    eval_df = pd.DataFrame(eval_rows)
    summary = {
        "model_file": "mean",
        "model_index": -1,
        "r2": float(eval_df["r2"].mean()),
    }
    summary_std = {
        "model_file": "std",
        "model_index": -1,
        "r2": float(eval_df["r2"].std(ddof=1)),
    }
    eval_df = pd.concat([eval_df, pd.DataFrame([summary, summary_std])], ignore_index=True)
    eval_df.to_csv(results_dir / "evaluation_results.csv", index=False)

    metadata_test = {
        "mode": "test",
        "loaded_models": [str(p) for p in model_paths],
        "source_model_root": str(train_run_root),
        "input_root": root_info["input_root"],
        "gt_root": root_info["gt_root"],
        "model_group": root_info["model_group"],
        "input_uncertainty": root_info["input_uncertainty"],
        "input_target": root_info.get("input_target", []),
        "label_column": label_col,
        "num_rows": int(len(df)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "feature_dimension": int(x.shape[1]),
        "input_features": spec.grad_columns,
        "dim_by_feature": spec.dim_by_column,
        "nonfinite_replaced": had_nonfinite,
        "nonfinite_stats": nonfinite_stats,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_test, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {out_dir}")
    print(eval_df)
    return out_dir
