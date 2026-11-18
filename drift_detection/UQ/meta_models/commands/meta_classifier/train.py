from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from tqdm import tqdm

from meta_models.commands.common import (
    build_feature_matrix,
    infer_feature_spec,
    load_training_dataframe,
    make_eval_dataframe,
    sanitize_feature_matrix,
    save_object,
)
from meta_models.commands.utils.plot import save_eval_plots
from meta_models.losses.meta_classifier import (
    CLASSIFIER_METRIC_COLUMNS,
    compute_ace,
    compute_ece,
    compute_fpr_at_tpr,
    evaluate_classifier,
)
from meta_models.models.meta_classifier import build_estimator, param_grid

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    SMOTE = None
    ImbPipeline = None


def apply_augmentation(
    x: np.ndarray,
    y: np.ndarray,
    augmentation: str,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if augmentation == "none":
        return x, y
    if augmentation == "smote":
        if SMOTE is None:
            raise ImportError("imblearn is required for augmentation='smote'.")
        sampler = SMOTE(random_state=int(random_seed))
        return sampler.fit_resample(x, y)
    raise ValueError(f"Unsupported augmentation: {augmentation}")


def _build_search(
    estimator,
    model_name: str,
    model_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    augmentation: str,
    random_seed: int,
) -> tuple[GridSearchCV, bool]:
    grid = param_grid(model_name)
    search_estimator = estimator
    strip_prefix = False
    if augmentation == "smote":
        if SMOTE is None or ImbPipeline is None:
            raise ImportError("imblearn is required for augmentation='smote'.")
        search_estimator = ImbPipeline(
            [
                ("smote", SMOTE(random_state=int(random_seed))),
                ("estimator", estimator),
            ]
        )
        grid = {f"estimator__{key}": value for key, value in grid.items()}
        strip_prefix = True
    elif augmentation != "none":
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    search = GridSearchCV(
        estimator=search_estimator,
        param_grid=grid,
        scoring=str(model_cfg.get("search_scoring", "roc_auc")),
        n_jobs=int(exp_cfg.get("n_jobs", 8)),
        cv=5,
        verbose=1,
    )
    return search, strip_prefix


def _strip_search_params(params: dict[str, Any], strip_prefix: bool) -> dict[str, Any]:
    if not strip_prefix:
        return dict(params)
    prefix = "estimator__"
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in params.items()}


def _append_summary_rows(eval_rows: list[dict[str, Any]], metric_cols: list[str]) -> pd.DataFrame:
    eval_df = pd.DataFrame(eval_rows)
    mean_row: dict[str, Any] = {"row_type": "mean", "split_index": -1}
    std_row: dict[str, Any] = {"row_type": "std", "split_index": -1}
    for col in metric_cols:
        mean_row[col] = float(eval_df[col].mean())
        std_row[col] = float(eval_df[col].std(ddof=1))
    return pd.concat([eval_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    auroc, ap = evaluate_classifier(y_true, y_score)
    return {
        "auroc": float(auroc),
        "ap": float(ap),
        "fpr95": float(compute_fpr_at_tpr(y_true, y_score)),
        "ece": float(compute_ece(y_true, y_score)),
        "ace": float(compute_ace(y_true, y_score)),
    }


def run_train(config: dict[str, Any], run_dir: Path) -> Path:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    exp_cfg = config["experiment"]

    out_dir = run_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    models_dir = out_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df, label_col, grad_columns, root_info = load_training_dataframe(dataset_cfg, task="classifier")
    y = df[label_col].astype(int).to_numpy()
    spec = infer_feature_spec(df, grad_columns)
    x = build_feature_matrix(df, spec)
    x, nonfinite_stats = sanitize_feature_matrix(x)
    had_nonfinite = any(v > 0 for v in nonfinite_stats.values())
    if had_nonfinite:
        warnings.warn(
            "Non-finite values found in input features and replaced with finite values: "
            f"{nonfinite_stats}"
        )

    model_name = str(model_cfg.get("type", "gb_classifier"))
    device = str(model_cfg.get("device", "cpu"))
    random_seed = int(model_cfg.get("random_seed", 42))
    estimator = build_estimator(model_name, device=device, random_seed=random_seed)
    best_params: dict[str, Any] = {}

    do_search = bool(model_cfg.get("search", False))
    augmentation = str(exp_cfg.get("augmentation", "none"))
    if do_search:
        search, strip_prefix = _build_search(
            estimator=estimator,
            model_name=model_name,
            model_cfg=model_cfg,
            exp_cfg=exp_cfg,
            augmentation=augmentation,
            random_seed=random_seed,
        )
        search.fit(x, y)
        best_params = dict(search.best_params_)
        estimator.set_params(**_strip_search_params(best_params, strip_prefix))

    eval_rows: list[dict[str, Any]] = []
    process = str(exp_cfg.get("process", "kfold")).strip().lower()
    used_num_fold = None
    used_split = None
    used_repeats = None

    if process == "kfold":
        kfold_cfg = exp_cfg.get("kfold", {})
        num_fold = int(kfold_cfg.get("num_fold", 10))
        if num_fold < 2:
            raise ValueError("experiment.kfold.num_fold must be >= 2.")
        used_num_fold = num_fold
        kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=random_seed)
        split_iter = tqdm(enumerate(kfold.split(x, y)), desc="Meta Classifier (kfold)", total=num_fold, unit="fold")
        for i, (train_idx, test_idx) in split_iter:
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            x_train, y_train = apply_augmentation(x_train, y_train, augmentation, random_seed=random_seed + i)

            estimator.fit(x_train, y_train)
            y_pred = estimator.predict_proba(x_test)[:, 1]

            metrics = _compute_metrics(y_test, y_pred)
            eval_rows.append(
                {
                    "row_type": "split",
                    "split_index": int(i),
                    **metrics,
                }
            )

            make_eval_dataframe(df.iloc[test_idx], y_test, y_pred).to_csv(results_dir / f"eval_data_{i}.csv", index=False)
            save_eval_plots(y_test, y_pred, out_dir=out_dir, model_name=f"model_{i}", label_col=label_col)
            save_object(estimator, models_dir / f"model_{i}")
    elif process == "repeat":
        repeat_cfg = exp_cfg.get("repeat", {})
        split = float(repeat_cfg.get("split", 0.3))
        repeats = int(repeat_cfg.get("repeats", 15))
        if not (0.0 < split < 1.0):
            raise ValueError("experiment.repeat.split must be in (0,1).")
        if repeats < 1:
            raise ValueError("experiment.repeat.repeats must be >= 1.")
        used_split = split
        used_repeats = repeats
        split_iter = tqdm(range(repeats), desc="Meta Classifier (repeat)", total=repeats, unit="split")
        for i in split_iter:
            indices = np.arange(len(y))
            train_idx, test_idx = train_test_split(
                indices,
                test_size=split,
                random_state=random_seed + i,
                stratify=y,
                shuffle=True,
            )
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            x_train, y_train = apply_augmentation(x_train, y_train, augmentation, random_seed=random_seed + i)

            estimator.fit(x_train, y_train)
            y_pred = estimator.predict_proba(x_test)[:, 1]

            metrics = _compute_metrics(y_test, y_pred)
            eval_rows.append(
                {
                    "row_type": "split",
                    "split_index": int(i),
                    **metrics,
                }
            )

            make_eval_dataframe(df.iloc[test_idx], y_test, y_pred).to_csv(results_dir / f"eval_data_{i}.csv", index=False)
            save_eval_plots(y_test, y_pred, out_dir=out_dir, model_name=f"model_{i}", label_col=label_col)
            save_object(estimator, models_dir / f"model_{i}")
    else:
        raise ValueError("experiment.process must be 'kfold' or 'repeat'.")

    eval_df = _append_summary_rows(eval_rows, CLASSIFIER_METRIC_COLUMNS)
    eval_df.to_csv(results_dir / "evaluation_results.csv", index=False)

    metadata = {
        "input_root": root_info["input_root"],
        "gt_root": root_info["gt_root"],
        "model_group": root_info["model_group"],
        "input_uncertainty": root_info["input_uncertainty"],
        "input_target": root_info.get("input_target", ""),
        "label_column": label_col,
        "model": model_name,
        "device": device,
        "search_scoring": str(model_cfg.get("search_scoring", "roc_auc")),
        "augmentation": augmentation,
        "feature_dimension": int(x.shape[1]),
        "num_rows": int(len(df)),
        "num_positive_tp": int(np.sum(y)),
        "input_features": grad_columns,
        "dim_by_feature": spec.dim_by_column,
        "best_params": best_params,
        "process": process,
        "num_fold": used_num_fold,
        "repeat_split": used_split,
        "repeat_repeats": used_repeats,
        "nonfinite_replaced": had_nonfinite,
        "nonfinite_stats": nonfinite_stats,
        "random_seed": random_seed,
        "shuffle": True,
        "search": do_search,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {out_dir}")
    print(eval_df)
    return out_dir
