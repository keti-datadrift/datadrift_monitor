from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
META_MODELS_ROOT = REPO_ROOT / "meta_models"

for import_path in (REPO_ROOT,):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

from sklearn.model_selection import StratifiedKFold, train_test_split

from meta_models.commands.common import (  # noqa: E402
    build_feature_matrix,
    infer_feature_spec,
    load_training_dataframe,
    sanitize_feature_matrix,
)
from meta_models.commands.meta_classifier.train import apply_augmentation  # noqa: E402
from meta_models.losses.meta_classifier import compute_ace, compute_ece, compute_fpr_at_tpr, evaluate_classifier  # noqa: E402
from meta_models.models.meta_classifier import build_estimator  # noqa: E402

# Edit these paths before running.
INPUT_ROOTS = [
    r"object_detectors/runs/faster_rcnn/predict/coco/05-29-2026_14;52_faster_rcnn_layer_grad_grid/05-30-2026_01;35_layer_grad_t-null__term-rpnb__rpnb-offset_l2-pred",
    r"object_detectors/runs/faster_rcnn/predict/coco/05-29-2026_14;52_faster_rcnn_layer_grad_grid/05-30-2026_03;33_layer_grad_t-null__term-rpno__rpno-abs-pred",
    r"object_detectors/runs/faster_rcnn/predict/coco/05-29-2026_14;52_faster_rcnn_layer_grad_grid/05-30-2026_06;27_layer_grad_t-null__term-roib__roib-box_l1-pred",
    r"object_detectors/runs/faster_rcnn/predict/coco/05-29-2026_14;52_faster_rcnn_layer_grad_grid/05-30-2026_08;17_layer_grad_t-null__term-roic__roic-bce-pred",
]
GT_ROOT = r"object_detectors/runs/faster_rcnn/predict/coco/05-27-2026_22;00_gt"
BASE_META_CLASSIFIER_CONFIG = (
    r"meta_models/configs/meta_classifier/train.yaml"
)

# Set to a fixed name to resume/read beside a previous run. Empty string creates a new timestamped root.
GRID_NAME = ""

MODEL_TYPE = "gb_classifier"
DEVICE = "cpu"
RANDOM_SEED = 42
AUGMENTATION = "none"
N_JOBS = 8

# repeat is usually closer to the current meta_classifier train config.
PROCESS = "repeat"  # repeat | kfold
REPEAT_SPLIT = 0.3
REPEATS = 15
NUM_FOLD = 5

# Use None for all combinations, or set a small int for a smoke test.
MAX_COMBINATIONS = None

# XGBClassifier search space. Keep this compact first; widening this is expensive.
PARAM_GRID = {
    "clf__n_estimators": [25, 50, 100],
    "clf__max_depth": [2, 3, 4],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__reg_alpha": [0.0, 0.5],
    "clf__reg_lambda": [1.0, 5.0],
}


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_path_part(value: str) -> str:
    return (
        "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
        .strip("_")
        .lower()
        or "unknown"
    )


def _parse_object_detector_root(root_path: Path) -> tuple[str, str]:
    parts = list(root_path.parts)
    try:
        runs_idx = parts.index("runs")
    except ValueError:
        return "unknown", "unknown"
    tail = parts[runs_idx + 1 :]
    if len(tail) >= 4:
        # object_detectors/runs/<model>/<mode>/<dataset>/<time>...
        return _safe_path_part(tail[0]), _safe_path_part(tail[2])
    if len(tail) >= 3:
        # object_detectors/runs/<mode>/<dataset>/<time>...
        return "unknown", _safe_path_part(tail[1])
    if len(tail) >= 2:
        return "unknown", _safe_path_part(tail[0])
    return "unknown", "unknown"


def _make_dataset_config(base_config: dict) -> dict:
    config = deepcopy(base_config)
    config["mode"] = "train"
    model_cfg = config.setdefault("model", {})
    model_cfg["type"] = MODEL_TYPE
    model_cfg["device"] = DEVICE
    model_cfg["random_seed"] = RANDOM_SEED
    model_cfg["search"] = False

    exp_cfg = config.setdefault("experiment", {})
    exp_cfg["process"] = PROCESS
    exp_cfg["augmentation"] = AUGMENTATION
    exp_cfg["n_jobs"] = N_JOBS
    exp_cfg.setdefault("repeat", {})
    exp_cfg["repeat"]["split"] = REPEAT_SPLIT
    exp_cfg["repeat"]["repeats"] = REPEATS
    exp_cfg.setdefault("kfold", {})
    exp_cfg["kfold"]["num_fold"] = NUM_FOLD

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["input_root"] = [str(_resolve_path(path)) for path in INPUT_ROOTS]
    dataset_cfg["gt_root"] = str(_resolve_path(GT_ROOT))
    return config


def _iter_param_combinations():
    keys = list(PARAM_GRID.keys())
    for values in product(*(PARAM_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def _evaluate_params(x: np.ndarray, y: np.ndarray, params: dict, config: dict) -> dict:
    model_cfg = config["model"]
    exp_cfg = config["experiment"]
    random_seed = int(model_cfg.get("random_seed", RANDOM_SEED))
    augmentation = str(exp_cfg.get("augmentation", AUGMENTATION))

    eval_rows = []
    process = str(exp_cfg.get("process", PROCESS)).strip().lower()
    if process == "kfold":
        num_fold = int(exp_cfg.get("kfold", {}).get("num_fold", NUM_FOLD))
        splitter = StratifiedKFold(
            n_splits=num_fold, shuffle=True, random_state=random_seed
        )
        split_iter = enumerate(splitter.split(x, y))
    elif process == "repeat":
        split = float(exp_cfg.get("repeat", {}).get("split", REPEAT_SPLIT))
        repeats = int(exp_cfg.get("repeat", {}).get("repeats", REPEATS))
        indices = np.arange(len(y))

        def _repeat_iter():
            for i in range(repeats):
                yield i, train_test_split(
                    indices,
                    test_size=split,
                    random_state=random_seed + i,
                    stratify=y,
                    shuffle=True,
                )

        split_iter = _repeat_iter()
    else:
        raise ValueError("PROCESS must be repeat or kfold.")

    for split_idx, split_data in split_iter:
        if process == "kfold":
            train_idx, test_idx = split_data
        else:
            train_idx, test_idx = split_data
        estimator = build_estimator(
            str(model_cfg.get("type", MODEL_TYPE)),
            device=str(model_cfg.get("device", DEVICE)),
            random_seed=random_seed + int(split_idx),
        )
        estimator.set_params(**params)
        try:
            estimator.set_params(clf__n_jobs=int(exp_cfg.get("n_jobs", N_JOBS)))
        except ValueError:
            pass

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        x_train, y_train = apply_augmentation(
            x_train,
            y_train,
            augmentation,
            random_seed=random_seed + int(split_idx),
        )
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict_proba(x_test)[:, 1]
        auroc, ap = evaluate_classifier(y_test, y_pred)
        eval_rows.append(
            {
                "auroc": float(auroc),
                "ap": float(ap),
                "fpr95": float(compute_fpr_at_tpr(y_test, y_pred)),
                "ece": float(compute_ece(y_test, y_pred)),
                "ace": float(compute_ace(y_test, y_pred)),
            }
        )

    out = {}
    for key in ("auroc", "ap", "fpr95", "ece", "ace"):
        values = np.asarray([row[key] for row in eval_rows], dtype=np.float64)
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def main() -> None:
    base_config_path = _resolve_path(BASE_META_CLASSIFIER_CONFIG)
    base_config = _load_yaml(base_config_path)
    config = _make_dataset_config(base_config)
    dataset_cfg = config["dataset"]

    input_root_paths = [_resolve_path(path) for path in INPUT_ROOTS]
    model_name, dataset_name = _parse_object_detector_root(input_root_paths[0])
    grid_name = (
        GRID_NAME.strip()
        or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_meta_classifier_grid"
    )
    run_dir = (
        META_MODELS_ROOT
        / "runs"
        / "meta_classifier"
        / model_name
        / "grid_search"
        / dataset_name
        / grid_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    df, label_col, feature_columns, root_info = load_training_dataframe(dataset_cfg)
    y = df[label_col].astype(int).to_numpy()
    spec = infer_feature_spec(df, feature_columns)
    x = build_feature_matrix(df, spec)
    x, nonfinite_stats = sanitize_feature_matrix(x)

    param_combos = list(_iter_param_combinations())
    if MAX_COMBINATIONS is not None:
        param_combos = param_combos[: int(MAX_COMBINATIONS)]
    print(f"Meta-classifier parameter combinations: {len(param_combos)}", flush=True)
    print(
        f"Rows={len(df)}, features={x.shape[1]}, positives={int(np.sum(y))}", flush=True
    )

    rows = []
    result_path = run_dir / "grid_results.csv"
    for idx, params in enumerate(param_combos, start=1):
        print(f"[{idx}/{len(param_combos)}] {params}", flush=True)
        metrics = _evaluate_params(x, y, params, config)
        row = {
            **{key.replace("clf__", ""): value for key, value in params.items()},
            **metrics,
        }
        rows.append(row)
        rows.sort(key=lambda r: (r["mean_auroc"], r["mean_ap"]), reverse=True)
        _write_csv(result_path, rows, list(rows[0].keys()))

    best = rows[0] if rows else {}
    metadata = {
        "input_root": [str(path) for path in input_root_paths],
        "gt_root": str(_resolve_path(GT_ROOT)),
        "model": MODEL_TYPE,
        "device": DEVICE,
        "process": PROCESS,
        "repeat_split": REPEAT_SPLIT,
        "repeats": REPEATS,
        "num_fold": NUM_FOLD,
        "augmentation": AUGMENTATION,
        "random_seed": RANDOM_SEED,
        "num_rows": int(len(df)),
        "num_positive_tp": int(np.sum(y)),
        "feature_dimension": int(x.shape[1]),
        "input_features": feature_columns,
        "dim_by_feature": spec.dim_by_column,
        "nonfinite_stats": nonfinite_stats,
        "param_grid": PARAM_GRID,
        "best": best,
        "root_info": root_info,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    best_params = {
        key: best.get(key.replace("clf__", ""))
        for key in PARAM_GRID
        if key.replace("clf__", "") in best
    }
    with open(run_dir / "best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False, allow_unicode=True)

    print(f"Saved grid results: {result_path}")
    print(f"Saved metadata: {run_dir / 'metadata.json'}")
    print(f"Best: {best}")


if __name__ == "__main__":
    main()
