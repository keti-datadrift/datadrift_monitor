from __future__ import annotations

import json
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGE_KEYS = ["image_id", "image_path", "raw_pred_idx"]
_NPZ_FEATURE_CACHE: dict[Path, Any] = {}

EVAL_CONTEXT_COLUMNS = [
    "image_id",
    "image_path",
    "pred_idx",
    "raw_pred_idx",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
    "pred_class",
    "tp",
    "max_iou",
    "gt_iou",
    "error_type",
    "best_same_class_iou",
    "best_any_class_iou",
    "best_same_class_gt_idx",
    "best_any_class_gt_idx",
    "best_same_class_gt_class",
    "best_any_class_gt_class",
    "matched_gt_idx",
    "is_duplicate",
    "is_localization_error",
    "is_classification_error",
]


@dataclass
class FeatureSpec:
    grad_columns: list[str]
    dim_by_column: dict[str, int]


def resolve_path_value(raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def normalize_input_roots(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return [value] if value else []
    if isinstance(raw_value, (list, tuple)):
        out: list[str] = []
        for item in raw_value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def parse_root_info(root_path: Path) -> tuple[str, str, str]:
    def _parse_tail(model_group: str, run_name: str) -> tuple[str, str, str]:
        match = re.match(r"^\d{2}-\d{2}-\d{4}_\d{2};\d{2}_(.+)$", run_name)
        tail = match.group(1) if match else run_name
        for cue_name in (
            "layer_grad",
            "class_probability",
            "mc_dropout",
            "meta_detect",
            "null_detect",
            "entropy",
            "energy",
            "ensemble",
            "score",
            "gt",
            "tp",
        ):
            if tail == cue_name:
                return model_group, cue_name, ""
            prefix = f"{cue_name}_"
            if tail.startswith(prefix):
                return model_group, cue_name, tail[len(prefix):]
        return model_group, tail, ""

    parent = root_path.parent
    if parent.parent.parent.parent.name == "runs":
        return _parse_tail(f"{parent.parent.parent.name}/{parent.name}", root_path.name)
    if parent.parent.parent.parent.parent.name == "runs":
        return _parse_tail(f"{parent.parent.parent.parent.name}/{parent.parent.name}", root_path.name)
    if parent.name == "runs":
        return _parse_tail("bbox_predictions", root_path.name)
    if parent.parent.name == "runs":
        return _parse_tail(parent.name, root_path.name)
    if parent.parent.parent.name == "runs":
        return _parse_tail(parent.name, root_path.name)
    if parent.parent.parent.parent.name == "runs":
        return _parse_tail(parent.parent.name, root_path.name)
    raise ValueError("dataset root must follow object_detectors/runs/{mode?}/{dataset?}/{time}_{cue}_{target?} ")


def _header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _read_csv_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(dict.fromkeys(columns)))


def _validate_unique_index(df: pd.DataFrame, keys: list[str], source_name: str) -> pd.MultiIndex:
    index = pd.MultiIndex.from_frame(df[keys])
    if index.is_unique:
        return index
    duplicate_mask = index.duplicated(keep=False)
    examples = df.loc[duplicate_mask, keys].head(5).to_dict("records")
    raise ValueError(f"{source_name} contains duplicate merge keys {keys}. Examples: {examples}")


def _resolve_npz_feature_references(df: pd.DataFrame, feature_columns: list[str], root: Path) -> None:
    root = Path(root)
    for col in feature_columns:
        def _resolve(value: Any) -> Any:
            if not isinstance(value, str) or "::" not in value:
                return value
            npz_part, array_key = value.split("::", 1)
            npz_path = Path(npz_part)
            if not npz_path.is_absolute():
                npz_path = (root / npz_path).resolve()
            return f"{npz_path}::{array_key}"

        df[col] = df[col].map(_resolve)


def flatten_numeric(obj: Any) -> list[float]:
    if obj is None:
        return []
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [float(obj)]
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return []
        if "::" in text:
            npz_path_raw, array_key = text.split("::", 1)
            npz_path = Path(npz_path_raw)
            if npz_path.is_file():
                npz_path = npz_path.resolve()
                data = _NPZ_FEATURE_CACHE.get(npz_path)
                if data is None:
                    data = np.load(npz_path, allow_pickle=False)
                    _NPZ_FEATURE_CACHE[npz_path] = data
                if array_key in data.files:
                    return data[array_key].astype(np.float32, copy=False).reshape(-1).astype(float).tolist()
            return []
        try:
            return flatten_numeric(json.loads(text))
        except Exception:
            return []
    if isinstance(obj, dict):
        out: list[float] = []
        for key in sorted(obj.keys()):
            out.extend(flatten_numeric(obj[key]))
        return out
    if isinstance(obj, (list, tuple)):
        out: list[float] = []
        for item in obj:
            out.extend(flatten_numeric(item))
        return out
    return []


def infer_feature_spec(df: pd.DataFrame, grad_columns: list[str]) -> FeatureSpec:
    dim_by_column: dict[str, int] = {}
    for col in grad_columns:
        inferred_dim = 0
        for value in df[col].values:
            vec = flatten_numeric(value)
            if vec:
                inferred_dim = len(vec)
                break
        dim_by_column[col] = inferred_dim
    return FeatureSpec(grad_columns=grad_columns, dim_by_column=dim_by_column)


def build_feature_matrix(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    active_columns = [col for col in spec.grad_columns if int(spec.dim_by_column.get(col, 0)) > 0]
    total_dim = int(sum(int(spec.dim_by_column[col]) for col in active_columns))
    x = np.zeros((len(df), total_dim), dtype=np.float32)
    offset = 0
    for col in active_columns:
        dim = int(spec.dim_by_column[col])
        series = df[col]
        if dim == 1 and pd.api.types.is_numeric_dtype(series):
            x[:, offset] = series.to_numpy(dtype=np.float32, copy=False)
        else:
            for row_idx, value in enumerate(series.values):
                vec = flatten_numeric(value)
                if not vec:
                    continue
                limit = min(len(vec), dim)
                x[row_idx, offset:offset + limit] = np.asarray(vec[:limit], dtype=np.float32)
        offset += dim
    return x


def sanitize_feature_matrix(x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    stats = {
        "nan_count": int(np.isnan(x).sum()),
        "posinf_count": int(np.isposinf(x).sum()),
        "neginf_count": int(np.isneginf(x).sum()),
    }
    if stats["nan_count"] == 0 and stats["posinf_count"] == 0 and stats["neginf_count"] == 0:
        return x, stats
    x_clean = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)
    return x_clean, stats


def save_object(obj: Any, path_without_suffix: Path) -> Path:
    if joblib is not None:
        out = path_without_suffix.with_suffix(".joblib")
        joblib.dump(obj, out)
        return out
    out = path_without_suffix.with_suffix(".pkl")
    with open(out, "wb") as f:
        pickle.dump(obj, f)
    return out


def load_object(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".joblib":
        if joblib is None:
            raise ImportError("joblib is required to load '.joblib' model files.")
        return joblib.load(path)
    if suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported model file suffix: {path.suffix}")


def make_eval_dataframe(df_subset: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    context_cols = [col for col in EVAL_CONTEXT_COLUMNS if col in df_subset.columns]
    out = df_subset[context_cols].reset_index(drop=True).copy()
    out.insert(0, "row_index", df_subset.index.to_numpy())
    out["y_test"] = y_true
    out["y_pred"] = y_pred
    return out


def _label_column(header: list[str], task: str) -> str:
    if task == "classifier":
        if "tp" not in header:
            raise ValueError("tp.csv missing required classifier label column: tp.")
        return "tp"
    if task == "regressor":
        if "gt_iou" in header:
            return "gt_iou"
        if "max_iou" in header:
            return "max_iou"
        raise ValueError("tp.csv missing required regressor label column: gt_iou or max_iou.")
    raise ValueError(f"Unsupported meta task: {task}")


def load_training_dataframe(dataset_cfg: dict[str, Any], task: str = "classifier") -> tuple[pd.DataFrame, str, list[str], dict[str, Any]]:
    input_root_raw_list = normalize_input_roots(dataset_cfg.get("input_root", ""))
    gt_root_raw = str(dataset_cfg.get("gt_root", "")).strip()
    if not input_root_raw_list or not gt_root_raw:
        raise ValueError("dataset.input_root (str or list[str]) and dataset.gt_root are required.")

    input_roots = [resolve_path_value(v) for v in input_root_raw_list]
    gt_root = resolve_path_value(gt_root_raw)
    input_infos = [parse_root_info(p) for p in input_roots]
    input_groups = {group for group, _cue, _target in input_infos}
    if len(input_groups) != 1:
        msg = f"All dataset.input_root entries must share one model group, got: {sorted(input_groups)}"
        warnings.warn(msg)
        raise ValueError(msg)
    input_group = next(iter(input_groups))

    gt_group, _gt_cue, _gt_target = parse_root_info(gt_root)
    if input_group != gt_group:
        msg = (
            "dataset.input_root and dataset.gt_root must have the same model group "
            f"(got '{input_group}' vs '{gt_group}')."
        )
        warnings.warn(msg)
        raise ValueError(msg)

    cue_to_csv = {
        "layer_grad": "layer_grad.csv",
        "score": "score.csv",
        "mc_dropout": "mc_dropout.csv",
        "meta_detect": "meta_detect.csv",
        "null_detect": "null_detect.csv",
        "class_probability": "class_probability.csv",
        "entropy": "entropy.csv",
        "energy": "energy.csv",
        "ensemble": "ensemble.csv",
    }
    gt_csv = gt_root / "tp.csv"
    if not gt_csv.is_file():
        raise FileNotFoundError(f"tp.csv not found: {gt_csv}")
    gt_header = _header(gt_csv)
    label_col = _label_column(gt_header, task)
    required_gt = set(MERGE_KEYS + [label_col])
    if not required_gt.issubset(gt_header):
        raise ValueError("tp.csv missing required join keys: image_id, image_path, raw_pred_idx.")
    gt_context = [col for col in EVAL_CONTEXT_COLUMNS if col in gt_header]
    gt_candidates = list(dict.fromkeys(MERGE_KEYS + [label_col, "pred_idx", "xmin", "ymin", "xmax", "ymax"] + gt_context))
    gt_usecols = [col for col in gt_candidates if col in gt_header]
    gt_df = _read_csv_columns(gt_csv, gt_usecols)
    gt_index = _validate_unique_index(gt_df, MERGE_KEYS, str(gt_csv))
    gt_df.index = gt_index

    feature_frames: list[pd.DataFrame] = []
    prefixed_feature_columns: list[str] = []
    input_uncertainties: list[str] = []
    input_targets: list[str] = []
    prefix_seen_count: dict[str, int] = {}

    for input_root, (_group, input_cue, input_target) in zip(input_roots, input_infos):
        input_csv_name = cue_to_csv.get(input_cue)
        if input_csv_name is None:
            raise ValueError(
                f"Unsupported input uncertainty '{input_cue}'. "
                "Supported uncertainties: layer_grad, class_probability, score, mc_dropout, meta_detect, null_detect, entropy, energy, ensemble."
            )
        input_csv = input_root / input_csv_name
        if not input_csv.is_file():
            raise FileNotFoundError(f"{input_csv_name} not found: {input_csv}")

        feature_header = _header(input_csv)
        if not set(MERGE_KEYS).issubset(feature_header):
            raise ValueError(
                f"Cannot match {input_csv_name} to tp.csv. "
                "Both files must contain raw_pred_idx join keys: image_id, image_path, raw_pred_idx."
            )
        meta_columns = {
            "image_id",
            "image_path",
            "pred_idx",
            "raw_pred_idx",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "score",
            "pred_class",
            "max_iou",
            "gt_iou",
            "tp",
        }
        if input_cue == "score":
            meta_columns.discard("score")

        feature_columns = [c for c in feature_header if c not in meta_columns]
        if input_cue in {"mc_dropout", "ensemble"}:
            feature_columns = [c for c in feature_columns if str(c).endswith("_std")]
        if not feature_columns:
            raise ValueError(f"No input feature columns found in {input_csv}")

        feature_df = _read_csv_columns(input_csv, MERGE_KEYS + feature_columns)
        feature_index = _validate_unique_index(feature_df, MERGE_KEYS, str(input_csv))
        if input_cue == "layer_grad":
            _resolve_npz_feature_references(feature_df, feature_columns, input_root)
        suffix_target = f"_{input_target}" if input_target else ""
        prefix_base = f"{input_cue}{suffix_target}__"
        seen_count = int(prefix_seen_count.get(prefix_base, 0))
        prefix_seen_count[prefix_base] = seen_count + 1
        prefix = prefix_base if seen_count == 0 else f"{input_cue}{suffix_target}__src{seen_count}__"
        rename_map = {c: f"{prefix}{c}" for c in feature_columns}
        frame = feature_df[feature_columns].rename(columns=rename_map)
        frame.index = feature_index
        feature_frames.append(frame)
        prefixed_feature_columns.extend(rename_map.values())
        input_uncertainties.append(input_cue)
        input_targets.append(input_target)

    merged = pd.concat([gt_df, *feature_frames], axis=1, join="inner", sort=False)
    if merged.empty:
        raise ValueError("Merged training dataframe is empty. Check input_root and gt_root pair.")
    merged = merged.reset_index(drop=True)
    if label_col not in merged.columns:
        raise ValueError(f"Ground-truth label column '{label_col}' is missing after merge.")

    input_features = [c for c in prefixed_feature_columns if c in merged.columns]
    if not input_features:
        raise ValueError("No input feature columns found after merge.")

    root_info = {
        "input_root": [str(p) for p in input_roots],
        "gt_root": str(gt_root),
        "model_group": input_group,
        "input_uncertainty": input_uncertainties,
        "input_target": input_targets,
    }
    return merged, label_col, input_features, root_info
