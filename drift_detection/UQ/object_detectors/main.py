import argparse
from copy import deepcopy
import os
from datetime import datetime
from pathlib import Path

import yaml


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataloaders.core.factory import load_config
from commands.run_predict import run_predict
from commands.run_train import run_train
from commands.utils.run_utils import create_run_dir, save_run_summary

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


def _resolve_path_value(raw_path):
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _resolve_config_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _resolve_run_dir(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _dataset_run_subdir(config):
    raw = config.get("dataset", {}).get("used_dataset", "unknown")
    if isinstance(raw, str):
        names = [raw.strip().lower()]
    elif isinstance(raw, (list, tuple)):
        names = [str(v).strip().lower() for v in raw if str(v).strip()]
    else:
        names = []
    if not names:
        return "unknown"
    safe_names = ["".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name) for name in names]
    return "-".join(safe_names)


def _model_run_subdir(config):
    raw = str(config.get("model", {}).get("type", "yolov5")).strip().lower()
    aliases = {
        "yolo": "yolov5",
        "yolo_v5": "yolov5",
        "faster-rcnn": "faster_rcnn",
        "frcnn": "faster_rcnn",
    }
    name = aliases.get(raw, raw or "unknown_model")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def _normalize_training_seeds(config):
    raw_seed = config.get("training", {}).get("seed")
    if raw_seed is None:
        return [None]
    if isinstance(raw_seed, (list, tuple)):
        if not raw_seed:
            return [None]
        return [int(seed) for seed in raw_seed]
    return [int(raw_seed)]


def _seed_tag(seed):
    return "seed_none" if seed is None else f"seed{int(seed)}"


def _save_effective_config(config, run_dir):
    with open(Path(run_dir) / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def _normalize_uncertainties(config):
    raw = config.get("output", {}).get("uncertainty", "gt")
    if isinstance(raw, (list, tuple)):
        values = [str(v).strip().lower() for v in raw if str(v).strip()]
    else:
        values = [str(raw).strip().lower()]
    if not values:
        raise ValueError("output.uncertainty is empty.")
    return values


def _active_uncertainty(config):
    raw = config.get("output", {}).get("uncertainty", "gt")
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError("_run_one_predict_uncertainty expects exactly one output.uncertainty value.")
        raw = raw[0]
    value = str(raw).strip().lower()
    return value or "gt"


def _layer_grad_target_tag(config):
    layer_grad_cfg = config.get("output", {}).get("layer_grad", {})
    grad_cfg = layer_grad_cfg.get("gradient", {})

    raw_target = grad_cfg.get("target", "cand_target")
    grad_target = str(raw_target).strip().lower() if raw_target is not None else "null_target"

    raw_vals = grad_cfg.get("scalar", [])
    scalar_list = [
        str(v).strip().lower()
        for v in (raw_vals if isinstance(raw_vals, list) else [raw_vals])
        if str(v).strip()
    ]
    scalar_tag = "-".join(scalar_list) if scalar_list else "loss"
    return f"{grad_target}_{scalar_tag}"


def _timestamped_child_dir(base_dir, uncertainty, target_tag=""):
    timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
    suffix = str(uncertainty).strip().lower()
    target = str(target_tag or "").strip().lower()
    name = f"{timestamp}_{suffix}" if not target else f"{timestamp}_{suffix}_{target}"
    return (Path(base_dir) / name).resolve()


def _run_one_predict_uncertainty(config, config_path, args_run_dir="", force_child_dir=False):
    uncertainty = _active_uncertainty(config)
    target_tag = _layer_grad_target_tag(config) if uncertainty == "layer_grad" else ""
    run_base_subdir = str(Path(_model_run_subdir(config)) / "predict" / _dataset_run_subdir(config))

    if uncertainty == "deterministic":
        deterministic_uncertainties = ["score", "class_probability", "entropy", "energy"]
        if args_run_dir:
            base_dir = _resolve_run_dir(args_run_dir)
            run_dir = {
                name: _timestamped_child_dir(base_dir, name)
                for name in deterministic_uncertainties
            }
        else:
            run_dir = {
                name: create_run_dir(
                    uncertainty=name,
                    target_value="",
                    base_subdir=run_base_subdir,
                ).resolve()
                for name in deterministic_uncertainties
            }
        for child_dir in run_dir.values():
            child_dir.mkdir(parents=True, exist_ok=True)
            _save_effective_config(config, child_dir)
        run_predict(config, run_dir)
        for name, child_dir in run_dir.items():
            save_run_summary(child_dir, uncertainty=name)
        return

    if args_run_dir:
        base_dir = _resolve_run_dir(args_run_dir)
        run_dir = _timestamped_child_dir(base_dir, uncertainty, target_tag) if force_child_dir else base_dir
    else:
        run_dir = create_run_dir(
            uncertainty=uncertainty,
            target_value=target_tag,
            base_subdir=run_base_subdir,
        ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_effective_config(config, run_dir)
    run_predict(config, run_dir)
    save_run_summary(run_dir, uncertainty=uncertainty)


def _normalize_config_paths(config):
    model_cfg = config.get("model", {})
    weight_path = model_cfg.get("weights")
    if isinstance(weight_path, str) and weight_path:
        model_cfg["weights"] = str(_resolve_path_value(weight_path))
    elif isinstance(weight_path, (list, tuple)):
        model_cfg["weights"] = [
            str(_resolve_path_value(v))
            for v in weight_path
            if isinstance(v, str) and v
        ]
    config_file = model_cfg.get("config_file")
    if isinstance(config_file, str) and config_file:
        model_cfg["config_file"] = str(_resolve_path_value(config_file))

    dataset_cfg = config.get("dataset", {})
    used_dataset = dataset_cfg.get("used_dataset")
    if isinstance(used_dataset, str):
        names = [used_dataset.strip()]
    elif isinstance(used_dataset, (list, tuple)):
        names = [str(v).strip() for v in used_dataset if str(v).strip()]
    else:
        names = []
    for name in names:
        if name in dataset_cfg:
            active_cfg = dataset_cfg[name]
            root = active_cfg.get("root")
            if isinstance(root, str) and root:
                active_cfg["root"] = str(_resolve_path_value(root))

    output_cfg = config.get("output", {})
    ensemble_cfg = output_cfg.get("ensemble")
    if isinstance(ensemble_cfg, dict):
        ensemble_weights = ensemble_cfg.get("weights")
        if isinstance(ensemble_weights, str) and ensemble_weights:
            ensemble_cfg["weights"] = str(_resolve_path_value(ensemble_weights))
        elif isinstance(ensemble_weights, (list, tuple)):
            ensemble_cfg["weights"] = [
                str(_resolve_path_value(v))
                for v in ensemble_weights
                if isinstance(v, str) and v
            ]

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="object_detectors/configs/yolov5/predict/coco.yaml")
    parser.add_argument("--run-dir", type=str, default="")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _normalize_config_paths(load_config(str(config_path)))
    mode = str(config.get("mode", "")).lower()
    if mode == "predict":
        uncertainties = _normalize_uncertainties(config)
        for idx, uncertainty in enumerate(uncertainties, start=1):
            run_config = deepcopy(config)
            run_config.setdefault("output", {})["uncertainty"] = uncertainty
            print(f"[predict] uncertainty={uncertainty} ({idx}/{len(uncertainties)})")
            _run_one_predict_uncertainty(
                config=run_config,
                config_path=config_path,
                args_run_dir=args.run_dir,
                force_child_dir=len(uncertainties) > 1,
            )
        return

    if mode == "train":
        seeds = _normalize_training_seeds(config)
        timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
        for seed in seeds:
            seed_config = deepcopy(config)
            seed_config.setdefault("training", {})["seed"] = seed
            if args.run_dir:
                base_run_dir = _resolve_run_dir(args.run_dir)
                run_dir = base_run_dir if len(seeds) == 1 else (base_run_dir / _seed_tag(seed)).resolve()
            else:
                model_name = _model_run_subdir(seed_config)
                dir_name = f"{timestamp}_{model_name}"
                if len(seeds) > 1:
                    dir_name = f"{dir_name}_{_seed_tag(seed)}"
                run_dir = (PROJECT_ROOT / "runs" / model_name / "train" / _dataset_run_subdir(seed_config) / dir_name).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            _save_effective_config(seed_config, run_dir)
            print(f"[train] run_dir={run_dir}")
            run_train(seed_config, run_dir)
        return

    raise ValueError(f"Unsupported mode: {mode}. Use 'predict' or 'train'.")


if __name__ == "__main__":
    main()
