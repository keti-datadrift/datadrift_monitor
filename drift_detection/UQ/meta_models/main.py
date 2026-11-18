from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meta_models.commands.common import normalize_input_roots, parse_root_info, resolve_path_value
from meta_models.commands.registry import resolve_runner
from meta_models.commands.utils.run import create_run_dir, save_used_config


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def _safe_name(text: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower())
    return value.strip("_") or "run"


def _resolve_compare_run_dir(config: dict[str, Any], config_path: Path, run_dir_arg: str) -> Path:
    compare_cfg = config.get("compare", {})
    run_roots = compare_cfg.get("run_roots", [])
    if not isinstance(run_roots, (list, tuple)) or len(run_roots) != 2:
        raise ValueError("compare.run_roots must be a list of exactly two paths.")
    run_a = resolve_path_value(str(run_roots[0]))
    run_b = resolve_path_value(str(run_roots[1]))
    name_a = str(compare_cfg.get("name_a", "")).strip() or run_a.name
    name_b = str(compare_cfg.get("name_b", "")).strip() or run_b.name
    if run_dir_arg:
        run_dir = resolve_path_value(run_dir_arg)
    else:
        run_dir = create_run_dir(
            task="meta_classifier",
            model_group="comparisons",
            cue="meta_classifier_compare",
            target_value=f"{_safe_name(name_a)}_vs_{_safe_name(name_b)}",
            mode_subdir="compare",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_used_config(config_path, run_dir)
    return run_dir.resolve()


def _resolve_train_test_run_dir(
    task: str,
    mode: str,
    config: dict[str, Any],
    config_path: Path,
    run_dir_arg: str,
) -> Path:
    dataset_cfg = config.get("dataset", {})
    input_root_raw_list = normalize_input_roots(dataset_cfg.get("input_root", ""))
    gt_root_raw = str(dataset_cfg.get("gt_root", "")).strip()
    if not input_root_raw_list or not gt_root_raw:
        raise ValueError("dataset.input_root (str or list[str]) and dataset.gt_root are required.")

    input_roots = [resolve_path_value(v) for v in input_root_raw_list]
    gt_root = resolve_path_value(gt_root_raw)
    parsed_inputs = [parse_root_info(p) for p in input_roots]
    input_groups = {group for group, _cue, _target in parsed_inputs}
    if len(input_groups) != 1:
        msg = f"All dataset.input_root entries must share one model group, got: {sorted(input_groups)}"
        warnings.warn(msg)
        raise ValueError(msg)
    input_group = next(iter(input_groups))
    cue_parts = [cue for _group, cue, _target in parsed_inputs]
    target_parts = [target for _group, _cue, target in parsed_inputs if target]
    input_cue = "+".join(cue_parts)
    input_target = "+".join(target_parts)

    gt_group, _gt_cue, _gt_target = parse_root_info(gt_root)
    if input_group != gt_group:
        msg = (
            "dataset.input_root and dataset.gt_root must have the same model group "
            f"(got '{input_group}' vs '{gt_group}')."
        )
        warnings.warn(msg)
        raise ValueError(msg)

    if run_dir_arg:
        run_dir = resolve_path_value(run_dir_arg)
    else:
        if mode == "train":
            cue_for_run = input_cue
        elif task == "meta_regressor":
            cue_for_run = f"{input_cue}_meta_regressor_test"
        else:
            cue_for_run = f"{input_cue}_test"
        run_dir = create_run_dir(
            task=task,
            model_group=input_group,
            cue=cue_for_run,
            target_value=input_target,
            mode_subdir=mode,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_used_config(config_path, run_dir)
    return run_dir.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="meta_models/configs/meta_classifier/train.yaml")
    parser.add_argument("--run-dir", type=str, default="")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(config_path)

    task = str(config.get("task", "")).strip().lower()
    if not task:
        raise ValueError("Config must define task: meta_classifier | meta_regressor.")
    mode = str(config.get("mode", "train")).strip().lower()
    runner = resolve_runner(task, mode)

    if mode == "compare":
        run_dir = _resolve_compare_run_dir(config, config_path, args.run_dir)
    else:
        run_dir = _resolve_train_test_run_dir(task, mode, config, config_path, args.run_dir)
    runner(config, run_dir)


if __name__ == "__main__":
    main()
