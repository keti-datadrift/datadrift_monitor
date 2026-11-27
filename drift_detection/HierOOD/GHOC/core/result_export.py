from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import torch


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    rows = {}
    for key, value in config.items():
        joined = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.update(_flatten_config(value, joined))
        else:
            rows[joined] = value
    return rows


def _to_csv_value(value: Any):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def export_result_to_csv(result_path: str | Path, output_dir: str | Path) -> list[Path]:
    result = torch.load(result_path, map_location="cpu", weights_only=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    summary_rows = []
    class_rows = []
    for split, metrics in result["results"].items():
        cgm_cfg = metrics.get("cgm", {}) or {}
        cgm_local_mode = cgm_cfg.get("local_mode", "density_softmax")
        if cgm_cfg.get("ood_density") == "positive_density_residual":
            cgm_ood_prior = "induced_residual_mass"
        elif cgm_local_mode == "binary_model_selection":
            cgm_ood_prior = "not_used_binary_symmetric"
        else:
            cgm_ood_prior = cgm_cfg.get("ood_prior")
        row = {
            "experiment_name": result.get("experiment_name"),
            "dataset": result.get("dataset"),
            "split": split,
            "acc": _to_csv_value(metrics.get("acc")),
            "balanced_acc": _to_csv_value(metrics.get("balanced_acc")),
            "avg_hdist": _to_csv_value(metrics.get("avg_hdist")),
            "balanced_hdist": _to_csv_value(metrics.get("balanced_hdist")),
            "prediction_mode": _to_csv_value(metrics.get("prediction_mode")),
            "score_type": _to_csv_value(metrics.get("score_type")),
            "temperature": _to_csv_value(metrics.get("temperature")),
            "kappa": _to_csv_value(metrics.get("kappa")),
            "ood_scale": _to_csv_value(metrics.get("ood_scale")),
            "cgm_enabled": _to_csv_value(cgm_cfg.get("enabled", False)),
            "cgm_ood_density": _to_csv_value(cgm_cfg.get("ood_density", "parent_mask")),
            "cgm_complement_reduce": _to_csv_value(cgm_cfg.get("complement_reduce")),
            "cgm_complement_weight": _to_csv_value(cgm_cfg.get("complement_weight")),
            "cgm_mask_type": _to_csv_value(cgm_cfg.get("mask_type")),
            "cgm_membership_type": _to_csv_value(cgm_cfg.get("membership_type")),
            "cgm_membership_probability": _to_csv_value(cgm_cfg.get("membership_probability")),
            "cgm_membership_correction": _to_csv_value(cgm_cfg.get("membership_correction")),
            "cgm_ood_base_cov_scale": _to_csv_value(cgm_cfg.get("ood_base_cov_scale", 1.0)),
            "cgm_mask_cov_scale": _to_csv_value(cgm_cfg.get("mask_cov_scale", 1.0)),
            "cgm_between_cov_scale": _to_csv_value(cgm_cfg.get("between_cov_scale")),
            "cgm_between_cov_estimator": _to_csv_value(cgm_cfg.get("between_cov_estimator")),
            "cgm_between_cov_shrinkage_strength": _to_csv_value(
                cgm_cfg.get("between_cov_shrinkage_strength")
            ),
            "cgm_product_mask_samples": _to_csv_value(cgm_cfg.get("product_mask_samples")),
            "cgm_random_effects_weight": _to_csv_value(cgm_cfg.get("random_effects_weight")),
            "cgm_parent_covariance_scales": _to_csv_value(cgm_cfg.get("parent_covariance_scales")),
            "cgm_parent_scale_weights": _to_csv_value(cgm_cfg.get("parent_scale_weights")),
            "cgm_local_mode": _to_csv_value(cgm_local_mode),
            "cgm_lambda": _to_csv_value(cgm_cfg.get("lambda")),
            "cgm_child_weight": _to_csv_value(cgm_cfg.get("child_weight")),
            "cgm_candidate_prior": _to_csv_value(cgm_cfg.get("candidate_prior", "uniform")),
            "cgm_ood_prior": _to_csv_value(cgm_ood_prior),
            "cgm_child_log_scale": _to_csv_value(cgm_cfg.get("child_log_scale")),
            "cgm_ood_log_scale": _to_csv_value(cgm_cfg.get("ood_log_scale")),
            "cgm_gate_log_scale": _to_csv_value(cgm_cfg.get("gate_log_scale")),
            "cgm_gate_bias": _to_csv_value(cgm_cfg.get("gate_bias")),
            "cgm_blend_weight": _to_csv_value(cgm_cfg.get("blend_weight")),
            "collapsed_ood": _to_csv_value(metrics.get("collapsed_ood")),
        }
        summary_rows.append(row)
        for class_name, class_hdist in metrics.get("class_hdists", {}).items():
            class_rows.append(
                {
                    "experiment_name": result.get("experiment_name"),
                    "dataset": result.get("dataset"),
                    "split": split,
                    "class_name": class_name,
                    "class_hdist": _to_csv_value(class_hdist),
                }
            )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    created.append(summary_path)

    class_path = output_dir / "class_hdists.csv"
    with class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0].keys()) if class_rows else [])
        if class_rows:
            writer.writeheader()
            writer.writerows(class_rows)
    created.append(class_path)

    config_path = output_dir / "config_flat.csv"
    flat_config = _flatten_config(result.get("config", {}))
    with config_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value"])
        writer.writeheader()
        for key, value in flat_config.items():
            writer.writerow({"key": key, "value": value})
    created.append(config_path)

    return created
