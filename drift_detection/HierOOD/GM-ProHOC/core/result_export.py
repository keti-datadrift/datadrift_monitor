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
            "alpha": _to_csv_value(metrics.get("alpha")),
            "beta": _to_csv_value(metrics.get("beta")),
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
