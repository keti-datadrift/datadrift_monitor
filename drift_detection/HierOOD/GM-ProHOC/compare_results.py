from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch


SCALAR_METRICS = ["acc", "balanced_acc", "avg_hdist", "balanced_hdist"]


def _to_python_scalar(value: Any):
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


def _discover_result_files(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(path.rglob("*.result"))


def _extract_dataset_name(result_path: Path, payload: Any) -> str:
    if isinstance(payload, dict) and "dataset" in payload:
        return str(payload["dataset"])
    stem = result_path.stem
    if stem.startswith("hinference-"):
        return stem.replace("hinference-", "", 1)
    return stem


def _normalize_gm_result(result_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    experiment_name = payload.get("experiment_name", result_path.parent.name)
    dataset = _extract_dataset_name(result_path, payload)
    rows = []
    for split, metrics in payload.get("results", {}).items():
        row = {
            "source": "gm_prohoc",
            "result_file": str(result_path),
            "dataset": dataset,
            "method": experiment_name,
            "split": split,
        }
        for metric in SCALAR_METRICS:
            row[metric] = _to_python_scalar(metrics.get(metric))
        rows.append(row)
    return rows


def _normalize_prohoc_result(result_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = _extract_dataset_name(result_path, payload)
    rows = []
    for method_name, split_metrics in payload.items():
        if not isinstance(split_metrics, dict):
            continue
        for split, metrics in split_metrics.items():
            if not isinstance(metrics, dict):
                continue
            row = {
                "source": "prohoc",
                "result_file": str(result_path),
                "dataset": dataset,
                "method": method_name,
                "split": split,
            }
            for metric in SCALAR_METRICS:
                row[metric] = _to_python_scalar(metrics.get(metric))
            rows.append(row)
    return rows


def load_result_rows(result_paths: list[Path], source_hint: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        payload = torch.load(result_path, map_location="cpu", weights_only=False)
        if source_hint == "gm" or (isinstance(payload, dict) and "results" in payload and "artifacts" in payload):
            rows.extend(_normalize_gm_result(result_path, payload))
        else:
            rows.extend(_normalize_prohoc_result(result_path, payload))
    return rows


def make_bar_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    created = []
    for metric in SCALAR_METRICS:
        plot_df = df.dropna(subset=[metric])
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = plot_df.copy()
        plot_df["label"] = plot_df["dataset"] + ":" + plot_df["split"] + ":" + plot_df["method"]
        ax.bar(plot_df["label"], plot_df[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        save_path = output_dir / f"{metric}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        created.append(save_path)
    return created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prohoc", help="Path to a ProHOC .result file or directory", default=r"C:\HOYOUNG\datadrift_1\drift_detection\HierOOD\GM-ProHOC\ckpts\fgvc-aircraft\hinference-fgvc-aircraft.result")
    parser.add_argument("--gm", help="Path to a GM-ProHOC .result file or directory", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    prohoc_files = _discover_result_files(args.prohoc) if args.prohoc else []
    gm_files = _discover_result_files(args.gm) if args.gm else []

    rows = []
    rows.extend(load_result_rows(prohoc_files, source_hint="prohoc"))
    rows.extend(load_result_rows(gm_files, source_hint="gm"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    summary_path = output_dir / "comparison_summary.csv"
    df.to_csv(summary_path, index=False)

    pivot = df.pivot_table(
        index=["dataset", "split", "method"],
        values=SCALAR_METRICS,
        aggfunc="first",
    ).reset_index()
    pivot_path = output_dir / "comparison_pivot.csv"
    pivot.to_csv(pivot_path, index=False)

    if not df.empty:
        make_bar_plots(df, output_dir)


if __name__ == "__main__":
    main()
