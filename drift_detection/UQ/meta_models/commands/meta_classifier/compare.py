from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from meta_models.losses.meta_classifier import CLASSIFIER_METRIC_COLUMNS, compute_ace, compute_ece, compute_fpr_at_tpr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path_value(raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _sanitize_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower())
    return out.strip("_") or "run"


@dataclass
class RunBundle:
    name: str
    root: Path
    label_col: str
    y_true: np.ndarray
    y_score: np.ndarray
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]


def _read_eval_data(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    eval_files = sorted(results_dir.glob("eval_data*.csv"))
    if not eval_files:
        raise FileNotFoundError(f"No eval_data*.csv found in: {results_dir}")
    ys_true: list[np.ndarray] = []
    ys_score: list[np.ndarray] = []
    for path in eval_files:
        df = pd.read_csv(path)
        if "y_test" not in df.columns or "y_pred" not in df.columns:
            continue
        ys_true.append(df["y_test"].astype(int).to_numpy())
        ys_score.append(df["y_pred"].astype(float).to_numpy())
    if not ys_true:
        raise ValueError(f"Could not load y_test/y_pred from eval_data files under: {results_dir}")
    return np.concatenate(ys_true), np.concatenate(ys_score)


def _extract_metric_rows(eval_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    numeric_cols = [c for c in CLASSIFIER_METRIC_COLUMNS if c in eval_df.columns]
    if not numeric_cols:
        return {}, {}
    mean_row = None
    std_row = None
    summary_mask = pd.Series(False, index=eval_df.index)
    if "row_type" in eval_df.columns:
        row_type = eval_df["row_type"].astype(str).str.lower()
        mean_hit = eval_df[row_type == "mean"]
        std_hit = eval_df[row_type == "std"]
        summary_mask = row_type.isin({"mean", "std"})
        if not mean_hit.empty:
            mean_row = mean_hit.iloc[0]
        if not std_hit.empty:
            std_row = std_hit.iloc[0]
    if "model_file" in eval_df.columns:
        mean_hit = eval_df[eval_df["model_file"].astype(str).str.lower() == "mean"]
        std_hit = eval_df[eval_df["model_file"].astype(str).str.lower() == "std"]
        summary_mask = summary_mask | eval_df["model_file"].astype(str).str.lower().isin({"mean", "std"})
        if not mean_hit.empty:
            mean_row = mean_hit.iloc[0]
        if not std_hit.empty:
            std_row = std_hit.iloc[0]
    legacy_index_cols = [c for c in eval_df.columns if str(c).startswith("Unnamed")]
    for col in legacy_index_cols:
        values = eval_df[col].astype(str).str.lower()
        summary_mask = summary_mask | values.isin({"mean", "std"})
        if mean_row is None:
            mean_hit = eval_df[values == "mean"]
            if not mean_hit.empty:
                mean_row = mean_hit.iloc[0]
        if std_row is None:
            std_hit = eval_df[values == "std"]
            if not std_hit.empty:
                std_row = std_hit.iloc[0]
    if mean_row is None:
        mean_row = eval_df.loc[~summary_mask, numeric_cols].mean(numeric_only=True)
    if std_row is None:
        std_row = eval_df.loc[~summary_mask, numeric_cols].std(numeric_only=True, ddof=1)
    mean_dict = {k: float(mean_row[k]) for k in numeric_cols}
    std_dict = {k: float(std_row[k]) for k in numeric_cols}
    return mean_dict, std_dict


def _load_run_bundle(run_root: Path, name: str) -> RunBundle:
    root = run_root.resolve()
    results_dir = root / "results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    label_col = str(metadata.get("label_column", "tp")).strip().lower()
    y_true, y_score = _read_eval_data(results_dir)
    eval_path = results_dir / "evaluation_results.csv"
    if eval_path.is_file():
        eval_df = pd.read_csv(eval_path)
        mean_dict, std_dict = _extract_metric_rows(eval_df)
    else:
        mean_dict, std_dict = {}, {}

    if not mean_dict:
        mean_dict = {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "ap": float(average_precision_score(y_true, y_score)),
            "fpr95": float(compute_fpr_at_tpr(y_true, y_score)),
            "ece": float(compute_ece(y_true, y_score)),
            "ace": float(compute_ace(y_true, y_score)),
        }
    for k, v in {
        "fpr95": float(compute_fpr_at_tpr(y_true, y_score)),
        "ece": float(compute_ece(y_true, y_score)),
        "ace": float(compute_ace(y_true, y_score)),
    }.items():
        mean_dict.setdefault(k, v)
    return RunBundle(
        name=name,
        root=root,
        label_col=label_col,
        y_true=y_true,
        y_score=y_score,
        metrics_mean=mean_dict,
        metrics_std=std_dict,
    )


def _plot_roc_compare(a: RunBundle, b: RunBundle, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    fa, ta, _ = roc_curve(a.y_true, a.y_score)
    fb, tb, _ = roc_curve(b.y_true, b.y_score)
    ax.plot(fa, ta, lw=2, label=f"{a.name} (AUROC={roc_auc_score(a.y_true, a.y_score):.4f})")
    ax.plot(fb, tb, lw=2, label=f"{b.name} (AUROC={roc_auc_score(b.y_true, b.y_score):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray", label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("ROC Compare")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pr_compare(a: RunBundle, b: RunBundle, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    pa, ra, _ = precision_recall_curve(a.y_true, a.y_score)
    pb, rb, _ = precision_recall_curve(b.y_true, b.y_score)
    ax.plot(ra, pa, lw=2, label=f"{a.name} (AP={average_precision_score(a.y_true, a.y_score):.4f})")
    ax.plot(rb, pb, lw=2, label=f"{b.name} (AP={average_precision_score(b.y_true, b.y_score):.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("PR Compare")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_score_distribution_compare(a: RunBundle, b: RunBundle, out_path: Path) -> None:
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6, 5))
        a_pos = a.y_score[a.y_true == 1]
        a_neg = a.y_score[a.y_true == 0]
        b_pos = b.y_score[b.y_true == 1]
        b_neg = b.y_score[b.y_true == 0]

        label_pos = "FP"
        label_neg = "non-FP"

        if a_neg.size > 0:
            sns.histplot(
                a_neg,
                bins=40,
                stat="density",
                alpha=0.25,
                element="step",
                fill=True,
                color="#1f77b4",
                linestyle="-",
                label=f"{a.name} {label_neg}",
                ax=ax,
            )
        if a_pos.size > 0:
            sns.histplot(
                a_pos,
                bins=40,
                stat="density",
                alpha=0.35,
                element="step",
                fill=True,
                color="#1f77b4",
                linestyle="--",
                label=f"{a.name} {label_pos}",
                ax=ax,
            )
        if b_neg.size > 0:
            sns.histplot(
                b_neg,
                bins=40,
                stat="density",
                alpha=0.25,
                element="step",
                fill=True,
                color="#d62728",
                linestyle="-",
                label=f"{b.name} {label_neg}",
                ax=ax,
            )
        if b_pos.size > 0:
            sns.histplot(
                b_pos,
                bins=40,
                stat="density",
                alpha=0.35,
                element="step",
                fill=True,
                color="#d62728",
                linestyle="--",
                label=f"{b.name} {label_pos}",
                ax=ax,
            )

        ax.set_title("Predicted Score Distribution Compare (FP / non-FP)")
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Density")
        ax.legend(loc="upper center", ncol=2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def _conditional_precision_curve(y_true: np.ndarray, y_score: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    scores = np.clip(np.asarray(y_score, dtype=np.float64).reshape(-1), 0.0, 1.0)
    labels = np.asarray(y_true, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(int(bins)):
        lo = edges[i]
        hi = edges[i + 1]
        if i == int(bins) - 1:
            m = (scores >= lo) & (scores <= hi)
        else:
            m = (scores >= lo) & (scores < hi)
        if not np.any(m):
            continue
        xs.append(float((lo + hi) * 0.5))
        ys.append(float(labels[m].mean()))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _plot_confidence_conditional_precision_compare(a: RunBundle, b: RunBundle, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    xa, ya = _conditional_precision_curve(a.y_true, a.y_score, bins=20)
    xb, yb = _conditional_precision_curve(b.y_true, b.y_score, bins=20)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", linewidth=1.5, label="Oracle")
    if xa.size:
        ax.plot(xa, ya, color="#4C78A8", marker="o", markersize=3.5, linewidth=2.0, label=a.name)
    if xb.size:
        ax.plot(xb, yb, color="#E45756", marker="o", markersize=3.5, linewidth=2.0, label=b.name)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Confidence vs Conditional Precision Compare")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Conditional Precision")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fp_score_conditional_precision_compare(a: RunBundle, b: RunBundle, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    a_fp_score = 1.0 - np.clip(a.y_score, 0.0, 1.0)
    b_fp_score = 1.0 - np.clip(b.y_score, 0.0, 1.0)
    a_fp_true = 1.0 - np.asarray(a.y_true, dtype=np.float64)
    b_fp_true = 1.0 - np.asarray(b.y_true, dtype=np.float64)
    xa, ya = _conditional_precision_curve(a_fp_true, a_fp_score, bins=20)
    xb, yb = _conditional_precision_curve(b_fp_true, b_fp_score, bins=20)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", linewidth=1.5, label="Oracle")
    if xa.size:
        ax.plot(xa, ya, color="#4C78A8", marker="o", markersize=3.5, linewidth=2.0, label=a.name)
    if xb.size:
        ax.plot(xb, yb, color="#E45756", marker="o", markersize=3.5, linewidth=2.0, label=b.name)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("FP Score vs FP Precision Compare")
    ax.set_xlabel("FP score")
    ax.set_ylabel("FP precision")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_compare(config: dict[str, Any], run_dir: Path) -> Path:
    compare_cfg = config.get("compare", {})
    raw_runs = compare_cfg.get("run_roots", [])
    if not isinstance(raw_runs, (list, tuple)) or len(raw_runs) != 2:
        raise ValueError("compare.run_roots must be a list of exactly two run paths.")
    run_a_root = _resolve_path_value(str(raw_runs[0]))
    run_b_root = _resolve_path_value(str(raw_runs[1]))

    name_a = str(compare_cfg.get("name_a", "")).strip() or run_a_root.name
    name_b = str(compare_cfg.get("name_b", "")).strip() or run_b_root.name

    bundle_a = _load_run_bundle(run_a_root, name_a)
    bundle_b = _load_run_bundle(run_b_root, name_b)

    out_dir = run_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    plots_dir = out_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        key
        for key in CLASSIFIER_METRIC_COLUMNS
        if key in bundle_a.metrics_mean or key in bundle_b.metrics_mean
    ]
    metric_keys.extend(
        sorted((set(bundle_a.metrics_mean.keys()) | set(bundle_b.metrics_mean.keys())) - set(metric_keys))
    )
    rows = []
    for key in metric_keys:
        a_mean = float(bundle_a.metrics_mean.get(key, np.nan))
        b_mean = float(bundle_b.metrics_mean.get(key, np.nan))
        a_std = float(bundle_a.metrics_std.get(key, np.nan))
        b_std = float(bundle_b.metrics_std.get(key, np.nan))
        rows.append(
            {
                "metric": key,
                f"{bundle_a.name}_mean": a_mean,
                f"{bundle_a.name}_std": a_std,
                f"{bundle_b.name}_mean": b_mean,
                f"{bundle_b.name}_std": b_std,
                "delta_mean_a_minus_b": a_mean - b_mean,
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "metrics_compare.csv", index=False)

    _plot_roc_compare(bundle_a, bundle_b, plots_dir / "roc_compare.png")
    _plot_pr_compare(bundle_a, bundle_b, plots_dir / "pr_compare.png")
    _plot_score_distribution_compare(bundle_a, bundle_b, plots_dir / "score_distribution_compare.png")
    _plot_confidence_conditional_precision_compare(
        bundle_a,
        bundle_b,
        plots_dir / "confidence_vs_conditional_precision_compare.png",
    )
    _plot_fp_score_conditional_precision_compare(
        bundle_a,
        bundle_b,
        plots_dir / "fp_score_vs_conditional_precision_compare.png",
    )

    summary = {
        "run_a": str(bundle_a.root),
        "run_b": str(bundle_b.root),
        "name_a": bundle_a.name,
        "name_b": bundle_b.name,
        "label_col_a": bundle_a.label_col,
        "label_col_b": bundle_b.label_col,
        "n_samples_a": int(bundle_a.y_true.size),
        "n_samples_b": int(bundle_b.y_true.size),
        "metrics_mean_a": bundle_a.metrics_mean,
        "metrics_mean_b": bundle_b.metrics_mean,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved compare outputs to: {out_dir}")
    return out_dir
