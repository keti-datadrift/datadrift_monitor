from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _matplotlib_path(path: Path) -> str:
    path = Path(path)
    if os.name != "nt":
        return str(path)
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def _savefig(fig, out_path: Path, dpi: int = 150) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_matplotlib_path(out_path), dpi=dpi)


def _save_curve_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray", label="Random")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right")
    except Exception as e:  # pragma: no cover - depends on label distribution
        ax.text(0.5, 0.5, f"ROC unavailable\n{e}", ha="center", va="center")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=150)
    plt.close(fig)


def _save_curve_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, lw=2, label=f"PR (AP={ap:.4f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower left")
    except Exception as e:  # pragma: no cover - depends on label distribution
        ax.text(0.5, 0.5, f"PR unavailable\n{e}", ha="center", va="center")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=150)
    plt.close(fig)


def _save_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
    pos_name: str,
    neg_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[finite].astype(np.int64)
    scores = scores[finite]
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if scores.size == 0 or (pos_scores.size == 0 and neg_scores.size == 0):
        ax.text(0.5, 0.5, "No scores to plot", ha="center", va="center")
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.nanmin(scores))
        hi = float(np.nanmax(scores))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 36)
        ax.hist(
            pos_scores,
            bins=bins,
            color="#4C78A8",
            alpha=0.45,
            edgecolor="none",
            label=f"{pos_name} (n={pos_scores.size})",
        )
        ax.hist(
            neg_scores,
            bins=bins,
            color="#E45756",
            alpha=0.45,
            edgecolor="none",
            label=f"{neg_name} (n={neg_scores.size})",
        )

    metric_text = ""
    if np.unique(labels).size == 2:
        try:
            auroc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            metric_text = f"AUROC={auroc:.3f}, AP={ap:.3f}"
        except Exception:
            metric_text = ""

    ax.set_title(f"{pos_name} vs {neg_name} Score Distribution", fontsize=12, pad=18)
    if metric_text:
        ax.text(
            0.5,
            1.015,
            metric_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    ax.set_xlabel(f"Predicted {pos_name} score")
    ax.set_ylabel("count")
    ax.set_xlim(lo, hi)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=150)
    plt.close(fig)


def _save_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    n = int(y_true.shape[0])
    if n == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
    else:
        order = np.argsort(-y_score)
        y_sorted = y_true[order].astype(np.float64)
        cum_tp = np.cumsum(y_sorted)
        k_all = np.arange(1, n + 1, dtype=np.int64)
        precision_all = cum_tp / k_all

        max_points = 1000
        if n > max_points:
            idx = np.unique(np.linspace(0, n - 1, num=max_points, dtype=np.int64))
            k_plot = k_all[idx]
            precision_plot = precision_all[idx]
        else:
            k_plot = k_all
            precision_plot = precision_all

        ax.plot(k_plot, precision_plot, lw=2)
        ax.set_xlim(1, n)
        ax.set_ylim(0.0, 1.0)

    ax.set_title("Precision@K")
    ax.set_xlabel("K (top-K by predicted FN score)")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=150)
    plt.close(fig)


def _save_confidence_vs_conditional_precision(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    scores = np.clip(np.asarray(y_score, dtype=np.float64).reshape(-1), 0.0, 1.0)
    labels = np.asarray(y_true, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    centers: list[float] = []
    precision_vals: list[float] = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if i == len(bins) - 2:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        centers.append(float((lo + hi) * 0.5))
        precision_vals.append(float(labels[mask].mean()))

    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", linewidth=1.5, label="Oracle")
    if centers:
        ax.plot(centers, precision_vals, color="#1f77b4", linewidth=2.0, marker="o", markersize=3.5, label="Observed")
    else:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Confidence vs Conditional Precision")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Conditional Precision")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=150)
    plt.close(fig)


def save_eval_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir: Path,
    model_name: str,
    label_col: str = "fn",
) -> None:
    plot_dir = out_dir / "plots" / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    label_key = str(label_col).strip().lower()
    if label_key == "tp":
        pos_name = "TP"
        neg_name = "FP"
        dist_name = "score_distribution_tp_vs_fp.png"
    else:
        pos_name = "FN"
        neg_name = "non-FN"
        dist_name = "score_distribution_fn_vs_nonfn.png"
    _save_curve_roc(y_true, y_score, plot_dir / "roc_curve.png")
    _save_curve_pr(y_true, y_score, plot_dir / "pr_curve.png")
    _save_score_distribution(y_true, y_score, plot_dir / dist_name, pos_name=pos_name, neg_name=neg_name)
    _save_precision_at_k(y_true, y_score, plot_dir / "precision_at_k.png")
    if label_key == "tp":
        _save_confidence_vs_conditional_precision(
            y_true,
            y_score,
            plot_dir / "confidence_vs_conditional_precision.png",
        )
