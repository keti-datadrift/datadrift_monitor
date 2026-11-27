from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Keep the existing title treatment, but make the supporting text legible
# when the figures are placed on presentation slides.
TITLE_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 18
SUMMARY_FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the CIFAR-100 trajectory with the largest accuracy gap "
            "between 20-exemplar NME and full-data prototype NME."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/cmpt/full_mean_oracle/cifar100"),
    )
    parser.add_argument(
        "--oracle-mode",
        choices=("all_seen", "old_only"),
        default="all_seen",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Plot only the 20-exemplar NME trajectory. This omits the "
            "full-data oracle, gap annotations, and summary box."
        ),
    )
    parser.add_argument(
        "--hide-summary",
        action="store_true",
        help="Omit the lower-left summary box from the oracle comparison.",
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Omit the plot legend.",
    )
    parser.add_argument(
        "--learner",
        type=str,
        default=None,
        help=(
            "Select a learner by its cleaned result name instead of choosing "
            "the learner with the largest oracle gap."
        ),
    )
    parser.add_argument(
        "--affine-results",
        type=Path,
        default=None,
        help=(
            "Optional matched CMPT result JSON. When provided, add its "
            "affine prototype-transformation accuracy trajectory."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _clean_learner_name(value: str) -> str:
    suffix = "-full-training-mean-oracle"
    return value[: -len(suffix)] if value.endswith(suffix) else value


def _load_candidates(root: Path, oracle_mode: str) -> list[dict]:
    candidates: list[dict] = []
    for path in sorted(root.glob("*/seed_1/results.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = [
            record
            for record in payload["records"]
            if int(record["session_id"]) > 0
        ]
        sessions = [int(record["session_id"]) for record in records]
        if sessions != list(range(1, 11)):
            raise ValueError(f"{path} does not contain S1-S10")
        baseline = [100.0 * float(record["baseline"]["accuracy"]) for record in records]
        oracle = [
            100.0
            * float(record["full_mean_oracle"][oracle_mode]["accuracy"])
            for record in records
        ]
        gaps = [right - left for left, right in zip(baseline, oracle)]
        candidates.append(
            {
                "learner": _clean_learner_name(str(payload["learner"])),
                "result_file": str(path.resolve()),
                "sessions": sessions,
                "baseline": baseline,
                "oracle": oracle,
                "gaps": gaps,
                "incremental_nme_average": sum(baseline) / len(baseline),
                "incremental_oracle_average": sum(oracle) / len(oracle),
                "mean_gap": sum(gaps) / len(gaps),
                "final_gap": gaps[-1],
                "maximum_gap": max(gaps),
                "maximum_gap_session": sessions[gaps.index(max(gaps))],
            }
        )
    if not candidates:
        raise FileNotFoundError(f"no full-mean oracle results below {root}")
    return candidates


def _load_affine_trajectory(
    path: Path,
    sessions: list[int],
    baseline: list[float],
) -> list[float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = {
        int(record["session_id"]): record
        for record in payload["records"]
        if int(record["session_id"]) > 0
    }
    if sorted(records) != sessions:
        raise ValueError(f"{path} does not contain the requested sessions")
    affine_baseline = [
        100.0 * float(records[session]["baseline"]["accuracy"])
        for session in sessions
    ]
    parity_error = max(
        abs(left - right)
        for left, right in zip(baseline, affine_baseline)
    )
    if parity_error > 1.0e-8:
        raise ValueError(
            "affine and oracle evaluations use different NME baselines: "
            f"maximum error={parity_error:.12g} pp"
        )
    return [
        100.0 * float(records[session]["cmpt"]["accuracy"])
        for session in sessions
    ]


def _draw(
    output: Path,
    selected: dict,
    candidate_count: int,
    *,
    baseline_only: bool = False,
    hide_summary: bool = False,
    hide_legend: bool = False,
    affine: list[float] | None = None,
) -> None:
    sessions = selected["sessions"]
    baseline = selected["baseline"]
    oracle = selected["oracle"]
    gaps = selected["gaps"]
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    axis.plot(
        sessions,
        baseline,
        color="#2166ac",
        marker="o",
        markersize=7,
        linewidth=2.6,
        label="20-exemplar NME",
        zorder=4,
    )
    if not baseline_only:
        axis.plot(
            sessions,
            oracle,
            color="#d6604d",
            marker="s",
            markersize=7,
            linewidth=2.6,
            label="Full-data prototype oracle (500/class)",
            zorder=5,
        )
        axis.fill_between(
            sessions,
            baseline,
            oracle,
            color="#f4a582",
            alpha=0.24,
            label="Oracle headroom",
            zorder=1,
        )
        if affine is None:
            for session, low, high, gap in zip(sessions, baseline, oracle, gaps):
                axis.annotate(
                    f"+{gap:.2f}",
                    xy=(session, 0.5 * (low + high)),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_FONT_SIZE,
                    color="#7f0000",
                    weight="bold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                        "pad": 0.8,
                    },
                    zorder=7,
                )
        else:
            axis.plot(
                sessions,
                affine,
                color="#1b9e77",
                marker="D",
                markersize=6.5,
                linewidth=2.6,
                label="Proposed method",
                zorder=6,
            )

    lower = min(baseline) - 3.0
    upper = max(baseline if baseline_only else oracle) + 3.0
    axis.set_ylim(lower, upper)
    axis.set_xticks(sessions, [f"S{value}" for value in sessions])
    axis.set_xlabel("Incremental session", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel(
        "Accuracy" if baseline_only else "Top-1 accuracy (%)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    axis.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axis.grid(alpha=0.23, linestyle="--")
    if not hide_legend:
        axis.legend(
            loc="upper right",
            framealpha=0.95,
            fontsize=LEGEND_FONT_SIZE,
        )
    if baseline_only:
        axis.set_title(
            "20-exemplar NME Accuracy Across Incremental Sessions\n"
            f"{selected['learner']}, CIFAR-100 B50-Inc5, seed 1",
            fontsize=TITLE_FONT_SIZE,
            weight="bold",
        )
    else:
        title = (
            "Prototype Recovery under Class-Incremental Learning"
            if affine is not None
            else "Full-data prototype oracle vs 20-exemplar NME"
        )
        axis.set_title(
            f"{title}\n"
            f"{selected['learner']}, CIFAR-100 B50-Inc5, seed 1",
            fontsize=TITLE_FONT_SIZE,
            weight="bold",
        )
        if not hide_summary:
            summary = (
                f"Largest incremental mean gap among {candidate_count} learners\n"
                f"20-exemplar mean: {selected['incremental_nme_average']:.3f}%\n"
                f"Full-data mean: {selected['incremental_oracle_average']:.3f}%\n"
                f"Mean headroom: +{selected['mean_gap']:.3f} pp\n"
                f"Final headroom: +{selected['final_gap']:.3f} pp"
            )
            axis.text(
                0.025,
                0.035,
                summary,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=SUMMARY_FONT_SIZE,
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": "white",
                    "edgecolor": "#888888",
                    "alpha": 0.94,
                },
                zorder=8,
            )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    candidates = _load_candidates(results_root, args.oracle_mode)
    ranking = sorted(candidates, key=lambda item: item["mean_gap"], reverse=True)
    if args.learner is None:
        selected = ranking[0]
    else:
        matches = [
            item
            for item in candidates
            if item["learner"].casefold() == args.learner.casefold()
        ]
        if len(matches) != 1:
            available = ", ".join(item["learner"] for item in candidates)
            raise ValueError(
                f"learner {args.learner!r} not found; available: {available}"
            )
        selected = matches[0]
    affine_path = (
        args.affine_results.expanduser().resolve()
        if args.affine_results is not None
        else None
    )
    affine = (
        _load_affine_trajectory(
            affine_path,
            selected["sessions"],
            selected["baseline"],
        )
        if affine_path is not None
        else None
    )
    output = args.output.expanduser().resolve()
    _draw(
        output,
        selected,
        len(ranking),
        baseline_only=args.baseline_only,
        hide_summary=args.hide_summary,
        hide_legend=args.hide_legend,
        affine=affine,
    )
    result = {
        "schema_version": 1,
        "plot_mode": "baseline_only" if args.baseline_only else "oracle_comparison",
        "summary_visible": not (args.baseline_only or args.hide_summary),
        "legend_visible": not args.hide_legend,
        "affine_result_file": str(affine_path) if affine_path else None,
        "affine_accuracy": affine,
        "selection_metric": "S1-S10 mean full-data accuracy minus NME accuracy",
        "oracle_mode": args.oracle_mode,
        "selected": selected,
        "ranking": [
            {
                "learner": item["learner"],
                "mean_gap": item["mean_gap"],
                "final_gap": item["final_gap"],
            }
            for item in ranking
        ],
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
        "oracle_warning": (
            "full-data prototypes revisit unavailable old-class training data"
        ),
    }
    metadata = output.with_suffix(".json")
    with metadata.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
