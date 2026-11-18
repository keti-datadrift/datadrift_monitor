import json
from datetime import datetime
from pathlib import Path

import numpy as np

RUN_PATHS = [
    r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_score",
    r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_class_probability",
    r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_entropy",
    r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_energy",
    r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;22_mc_dropout",
    r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;33_meta_detect",
    r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;51_null_detect"
]

COMBINED_LAYER_GRAD_TERM_RUNS = [
    {
        "label": "layer_grad cand/box_l1+kl+bce",
        "term_paths": [
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-cand__term-bbox__b-box_l1-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-cand__term-cls__c-kl-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-cand__term-obj__o-bce-pred",
        ],
    },
        {
        "label": "layer_grad null/box_l1+kl+bce",
        "term_paths": [
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-null__term-bbox__b-box_l1-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-null__term-cls__c-kl-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-null__term-obj__o-bce-pred",
        ],
    },
]
# COMBINED_LAYER_GRAD_TERM_RUNS = []

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
OUTPUT_FILENAME = "uncertainty_timing_per_prediction.png"
METRIC = "mean_stage_ms_per_prediction"
TITLE = "Uncertainty Timing Comparison"
FIGSIZE = (11.8, 5.5)
USE_BROKEN_AXIS = False

DEFAULT_STAGE_ORDER = [
    "detector_inference_sec",
    "candidate_search_sec",
    "prediction_matching_sec",
    "loss_compute_sec",
    "backpropagation_sec",
    "feature_compute_sec",
]

STAGE_LABELS = {
    "detector_inference_sec": "Detector inference",
    "candidate_search_sec": "Candidate search",
    "prediction_matching_sec": "Prediction matching",
    "loss_compute_sec": "Loss compute",
    "backpropagation_sec": "Backpropagation",
    "feature_compute_sec": "Feature compute",
}

STAGE_COLORS = {
    "detector_inference_sec": "#4C78A8",
    "candidate_search_sec": "#F58518",
    "prediction_matching_sec": "#54A24B",
    "loss_compute_sec": "#E45756",
    "backpropagation_sec": "#B279A2",
    "feature_compute_sec": "#72B7B2",
}


def find_timing_jsons(paths):
    timing_paths = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.name.endswith("_timing.json"):
            timing_paths.append(path)
        elif path.is_dir():
            timing_paths.extend(path.rglob("*_timing.json"))
    return sorted(set(p.resolve() for p in timing_paths))


def find_timing_jsons_in_config_order(paths):
    timing_paths = []
    seen = set()
    for raw_path in paths:
        for timing_path in find_timing_jsons([raw_path]):
            if timing_path in seen:
                continue
            seen.add(timing_path)
            timing_paths.append(timing_path)
    return timing_paths


def load_timing_record(path, metric):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = data.get(metric)
    if not isinstance(values, dict):
        raise ValueError(f"{path} does not contain a '{metric}' object.")

    run_dir = path.parent.parent
    uncertainty = str(data.get("uncertainty") or path.name.replace("_timing.json", ""))
    return {
        "path": path,
        "run_dir": run_dir,
        "uncertainty": uncertainty,
        "label": f"{uncertainty}\n{run_dir.name}",
        "values": {str(k): float(v) for k, v in values.items()},
        "total_predictions": int(data.get("total_predictions", 0)),
    }


def load_combined_layer_grad_term_record(group, metric):
    label = str(group.get("label") or "layer_grad\ncombined terms")
    term_paths = list(group.get("term_paths") or [])
    if not term_paths:
        raise ValueError(f"Combined timing group '{label}' has no term_paths.")

    timing_paths = find_timing_jsons_in_config_order(term_paths)
    if not timing_paths:
        raise FileNotFoundError(
            f"No *_timing.json files found for combined timing group '{label}'."
        )

    records = [load_timing_record(path, metric) for path in timing_paths]
    stages = ordered_stages(records)
    values = {}
    for stage in stages:
        stage_values = [record["values"].get(stage, 0.0) for record in records]
        if stage in {"detector_inference_sec", "candidate_search_sec"}:
            nonzero = [value for value in stage_values if value > 0.0]
            values[stage] = float(max(nonzero)) if nonzero else 0.0
        else:
            values[stage] = float(sum(stage_values))

    total_predictions = min(
        (record["total_predictions"] for record in records), default=0
    )
    return {
        "path": timing_paths[0],
        "run_dir": timing_paths[0].parent.parent,
        "uncertainty": "layer_grad",
        "label": label,
        "values": values,
        "total_predictions": int(total_predictions),
        "combined_from": timing_paths,
    }


def ordered_stages(records):
    seen = []
    for stage in DEFAULT_STAGE_ORDER:
        if any(stage in record["values"] for record in records):
            seen.append(stage)
    for record in records:
        for stage in record["values"]:
            if stage not in seen:
                seen.append(stage)
    return seen


def broken_axis_limits(totals):
    positive = sorted([float(v) for v in totals if float(v) > 0.0], reverse=True)
    if len(positive) < 2:
        return None
    largest, second = positive[0], positive[1]
    if second <= 0.0 or largest < second * 2.0:
        return None
    lower_top = second * 1.25
    upper_bottom = largest * 0.88
    upper_top = largest * 1.08
    if lower_top >= upper_bottom:
        return None
    return lower_top, upper_bottom, upper_top


def plot_stacked_timing(records, output_path, title, figsize):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to draw timing plots. Install it in the environment used for visualization."
        ) from exc

    if not records:
        raise ValueError("No timing records to plot.")

    stages = ordered_stages(records)
    labels = [record["label"] for record in records]
    x = list(range(len(records)))
    totals = [
        sum(record["values"].get(stage, 0.0) for stage in stages) for record in records
    ]
    break_limits = broken_axis_limits(totals) if USE_BROKEN_AXIS else None

    fig_width = max(figsize[0], 1.1 * len(records) + 3.0)
    if break_limits is None:
        fig, ax = plt.subplots(figsize=(fig_width, figsize[1]))
        axes = [ax]
    else:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(fig_width, figsize[1] + 0.4),
            gridspec_kw={"height_ratios": [0.75, 2.5], "hspace": 0.06},
        )
        axes = [ax_top, ax_bottom]

    def draw_bars(ax, show_legend=False):
        bottoms = [0.0 for _ in records]
        for stage in stages:
            heights = [record["values"].get(stage, 0.0) for record in records]
            if not any(value > 0.0 for value in heights):
                continue
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                label=STAGE_LABELS.get(stage, stage),
                color=STAGE_COLORS.get(stage),
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        if show_legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        return bottoms

    def draw_axis_break_marks(fig, ax_bottom, ax_top):
        top_box = ax_top.get_position()
        bottom_box = ax_bottom.get_position()
        x0 = bottom_box.x0
        dx = 0.012
        dy = 0.012
        kwargs = dict(
            transform=fig.transFigure, color="#333333", clip_on=False, linewidth=1.5
        )
        fig.lines.append(
            plt.Line2D([x0 - dx, x0 + dx], [top_box.y0 - dy, top_box.y0 + dy], **kwargs)
        )
        fig.lines.append(
            plt.Line2D(
                [x0 - dx, x0 + dx], [bottom_box.y1 - dy, bottom_box.y1 + dy], **kwargs
            )
        )

    if break_limits is None:
        ax = axes[0]
        bottoms = draw_bars(ax, show_legend=True)
        for idx, total in enumerate(bottoms):
            ax.text(idx, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Mean stage time per prediction (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.yaxis.set_major_locator(MultipleLocator(10.0))
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    else:
        lower_top, upper_bottom, upper_top = break_limits
        ax_top, ax_bottom = axes
        draw_bars(ax_top, show_legend=True)
        draw_bars(ax_bottom, show_legend=False)

        ax_bottom.set_ylim(0.0, lower_top)
        ax_top.set_ylim(upper_bottom, upper_top)
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()

        for idx, total in enumerate(totals):
            target_ax = ax_top if total > upper_bottom else ax_bottom
            target_ax.text(
                idx, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8
            )

        ax_top.set_title(title)
        ax_bottom.set_ylabel("Mean stage time per prediction (ms)")
        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels(labels, rotation=35, ha="right")
        for ax in axes:
            ax.yaxis.set_major_locator(MultipleLocator(10.0))
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

        draw_axis_break_marks(fig, ax_bottom, ax_top)

    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return output_path


def make_output_path():
    timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
    return Path(OUTPUT_ROOT) / f"{timestamp}_uncertainty_timing" / OUTPUT_FILENAME


def main():
    timing_paths = find_timing_jsons_in_config_order(RUN_PATHS)
    if not timing_paths and not COMBINED_LAYER_GRAD_TERM_RUNS:
        searched = ", ".join(str(Path(p)) for p in RUN_PATHS)
        raise FileNotFoundError(f"No *_timing.json files found under: {searched}")

    records = [load_timing_record(path, METRIC) for path in timing_paths]
    records.extend(
        load_combined_layer_grad_term_record(group, METRIC)
        for group in COMBINED_LAYER_GRAD_TERM_RUNS
    )
    if not records:
        raise ValueError("No timing records to plot.")

    output_path = plot_stacked_timing(records, make_output_path(), TITLE, FIGSIZE)
    print(f"Saved timing plot: {output_path}")


if __name__ == "__main__":
    main()
