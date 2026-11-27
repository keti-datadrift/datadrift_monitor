import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

METHOD_RUNS = [
    {
        "label": "Score",
        "method_type": "Deterministic",
        "od_run": r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_score",
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_18;20_score",
        "input_dim": 1,
    },
    {
        "label": "Class Prob.",
        "method_type": "Deterministic",
        "od_run": r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_class_probability",
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_18;21_class_probability",
        "input_dim": 80,
    },
    {
        "label": "Entropy",
        "method_type": "Deterministic",
        "od_run": r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_entropy",
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_18;22_entropy",
        "input_dim": 1,
    },
    {
        "label": "Energy",
        "method_type": "Deterministic",
        "od_run": r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;09_energy",
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_18;23_energy",
        "input_dim": 1,
    },
    {
        "label": "MC Dropout",
        "method_type": "Stochastic",
        "od_run": r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;22_mc_dropout",
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_18;24_mc_dropout",
        "input_dim": 85,
    },
    {
        "label": "MetaDetect",
        "method_type": "Deterministic",
        "od_run": r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;33_meta_detect",
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_18;25_meta_detect",
        "input_dim": 121,
    },
    {
        "label": "NullDetect",
        "method_type": "Deterministic",
        "od_run": r"object_detectors/runs/yolov5/predict/coco/06-05-2026_17;51_null_detect",
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_18;27_null_detect",
        "input_dim": 94,
    },
    {
        "label": "LayerGrad-cand",
        "method_type": "Gradient",
        "od_runs": [
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-cand__term-bbox__b-box_l1-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-cand__term-cls__c-kl-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-cand__term-obj__o-bce-pred",
        ],
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_20;56_layer_grad_t-cand__b-box_l1-pred__c-kl-pred__o-bce-pred",
        "input_dim": 54,
    },
    {
        "label": "LayerGrad-null",
        "method_type": "Gradient",
        "od_runs": [
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-null__term-bbox__b-box_l1-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-null__term-cls__c-kl-pred",
            r"object_detectors/runs/yolov5/predict/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_19;12_layer_grad_t-null__term-obj__o-bce-pred",
        ],
        "meta_run": r"meta_models/meta_classifier/runs/yolov5/train/coco/06-05-2026_19;12_layer_grad_grid/06-05-2026_21;04_layer_grad_t-null__b-box_l1-pred__c-kl-pred__o-bce-pred",
        "input_dim": 54,
    },
]

BASE_COLUMNS = {
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
    "error_type",
    "matched_gt_idx",
    "matched_iou",
}

METHOD_COLORS = {
    "Deterministic": "#64748b",
    "Stochastic": "#8b5cf6",
    "Gradient": "#ef7777",
}

WIDTH = 1120
HEIGHT = 620
PLOT_LEFT = 90
PLOT_TOP = 42
PLOT_RIGHT = 935
PLOT_BOTTOM = 538
Y_MIN = 0.54
Y_MAX = 1.0
Y_TICK_STEP = 0.02
X_TICK_STEP = 2.0
SIZE_LEGEND_DIMS = [1, 54, 88, 121]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def find_single_file(root, pattern, label):
    root = resolve_path(root)
    if not root.exists():
        raise FileNotFoundError(f"{label}: path does not exist: {root}")
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"{label}: no {pattern} found under: {root}")
    return matches[0]


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    values = [float(v) for v in values]
    if not values:
        raise ValueError("Cannot compute mean of empty values.")
    return sum(values) / len(values)


def load_eval_metrics(meta_run, label, grid_filter=None):
    meta_path = resolve_path(meta_run)
    if grid_filter:
        grid_path = meta_path / "grid_results.csv"
        if not grid_path.exists():
            raise FileNotFoundError(f"{label}: missing grid_results.csv: {grid_path}")
        rows = read_csv_rows(grid_path)
        matches = [
            row
            for row in rows
            if all(str(row.get(key, "")) == str(value) for key, value in grid_filter.items())
        ]
        if len(matches) != 1:
            raise ValueError(f"{label}: expected one grid_results row for {grid_filter}, found {len(matches)}.")
        row = matches[0]
        return float(row["auroc"]), float(row["ap"])

    eval_path = meta_path / "results" / "evaluation_results.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"{label}: missing evaluation_results.csv: {eval_path}")
    rows = read_csv_rows(eval_path)
    index_key = next((key for key in rows[0].keys() if key not in {"auroc", "ap", "fpr95", "ece", "ace"}), None) if rows else None
    if index_key is not None:
        for row in rows:
            if str(row.get(index_key, "")).strip().lower() == "mean":
                return float(row["auroc"]), float(row["ap"])
        rows = [row for row in rows if str(row.get(index_key, "")).strip().lower() not in {"mean", "std"}]
    return mean(row["auroc"] for row in rows), mean(row["ap"] for row in rows)


def load_timing_values(od_run, label):
    timing_path = find_single_file(od_run, "*_timing.json", label)
    with open(timing_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    values = data.get("mean_stage_ms_per_prediction")
    if not isinstance(values, dict):
        raise ValueError(f"{label}: timing json has no mean_stage_ms_per_prediction: {timing_path}")
    return {str(key): float(value) for key, value in values.items()}


def method_time_ms(method):
    label = method["label"]
    if "od_runs" not in method:
        values = load_timing_values(method["od_run"], label)
        return float(sum(values.values()))

    term_values = [load_timing_values(path, label) for path in method["od_runs"]]
    stages = sorted({stage for values in term_values for stage in values})
    total = 0.0
    for stage in stages:
        vals = [values.get(stage, 0.0) for values in term_values]
        if stage in {"detector_inference_sec", "candidate_search_sec"}:
            total += max(vals)
        else:
            total += sum(vals)
    return float(total)


def find_run_csv(root, label):
    root = resolve_path(root)
    if not root.exists():
        raise FileNotFoundError(f"{label}: path does not exist: {root}")
    candidates = [
        path for path in root.glob("*.csv")
        if path.name not in {"used_config.csv"} and not path.name.endswith("_timing.csv")
    ]
    if not candidates:
        raise FileNotFoundError(f"{label}: no output csv found directly under: {root}")
    return sorted(candidates)[0]


def feature_dim_from_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    return sum(1 for column in header if column not in BASE_COLUMNS)


def method_input_dim(method):
    if method.get("input_dim") is not None:
        return int(method["input_dim"])
    label = method["label"]
    if "od_runs" in method:
        return sum(feature_dim_from_csv(find_run_csv(path, label)) for path in method["od_runs"])
    return feature_dim_from_csv(find_run_csv(method["od_run"], label))


def collect_points(methods):
    points = []
    for method in methods:
        label = method["label"]
        auroc, ap = load_eval_metrics(method["meta_run"], label, method.get("grid_filter"))
        point = {
            "label": label,
            "method_type": method["method_type"],
            "time_ms_per_prediction": method_time_ms(method),
            "auroc": auroc,
            "ap": ap,
            "input_dim": method_input_dim(method),
            "od_run": method.get("od_run", ""),
            "od_runs": " | ".join(method.get("od_runs", [])),
            "meta_run": method["meta_run"],
        }
        points.append(point)
    return points


def nice_x_max(values):
    max_value = max(values)
    return max(2.0, math.ceil(max_value / X_TICK_STEP) * X_TICK_STEP)


def x_map(value, x_max):
    return PLOT_LEFT + (float(value) / x_max) * (PLOT_RIGHT - PLOT_LEFT)


def y_map(value):
    return PLOT_BOTTOM - ((float(value) - Y_MIN) / (Y_MAX - Y_MIN)) * (PLOT_BOTTOM - PLOT_TOP)


def radius_for_dim(dim):
    dim = max(float(dim), 1.0)
    min_r = 6.0
    max_r = 32.0
    max_dim = max(float(max(SIZE_LEGEND_DIMS)), 1.0)
    return min_r + (math.sqrt(dim) - 1.0) / (math.sqrt(max_dim) - 1.0) * (max_r - min_r)


def svg_text(x, y, text, size=12, fill="#374151", anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial" '
        f'font-size="{size}" fill="{fill}"{extra}>{html.escape(str(text))}</text>'
    )


def draw_svg(points, output_path, with_labels):
    x_max = nice_x_max([point["time_ms_per_prediction"] for point in points])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    y_tick = Y_MIN
    while y_tick <= Y_MAX + 1e-9:
        y = y_map(y_tick)
        lines.append(f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_RIGHT}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(svg_text(78, y + 4, f"{y_tick:.2f}", fill="#6b7280", anchor="end"))
        y_tick += Y_TICK_STEP

    tick = X_TICK_STEP
    while tick <= x_max + 1e-9:
        x = x_map(tick, x_max)
        lines.append(f'<line x1="{x:.1f}" y1="{PLOT_TOP}" x2="{x:.1f}" y2="{PLOT_BOTTOM}" stroke="#f1f5f9"/>')
        lines.append(svg_text(f"{x:.1f}", 560, f"{tick:.1f}", fill="#6b7280", anchor="middle"))
        tick += X_TICK_STEP

    lines.append(f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(f'<line x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(svg_text(512.5, 596, "Mean time per prediction (ms)", size=14, fill="#111827", anchor="middle"))
    lines.append(svg_text(24, 290.0, "AUROC", size=14, fill="#111827", anchor="middle", extra=' transform="rotate(-90 24 290.0)"'))

    for point in points:
        x = x_map(point["time_ms_per_prediction"], x_max)
        y = y_map(point["auroc"])
        r = radius_for_dim(point["input_dim"])
        color = METHOD_COLORS.get(point["method_type"], "#64748b")
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            'fill-opacity="0.88" stroke="white" stroke-width="1.4"/>'
        )

    if with_labels:
        for point in points:
            x = x_map(point["time_ms_per_prediction"], x_max)
            y = y_map(point["auroc"])
            r = radius_for_dim(point["input_dim"])
            dy = -10 if point["label"] in {"Class Prob.", "MetaDetect", "LayerGrad-null"} else 16
            lines.append(svg_text(f"{x + r * 0.35:.1f}", f"{y + dy:.1f}", point["label"], size=12, fill="#111827"))

    for idx, dim in enumerate(SIZE_LEGEND_DIMS):
        y = 80 + idx * 46
        r = radius_for_dim(dim)
        lines.append(f'<circle cx="981" cy="{y}" r="{r:.1f}" fill="#cbd5e1" fill-opacity="0.62" stroke="white"/>')
        lines.append(svg_text(1015, y + 4, dim, size=12, fill="#374151"))

    legend_y = 300
    for idx, method_type in enumerate(["Deterministic", "Stochastic", "Gradient"]):
        y = legend_y + idx * 26
        color = METHOD_COLORS[method_type]
        lines.append(f'<circle cx="973" cy="{y}" r="7" fill="{color}" fill-opacity="0.9" stroke="white"/>')
        lines.append(svg_text(989, y + 4, method_type, size=12, fill="#374151"))

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_points_csv(points, output_path):
    fieldnames = [
        "label",
        "method_type",
        "time_ms_per_prediction",
        "auroc",
        "ap",
        "input_dim",
        "od_run",
        "od_runs",
        "meta_run",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point)


def make_output_dir():
    timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
    path = OUTPUT_ROOT / f"{timestamp}_yolov5_method_bubble_plot_refined"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    points = collect_points(METHOD_RUNS)
    output_dir = make_output_dir()
    write_points_csv(points, output_dir / "bubble_plot_points.csv")
    draw_svg(points, output_dir / "bubble_auroc_time_dim_refined_labeled_stronger_size.svg", with_labels=True)
    draw_svg(points, output_dir / "bubble_auroc_time_dim_refined_no_labels_stronger_size.svg", with_labels=False)
    print(f"Saved bubble plot outputs: {output_dir}")


if __name__ == "__main__":
    main()
