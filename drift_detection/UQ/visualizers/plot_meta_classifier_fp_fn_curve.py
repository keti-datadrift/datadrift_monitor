import csv
import html
import struct
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

RUNS = [
    {
        "label": "Score",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_12;44_score",
    },
    {
        "label": "Class Prob.",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_12;48_class_probability",
    },
    {
        "label": "Entropy",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_13;01_entropy",
    },
    {
        "label": "Energy",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_13;05_energy",
    },
    {
        "label": "MC Dropout",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_13;09_mc_dropout",
    },
    {
        "label": "Ensemble",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_13;21_ensemble",
    },
    {
        "label": "MetaDetect",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_13;35_meta_detect",
    },
    {
        "label": "NullDetect",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_14;07_null_detect",
    },
    {
        "label": "MetaDetect+LG cand",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-22-2026_13;22_meta_detect_layer_grad_layer_grad_layer_grad_t-cand__term-bbox__b-l1-pred_t-cand__ter_ef450e3012",
    },
    {
        "label": "NullDetect+LG null",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-22-2026_13;29_null_detect_layer_grad_layer_grad_layer_grad_t-null__term-bbox__b-l1-pred_t-null__ter_8a46228cb6",
    },
    {
        "label": "MetaDetect+LG null",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-22-2026_14;11_meta_detect_layer_grad_layer_grad_layer_grad_t-null__term-bbox__b-l1-pred_t-null__ter_0ac2797b38",
    },
    {
        "label": "NullDetect+LG cand",
        "run_root": r"meta_models/runs/meta_classifier/fcos/train/coco/06-22-2026_14;19_null_detect_layer_grad_layer_grad_layer_grad_t-cand__term-bbox__b-l1-pred_t-cand__ter_148720dece",
    },
]

MAX_CURVE_POINTS = 2500
SAMPLE_SIZE = 1000
SAMPLE_SEED = 20260623
CSV_CHUNKSIZE = 200000


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _priority_sample_arrays(current, incoming, rng, sample_size):
    cur_keys, cur_labels, cur_scores = current
    labels, scores = incoming
    if labels.size == 0:
        return current
    keys = rng.random(labels.size)
    if cur_labels.size:
        keys = np.concatenate([cur_keys, keys])
        labels = np.concatenate([cur_labels, labels])
        scores = np.concatenate([cur_scores, scores])
    if labels.size > sample_size:
        keep = np.argpartition(keys, sample_size - 1)[:sample_size]
        keys = keys[keep]
        labels = labels[keep]
        scores = scores[keep]
    return keys, labels, scores


def _sample_eval_data(files, sample_size, seed):
    rng = np.random.default_rng(seed)
    reservoir = (
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
    )
    total_rows = 0
    if pd is not None:
        for path in files:
            for chunk in pd.read_csv(path, usecols=["y_test", "y_pred"], chunksize=CSV_CHUNKSIZE):
                labels = chunk["y_test"].to_numpy(dtype=np.int64)
                scores = chunk["y_pred"].to_numpy(dtype=np.float64)
                total_rows += labels.size
                reservoir = _priority_sample_arrays(reservoir, (labels, scores), rng, sample_size)
    else:
        for path in files:
            batch_labels = []
            batch_scores = []
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "y_test" not in reader.fieldnames or "y_pred" not in reader.fieldnames:
                    raise ValueError(f"{path} must contain y_test and y_pred columns.")
                for row in reader:
                    batch_labels.append(int(float(row["y_test"])))
                    batch_scores.append(float(row["y_pred"]))
                    total_rows += 1
                    if len(batch_labels) >= CSV_CHUNKSIZE:
                        labels = np.asarray(batch_labels, dtype=np.int64)
                        scores = np.asarray(batch_scores, dtype=np.float64)
                        reservoir = _priority_sample_arrays(reservoir, (labels, scores), rng, sample_size)
                        batch_labels.clear()
                        batch_scores.clear()
            if batch_labels:
                labels = np.asarray(batch_labels, dtype=np.int64)
                scores = np.asarray(batch_scores, dtype=np.float64)
                reservoir = _priority_sample_arrays(reservoir, (labels, scores), rng, sample_size)
    _keys, labels, scores = reservoir
    if labels.size == 0:
        raise ValueError("No eval rows loaded.")
    return labels, scores, total_rows


def read_eval_data(run_root, sample_size=SAMPLE_SIZE, seed=SAMPLE_SEED):
    results_dir = resolve_path(run_root) / "results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")
    files = sorted(results_dir.glob("eval_data*.csv"))
    if not files:
        raise FileNotFoundError(f"No eval_data*.csv found in: {results_dir}")
    if sample_size is not None and sample_size > 0:
        return _sample_eval_data(files, int(sample_size), int(seed))
    if pd is not None:
        frames = [pd.read_csv(path, usecols=["y_test", "y_pred"]) for path in files]
        if not frames:
            raise ValueError(f"No eval rows loaded from: {results_dir}")
        df = pd.concat(frames, ignore_index=True)
        labels = df["y_test"].to_numpy(dtype=np.int64)
        scores = df["y_pred"].to_numpy(dtype=np.float64)
        return labels, scores, int(labels.size)
    labels = []
    scores = []
    for path in files:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "y_test" not in reader.fieldnames or "y_pred" not in reader.fieldnames:
                raise ValueError(f"{path} must contain y_test and y_pred columns.")
            for row in reader:
                labels.append(int(float(row["y_test"])))
                scores.append(float(row["y_pred"]))
    if not labels:
        raise ValueError(f"No eval rows loaded from: {results_dir}")
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    return labels, scores, int(labels.size)


def build_curve(labels, scores, max_points=MAX_CURVE_POINTS):
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = np.clip(scores[finite], 0.0, 1.0)
    if labels.size == 0:
        raise ValueError("No finite y_pred scores are available.")
    tp_total = int(np.sum(labels == 1))
    fp_total = int(np.sum(labels == 0))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_tp = (labels[order] == 1).astype(np.int64)
    sorted_fp = (labels[order] == 0).astype(np.int64)
    tp_cum = np.cumsum(sorted_tp)
    fp_cum = np.cumsum(sorted_fp)
    last_by_score = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
    thresholds = sorted_scores[last_by_score][::-1]
    fp_keep = fp_cum[last_by_score].astype(np.int64)[::-1]
    tp_keep = tp_cum[last_by_score].astype(np.int64)[::-1]
    fn_count = tp_total - tp_keep
    kept_count = fp_keep + tp_keep
    dropped_count = labels.size - kept_count

    x_full = np.concatenate(([fp_total], fp_keep.astype(np.float64), [0.0]))
    y_full = np.concatenate(([0.0], fn_count.astype(np.float64), [float(tp_total)]))
    order = np.argsort(x_full, kind="mergesort")
    area = float(np.trapezoid(y_full[order], x_full[order])) if x_full.size >= 2 else 0.0

    if thresholds.size + 2 > max_points:
        budget = max(int(max_points) - 2, 1)
        keep_indices = np.unique(np.linspace(0, thresholds.size - 1, budget, dtype=np.int64))
        thresholds = thresholds[keep_indices]
        fp_keep = fp_keep[keep_indices]
        fn_count = fn_count[keep_indices]
        kept_count = kept_count[keep_indices]
        dropped_count = dropped_count[keep_indices]

    rows = [
        {
            "threshold": "-inf",
            "fp_count": fp_total,
            "fn_count": 0,
            "kept_count": int(labels.size),
            "dropped_count": 0,
            "tp_total": tp_total,
            "fp_total": fp_total,
        }
    ]
    for threshold, fp, fn, kept, dropped in zip(thresholds, fp_keep, fn_count, kept_count, dropped_count):
        rows.append(
            {
                "threshold": float(threshold),
                "fp_count": int(fp),
                "fn_count": int(fn),
                "kept_count": int(kept),
                "dropped_count": int(dropped),
                "tp_total": tp_total,
                "fp_total": fp_total,
            }
        )
    rows.append(
        {
            "threshold": "inf",
            "fp_count": 0,
            "fn_count": tp_total,
            "kept_count": 0,
            "dropped_count": int(labels.size),
            "tp_total": tp_total,
            "fp_total": fp_total,
        }
    )
    return rows, area


def downsample_curve(rows, max_points):
    if len(rows) <= max_points:
        return rows
    indices = np.unique(np.linspace(0, len(rows) - 1, int(max_points), dtype=np.int64))
    return [rows[int(i)] for i in indices]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_geometry(curves):
    max_x = max(max(row["fp_count"] for row in item["curve"]) for item in curves)
    max_y = max(max(row["fn_count"] for row in item["curve"]) for item in curves)
    return max(max_x, 1), max(max_y, 1)


def write_svg(curves, out_path):
    width = 1180
    height = 720
    left = 92
    top = 44
    right = 900
    bottom = 626
    max_x, max_y = _plot_geometry(curves)
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
        "#65a30d",
        "#7c2d12",
        "#0f766e",
        "#a21caf",
    ]

    def x_map(value):
        return left + (float(value) / float(max_x)) * (right - left)

    def y_map(value):
        return bottom - (float(value) / float(max_y)) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="590" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Meta Classifier TP-Probability Postprocessing</text>',
    ]
    for i in range(6):
        tx = max_x * i / 5.0
        x = x_map(tx)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{int(round(tx))}</text>')
        ty = max_y * i / 5.0
        y = y_map(ty)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{int(round(ty))}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827" stroke-width="1.5"/>')
    parts.append(f'<text x="{(left + right) / 2:.2f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">FP remaining</text>')
    parts.append(f'<text x="24" y="{(top + bottom) / 2:.2f}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827" transform="rotate(-90 24 {(top + bottom) / 2:.2f})">TP dropped / added FN</text>')
    for idx, item in enumerate(curves):
        color = colors[idx % len(colors)]
        plot_rows = downsample_curve(item["curve"], MAX_CURVE_POINTS)
        points = " ".join(f'{x_map(row["fp_count"]):.2f},{y_map(row["fn_count"]):.2f}' for row in plot_rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
    legend_x = right + 28
    legend_y = top + 12
    for idx, item in enumerate(curves):
        y = legend_y + idx * 25
        color = colors[idx % len(colors)]
        label = html.escape(str(item["label"]))
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 38}" y="{y + 4}" font-family="Arial" font-size="12" fill="#111827">{label}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_png_if_available(curves, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_basic_png(curves, out_path)
        print(f"Matplotlib is unavailable ({exc}); wrote a basic PNG fallback.")
        return
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    cmap = plt.get_cmap("tab20")
    for idx, item in enumerate(curves):
        plot_rows = downsample_curve(item["curve"], MAX_CURVE_POINTS)
        x = [row["fp_count"] for row in plot_rows]
        y = [row["fn_count"] for row in plot_rows]
        ax.plot(x, y, linewidth=2.0, color=cmap(idx % 20), label=item["label"])
    ax.set_xlabel("FP remaining")
    ax.set_ylabel("TP dropped / added FN")
    ax.set_title("Meta Classifier TP-Probability Postprocessing")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _draw_line(image, x0, y0, x1, y1, color, thickness=1):
    h, w = image.shape[:2]
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    radius = max(int(thickness) // 2, 0)
    while True:
        for yy in range(max(0, y0 - radius), min(h, y0 + radius + 1)):
            for xx in range(max(0, x0 - radius), min(w, x0 + radius + 1)):
                image[yy, xx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _write_png_rgb(path, image):
    h, w = image.shape[:2]
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(raw, 6)),
        chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(png))


def write_basic_png(curves, out_path):
    width = 1180
    height = 720
    left = 92
    top = 44
    right = 900
    bottom = 626
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    max_x, max_y = _plot_geometry(curves)
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
        "#65a30d",
        "#7c2d12",
        "#0f766e",
        "#a21caf",
    ]

    def x_map(value):
        return left + (float(value) / float(max_x)) * (right - left)

    def y_map(value):
        return bottom - (float(value) / float(max_y)) * (bottom - top)

    grid_color = (229, 231, 235)
    axis_color = (17, 24, 39)
    for i in range(6):
        x = x_map(max_x * i / 5.0)
        y = y_map(max_y * i / 5.0)
        _draw_line(image, x, top, x, bottom, grid_color, thickness=1)
        _draw_line(image, left, y, right, y, grid_color, thickness=1)
    _draw_line(image, left, bottom, right, bottom, axis_color, thickness=2)
    _draw_line(image, left, top, left, bottom, axis_color, thickness=2)
    for idx, item in enumerate(curves):
        color = _hex_to_rgb(colors[idx % len(colors)])
        rows = item["curve"]
        for prev, cur in zip(rows, rows[1:]):
            _draw_line(
                image,
                x_map(prev["fp_count"]),
                y_map(prev["fn_count"]),
                x_map(cur["fp_count"]),
                y_map(cur["fn_count"]),
                color,
                thickness=2,
            )
        legend_y = top + 14 + idx * 18
        _draw_line(image, right + 28, legend_y, right + 62, legend_y, color, thickness=4)
    _write_png_rgb(out_path, image)


def main():
    if not RUNS:
        raise ValueError("Set RUNS at the top of this script before running.")
    out_dir = OUTPUT_ROOT / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_meta_classifier_fp_fn_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_point_rows = []
    summary_rows = []
    curves = []
    for run_index, run in enumerate(RUNS, start=1):
        label = str(run["label"])
        run_root = resolve_path(run["run_root"])
        print(f"[{run_index}/{len(RUNS)}] Loading {label}: {run_root}", flush=True)
        labels, scores, total_rows = read_eval_data(run_root)
        print(
            f"[{run_index}/{len(RUNS)}] Building curve for {label}: "
            f"{labels.size} sampled rows from {total_rows} total rows",
            flush=True,
        )
        curve, area = build_curve(labels, scores)
        curves.append({"label": label, "curve": curve})
        for row in curve:
            all_point_rows.append({"label": label, **row})
        summary_rows.append(
            {
                "label": label,
                "run_root": str(run_root),
                "num_eval_rows": int(labels.size),
                "source_eval_rows": int(total_rows),
                "sample_size": int(SAMPLE_SIZE) if SAMPLE_SIZE else "",
                "sample_seed": int(SAMPLE_SEED) if SAMPLE_SIZE else "",
                "tp_total": int(np.sum(labels == 1)),
                "fp_total": int(np.sum(labels == 0)),
                "auc_fp_fn": area,
            }
        )
    write_csv(
        out_dir / "fp_fn_curve_points.csv",
        all_point_rows,
        ["label", "threshold", "fp_count", "fn_count", "kept_count", "dropped_count", "tp_total", "fp_total"],
    )
    write_csv(
        out_dir / "run_summary.csv",
        summary_rows,
        [
            "label",
            "run_root",
            "num_eval_rows",
            "source_eval_rows",
            "sample_size",
            "sample_seed",
            "tp_total",
            "fp_total",
            "auc_fp_fn",
        ],
    )
    write_svg(curves, out_dir / "fp_fn_curve.svg")
    write_png_if_available(curves, out_dir / "fp_fn_curve.png")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
