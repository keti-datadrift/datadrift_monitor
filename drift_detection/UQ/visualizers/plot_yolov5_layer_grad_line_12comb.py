import csv
import html
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

GRID_RESULTS = r"meta_models/meta_classifier/runs/yolov5/train/coco/05-27-2026_13;17_layer_grad_grid/grid_results.csv"

BBOX_LOSS_ORDER = ["box_l1", "box_l2", "l1", "l2"]
CLS_OPTION_ORDER = [
    ("bcewithlogits", "pred_to_target"),
    ("kl", "pred_to_target"),
    ("kl", "target_to_pred"),
]
TERM_OPTION_ORDER = [
    ("bcewithlogits", "pred_to_target"),
    ("abs_diff", "pred_to_target"),
]

WIDTH = 1100
HEIGHT = 470
PLOT_LEFT = 84
PLOT_TOP = 36
PLOT_RIGHT = 1064
PLOT_BOTTOM = 394
Y_MIN = 0.89
Y_MAX = 0.95
Y_TICK_STEP = 0.01


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_grid_results(path):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing grid results CSV: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_schema(rows):
    if not rows:
        raise ValueError("grid_results.csv is empty.")
    if "obj_loss" in rows[0] and "obj_direction" in rows[0]:
        return {
            "model_name": "yolov5",
            "term_loss_col": "obj_loss",
            "term_direction_col": "obj_direction",
            "term_label": "obj",
        }
    if "cnt_loss" in rows[0] and "cnt_direction" in rows[0]:
        return {
            "model_name": "fcos",
            "term_loss_col": "cnt_loss",
            "term_direction_col": "cnt_direction",
            "term_label": "cnt",
        }
    raise ValueError("grid_results.csv must contain obj_loss/obj_direction or cnt_loss/cnt_direction columns.")


def ordered_present(values, order):
    value_set = set(values)
    return [value for value in order if value in value_set]


def ordered_option_present(rows, loss_col, direction_col, order):
    present = {(row[loss_col], row[direction_col]) for row in rows}
    return [option for option in order if option in present]


def row_key(row, schema):
    return (
        row["target"],
        row["bbox_loss"],
        row["bbox_direction"],
        row["cls_loss"],
        row["cls_direction"],
        row[schema["term_loss_col"]],
        row[schema["term_direction_col"]],
    )


def build_combinations(rows, schema):
    target_rows = [row for row in rows if row.get("target") in {"cand_target", "null_target"}]
    bbox_losses = ordered_present([row["bbox_loss"] for row in target_rows], BBOX_LOSS_ORDER)
    cls_options = ordered_option_present(target_rows, "cls_loss", "cls_direction", CLS_OPTION_ORDER)
    term_options = ordered_option_present(
        target_rows,
        schema["term_loss_col"],
        schema["term_direction_col"],
        TERM_OPTION_ORDER,
    )
    if not bbox_losses or not cls_options or not term_options:
        raise ValueError("Could not infer the 12 loss combinations from grid_results.csv.")
    combinations = []
    for bbox_loss in bbox_losses:
        for cls_loss, cls_direction in cls_options:
            for term_loss, term_direction in term_options:
                combinations.append(
                    {
                        "bbox_loss": bbox_loss,
                        "bbox_direction": "pred_to_target",
                        "cls_loss": cls_loss,
                        "cls_direction": cls_direction,
                        schema["term_loss_col"]: term_loss,
                        schema["term_direction_col"]: term_direction,
                    }
                )
    if len(combinations) != 12:
        raise ValueError(f"Expected 12 combinations, found {len(combinations)}.")
    return combinations


def collect_points(rows, schema):
    rows_by_key = {row_key(row, schema): row for row in rows}
    cand_values = []
    null_values = []
    records = []
    for idx, combo in enumerate(build_combinations(rows, schema), start=1):
        base = (
            combo["bbox_loss"],
            combo["bbox_direction"],
            combo["cls_loss"],
            combo["cls_direction"],
            combo[schema["term_loss_col"]],
            combo[schema["term_direction_col"]],
        )
        cand_key = ("cand_target",) + base
        null_key = ("null_target",) + base
        if cand_key not in rows_by_key:
            raise KeyError(f"Missing cand_target row: {cand_key}")
        if null_key not in rows_by_key:
            raise KeyError(f"Missing null_target row: {null_key}")
        cand = float(rows_by_key[cand_key]["auroc"])
        null = float(rows_by_key[null_key]["auroc"])
        cand_values.append(cand)
        null_values.append(null)
        records.append({"idx": idx, "cand_auroc": cand, "null_auroc": null, **combo})
    return cand_values, null_values, records


def x_map(index, count):
    if count <= 1:
        return (PLOT_LEFT + PLOT_RIGHT) / 2
    return PLOT_LEFT + ((index - 1) / (count - 1)) * (PLOT_RIGHT - PLOT_LEFT)


def resolve_y_axis(values):
    min_value = min(values)
    max_value = max(values)
    if Y_MIN <= min_value and max_value <= Y_MAX:
        y_min = Y_MIN
        y_max = Y_MAX
    else:
        y_min = (int((min_value - Y_TICK_STEP) / Y_TICK_STEP) * Y_TICK_STEP)
        y_max = (int((max_value + Y_TICK_STEP * 1.999) / Y_TICK_STEP) * Y_TICK_STEP)
    ticks = []
    tick = y_min
    while tick <= y_max + 1e-9:
        ticks.append(round(tick, 2))
        tick += Y_TICK_STEP
    return y_min, y_max, ticks


def y_map_for_axis(value, y_min, y_max):
    return PLOT_BOTTOM - ((float(value) - y_min) / (y_max - y_min)) * (PLOT_BOTTOM - PLOT_TOP)


def svg_text(x, y, text, size=12, fill="#374151", anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial" '
        f'font-size="{size}" fill="{fill}"{extra}>{html.escape(str(text))}</text>'
    )


def path_d(xs, ys):
    parts = [f"M {xs[0]:.1f} {ys[0]:.1f}"]
    parts.extend(f"L {x:.1f} {y:.1f}" for x, y in zip(xs[1:], ys[1:]))
    return " ".join(parts)


def draw_svg(cand_values, null_values, output_path):
    count = len(cand_values)
    y_min, y_max, y_ticks = resolve_y_axis(cand_values + null_values)
    xs = [x_map(i, count) for i in range(1, count + 1)]
    cand_ys = [y_map_for_axis(value, y_min, y_max) for value in cand_values]
    null_ys = [y_map_for_axis(value, y_min, y_max) for value in null_values]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
    ]

    for tick in y_ticks:
        y = y_map_for_axis(tick, y_min, y_max)
        lines.append(f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_RIGHT}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(svg_text(72, y + 4, f"{tick:.2f}", fill="#6b7280", anchor="end"))

    lines.append(f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(f'<line x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(svg_text(22, 215.0, "AUROC", size=14, fill="#111827", anchor="middle", extra=' transform="rotate(-90 22 215.0)"'))

    for idx, x in enumerate(xs, start=1):
        lines.append(f'<line x1="{x:.1f}" y1="{PLOT_BOTTOM}" x2="{x:.1f}" y2="{PLOT_BOTTOM + 5}" stroke="#111827" stroke-width="1"/>')
        lines.append(svg_text(f"{x:.1f}", 417, idx, fill="#374151", anchor="middle"))
    lines.append(svg_text(574.0, 452, "12 loss combinations", size=14, fill="#111827", anchor="middle"))

    for i in range(count - 1):
        points = (
            f"{xs[i]:.1f},{cand_ys[i]:.1f} "
            f"{xs[i + 1]:.1f},{cand_ys[i + 1]:.1f} "
            f"{xs[i + 1]:.1f},{null_ys[i + 1]:.1f} "
            f"{xs[i]:.1f},{null_ys[i]:.1f}"
        )
        lines.append(f'<polygon points="{points}" fill="#fee2e2" opacity="0.95"/>')

    lines.append(f'<path d="{path_d(xs, cand_ys)}" fill="none" stroke="#334155" stroke-width="2.5"/>')
    lines.append(f'<path d="{path_d(xs, null_ys)}" fill="none" stroke="#dc2626" stroke-width="3.6"/>')

    for x, y in zip(xs, cand_ys):
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="#334155" stroke="white" stroke-width="1"/>')
    for x, y in zip(xs, null_ys):
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" fill="#dc2626" stroke="white" stroke-width="1"/>')

    lines.append('<line x1="850" y1="34" x2="880" y2="34" stroke="#334155" stroke-width="3"/><circle cx="865" cy="34" r="4" fill="#334155"/>')
    lines.append(svg_text(892, 38, "GradScore", size=14, fill="#111827"))
    lines.append('<line x1="980" y1="34" x2="1010" y2="34" stroke="#dc2626" stroke-width="3.5"/><circle cx="995" cy="34" r="4" fill="#dc2626"/>')
    lines.append(svg_text(1022, 38, "Ours", size=14, fill="#111827"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_records(records, output_path, schema):
    fieldnames = [
        "idx",
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        schema["term_loss_col"],
        schema["term_direction_col"],
        "cand_auroc",
        "null_auroc",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def make_output_dir(model_name):
    timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
    path = OUTPUT_ROOT / f"{timestamp}_{model_name}_layer_grad_line_style_final_12comb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    grid_results = sys.argv[1] if len(sys.argv) > 1 else GRID_RESULTS
    rows = read_grid_results(grid_results)
    schema = infer_schema(rows)
    cand_values, null_values, records = collect_points(rows, schema)
    output_dir = make_output_dir(schema["model_name"])
    draw_svg(cand_values, null_values, output_dir / "style_2_clean_12comb_no_signed_obj.svg")
    write_records(records, output_dir / "style_2_clean_12comb_no_signed_obj_points.csv", schema)
    print(f"Saved layer-grad line plot outputs: {output_dir}")


if __name__ == "__main__":
    main()
