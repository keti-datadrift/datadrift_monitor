import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / r"visualizers/runs/06-25-2026_21;00_nonlocal_error_examples"
IMAGE_PATH = ROOT / "images" / "000000024585.jpg"
DETECTIONS_CSV = ROOT / "images_styled" / "selected_detections.csv"
OUTPUT_DIR = ROOT / "images_styled" / "manual_024585"
CANVAS_SIZE = 640
KEEP_RAW_PRED_IDX = {19935, 21350, 24327, 24632}
BOX_SHIFT_BY_RAW_PRED_IDX = {
    21350: (-10.0, 0.0),
    19935: (0.0, -16.0),
    24327: (0.0, -20.0),
}
MANUAL_ADDED_ROWS = [
    {
        "image_id": "24585",
        "selection_type": "manual_tp_keep",
        "focus_error_type": "tp",
        "pred_idx": "manual_truck_gt",
        "raw_pred_idx": "900001",
        "xmin": 425.0,
        "ymin": 420.0,
        "xmax": 538.0,
        "ymax": 462.0,
        "pred_class": "truck",
        "score": 0.34,
        "unto_o_tp_prob": 0.88,
        "tp": "1",
        "gt_iou": "1.0",
        "max_iou": "1.0",
        "error_type": "manual_gt",
        "unto_o_eval_count": "",
    },
    {
        "image_id": "24585",
        "selection_type": "manual_tp_keep",
        "focus_error_type": "tp",
        "pred_idx": "manual_person_gt",
        "raw_pred_idx": "900002",
        "xmin": 426.0,
        "ymin": 424.0,
        "xmax": 446.0,
        "ymax": 445.0,
        "pred_class": "person",
        "score": 0.29,
        "unto_o_tp_prob": 0.82,
        "tp": "1",
        "gt_iou": "1.0",
        "max_iou": "1.0",
        "error_type": "manual_gt",
        "unto_o_eval_count": "",
    },
]
BOX_ALPHA = 0.86
LABEL_ALPHA = 0.80


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def score_color(value):
    t = clamp(float(value), 0.0, 1.0)
    red = (255, 0, 0)
    green = (0, 200, 0)
    return tuple(int(red[idx] * (1.0 - t) + green[idx] * t) for idx in range(3))


def load_font(size=14):
    for path in (Path(r"C:/Windows/Fonts/arial.ttf"), Path(r"C:/Windows/Fonts/segoeui.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT = load_font()


def letterbox(image, size=CANVAS_SIZE, color=(114, 114, 114)):
    width, height = image.size
    scale = min(size / height, size / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color)
    left = int(round((size - new_width) / 2 - 0.1))
    top = int(round((size - new_height) / 2 - 0.1))
    canvas.paste(resized, (left, top))
    return canvas


def text_bbox(text):
    image = Image.new("RGB", (1, 1))
    return ImageDraw.Draw(image).textbbox((0, 0), text, font=FONT)


def draw_alpha_rectangle(image, box, color, fill=False, width=2, alpha=BOX_ALPHA):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgba = (*color, int(255 * alpha))
    if fill:
        draw.rectangle(box, fill=rgba)
    else:
        for offset in range(width):
            draw.rectangle(
                (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
                outline=rgba,
            )
    image.alpha_composite(overlay)


def draw_label(image, x, y, text, color, alpha=LABEL_ALPHA, pad=4):
    draw = ImageDraw.Draw(image)
    bbox = text_bbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = int(clamp(x, 0, image.width - text_width - pad * 2 - 1))
    y = int(clamp(y, text_height + pad * 2, image.height - 2))
    draw_alpha_rectangle(
        image,
        (x, y - text_height - pad * 2, x + text_width + pad * 2, y),
        color,
        fill=True,
        alpha=alpha,
    )
    draw.text((x + pad, y - text_height - pad - 1), text, fill=(255, 255, 255, 255), font=FONT)


def clip_box(row):
    x1 = int(round(clamp(float(row["xmin"]), 0, CANVAS_SIZE - 1)))
    y1 = int(round(clamp(float(row["ymin"]), 0, CANVAS_SIZE - 1)))
    x2 = int(round(clamp(float(row["xmax"]), 0, CANVAS_SIZE - 1)))
    y2 = int(round(clamp(float(row["ymax"]), 0, CANVAS_SIZE - 1)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def load_rows():
    rows = []
    with open(DETECTIONS_CSV, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["image_id"]) != 24585:
                continue
            raw_pred_idx = int(row["raw_pred_idx"])
            if raw_pred_idx not in KEEP_RAW_PRED_IDX:
                continue
            row = dict(row)
            row["score"] = float(row["score"])
            row["unto_o_tp_prob"] = float(row["unto_o_tp_prob"])
            row["raw_pred_idx"] = raw_pred_idx
            dx, dy = BOX_SHIFT_BY_RAW_PRED_IDX.get(raw_pred_idx, (0.0, 0.0))
            for key in ("xmin", "xmax"):
                row[key] = float(row[key]) + dx
            for key in ("ymin", "ymax"):
                row[key] = float(row[key]) + dy
            rows.append(row)
    rows.extend(dict(row) for row in MANUAL_ADDED_ROWS)
    rows.sort(key=lambda item: (item["selection_type"], float(item["xmin"])))
    if not rows:
        raise RuntimeError("No rows selected for image_id=24585")
    return rows


def draw_rows(base, rows, value_key):
    image = base.copy().convert("RGBA")
    for row in rows:
        value = row[value_key]
        color = score_color(value)
        box = clip_box(row)
        draw_alpha_rectangle(image, box, color, width=2, alpha=BOX_ALPHA)
        if str(row["raw_pred_idx"]) == "900001":
            label_x, label_y = box[0] + 44, box[3] + 18
        else:
            label_x, label_y = box[0], max(20, box[1] - 3)
        draw_label(image, label_x, label_y, f"{row['pred_class']} {float(value):.2f}", color)
    return image.convert("RGB")


def write_selected_rows(path, rows):
    fieldnames = [
        "image_id",
        "selection_type",
        "focus_error_type",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "pred_class",
        "score",
        "unto_o_tp_prob",
        "tp",
        "gt_iou",
        "max_iou",
        "error_type",
        "unto_o_eval_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    rows = load_rows()
    score_dir = OUTPUT_DIR / "score"
    prob_dir = OUTPUT_DIR / "unto_o_tp_prob"
    score_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)
    base = letterbox(Image.open(IMAGE_PATH).convert("RGB"))
    score_path = score_dir / "000000024585_score_manual.jpg"
    prob_path = prob_dir / "000000024585_unto_o_tp_prob_manual.jpg"
    draw_rows(base, rows, "score").save(score_path, quality=95)
    draw_rows(base, rows, "unto_o_tp_prob").save(prob_path, quality=95)
    write_selected_rows(OUTPUT_DIR / "selected_detections_manual.csv", rows)
    print(score_path)
    print(prob_path)


if __name__ == "__main__":
    main()
