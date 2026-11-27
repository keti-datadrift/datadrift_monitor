import csv
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / r"visualizers/runs/06-25-2026_21;17_fcos_nonlocal_error_examples"
IMAGE_DIR = ROOT / "images"
DETECTION_CSV = ROOT / "candidate_detections.csv"
OUTPUT_DIR = ROOT / "images_styled"
MIN_SIZE = 800
MAX_SIZE = 1333
BOX_ALPHA = 0.86
LABEL_ALPHA = 0.80
MAX_PER_GROUP = 8


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def score_color_rgb(value):
    t = clamp(float(value), 0.0, 1.0)
    red = (255, 0, 0)
    green = (0, 200, 0)
    return tuple(int(red[i] * (1.0 - t) + green[i] * t) for i in range(3))


def load_font(size=14):
    candidates = [
        Path(r"C:/Windows/Fonts/arial.ttf"),
        Path(r"C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT = load_font(14)


def fcos_resize_size(width, height, min_size=MIN_SIZE, max_size=MAX_SIZE):
    min_original = min(width, height)
    max_original = max(width, height)
    size = min_size
    if max_original / min_original * size > max_size:
        size = int(round(max_size * min_original / max_original))
    if (width <= height and width == size) or (height <= width and height == size):
        return width, height
    if width < height:
        ow = size
        oh = int(size * height / width)
    else:
        oh = size
        ow = int(size * width / height)
    return ow, oh


def alpha_line(image, points, fill, width=3, alpha=BOX_ALPHA):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgba = (*fill, int(255 * alpha))
    draw.line(points, fill=rgba, width=width, joint="curve")
    image.alpha_composite(overlay)


def alpha_rect(image, box, fill, alpha=LABEL_ALPHA):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(box, fill=(*fill, int(255 * alpha)))
    image.alpha_composite(overlay)


def text_bbox(text):
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    return draw.textbbox((0, 0), text, font=FONT)


def draw_label(image, x, y, text, color):
    draw = ImageDraw.Draw(image)
    bbox = text_bbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 4
    x = int(clamp(x, 0, image.width - tw - pad * 2 - 1))
    y = int(clamp(y, th + pad * 2, image.height - 2))
    alpha_rect(image, (x, y - th - pad * 2, x + tw + pad * 2, y), color, LABEL_ALPHA)
    draw.text((x + pad, y - th - pad - 1), text, fill=(255, 255, 255, 255), font=FONT)


def clip_box(row, width, height):
    x1 = int(round(clamp(float(row["xmin"]), 0, width - 1)))
    y1 = int(round(clamp(float(row["ymin"]), 0, height - 1)))
    x2 = int(round(clamp(float(row["xmax"]), 0, width - 1)))
    y2 = int(round(clamp(float(row["ymax"]), 0, height - 1)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def load_rows(image_ids):
    image_ids = set(image_ids)
    df = pd.read_csv(DETECTION_CSV)
    df = df[df["image_id"].isin([int(x) for x in image_ids])].copy()
    if df.empty:
        return {image_id: [] for image_id in image_ids}
    order = {"tp_keep": 0, "classification_filter": 1, "background_filter": 2, "localization_filter": 3}
    df["selection_order"] = df["selection_type"].map(order).fillna(9)
    df["gap"] = (df["unto_o_tp_prob"] - df["score"]).abs()
    df = df.sort_values(["image_id", "selection_order", "gap"], ascending=[True, True, False])
    df = df.groupby(["image_id", "selection_type"], group_keys=False).head(MAX_PER_GROUP)
    rows = {image_id: [] for image_id in image_ids}
    for row in df.drop(columns=["selection_order", "gap"]).to_dict("records"):
        rows[str(int(row["image_id"]))].append(row)
    return rows


def draw_rows(base, rows, value_key):
    image = base.copy().convert("RGBA")
    for row in rows:
        value = float(row[value_key])
        color = score_color_rgb(value)
        x1, y1, x2, y2 = clip_box(row, image.width, image.height)
        alpha_line(image, [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], color, width=3, alpha=BOX_ALPHA)
        draw_label(image, x1, max(22, y1 - 3), f"{row['pred_class']} {value:.2f}", color)
    return image.convert("RGB")


def make_contact_sheet(paths, out_path):
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((320, 320), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (320, 320), (128, 128, 128))
        canvas.paste(image, ((320 - image.width) // 2, (320 - image.height) // 2))
        thumbs.append(canvas)
    if not thumbs:
        return
    while len(thumbs) % 3:
        thumbs.append(Image.new("RGB", (320, 320), (240, 240, 240)))
    sheet = Image.new("RGB", (960, 320 * (len(thumbs) // 3)), (128, 128, 128))
    for idx, thumb in enumerate(thumbs):
        x = (idx % 3) * 320
        y = (idx // 3) * 320
        sheet.paste(thumb, (x, y))
    sheet.save(out_path, quality=95)


def write_selected_csv(path, rows_by_image):
    fieldnames = [
        "image_id", "selection_type", "focus_error_type", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "pred_class", "score", "unto_o_tp_prob",
        "tp", "gt_iou", "max_iou", "error_type", "unto_o_eval_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_id in sorted(rows_by_image, key=lambda x: int(x)):
            for row in rows_by_image[image_id]:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    image_ids = [str(int(path.stem)) for path in image_paths if path.stem.isdigit()]
    rows_by_image = load_rows(image_ids)
    score_dir = OUTPUT_DIR / "score"
    unto_dir = OUTPUT_DIR / "unto_o_tp_prob"
    sheet_dir = OUTPUT_DIR / "contact_sheets"
    for path in (score_dir, unto_dir, sheet_dir):
        path.mkdir(parents=True, exist_ok=True)
    score_paths = []
    unto_paths = []
    rendered = 0
    for src in image_paths:
        image_id = str(int(src.stem))
        rows = rows_by_image.get(image_id, [])
        if not rows:
            print(f"[WARN] no rows for image_id={image_id}")
            continue
        original = Image.open(src).convert("RGB")
        resized_size = fcos_resize_size(*original.size)
        base = original.resize(resized_size, Image.Resampling.BILINEAR)
        score_out = score_dir / f"{int(image_id):012d}_score.jpg"
        unto_out = unto_dir / f"{int(image_id):012d}_unto_o_tp_prob.jpg"
        draw_rows(base, rows, "score").save(score_out, quality=95)
        draw_rows(base, rows, "unto_o_tp_prob").save(unto_out, quality=95)
        score_paths.append(score_out)
        unto_paths.append(unto_out)
        rendered += 1
    write_selected_csv(OUTPUT_DIR / "selected_detections.csv", rows_by_image)
    make_contact_sheet(score_paths, sheet_dir / "score_contact.jpg")
    make_contact_sheet(unto_paths, sheet_dir / "unto_o_tp_prob_contact.jpg")
    print(f"rendered images: {rendered}")
    print(f"saved outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
