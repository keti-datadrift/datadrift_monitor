import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / r"visualizers/runs/06-25-2026_21;00_nonlocal_error_examples"
IMAGE_DIR = ROOT / "images"
DETECTION_CSV = ROOT / "candidate_detections.csv"
OUTPUT_DIR = ROOT / "images_styled"
TP_CSV = REPO_ROOT / r"object_detectors/runs/yolov5/predict/coco/06-15-2026_18;54_gt/tp.csv"
UNTO_O_RESULTS = REPO_ROOT / r"meta_models/runs/meta_classifier/yolov5/train/coco/06-17-2026_00;42_null_detect/results"
CANVAS_SIZE = 640
MAX_PER_GROUP = 8
BG_IOU_MAX = 0.05


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def score_color(value):
    t = clamp(float(value), 0.0, 1.0)
    red = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 200, 0], dtype=np.float32)
    color = red * (1.0 - t) + green * t
    return tuple(int(v) for v in color)


def letterbox(image, size=CANVAS_SIZE, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    left = int(round((size - new_w) / 2 - 0.1))
    top = int(round((size - new_h) / 2 - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def text_size(text, scale=0.42, thickness=1):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h, baseline


def draw_label(image, x, y, text, color, alpha=0.80, scale=0.42, text_color=(255, 255, 255), pad=4):
    h_img, w_img = image.shape[:2]
    tw, th, base = text_size(text, scale)
    x = int(clamp(x, 0, w_img - tw - pad * 2 - 1))
    y = int(clamp(y, th + pad * 2, h_img - 2))
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    cv2.putText(
        image,
        text,
        (x + pad, y - pad - base // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        1,
        cv2.LINE_AA,
    )


def clip_box(row):
    x1 = int(round(clamp(float(row["xmin"]), 0, CANVAS_SIZE - 1)))
    y1 = int(round(clamp(float(row["ymin"]), 0, CANVAS_SIZE - 1)))
    x2 = int(round(clamp(float(row["xmax"]), 0, CANVAS_SIZE - 1)))
    y2 = int(round(clamp(float(row["ymax"]), 0, CANVAS_SIZE - 1)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def available_image_ids():
    ids = []
    for path in sorted(IMAGE_DIR.glob("*.jpg")):
        if path.stem.isdigit():
            ids.append(str(int(path.stem)))
    if not ids:
        raise RuntimeError(f"No input images found in {IMAGE_DIR}")
    return ids


def prediction_key(row):
    return str(int(row["image_id"])), Path(str(row["image_path"])).name, str(int(row["raw_pred_idx"]))


def focus_error_type(row):
    if int(row["tp"]) == 1:
        return "tp"
    if row["error_type"] == "classification_error":
        return "classification_error"
    if float(row["max_iou"]) <= BG_IOU_MAX:
        return "background_error"
    return "localization_error"


def load_unto_o_probabilities(image_ids):
    image_ids = set(image_ids)
    sums = defaultdict(float)
    counts = defaultdict(int)
    for path in sorted(UNTO_O_RESULTS.glob("eval_data_*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image_id = str(int(row["image_id"]))
                if image_id not in image_ids:
                    continue
                key = prediction_key(row)
                sums[key] += float(row["y_pred"])
                counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}, dict(counts)


def select_nonlocal_rows(rows):
    tp_keep = [
        row for row in rows
        if row["tp"] == 1 and row["unto_o_tp_prob"] - row["score"] >= 0.20 and row["unto_o_tp_prob"] >= 0.65
    ]
    classification = [
        row for row in rows
        if row["focus_error_type"] == "classification_error"
        and row["score"] - row["unto_o_tp_prob"] >= 0.20
        and row["unto_o_tp_prob"] <= 0.35
    ]
    background = [
        row for row in rows
        if row["focus_error_type"] == "background_error"
        and row["score"] - row["unto_o_tp_prob"] >= 0.20
        and row["unto_o_tp_prob"] <= 0.35
    ]
    tp_keep.sort(key=lambda row: (row["unto_o_tp_prob"] - row["score"], row["unto_o_tp_prob"]), reverse=True)
    classification.sort(key=lambda row: (row["score"] - row["unto_o_tp_prob"], row["score"]), reverse=True)
    background.sort(key=lambda row: (row["score"] - row["unto_o_tp_prob"], row["score"]), reverse=True)
    selected = []
    for selection_type, group in (
        ("tp_keep", tp_keep),
        ("classification_filter", classification),
        ("background_filter", background),
    ):
        for row in group[:MAX_PER_GROUP]:
            row = dict(row)
            row["selection_type"] = selection_type
            selected.append(row)
    selected.sort(key=lambda row: (row["selection_type"], -abs(row["unto_o_tp_prob"] - row["score"])))
    return selected


def supplement_missing_rows(rows_by_image):
    missing_ids = [image_id for image_id, rows in rows_by_image.items() if not rows]
    if not missing_ids:
        return rows_by_image
    probabilities, eval_counts = load_unto_o_probabilities(missing_ids)
    collected = {image_id: [] for image_id in missing_ids}
    with open(TP_CSV, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = str(int(row["image_id"]))
            if image_id not in collected:
                continue
            key = prediction_key(row)
            if key not in probabilities:
                continue
            row = dict(row)
            row["image_id"] = image_id
            row["pred_idx"] = int(row["pred_idx"])
            row["raw_pred_idx"] = int(row["raw_pred_idx"])
            row["score"] = float(row["score"])
            row["tp"] = int(row["tp"])
            row["gt_iou"] = float(row["gt_iou"])
            row["max_iou"] = float(row["max_iou"])
            row["unto_o_tp_prob"] = float(probabilities[key])
            row["unto_o_eval_count"] = int(eval_counts[key])
            row["focus_error_type"] = focus_error_type(row)
            collected[image_id].append(row)
    for image_id, rows in collected.items():
        rows_by_image[image_id] = select_nonlocal_rows(rows)
    return rows_by_image


def load_selected_rows(image_ids):
    image_ids = set(image_ids)
    rows_by_image = {image_id: [] for image_id in image_ids}
    with open(DETECTION_CSV, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = str(int(row["image_id"]))
            if image_id not in image_ids:
                continue
            row = dict(row)
            row["image_id"] = image_id
            row["pred_idx"] = int(row["pred_idx"])
            row["raw_pred_idx"] = int(row["raw_pred_idx"])
            row["score"] = float(row["score"])
            row["unto_o_tp_prob"] = float(row["unto_o_tp_prob"])
            row["tp"] = int(row["tp"])
            row["max_iou"] = float(row["max_iou"])
            row["gt_iou"] = float(row["gt_iou"])
            rows_by_image[image_id].append(row)
    for rows in rows_by_image.values():
        rows.sort(key=lambda row: (row["selection_type"], -abs(row["unto_o_tp_prob"] - row["score"])))
    return supplement_missing_rows(rows_by_image)


def draw_rows(base, rows, value_key):
    image = base.copy()
    for row in rows:
        value = row[value_key]
        color = score_color(value)
        x1, y1, x2, y2 = clip_box(row)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0, dst=image)
        label = f"{row['pred_class']} {value:.2f}"
        draw_label(image, x1, max(20, y1 - 3), label, color, alpha=0.80, scale=0.42)
    return image


def write_selected_csv(path, selected_by_image):
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
        for image_id in sorted(selected_by_image, key=lambda x: int(x)):
            for row in selected_by_image[image_id]:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_contact_sheet(paths, out_path):
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA))
    if not images:
        return
    while len(images) % 3:
        images.append(np.full_like(images[0], 240))
    rows = []
    for idx in range(0, len(images), 3):
        rows.append(cv2.hconcat(images[idx : idx + 3]))
    cv2.imwrite(str(out_path), cv2.vconcat(rows))


def main():
    image_ids = available_image_ids()
    selected_by_image = load_selected_rows(image_ids)
    score_dir = OUTPUT_DIR / "score"
    unto_dir = OUTPUT_DIR / "unto_o_tp_prob"
    sheet_dir = OUTPUT_DIR / "contact_sheets"
    for path in (score_dir, unto_dir, sheet_dir):
        path.mkdir(parents=True, exist_ok=True)

    score_paths = []
    unto_paths = []
    rendered = 0
    for image_id in image_ids:
        rows = selected_by_image.get(image_id, [])
        if not rows:
            print(f"[WARN] no selected detections for image_id={image_id}")
            continue
        image_path = IMAGE_DIR / f"{int(image_id):012d}.jpg"
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        base = letterbox(original, CANVAS_SIZE)
        score_out = score_dir / f"{int(image_id):012d}_score.jpg"
        unto_out = unto_dir / f"{int(image_id):012d}_unto_o_tp_prob.jpg"
        cv2.imwrite(str(score_out), draw_rows(base, rows, "score"))
        cv2.imwrite(str(unto_out), draw_rows(base, rows, "unto_o_tp_prob"))
        score_paths.append(score_out)
        unto_paths.append(unto_out)
        rendered += 1

    write_selected_csv(OUTPUT_DIR / "selected_detections.csv", selected_by_image)
    make_contact_sheet(score_paths, sheet_dir / "score_contact.jpg")
    make_contact_sheet(unto_paths, sheet_dir / "unto_o_tp_prob_contact.jpg")
    print(f"rendered images: {rendered}")
    print(f"saved outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
