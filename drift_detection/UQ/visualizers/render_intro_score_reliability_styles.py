import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent / "runs" / "06-25-2026_17;06_intro_score_reliability_examples"
IMAGE_DIR = ROOT / "images"
DETECTION_CSV = ROOT / "candidate_detections.csv"
OUTPUT_DIR = ROOT / "images_styled"
CANVAS_SIZE = 640
UNTO_O_RUN_ROOT = Path(
    r"meta_models/runs/meta_classifier/yolov5/train/coco/06-17-2026_00;42_null_detect"
)

CASE_ORDER = [
    "high_score_tp",
    "high_score_fp",
    "high_score_bad_localization",
    "low_score_tp",
]

CASE_LABELS = {
    "high_score_tp": "High-score TP",
    "high_score_fp": "High-score FP",
    "high_score_bad_localization": "High-score poor loc.",
    "low_score_tp": "Low-score TP",
}

CASE_SHORT = {
    "high_score_tp": "TP high",
    "high_score_fp": "FP high",
    "high_score_bad_localization": "Poor loc.",
    "low_score_tp": "TP low",
}

CASE_COLORS = {
    "high_score_tp": (82, 196, 26),
    "high_score_fp": (68, 68, 239),
    "high_score_bad_localization": (11, 158, 245),
    "low_score_tp": (246, 130, 59),
}

STYLE_DESCRIPTIONS = {
    "style_1_clean_paper": "Thin colored boxes, compact class/score tags, white legend.",
    "style_1_score_gradient": "Style 1 without category legend, colored only by detection score from red to green.",
    "style_1_unto_o_tp_prob": "Style 1 without category legend, colored and labeled by UnTO-O TP probability.",
    "style_2_dark_focus": "Dimmed image with high-contrast tags for slide-style emphasis.",
    "style_3_score_cards": "Score-first labels that highlight the detection score as the unreliable signal.",
    "style_4_callout": "External callout labels with leader lines to reduce box-label overlap.",
    "style_5_minimal": "Minimal boxes and tiny labels for a less cluttered paper figure.",
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def read_rows():
    rows_by_image = defaultdict(list)
    with open(DETECTION_CSV, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = int(row["image_id"])
            row["image_id"] = image_id
            row["pred_idx"] = int(row["pred_idx"])
            row["score"] = float(row["score"])
            row["gt_iou"] = float(row["gt_iou"])
            row["xmin"] = float(row["xmin"])
            row["ymin"] = float(row["ymin"])
            row["xmax"] = float(row["xmax"])
            row["ymax"] = float(row["ymax"])
            row["tp"] = int(row["tp"])
            rows_by_image[image_id].append(row)
    for rows in rows_by_image.values():
        rows.sort(key=lambda r: (CASE_ORDER.index(r["case_type"]), -r["score"], -r["gt_iou"]))
    return rows_by_image


def prediction_key(row):
    return (str(row["image_id"]), Path(str(row["image_path"])).name, str(row["raw_pred_idx"]))


def attach_unto_o_tp_prob(rows_by_image):
    wanted = {}
    for rows in rows_by_image.values():
        for row in rows:
            wanted[prediction_key(row)] = row
            row["unto_o_tp_prob_values"] = []

    results_dir = UNTO_O_RUN_ROOT / "results"
    eval_files = sorted(results_dir.glob("eval_data_*.csv"))
    if not eval_files:
        raise FileNotFoundError(f"No eval_data_*.csv files found under {results_dir}")

    for path in eval_files:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["image_id"]), Path(str(row["image_path"])).name, str(row["raw_pred_idx"]))
                target = wanted.get(key)
                if target is not None:
                    target["unto_o_tp_prob_values"].append(float(row["y_pred"]))

    missing = []
    for row in wanted.values():
        values = row["unto_o_tp_prob_values"]
        if not values:
            missing.append(prediction_key(row))
            continue
        row["unto_o_tp_prob"] = float(np.mean(values))
        row["unto_o_eval_count"] = int(len(values))
    if missing:
        sample = "\n".join(str(item) for item in missing[:10])
        raise RuntimeError(f"Missing UnTO-O TP probability for {len(missing)} candidate detections:\n{sample}")


def image_path_for_id(image_id):
    path = IMAGE_DIR / f"{image_id:012d}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def clip_box(row):
    x1 = int(round(clamp(row["xmin"], 0, CANVAS_SIZE - 1)))
    y1 = int(round(clamp(row["ymin"], 0, CANVAS_SIZE - 1)))
    x2 = int(round(clamp(row["xmax"], 0, CANVAS_SIZE - 1)))
    y2 = int(round(clamp(row["ymax"], 0, CANVAS_SIZE - 1)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def text_size(text, scale=0.45, thickness=1):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h, baseline


def draw_filled_label(image, x, y, text, color, scale=0.45, text_color=(255, 255, 255), pad=4):
    h_img, w_img = image.shape[:2]
    tw, th, base = text_size(text, scale)
    x = int(clamp(x, 0, w_img - tw - pad * 2 - 1))
    y = int(clamp(y, th + pad * 2, h_img - 2))
    cv2.rectangle(image, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.putText(image, text, (x + pad, y - pad - base // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


def draw_translucent_label(image, x, y, text, color, alpha=0.68, scale=0.45, text_color=(255, 255, 255), pad=4):
    h_img, w_img = image.shape[:2]
    tw, th, base = text_size(text, scale)
    x = int(clamp(x, 0, w_img - tw - pad * 2 - 1))
    y = int(clamp(y, th + pad * 2, h_img - 2))
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    cv2.putText(image, text, (x + pad, y - pad - base // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


def score_color(score):
    t = clamp((float(score) - 0.25) / 0.70, 0.0, 1.0)
    red = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 200, 0], dtype=np.float32)
    color = red * (1.0 - t) + green * t
    return tuple(int(v) for v in color)


def blend_rect(image, pt1, pt2, color, alpha):
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_legend(image, style="light"):
    x, y = 14, 16
    line_h = 23
    bg = (255, 255, 255) if style == "light" else (28, 28, 28)
    fg = (20, 20, 20) if style == "light" else (245, 245, 245)
    blend_rect(image, (8, 8), (250, 8 + line_h * len(CASE_ORDER) + 11), bg, 0.82)
    for idx, case_type in enumerate(CASE_ORDER):
        yy = y + idx * line_h
        color = CASE_COLORS[case_type]
        cv2.rectangle(image, (x, yy), (x + 15, yy + 15), color, -1)
        cv2.putText(image, CASE_LABELS[case_type], (x + 23, yy + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg, 1, cv2.LINE_AA)


def draw_clean_paper(base, rows):
    image = base.copy()
    draw_legend(image, "light")
    for row in rows:
        x1, y1, x2, y2 = clip_box(row)
        color = CASE_COLORS[row["case_type"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{row['pred_class']} {row['score']:.2f}"
        draw_filled_label(image, x1, max(20, y1 - 3), label, color, scale=0.42)
    return image


def draw_score_gradient(base, rows):
    image = base.copy()
    for row in rows:
        x1, y1, x2, y2 = clip_box(row)
        color = score_color(row["score"])
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0, dst=image)
        label = f"{row['pred_class']} {row['score']:.2f}"
        draw_translucent_label(image, x1, max(20, y1 - 3), label, color, alpha=0.80, scale=0.42)
    return image


def draw_unto_o_tp_prob(base, rows):
    image = base.copy()
    for row in rows:
        x1, y1, x2, y2 = clip_box(row)
        prob = float(row["unto_o_tp_prob"])
        color = score_color(prob)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0, dst=image)
        label = f"{row['pred_class']} {prob:.2f}"
        draw_translucent_label(image, x1, max(20, y1 - 3), label, color, alpha=0.80, scale=0.42)
    return image


def draw_dark_focus(base, rows):
    image = cv2.addWeighted(base, 0.46, np.zeros_like(base), 0.54, 0)
    draw_legend(image, "dark")
    for row in rows:
        x1, y1, x2, y2 = clip_box(row)
        color = CASE_COLORS[row["case_type"]]
        blend_rect(image, (x1, y1), (x2, y2), color, 0.10)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(image, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (255, 255, 255), 1)
        label = f"{CASE_SHORT[row['case_type']]} | {row['score']:.2f}"
        draw_filled_label(image, x1, min(CANVAS_SIZE - 4, y2 + 20), label, color, scale=0.43)
    return image


def draw_score_cards(base, rows):
    image = base.copy()
    blend_rect(image, (0, 0), (CANVAS_SIZE, 44), (255, 255, 255), 0.85)
    cv2.putText(image, "Detection score is not reliability", (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    for row in rows:
        x1, y1, x2, y2 = clip_box(row)
        color = CASE_COLORS[row["case_type"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        score_text = f"{row['score']:.2f}"
        tag_text = CASE_SHORT[row["case_type"]]
        card_w = max(78, text_size(tag_text, 0.36)[0] + 16)
        card_h = 42
        cx = int(clamp(x1, 0, CANVAS_SIZE - card_w - 1))
        cy = int(clamp(y1 - card_h - 4, 48, CANVAS_SIZE - card_h - 1))
        blend_rect(image, (cx, cy), (cx + card_w, cy + card_h), (255, 255, 255), 0.92)
        cv2.rectangle(image, (cx, cy), (cx + card_w, cy + card_h), color, 2)
        cv2.putText(image, score_text, (cx + 8, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        cv2.putText(image, tag_text, (cx + 8, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (45, 45, 45), 1, cv2.LINE_AA)
    return image


def draw_callout(base, rows):
    image = base.copy()
    selected = rows[:]
    left_rows = selected[::2]
    right_rows = selected[1::2]
    y_slots_left = np.linspace(70, 560, max(1, len(left_rows))).astype(int)
    y_slots_right = np.linspace(70, 560, max(1, len(right_rows))).astype(int)
    slot_map = {}
    for row, yy in zip(left_rows, y_slots_left):
        slot_map[id(row)] = (14, int(yy), "left")
    for row, yy in zip(right_rows, y_slots_right):
        slot_map[id(row)] = (430, int(yy), "right")
    for row in selected:
        x1, y1, x2, y2 = clip_box(row)
        color = CASE_COLORS[row["case_type"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        lx, ly, side = slot_map[id(row)]
        label = f"{CASE_SHORT[row['case_type']]}  {row['score']:.2f}"
        tw, th, _ = text_size(label, 0.42)
        card_w = max(178, tw + 20)
        card_h = 28
        target_x = lx + card_w if side == "left" else lx
        cv2.line(image, (cx, cy), (target_x, ly), color, 1, cv2.LINE_AA)
        blend_rect(image, (lx, ly - card_h + 6), (lx + card_w, ly + 6), (255, 255, 255), 0.90)
        cv2.rectangle(image, (lx, ly - card_h + 6), (lx + card_w, ly + 6), color, 2)
        cv2.putText(image, label, (lx + 9, ly - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
    return image


def draw_minimal(base, rows):
    image = base.copy()
    for row in rows:
        x1, y1, x2, y2 = clip_box(row)
        color = CASE_COLORS[row["case_type"]]
        thickness = 2 if row["case_type"] in ("high_score_fp", "high_score_bad_localization") else 1
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{row['score']:.2f}"
        cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return image


STYLE_RENDERERS = {
    "style_1_clean_paper": draw_clean_paper,
    "style_1_score_gradient": draw_score_gradient,
    "style_1_unto_o_tp_prob": draw_unto_o_tp_prob,
    "style_2_dark_focus": draw_dark_focus,
    "style_3_score_cards": draw_score_cards,
    "style_4_callout": draw_callout,
    "style_5_minimal": draw_minimal,
}


def make_contact_sheet(paths, out_path):
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA))
    if not images:
        return
    while len(images) % 3:
        images.append(np.full_like(images[0], 240))
    rows = []
    for i in range(0, len(images), 3):
        rows.append(cv2.hconcat(images[i : i + 3]))
    sheet = cv2.vconcat(rows)
    cv2.imwrite(str(out_path), sheet)


def write_style_guide():
    path = OUTPUT_DIR / "style_guide.md"
    lines = ["# Intro Reliability Figure Style Options", ""]
    for name, description in STYLE_DESCRIPTIONS.items():
        lines.append(f"- `{name}`: {description}")
    lines.append("")
    lines.append("Score gradient:")
    lines.append("- `style_1_score_gradient`: low score is red, middle score is yellow, high score is green.")
    lines.append("- `style_1_unto_o_tp_prob`: low UnTO-O TP probability is red, high probability is green.")
    lines.append("")
    lines.append("Colors:")
    for case_type in CASE_ORDER:
        lines.append(f"- {CASE_LABELS[case_type]}: BGR {CASE_COLORS[case_type]}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows_by_image = read_rows()
    attach_unto_o_tp_prob(rows_by_image)
    available_ids = {
        int(path.stem)
        for path in IMAGE_DIR.glob("*.jpg")
        if path.stem.isdigit()
    }
    image_ids = [
        image_id
        for image_id in sorted(rows_by_image.keys(), key=lambda x: min(int(r["recommended_rank"]) for r in rows_by_image[x]))
        if image_id in available_ids
    ]
    if not image_ids:
        raise RuntimeError(f"No matching images found in {IMAGE_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contact_dir = OUTPUT_DIR / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    written_by_style = defaultdict(list)
    for image_id in image_ids:
        image_path = image_path_for_id(image_id)
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        base = letterbox(original, CANVAS_SIZE)
        rows = rows_by_image[image_id]
        for style_name, renderer in STYLE_RENDERERS.items():
            out_dir = OUTPUT_DIR / style_name
            out_dir.mkdir(parents=True, exist_ok=True)
            rendered = renderer(base, rows)
            out_path = out_dir / f"{image_id:012d}_{style_name}.jpg"
            cv2.imwrite(str(out_path), rendered)
            written_by_style[style_name].append(out_path)
    for style_name, paths in written_by_style.items():
        make_contact_sheet(paths, contact_dir / f"{style_name}_contact.jpg")
    write_style_guide()
    print(f"Saved styled images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
