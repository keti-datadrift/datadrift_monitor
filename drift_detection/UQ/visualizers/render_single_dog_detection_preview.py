import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path('visualizers/runs/06-26-2026_16;57_single_dog_detection_candidates')
IMAGE_PATH = ROOT / 'images' / 'coco_000000029306.jpg'
CSV_PATH = ROOT / 'single_dog_candidates.csv'
OUT_DIR = ROOT / 'rendered'
CANVAS_SIZE = 640
BOX_ALPHA = 0.86
LABEL_ALPHA = 0.80


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def score_color(value):
    t = clamp(float(value), 0.0, 1.0)
    red = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 200, 0], dtype=np.float32)
    color = red * (1.0 - t) + green * t
    return tuple(int(v) for v in color)


def letterbox(image, size=CANVAS_SIZE, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    left = int(round((size - nw) / 2 - 0.1))
    top = int(round((size - nh) / 2 - 0.1))
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def draw_label(image, x, y, text, color):
    scale = 0.46
    pad = 4
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(clamp(x, 0, image.shape[1] - tw - pad * 2 - 1))
    y = int(clamp(y, th + pad * 2, image.shape[0] - 2))
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.addWeighted(overlay, LABEL_ALPHA, image, 1 - LABEL_ALPHA, 0, dst=image)
    cv2.putText(image, text, (x + pad, y - pad - base // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 1, cv2.LINE_AA)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'failed to read {IMAGE_PATH}')
    base = letterbox(image)
    df = pd.read_csv(CSV_PATH)
    row = df[(df['dataset'] == 'coco') & (df['image_id'] == 29306)].iloc[0]
    color = score_color(row['score'])
    x1 = int(round(clamp(row['xmin'], 0, CANVAS_SIZE - 1)))
    y1 = int(round(clamp(row['ymin'], 0, CANVAS_SIZE - 1)))
    x2 = int(round(clamp(row['xmax'], 0, CANVAS_SIZE - 1)))
    y2 = int(round(clamp(row['ymax'], 0, CANVAS_SIZE - 1)))
    overlay = base.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
    cv2.addWeighted(overlay, BOX_ALPHA, base, 1 - BOX_ALPHA, 0, dst=base)
    draw_label(base, x1, max(22, y1 - 3), f"dog {row['score']:.2f}", color)
    out = OUT_DIR / 'coco_000000029306_dog_detection.jpg'
    cv2.imwrite(str(out), base)
    selected = OUT_DIR / 'coco_000000029306_detection_row.csv'
    row.to_frame().T.to_csv(selected, index=False)
    print(out)
    print(selected)


if __name__ == '__main__':
    main()
