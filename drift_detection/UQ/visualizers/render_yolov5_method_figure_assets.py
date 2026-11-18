from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "object_detectors"))

from models.yolov5.detector import YOLOV5TorchObjectDetector


RUN_ROOT = REPO_ROOT / "visualizers" / "runs" / "06-26-2026_16;57_single_dog_detection_candidates"
IMAGE_PATH = RUN_ROOT / "images" / "coco_000000029306.jpg"
WEIGHT_PATH = REPO_ROOT / "object_detectors" / "models" / "yolov5" / "weights" / "coco" / "yolov5x.pt"
OUTPUT_DIR = RUN_ROOT / "method_figure_assets"
THICK_OUTPUT_DIR = RUN_ROOT / "method_figure_assets_thick"

DOG_CLASS_INDEX = 16
FINAL_COLOR = (0, 210, 45)
CANDIDATE_COLOR = (255, 220, 0)
SECOND_CANDIDATE_COLOR = (255, 145, 0)
SOURCE_COLOR = (255, 220, 0)
LOCATION_COLOR = (255, 220, 0)
CANDIDATE_RENDER_OFFSETS = [(-18.0, -18.0), (18.0, 18.0)]
BOX_ALPHA = 0.86
LABEL_ALPHA = 0.80
CANVAS_SIZE = 640
THICK_SOURCE_WIDTH = 18
THICK_BOX_WIDTH = 15
THICK_LOCATION_OUTER_RADIUS = 32
THICK_LOCATION_INNER_RADIUS = 24
THICK_CANDIDATE_RENDER_SCALES = [0.62, 0.95]


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


def normalize_map(x):
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def xywh_to_xyxy_np(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    return out


def box_iou_np(box, boxes):
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)
    inter = iw * ih
    box_area = np.maximum(box[2] - box[0], 0.0) * np.maximum(box[3] - box[1], 0.0)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return inter / np.maximum(box_area + areas - inter, 1e-9)


def to_rgba(image_rgb):
    return Image.fromarray(image_rgb.astype(np.uint8), "RGB").convert("RGBA")


def text_bbox(text):
    image = Image.new("RGB", (1, 1))
    return ImageDraw.Draw(image).textbbox((0, 0), text, font=FONT)


def draw_alpha_rectangle(image, box, color, fill=False, width=2, alpha=BOX_ALPHA):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgba = (*color, int(255 * alpha))
    box = tuple(int(round(v)) for v in box)
    if fill:
        draw.rectangle(box, fill=rgba)
    else:
        for offset in range(width):
            draw.rectangle(
                (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
                outline=rgba,
            )
    image.alpha_composite(overlay)


def draw_alpha_line_box(image, box, color, width, alpha=BOX_ALPHA):
    draw_alpha_rectangle(image, box, color, fill=False, width=width, alpha=alpha)


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


def clip_box(box, width=CANVAS_SIZE, height=CANVAS_SIZE):
    x1 = int(round(clamp(float(box[0]), 0, width - 1)))
    y1 = int(round(clamp(float(box[1]), 0, height - 1)))
    x2 = int(round(clamp(float(box[2]), 0, width - 1)))
    y2 = int(round(clamp(float(box[3]), 0, height - 1)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def draw_box_with_label(image, box, color, text):
    clipped = clip_box(box, image.width, image.height)
    draw_alpha_rectangle(image, clipped, color, width=2, alpha=BOX_ALPHA)
    draw_label(image, clipped[0], max(20, clipped[1] - 3), text, color)


def draw_thick_box(image, box, color):
    draw_alpha_line_box(image, clip_box(box, image.width, image.height), color, width=THICK_BOX_WIDTH, alpha=BOX_ALPHA)


def shifted_box(box, dx, dy):
    out = np.asarray(box, dtype=np.float32).copy()
    out[[0, 2]] += dx
    out[[1, 3]] += dy
    return out


def scale_box_about_center(box, scale):
    out = np.asarray(box, dtype=np.float32).copy()
    cx = (out[0] + out[2]) / 2.0
    cy = (out[1] + out[3]) / 2.0
    half_w = (out[2] - out[0]) * float(scale) / 2.0
    half_h = (out[3] - out[1]) * float(scale) / 2.0
    return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float32)


def draw_dashed_box(image, box, color, width=3, dash=18, gap=10, alpha=BOX_ALPHA):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgba = (*color, int(255 * alpha))
    x1, y1, x2, y2 = [int(round(v)) for v in clip_box(box, image.width, image.height)]
    segments = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for start, end in segments:
        sx, sy = start
        ex, ey = end
        length = int(np.hypot(ex - sx, ey - sy))
        if length <= 0:
            continue
        for offset in range(0, length, dash + gap):
            end_offset = min(offset + dash, length)
            t0 = offset / length
            t1 = end_offset / length
            p0 = (int(round(sx + (ex - sx) * t0)), int(round(sy + (ey - sy) * t0)))
            p1 = (int(round(sx + (ex - sx) * t1)), int(round(sy + (ey - sy) * t1)))
            draw.line((p0, p1), fill=rgba, width=width)
    image.alpha_composite(overlay)


def draw_dashed_box_thick(image, box, color):
    draw_dashed_box(image, box, color, width=THICK_SOURCE_WIDTH, dash=30, gap=16, alpha=BOX_ALPHA)


def draw_location_circle(image, xy, color=LOCATION_COLOR):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = [int(round(v)) for v in xy]
    draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(255, 255, 255, 255))
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(*color, int(255 * BOX_ALPHA)))
    image.alpha_composite(overlay)


def draw_location_circle_thick(image, xy, color=LOCATION_COLOR):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = [int(round(v)) for v in xy]
    draw.ellipse(
        (
            x - THICK_LOCATION_OUTER_RADIUS,
            y - THICK_LOCATION_OUTER_RADIUS,
            x + THICK_LOCATION_OUTER_RADIUS,
            y + THICK_LOCATION_OUTER_RADIUS,
        ),
        fill=(255, 255, 255, 255),
    )
    draw.ellipse(
        (
            x - THICK_LOCATION_INNER_RADIUS,
            y - THICK_LOCATION_INNER_RADIUS,
            x + THICK_LOCATION_INNER_RADIUS,
            y + THICK_LOCATION_INNER_RADIUS,
        ),
        fill=(*color, int(255 * BOX_ALPHA)),
    )
    image.alpha_composite(overlay)


def save_rgb(path, img):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_pil_rgb(path, image):
    image.convert("RGB").save(path, quality=95)


def letterbox_content_bounds(original_shape, ratio, pad):
    original_h, original_w = original_shape[:2]
    content_w = int(round(original_w * ratio[0]))
    content_h = int(round(original_h * ratio[1]))
    left = int(round(pad[0] - 0.1))
    top = int(round(pad[1] - 0.1))
    return left, top, left + content_w, top + content_h


def crop_letterbox_rgb(image_rgb, original_shape, ratio, pad):
    left, top, right, bottom = letterbox_content_bounds(original_shape, ratio, pad)
    cropped = image_rgb[top:bottom, left:right].copy()
    original_h, original_w = original_shape[:2]
    if cropped.shape[0] != original_h or cropped.shape[1] != original_w:
        cropped = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def crop_letterbox_pil(image, original_shape, ratio, pad):
    left, top, right, bottom = letterbox_content_bounds(original_shape, ratio, pad)
    cropped = image.crop((left, top, right, bottom))
    original_h, original_w = original_shape[:2]
    if cropped.size != (original_w, original_h):
        cropped = cropped.resize((original_w, original_h), Image.Resampling.BILINEAR)
    return cropped


def letterbox_box_to_original(box, ratio, pad, original_shape):
    out = np.asarray(box, dtype=np.float32).copy()
    out[[0, 2]] = (out[[0, 2]] - float(pad[0])) / float(ratio[0])
    out[[1, 3]] = (out[[1, 3]] - float(pad[1])) / float(ratio[1])
    original_h, original_w = original_shape[:2]
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, float(original_w - 1))
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, float(original_h - 1))
    return out


def letterbox_point_to_original(point, ratio, pad, original_shape):
    x = (float(point[0]) - float(pad[0])) / float(ratio[0])
    y = (float(point[1]) - float(pad[1])) / float(ratio[1])
    original_h, original_w = original_shape[:2]
    return np.array([clamp(x, 0.0, float(original_w - 1)), clamp(y, 0.0, float(original_h - 1))], dtype=np.float32)


def raw_index_to_source(raw_idx, feature_shapes):
    cursor = 0
    for level, (_, _, ny, nx, _) in enumerate(feature_shapes):
        count = 3 * ny * nx
        if raw_idx < cursor + count:
            local = raw_idx - cursor
            anchor_idx = local // (ny * nx)
            rem = local % (ny * nx)
            grid_y = rem // nx
            grid_x = rem % nx
            return level, int(anchor_idx), int(grid_x), int(grid_y), int(nx), int(ny)
        cursor += count
    raise ValueError(f"raw_pred_idx {raw_idx} is outside raw prediction range {cursor}.")


def channel_to_viridis(channel):
    normalized = normalize_map(channel)
    resized = cv2.resize(normalized, (640, 640), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def channel_overlay(channel, image_rgb):
    heat = channel_to_viridis(channel)
    return cv2.addWeighted(image_rgb, 0.54, heat, 0.46, 0)


def level_start_index(level, feature_shapes):
    start = 0
    for current_level, (_, _, ny, nx, _) in enumerate(feature_shapes):
        if current_level == level:
            return start
        start += 3 * ny * nx
    raise ValueError(f"Invalid level {level}.")


def same_location_raw_indices(level, grid_x, grid_y, nx, ny, feature_shapes):
    start = level_start_index(level, feature_shapes)
    return [start + anchor_idx * ny * nx + grid_y * nx + grid_x for anchor_idx in range(3)]


def select_context_candidate_indices(decoded_boxes, dog_scores, final_box, final_raw_idx):
    ious = box_iou_np(final_box, decoded_boxes)
    final_center = np.array([(final_box[0] + final_box[2]) / 2.0, (final_box[1] + final_box[3]) / 2.0], dtype=np.float32)
    centers = np.stack(
        [
            (decoded_boxes[:, 0] + decoded_boxes[:, 2]) / 2.0,
            (decoded_boxes[:, 1] + decoded_boxes[:, 3]) / 2.0,
        ],
        axis=1,
    )
    distances = np.linalg.norm(centers - final_center, axis=1)
    base_mask = (
        (np.arange(decoded_boxes.shape[0]) != final_raw_idx)
        & (distances < 320.0)
        & (ious > 0.01)
        & (dog_scores > 0.01)
    )
    candidates = np.flatnonzero(base_mask)
    if candidates.size < 2:
        raise RuntimeError("Could not find enough nearby raw dog predictions for figure rendering.")
    selected = []
    for target_score in (0.75, 0.45):
        available = [idx for idx in candidates.tolist() if idx not in selected]
        best = min(
            available,
            key=lambda idx: (
                abs(float(dog_scores[idx]) - target_score),
                distances[idx] / 640.0,
                -ious[idx],
            ),
        )
        selected.append(best)
    return selected, ious, distances


def select_final_dog_detection(detector, prediction, logits):
    detections, _selected_logits, _objectness, selected_indices = detector.non_max_suppression(
        prediction,
        logits,
        conf_thres=detector.confidence,
        iou_thres=detector.iou_thresh,
        agnostic=detector.agnostic,
        return_indices=True,
    )
    det = detections[0]
    raw_indices = selected_indices[0]
    if det.numel() == 0:
        raise RuntimeError("No YOLOv5 detections found on the cropped dog image.")
    dog_mask = det[:, 5].long() == DOG_CLASS_INDEX
    if not dog_mask.any():
        raise RuntimeError("No dog detection found on the cropped dog image.")
    dog_rows = torch.nonzero(dog_mask, as_tuple=False).view(-1)
    best_local = dog_rows[det[dog_rows, 4].argmax()]
    final_box = det[best_local, :4].detach().cpu().numpy().astype(np.float32)
    final_score = float(det[best_local, 4].detach().cpu().item())
    raw_idx = int(raw_indices[best_local].detach().cpu().item())
    return raw_idx, final_box, final_score


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THICK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLOV5TorchObjectDetector(
        str(WEIGHT_PATH),
        device=device,
        img_size=(640, 640),
        confidence=0.25,
        iou_thresh=0.45,
        agnostic_nms=True,
        fuse=True,
    )
    detector.eval()

    source_features = {}
    detect = detector.model.model[-1]
    handles = []

    def capture_source_feature(level):
        def hook(module, inputs, output):
            source_features[level] = inputs[0].detach().cpu()
        return hook

    for level, conv in enumerate(detect.m):
        handles.append(conv.register_forward_hook(capture_source_feature(level)))

    image_bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(str(IMAGE_PATH))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape
    letterbox_rgb, ratio, pad = detector.yolo_resize(image_rgb, new_shape=(640, 640), auto=False)
    tensor = torch.from_numpy(letterbox_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        prediction, logits, detect_features, anchor_priors = detector.model(tensor, augment=False)

    for handle in handles:
        handle.remove()

    prediction_np = prediction[0].detach().cpu().numpy().astype(np.float32)
    raw_idx, final_box, final_score = select_final_dog_detection(detector, prediction, logits)
    feature_shapes = [tuple(x.shape) for x in detect_features]
    level, anchor_idx, grid_x, grid_y, nx, ny = raw_index_to_source(raw_idx, feature_shapes)
    anchor_raw_indices = same_location_raw_indices(level, grid_x, grid_y, nx, ny, feature_shapes)
    anchor_priors_np = anchor_priors[0, anchor_raw_indices].detach().cpu().numpy().astype(np.float32)
    anchor_boxes_xyxy = xywh_to_xyxy_np(anchor_priors_np)
    prior_xywh = anchor_priors_np[anchor_idx]
    source_xy = prior_xywh[:2]
    source_anchor_xyxy = anchor_boxes_xyxy[anchor_idx]
    base_original = image_rgb.copy()
    anchor_boxes_original = np.stack(
        [letterbox_box_to_original(box, ratio, pad, original_shape) for box in anchor_boxes_xyxy],
        axis=0,
    )
    source_anchor_original = anchor_boxes_original[anchor_idx]
    source_xy_original = letterbox_point_to_original(source_xy, ratio, pad, original_shape)

    p5 = source_features[level][0].float()
    channel_count = int(p5.shape[0])
    channel_indices = [0, channel_count // 3, (2 * channel_count) // 3, channel_count - 1]

    save_rgb(OUTPUT_DIR / "01_base_image.jpg", base_original)
    save_rgb(THICK_OUTPUT_DIR / "01_base_image.jpg", base_original)
    for channel_idx in channel_indices:
        overlay_img = crop_letterbox_rgb(channel_overlay(p5[channel_idx].numpy(), letterbox_rgb), original_shape, ratio, pad)
        save_rgb(OUTPUT_DIR / f"02_p5_feature_ch_{channel_idx:04d}.jpg", overlay_img)
        save_rgb(THICK_OUTPUT_DIR / f"02_p5_feature_ch_{channel_idx:04d}.jpg", overlay_img)

    first_overlay = crop_letterbox_rgb(channel_overlay(p5[channel_indices[0]].numpy(), letterbox_rgb), original_shape, ratio, pad)
    source_feature = to_rgba(first_overlay)
    for anchor_box in anchor_boxes_original:
        draw_dashed_box(source_feature, anchor_box, SOURCE_COLOR)
    draw_location_circle(source_feature, source_xy_original)
    save_pil_rgb(OUTPUT_DIR / "03_p5_feature_source_anchor_location.jpg", source_feature)

    thick_source_feature = to_rgba(first_overlay)
    for anchor_box in anchor_boxes_original:
        draw_dashed_box_thick(thick_source_feature, anchor_box, SOURCE_COLOR)
    draw_location_circle_thick(thick_source_feature, source_xy_original)
    save_pil_rgb(THICK_OUTPUT_DIR / "03_p5_feature_source_anchor_location.jpg", thick_source_feature)

    single_source_feature = to_rgba(first_overlay)
    draw_dashed_box(single_source_feature, source_anchor_original, SOURCE_COLOR)
    draw_location_circle(single_source_feature, source_xy_original)
    save_pil_rgb(OUTPUT_DIR / "03b_p5_feature_single_source_anchor_location.jpg", single_source_feature)

    thick_single_source_feature = to_rgba(first_overlay)
    draw_dashed_box_thick(thick_single_source_feature, source_anchor_original, SOURCE_COLOR)
    draw_location_circle_thick(thick_single_source_feature, source_xy_original)
    save_pil_rgb(THICK_OUTPUT_DIR / "03b_p5_feature_single_source_anchor_location.jpg", thick_single_source_feature)

    decoded_source = to_rgba(base_original)
    draw_dashed_box(decoded_source, source_anchor_original, SOURCE_COLOR)
    draw_location_circle(decoded_source, source_xy_original)
    save_pil_rgb(OUTPUT_DIR / "04_base_decoded_source_anchor_location.jpg", decoded_source)

    thick_decoded_source = to_rgba(base_original)
    draw_dashed_box_thick(thick_decoded_source, source_anchor_original, SOURCE_COLOR)
    draw_location_circle_thick(thick_decoded_source, source_xy_original)
    save_pil_rgb(THICK_OUTPUT_DIR / "04_base_decoded_source_anchor_location.jpg", thick_decoded_source)

    decoded_boxes = xywh_to_xyxy_np(prediction_np[:, :4])
    decoded_boxes_original = np.stack(
        [letterbox_box_to_original(box, ratio, pad, original_shape) for box in decoded_boxes],
        axis=0,
    )
    final_box_original = letterbox_box_to_original(final_box, ratio, pad, original_shape)
    class_probs = prediction_np[:, 5:]
    dog_scores = prediction_np[:, 4] * class_probs[:, DOG_CLASS_INDEX]
    anchor_iou = box_iou_np(final_box, decoded_boxes[np.array(anchor_raw_indices, dtype=np.int64)])
    context_indices, context_iou, context_distances = select_context_candidate_indices(decoded_boxes, dog_scores, final_box, raw_idx)
    three_boxes = to_rgba(base_original)
    context_offsets = {}
    for draw_order, current_raw_idx in enumerate(context_indices):
        dx, dy = CANDIDATE_RENDER_OFFSETS[draw_order]
        context_offsets[current_raw_idx] = (dx, dy)
        color = score_color(dog_scores[current_raw_idx])
        render_box = shifted_box(decoded_boxes_original[current_raw_idx], dx, dy)
        draw_box_with_label(three_boxes, render_box, color, f"dog {dog_scores[current_raw_idx]:.2f}")
    draw_box_with_label(
        three_boxes,
        decoded_boxes_original[anchor_raw_indices[anchor_idx]],
        score_color(final_score),
        f"dog {final_score:.2f}",
    )
    save_pil_rgb(OUTPUT_DIR / "05_base_three_detection_boxes.jpg", three_boxes)

    thick_three_boxes = to_rgba(base_original)
    for draw_order, current_raw_idx in enumerate(context_indices):
        dx, dy = CANDIDATE_RENDER_OFFSETS[draw_order]
        color = score_color(dog_scores[current_raw_idx])
        render_box = shifted_box(decoded_boxes_original[current_raw_idx], dx, dy)
        render_box = scale_box_about_center(render_box, THICK_CANDIDATE_RENDER_SCALES[draw_order])
        draw_thick_box(thick_three_boxes, render_box, color)
    draw_thick_box(thick_three_boxes, decoded_boxes_original[anchor_raw_indices[anchor_idx]], score_color(final_score))
    save_pil_rgb(THICK_OUTPUT_DIR / "05_base_three_detection_boxes.jpg", thick_three_boxes)

    final_only = to_rgba(base_original)
    draw_box_with_label(final_only, final_box_original, score_color(final_score), f"dog {final_score:.2f}")
    save_pil_rgb(OUTPUT_DIR / "06_base_final_detection_box.jpg", final_only)

    thick_final_only = to_rgba(base_original)
    draw_thick_box(thick_final_only, final_box_original, score_color(final_score))
    save_pil_rgb(THICK_OUTPUT_DIR / "06_base_final_detection_box.jpg", thick_final_only)

    metadata = {
        "image_id": 29306,
        "raw_pred_idx": raw_idx,
        "source_level": level,
        "source_anchor_idx": anchor_idx,
        "source_grid_x": grid_x,
        "source_grid_y": grid_y,
        "source_x": float(source_xy[0]),
        "source_y": float(source_xy[1]),
        "source_anchor_xmin": float(source_anchor_xyxy[0]),
        "source_anchor_ymin": float(source_anchor_xyxy[1]),
        "source_anchor_xmax": float(source_anchor_xyxy[2]),
        "source_anchor_ymax": float(source_anchor_xyxy[3]),
        "source_original_x": float(source_xy_original[0]),
        "source_original_y": float(source_xy_original[1]),
        "source_anchor_original_xmin": float(source_anchor_original[0]),
        "source_anchor_original_ymin": float(source_anchor_original[1]),
        "source_anchor_original_xmax": float(source_anchor_original[2]),
        "source_anchor_original_ymax": float(source_anchor_original[3]),
        "final_xmin": float(final_box[0]),
        "final_ymin": float(final_box[1]),
        "final_xmax": float(final_box[2]),
        "final_ymax": float(final_box[3]),
        "final_original_xmin": float(final_box_original[0]),
        "final_original_ymin": float(final_box_original[1]),
        "final_original_xmax": float(final_box_original[2]),
        "final_original_ymax": float(final_box_original[3]),
        "final_score": float(final_score),
        "crop_x0": 0,
        "crop_y0": 0,
        "crop_width": int(image_rgb.shape[1]),
        "crop_height": int(image_rgb.shape[0]),
        "letterbox_ratio": float(ratio[0]),
        "letterbox_pad_x": float(pad[0]),
        "letterbox_pad_y": float(pad[1]),
    }
    pd.DataFrame([metadata]).to_csv(OUTPUT_DIR / "source_anchor_metadata.csv", index=False)

    anchor_rows = []
    for current_anchor_idx, current_raw_idx in enumerate(anchor_raw_indices):
        prediction_box = decoded_boxes[current_raw_idx]
        prediction_box_original = decoded_boxes_original[current_raw_idx]
        anchor_rows.append({
            "anchor_idx": int(current_anchor_idx),
            "raw_pred_idx": int(current_raw_idx),
            "is_final_source_anchor": bool(current_anchor_idx == anchor_idx),
            "dog_score": float(dog_scores[current_raw_idx]),
            "iou_with_final": float(anchor_iou[current_anchor_idx]),
            "xmin": float(prediction_box[0]),
            "ymin": float(prediction_box[1]),
            "xmax": float(prediction_box[2]),
            "ymax": float(prediction_box[3]),
            "original_xmin": float(prediction_box_original[0]),
            "original_ymin": float(prediction_box_original[1]),
            "original_xmax": float(prediction_box_original[2]),
            "original_ymax": float(prediction_box_original[3]),
        })
    pd.DataFrame(anchor_rows).to_csv(OUTPUT_DIR / "same_location_anchor_predictions.csv", index=False)
    context_rows = []
    for current_raw_idx in context_indices:
        prediction_box = decoded_boxes[current_raw_idx]
        prediction_box_original = decoded_boxes_original[current_raw_idx]
        context_rows.append({
            "raw_pred_idx": int(current_raw_idx),
            "dog_score": float(dog_scores[current_raw_idx]),
            "iou_with_final": float(context_iou[current_raw_idx]),
            "center_distance_to_final": float(context_distances[current_raw_idx]),
            "render_dx": float(context_offsets[current_raw_idx][0]),
            "render_dy": float(context_offsets[current_raw_idx][1]),
            "xmin": float(prediction_box[0]),
            "ymin": float(prediction_box[1]),
            "xmax": float(prediction_box[2]),
            "ymax": float(prediction_box[3]),
            "original_xmin": float(prediction_box_original[0]),
            "original_ymin": float(prediction_box_original[1]),
            "original_xmax": float(prediction_box_original[2]),
            "original_ymax": float(prediction_box_original[3]),
        })
    pd.DataFrame(context_rows).to_csv(OUTPUT_DIR / "context_candidate_predictions.csv", index=False)
    stale_candidate_csv = OUTPUT_DIR / "neighbor_candidate_boxes.csv"
    if stale_candidate_csv.exists():
        stale_candidate_csv.unlink()
    print(f"saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
