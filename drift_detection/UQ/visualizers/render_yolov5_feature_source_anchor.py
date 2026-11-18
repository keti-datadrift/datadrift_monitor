from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "object_detectors"))

from models.yolov5.detector import YOLOV5TorchObjectDetector


RUN_ROOT = REPO_ROOT / "visualizers" / "runs" / "06-26-2026_16;57_single_dog_detection_candidates"
IMAGE_PATH = RUN_ROOT / "images" / "coco_000000029306.jpg"
DETECTION_CSV = RUN_ROOT / "rendered" / "coco_000000029306_detection_row.csv"
WEIGHT_PATH = REPO_ROOT / "object_detectors" / "models" / "yolov5" / "weights" / "coco" / "yolov5x.pt"
OUTPUT_DIR = RUN_ROOT / "feature_source_anchor"


def normalize_map(x):
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return np.array([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], dtype=np.float32)


def draw_box(img, box, color, label=None, thickness=3, alpha=0.88):
    overlay = img.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.62
        text_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, text_thickness)
        ty = max(0, y1 - th - baseline - 5)
        tx = max(0, min(x1, img.shape[1] - tw - 8))
        label_overlay = img.copy()
        cv2.rectangle(label_overlay, (tx, ty), (tx + tw + 8, ty + th + baseline + 7), color, -1)
        img[:] = cv2.addWeighted(label_overlay, 0.80, img, 0.20, 0)
        cv2.putText(img, label, (tx + 4, ty + th + 1), font, scale, (255, 255, 255), text_thickness, cv2.LINE_AA)


def draw_dashed_box(img, box, color, thickness=2, dash=14, gap=9):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    segments = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for start, end in segments:
        x_start, y_start = start
        x_end, y_end = end
        length = int(np.hypot(x_end - x_start, y_end - y_start))
        if length <= 0:
            continue
        for offset in range(0, length, dash + gap):
            offset_end = min(offset + dash, length)
            t0 = offset / length
            t1 = offset_end / length
            p0 = (int(round(x_start + (x_end - x_start) * t0)), int(round(y_start + (y_end - y_start) * t0)))
            p1 = (int(round(x_start + (x_end - x_start) * t1)), int(round(y_start + (y_end - y_start) * t1)))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def put_label(img, text, xy, color):
    x, y = [int(round(v)) for v in xy]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, img.shape[1] - tw - 8))
    y = max(0, min(y, img.shape[0] - th - baseline - 7))
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + tw + 8, y + th + baseline + 7), color, -1)
    img[:] = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)
    cv2.putText(img, text, (x + 4, y + th + 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_source_location(img, xy, color=(255, 0, 255)):
    x, y = [int(round(v)) for v in xy]
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, markerSize=24, thickness=3, line_type=cv2.LINE_AA)
    cv2.circle(img, (x, y), 6, color, -1, cv2.LINE_AA)


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
    raise ValueError(f"raw_pred_idx {raw_idx} is outside YOLOv5 raw prediction range {cursor}.")


def make_contact_sheet(paths, output_path):
    cells = []
    for title, path in paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (520, 520), interpolation=cv2.INTER_AREA)
        canvas = np.full((570, 520, 3), 255, dtype=np.uint8)
        canvas[50:, :, :] = img
        cv2.putText(canvas, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
        cells.append(canvas)
    sheet = np.concatenate([np.concatenate(cells[:2], axis=1), np.concatenate(cells[2:], axis=1)], axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def channel_to_rgb(channel, tile_size):
    x = normalize_map(channel)
    x = cv2.resize(x, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def select_channels(feature, count, grid_xy=None):
    c, h, w = feature.shape
    flat_score = feature.abs().reshape(c, -1).mean(dim=1)
    if grid_xy is not None:
        gx, gy = grid_xy
        if 0 <= gx < w and 0 <= gy < h:
            local_score = feature[:, gy, gx].abs()
            score = local_score * 0.7 + flat_score * 0.3
        else:
            score = flat_score
    else:
        score = flat_score
    k = min(count, c)
    return torch.topk(score, k=k).indices.cpu().numpy().astype(int).tolist()


def make_channel_grid(title, feature, channel_indices, output_path, tile_size=128, source_cell=None):
    title_h = 46
    label_h = 26
    gap = 3
    tiles = []
    for ch in channel_indices:
        tile = channel_to_rgb(feature[ch].numpy(), tile_size)
        if source_cell is not None:
            gx, gy, fw, fh = source_cell
            x1 = int(round(gx * tile_size / fw))
            y1 = int(round(gy * tile_size / fh))
            x2 = int(round((gx + 1) * tile_size / fw))
            y2 = int(round((gy + 1) * tile_size / fh))
            cv2.rectangle(tile, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(tile, ((x1 + x2) // 2, (y1 + y2) // 2), (255, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        canvas = np.full((tile_size + label_h, tile_size, 3), 255, dtype=np.uint8)
        canvas[:tile_size] = tile
        cv2.putText(canvas, f"ch {ch}", (7, tile_size + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
        tiles.append(canvas)
    row_w = len(tiles) * tile_size + (len(tiles) - 1) * gap
    out = np.full((title_h + tile_size + label_h, row_w, 3), 255, dtype=np.uint8)
    cv2.putText(out, title, (max(8, row_w // 2 - len(title) * 7), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (30, 30, 30), 2, cv2.LINE_AA)
    x = 0
    for tile in tiles:
        out[title_h:, x:x + tile_size] = tile
        x += tile_size + gap
    cv2.imwrite(str(output_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def make_pyramid_channel_grid(rows, output_path):
    row_images = []
    max_w = 0
    for title, feature, channels, source_cell in rows:
        tmp = output_path.with_name(f"_{title.replace(' ', '_').replace('/', '_')}.jpg")
        make_channel_grid(title, feature, channels, tmp, tile_size=128, source_cell=source_cell)
        img = cv2.imread(str(tmp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tmp.unlink()
        row_images.append(img)
        max_w = max(max_w, img.shape[1])
    padded_rows = []
    for img in row_images:
        canvas = np.full((img.shape[0] + 22, max_w, 3), 255, dtype=np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        padded_rows.append(canvas)
    sheet = np.concatenate(padded_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.read_csv(DETECTION_CSV).iloc[0]
    raw_idx = int(row["raw_pred_idx"])
    final_box = np.array([row["xmin"], row["ymin"], row["xmax"], row["ymax"]], dtype=np.float32)
    score = float(row["score"])

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
    letterbox_rgb, ratio, pad = detector.yolo_resize(image_rgb, new_shape=(640, 640), auto=False)
    tensor = torch.from_numpy(letterbox_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        prediction, logits, detect_features, anchor_priors = detector.model(tensor, augment=False)

    for handle in handles:
        handle.remove()

    feature_shapes = [tuple(x.shape) for x in detect_features]
    level, anchor_idx, grid_x, grid_y, nx, ny = raw_index_to_source(raw_idx, feature_shapes)
    prior_xywh = anchor_priors[0, raw_idx].detach().cpu().numpy().astype(np.float32)
    prior_xyxy = xywh_to_xyxy(prior_xywh)
    source_xy = prior_xywh[:2]
    stride = 640.0 / nx

    feature = source_features[level][0].float()
    activation = normalize_map(feature.abs().mean(dim=0).numpy())
    activation_up = cv2.resize(activation, (640, 640), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap((activation_up * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    letterbox_vis = letterbox_rgb.copy()
    overlay = cv2.addWeighted(letterbox_vis, 0.48, heat, 0.52, 0)

    cell_box = np.array([grid_x * stride, grid_y * stride, (grid_x + 1) * stride, (grid_y + 1) * stride], dtype=np.float32)

    feature_only = heat.copy()
    draw_box(feature_only, cell_box, (255, 255, 255), "source cell", thickness=3, alpha=0.95)
    draw_dashed_box(feature_only, prior_xyxy, (255, 230, 0), thickness=3)
    draw_source_location(feature_only, source_xy, (255, 0, 255))
    put_label(feature_only, f"L{level} stride {int(stride)}  grid ({grid_x},{grid_y})  anchor {anchor_idx}", (14, 18), (40, 40, 40))

    feature_overlay = overlay.copy()
    draw_box(feature_overlay, final_box, (30, 210, 55), f"final dog {score:.2f}", thickness=4, alpha=0.88)
    draw_box(feature_overlay, cell_box, (255, 255, 255), "source feature cell", thickness=3, alpha=0.85)
    draw_dashed_box(feature_overlay, prior_xyxy, (255, 230, 0), thickness=3)
    draw_source_location(feature_overlay, source_xy, (255, 0, 255))
    put_label(feature_overlay, "source anchor", (prior_xyxy[0], prior_xyxy[1] - 28), (170, 145, 0))
    put_label(feature_overlay, "source location", (source_xy[0] + 12, source_xy[1] + 10), (170, 0, 170))

    source_on_image = letterbox_rgb.copy()
    draw_box(source_on_image, final_box, (30, 210, 55), f"final dog {score:.2f}", thickness=4, alpha=0.88)
    draw_box(source_on_image, cell_box, (255, 255, 255), "feature cell", thickness=3, alpha=0.72)
    draw_dashed_box(source_on_image, prior_xyxy, (255, 230, 0), thickness=3)
    draw_source_location(source_on_image, source_xy, (255, 0, 255))
    put_label(source_on_image, "decoded source anchor", (prior_xyxy[0], prior_xyxy[1] - 28), (170, 145, 0))
    put_label(source_on_image, "decoded source location", (source_xy[0] + 12, source_xy[1] + 10), (170, 0, 170))

    decoded_only = letterbox_rgb.copy()
    draw_dashed_box(decoded_only, prior_xyxy, (255, 230, 0), thickness=4)
    draw_source_location(decoded_only, source_xy, (255, 0, 255))
    put_label(decoded_only, "source anchor decoded to image", (prior_xyxy[0], prior_xyxy[1] - 28), (170, 145, 0))
    put_label(decoded_only, "source location decoded to image", (source_xy[0] + 12, source_xy[1] + 10), (170, 0, 170))

    outputs = {
        "feature_map_source_cell.jpg": feature_only,
        "feature_map_overlay_source.jpg": feature_overlay,
        "image_with_final_source_anchor_location.jpg": source_on_image,
        "decoded_source_anchor_location.jpg": decoded_only,
    }
    for name, img in outputs.items():
        cv2.imwrite(str(OUTPUT_DIR / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    channel_dir = OUTPUT_DIR / "channel_grids"
    channel_dir.mkdir(parents=True, exist_ok=True)
    pyramid_rows = []
    for feature_level in sorted(source_features):
        feat = source_features[feature_level][0].float()
        _, fh, fw = feat.shape
        title = f"P{feature_level + 3} detection feature map ({fw}x{fh})"
        if feature_level == level:
            channels = select_channels(feat, 8, grid_xy=(grid_x, grid_y))
            source_cell = (grid_x, grid_y, fw, fh)
            title += f"  source level, grid ({grid_x},{grid_y})"
        else:
            channels = select_channels(feat, 8)
            source_cell = None
        make_channel_grid(title, feat, channels, channel_dir / f"p{feature_level + 3}_top_channels.jpg", tile_size=148, source_cell=source_cell)
        pyramid_rows.append((title, feat, channels, source_cell))
    make_pyramid_channel_grid(pyramid_rows, channel_dir / "yolov5_pyramid_top_channels.jpg")

    pd.DataFrame([{
        "image_id": int(row["image_id"]),
        "raw_pred_idx": raw_idx,
        "level": level,
        "anchor_idx": anchor_idx,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "feature_w": nx,
        "feature_h": ny,
        "stride": stride,
        "source_x": float(source_xy[0]),
        "source_y": float(source_xy[1]),
        "anchor_w": float(prior_xywh[2]),
        "anchor_h": float(prior_xywh[3]),
        "anchor_xmin": float(prior_xyxy[0]),
        "anchor_ymin": float(prior_xyxy[1]),
        "anchor_xmax": float(prior_xyxy[2]),
        "anchor_ymax": float(prior_xyxy[3]),
        "final_xmin": float(final_box[0]),
        "final_ymin": float(final_box[1]),
        "final_xmax": float(final_box[2]),
        "final_ymax": float(final_box[3]),
        "final_score": score,
        "letterbox_ratio": float(ratio[0]),
        "letterbox_pad_x": float(pad[0]),
        "letterbox_pad_y": float(pad[1]),
    }]).to_csv(OUTPUT_DIR / "source_anchor_metadata.csv", index=False)

    make_contact_sheet(
        [
            ("feature map + source cell", OUTPUT_DIR / "feature_map_source_cell.jpg"),
            ("feature map overlay", OUTPUT_DIR / "feature_map_overlay_source.jpg"),
            ("image + final/source", OUTPUT_DIR / "image_with_final_source_anchor_location.jpg"),
            ("decoded source only", OUTPUT_DIR / "decoded_source_anchor_location.jpg"),
        ],
        OUTPUT_DIR / "contact_sheet.jpg",
    )

    print(f"saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
