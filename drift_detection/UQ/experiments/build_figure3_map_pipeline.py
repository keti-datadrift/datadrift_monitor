from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_DETECTORS_ROOT = REPO_ROOT / "object_detectors"
META_MODELS_ROOT = REPO_ROOT / "meta_models"
for _path in (REPO_ROOT, OBJECT_DETECTORS_ROOT, META_MODELS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from commands.predict.yolov5.mc_dropout import (  # noqa: E402
    _forward_yolov5_features_to_head,
    _forward_yolov5_head_from_cache,
    _mc_feature_tensor,
)
from commands.utils.predict_utils import (  # noqa: E402
    _class_loss_tensor,
    _flatten_raw_prediction_layers,
    _objectness_loss_tensor,
    _xywh_to_xyxy_tensor,
    build_detector,
    build_layer_target_scalar_bbox,
    build_yolo_candidate_cache,
    enable_forced_mc_dropout_on_yolov5_head,
    format_gradient_output,
    get_prediction_class_probs,
    preprocess_with_letterbox,
    resolve_layer_parameter,
    yolo_candidate_mask_from_cache,
)
from meta_models.losses.meta_classifier import compute_fpr_at_tpr  # noqa: E402
from meta_models.models.meta_classifier import build_estimator  # noqa: E402

BASE_COLUMNS = [
    "image_id", "image_path", "raw_pred_idx",
    "xmin", "ymin", "xmax", "ymax",
    "coco_xmin", "coco_ymin", "coco_xmax", "coco_ymax",
    "score", "pred_class", "category_id",
]
IOU_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.50, 0.96, 0.05)]
DEFAULT_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.00, 1.0001, 0.05)]
METHOD_ALIASES = {
    "score": "score", "mc": "mc_dropout", "mc_dropout": "mc_dropout",
    "ensemble": "ensemble", "metadetect": "meta_detect", "meta_detect": "meta_detect", "md": "meta_detect",
    "gradscore": "gradscore", "gs": "gradscore",
    "unto_o": "unto_o", "unto-o": "unto_o", "unto_g": "unto_g", "unto-g": "unto_g",
    "md_unto_g": "md_unto_g", "md+unto-g": "md_unto_g",
}
LABEL_COLUMNS = ["max_iou"] + [f"tp_iou{int(t * 100):02d}" for t in IOU_THRESHOLDS]
FEATURE_META_COLUMNS = set(BASE_COLUMNS + LABEL_COLUMNS)


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    path: Path


@dataclass
class ForwardPack:
    infer_batch: torch.Tensor
    ratios: list[tuple[float, float]]
    pads: list[tuple[float, float]]
    raw_prediction: torch.Tensor
    raw_logits: torch.Tensor | None
    raw_anchor_priors: torch.Tensor | None
    raw_flat: torch.Tensor | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Figure 3 raw-detection COCO mAP pipeline outputs.")
    parser.add_argument("--config", default="object_detectors/runs/yolov5/predict/coco/06-30-2026_14;31_score/used_config.yaml")
    parser.add_argument("--sample-images", default="documents/figures/figure 3/figure4_sample_images.csv")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--annotation", default="D:/DataDrift/datasets/COCO/annotations/instances_train2017.json")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default="documents/figures/figure 3/raw_map_pipeline")
    parser.add_argument("--methods", nargs="+", default=["score", "mc_dropout", "ensemble", "meta_detect", "gradscore", "unto_o", "unto_g", "md_unto_g"])
    parser.add_argument("--thresholds", default=",".join(f"{x:.2f}" for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--raw-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-nms-candidates", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--meta-model", default="gb_classifier", choices=["gb_classifier", "logistic"])
    parser.add_argument("--meta-device", default="cpu")
    parser.add_argument("--mc-runs", type=int, default=30)
    parser.add_argument("--mc-dropout-rate", type=float, default=0.5)
    parser.add_argument("--gradient-layers", nargs="+", default=["model.24.m.0", "model.24.m.1", "model.24.m.2"])
    parser.add_argument("--gradient-reductions", nargs="+", default=["l1_norm", "l2_norm", "min", "max", "mean", "std"])
    parser.add_argument("--gradient-max-rows", type=int, default=None)
    parser.add_argument("--skip-feature-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-map", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_raw(config: dict[str, Any], raw_conf: float, nms_iou: float) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("model", {})["confidence_threshold"] = float(raw_conf)
    cfg.setdefault("model", {})["iou_threshold"] = float(nms_iou)
    return cfg


def normalize_methods(methods: Iterable[str]) -> list[str]:
    out: list[str] = []
    for method in methods:
        key = str(method).strip().lower().replace("-", "_")
        norm = METHOD_ALIASES.get(key, METHOD_ALIASES.get(str(method).strip().lower()))
        if norm is None:
            raise ValueError(f"Unsupported method: {method}")
        if norm not in out:
            out.append(norm)
    if "md_unto_g" in out:
        for dep in ("meta_detect", "unto_g"):
            if dep not in out:
                out.append(dep)
    return out


def parse_thresholds(text: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("No thresholds provided.")
    return values


def load_sample_ids(path: Path, num_images: int, limit_images: int | None) -> list[int]:
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"Missing image_id column in {path}")
    n = min(int(num_images), int(limit_images)) if limit_images is not None else int(num_images)
    ids = df["image_id"].astype(int).tolist()[:n]
    if len(ids) != n:
        raise ValueError(f"Requested {n} images, found {len(ids)}")
    return ids


def image_root_candidates(config: dict[str, Any], annotation: Path, explicit: str) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        roots.append(resolve_path(explicit))
    coco_cfg = config.get("dataset", {}).get("coco", {})
    root = coco_cfg.get("root", "")
    split = coco_cfg.get("train_split", "train2017")
    image_dir = coco_cfg.get("image_dir", "")
    if root:
        base = Path(root)
        if image_dir:
            roots.extend([base / image_dir / split, base / image_dir])
        roots.append(base / split)
    roots.extend([
        annotation.parent.parent / "train2017",
        annotation.parent.parent / "images" / "train2017",
        Path("D:/DataDrift/datasets/COCO/train2017"),
        Path("D:/DataDrift/datasets/COCO/images/train2017"),
        Path("D:/SEONGJIN/datasets/COCO/train2017"),
        Path("D:/SEONGJIN/datasets/COCO/images/train2017"),
    ])
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_images(coco: COCO, image_ids: list[int], config: dict[str, Any], annotation: Path, image_root: str) -> list[ImageRecord]:
    roots = image_root_candidates(config, annotation, image_root)
    records: list[ImageRecord] = []
    for image_id in image_ids:
        info = coco.imgs[int(image_id)]
        found = None
        for root in roots:
            path = root / info["file_name"]
            if path.is_file():
                found = path
                break
        if found is None:
            raise FileNotFoundError(f"Could not find {info['file_name']} in: {'; '.join(map(str, roots))}")
        records.append(ImageRecord(int(image_id), str(info["file_name"]), int(info["width"]), int(info["height"]), found))
    return records


def load_image(path: Path) -> torch.Tensor:
    if cv2 is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        if Image is None:
            raise ImportError("Either opencv-python or Pillow is required to read images.")
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"))
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def chunks(items: list[Any], size: int):
    for start in range(0, len(items), int(size)):
        yield items[start : start + int(size)]


def forward_batch(detector, records: list[ImageRecord], device: torch.device, requires_grad: bool) -> ForwardPack:
    infer_parts, ratios, pads = [], [], []
    for record in records:
        infer, ratio, pad, _ = preprocess_with_letterbox(
            detector, load_image(record.path), device, requires_grad=requires_grad, auto=False
        )
        infer_parts.append(infer)
        ratios.append(tuple(float(v) for v in ratio))
        pads.append(tuple(float(v) for v in pad))
    infer_batch = torch.cat(infer_parts, dim=0)
    model_output = detector.model(infer_batch, augment=False)
    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
    raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
    raw_layers = model_output[2] if isinstance(model_output, (tuple, list)) and len(model_output) > 2 else None
    raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
    raw_flat = _flatten_raw_prediction_layers(raw_layers)
    return ForwardPack(infer_batch, ratios, pads, raw_prediction, raw_logits, raw_anchor_priors, raw_flat)


def restore_box(box: torch.Tensor, ratio: tuple[float, float], pad: tuple[float, float], width: int, height: int) -> list[float]:
    rw, rh = ratio
    pw, ph = pad
    x1 = min(max((float(box[0]) - pw) / rw, 0.0), float(width))
    y1 = min(max((float(box[1]) - ph) / rh, 0.0), float(height))
    x2 = min(max((float(box[2]) - pw) / rw, 0.0), float(width))
    y2 = min(max((float(box[3]) - ph) / rh, 0.0), float(height))
    return [x1, y1, x2, y2]


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return inter / np.clip(area1 + area2 - inter, 1e-12, None)


def gt_by_image_class(coco: COCO, image_ids: list[int]) -> dict[int, dict[int, np.ndarray]]:
    tmp: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_ids)):
        if int(ann.get("iscrowd", 0)):
            continue
        x, y, w, h = ann["bbox"]
        if w > 0 and h > 0:
            tmp[int(ann["image_id"])][int(ann["category_id"])].append([x, y, x + w, y + h])
    return {i: {c: np.asarray(b, dtype=np.float32) for c, b in cls.items()} for i, cls in tmp.items()}


def raw_rows_for_sample(detector, record: ImageRecord, sample_idx: int, pack: ForwardPack, raw_conf: float, max_nms: int, cat_name_to_id: dict[str, int], gt_map: dict[int, dict[int, np.ndarray]]) -> list[dict[str, Any]]:
    pred = pack.raw_prediction[sample_idx].detach().float()
    if pred.numel() == 0:
        return []
    obj = pred[:, 4]
    probs = pred[:, 5:]
    cls_conf, cls_idx = probs.max(dim=1) if probs.numel() else (torch.ones_like(obj), torch.zeros_like(obj, dtype=torch.long))
    score = obj * cls_conf
    keep = torch.nonzero((obj > raw_conf) & (score > raw_conf), as_tuple=False).flatten()
    if int(keep.numel()) > int(max_nms):
        keep = keep[torch.argsort(score[keep], descending=True)[: int(max_nms)]]
    xyxy = _xywh_to_xyxy_tensor(pred[:, :4])
    rows: list[dict[str, Any]] = []
    for raw_idx in keep.detach().cpu().tolist():
        raw_idx = int(raw_idx)
        class_idx = int(cls_idx[raw_idx].detach().cpu().item())
        pred_class = detector.names[class_idx] if detector.names is not None else str(class_idx)
        category_id = int(cat_name_to_id.get(str(pred_class), -1))
        letter = xyxy[raw_idx].detach().cpu().float()
        coco_box = restore_box(letter, pack.ratios[sample_idx], pack.pads[sample_idx], record.width, record.height)
        same_gt = gt_map.get(record.image_id, {}).get(category_id, np.zeros((0, 4), dtype=np.float32))
        max_iou = float(iou_one_to_many(np.asarray(coco_box, dtype=np.float32), same_gt).max()) if same_gt.size else 0.0
        row = {
            "image_id": record.image_id, "image_path": str(record.path), "raw_pred_idx": raw_idx,
            "xmin": float(letter[0]), "ymin": float(letter[1]), "xmax": float(letter[2]), "ymax": float(letter[3]),
            "coco_xmin": coco_box[0], "coco_ymin": coco_box[1], "coco_xmax": coco_box[2], "coco_ymax": coco_box[3],
            "score": float(score[raw_idx].detach().cpu().item()), "pred_class": str(pred_class), "category_id": category_id,
            "max_iou": max_iou,
        }
        for thr in IOU_THRESHOLDS:
            row[f"tp_iou{int(thr * 100):02d}"] = int(max_iou >= thr)
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    append = path.exists()
    with open(path, "a" if append else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def base_fields() -> list[str]:
    return BASE_COLUMNS + LABEL_COLUMNS


def score_fields() -> list[str]:
    return BASE_COLUMNS + ["score_indicator"]


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "score_indicator": r["score"]} for r in rows]


def mean_std_fields(num_classes: int) -> list[str]:
    fields = BASE_COLUMNS + ["xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean", "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std"]
    for i in range(num_classes):
        fields += [f"prob_{i}_mean", f"prob_{i}_std"]
    return fields


def raw_feature_tensor(detector, raw_prediction: torch.Tensor) -> torch.Tensor:
    pred = raw_prediction.detach().float()
    return torch.cat([_xywh_to_xyxy_tensor(pred[..., :4]), pred[..., 4:5], get_prediction_class_probs(detector, pred).detach().float()], dim=2)


def rows_from_mean_std(rows: list[dict[str, Any]], mean: torch.Tensor, std: torch.Tensor, sample_idx: int, num_classes: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        idx = int(row["raw_pred_idx"])
        mv = mean[sample_idx, idx].detach().cpu().float().numpy()
        sv = std[sample_idx, idx].detach().cpu().float().numpy()
        new = dict(row)
        for name, value in zip(["xmin", "ymin", "xmax", "ymax", "score"], mv[:5]):
            new[f"{name}_mean"] = float(value)
        for name, value in zip(["xmin", "ymin", "xmax", "ymax", "score"], sv[:5]):
            new[f"{name}_std"] = float(value)
        for i in range(num_classes):
            new[f"prob_{i}_mean"] = float(mv[5 + i]) if 5 + i < len(mv) else 0.0
            new[f"prob_{i}_std"] = float(sv[5 + i]) if 5 + i < len(sv) else 0.0
        out.append(new)
    return out


def to_float(v: Any) -> float:
    return float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)


def stat4(v: torch.Tensor, device: torch.device):
    if v is None or v.numel() == 0:
        z = torch.zeros((), dtype=torch.float32, device=device)
        return z, z, z, z
    x = v.detach().float().reshape(-1)
    return x.min(), x.max(), x.mean(), x.std(unbiased=False)


def meta_detect_fields(num_classes: int) -> list[str]:
    return BASE_COLUMNS + ["prob_sum"] + [f"prob_{i}" for i in range(num_classes)] + [
        "num_candidate_boxes", "x_min", "x_max", "x_mean", "x_std", "y_min", "y_max", "y_mean", "y_std",
        "w_min", "w_max", "w_mean", "w_std", "h_min", "h_max", "h_mean", "h_std",
        "size", "size_min", "size_max", "size_mean", "size_std", "circum", "circum_min", "circum_max", "circum_mean", "circum_std",
        "size_circum", "size_circum_min", "size_circum_max", "size_circum_mean", "size_circum_std",
        "score_min", "score_max", "score_mean", "score_std", "iou_pb_min", "iou_pb_max", "iou_pb_mean", "iou_pb_std",
    ]


def build_meta_detect_rows(pack: ForwardPack, rows: list[dict[str, Any]], sample_idx: int, num_classes: int, iou_threshold: float) -> list[dict[str, Any]]:
    device = pack.raw_prediction.device
    pred = pack.raw_prediction[sample_idx].detach().float()
    cache = build_yolo_candidate_cache(pred, 0.0)
    out = []
    for row in rows:
        raw_idx = int(row["raw_pred_idx"])
        mask, ious = yolo_candidate_mask_from_cache(cache, raw_idx, iou_threshold)
        if mask is None:
            mask = torch.zeros((pred.shape[0],), dtype=torch.bool, device=device)
            ious = torch.zeros((pred.shape[0],), dtype=torch.float32, device=device)
        boxes = cache.raw_xyxy[mask]
        scores = cache.raw_score[mask]
        cand_ious = ious[mask]
        probs = pred[raw_idx, 5:].detach().float() if pred.shape[1] > 5 else torch.zeros((0,), device=device)
        new = dict(row)
        new["prob_sum"] = to_float(probs.sum() if probs.numel() else torch.zeros((), device=device))
        for i in range(num_classes):
            new[f"prob_{i}"] = to_float(probs[i]) if i < probs.shape[0] else 0.0
        if boxes.numel():
            x = 0.5 * (boxes[:, 0] + boxes[:, 2]); y = 0.5 * (boxes[:, 1] + boxes[:, 3])
            w = torch.abs(boxes[:, 2] - boxes[:, 0]); h = torch.abs(boxes[:, 3] - boxes[:, 1])
        else:
            x = y = w = h = torch.zeros((0,), device=device)
        size = w * h; circum = w + h; sc = size / circum.clamp(min=1e-12)
        iou_pb = torch.where(cand_ious == 1.0, torch.zeros_like(cand_ious), cand_ious)
        iou_pb = iou_pb[iou_pb > 0]
        fbox = cache.raw_xyxy[raw_idx]
        fw = torch.abs(fbox[2] - fbox[0]); fh = torch.abs(fbox[3] - fbox[1])
        fsize = fw * fh; fcircum = fw + fh
        new.update({"num_candidate_boxes": float(boxes.shape[0]), "size": to_float(fsize), "circum": to_float(fcircum), "size_circum": to_float(fsize / fcircum.clamp(min=1e-12))})
        for prefix, values in [("x", x), ("y", y), ("w", w), ("h", h), ("size", size), ("circum", circum), ("size_circum", sc), ("score", scores), ("iou_pb", iou_pb)]:
            mn, mx, mean, std = stat4(values, device)
            new[f"{prefix}_min"] = to_float(mn); new[f"{prefix}_max"] = to_float(mx); new[f"{prefix}_mean"] = to_float(mean); new[f"{prefix}_std"] = to_float(std)
        out.append(new)
    return out


def shape_features(box: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
    fx = 0.5 * (box[0] + box[2]); fy = 0.5 * (box[1] + box[3]); fw = torch.abs(box[2] - box[0]); fh = torch.abs(box[3] - box[1])
    rx = 0.5 * (ref[0] + ref[2]); ry = 0.5 * (ref[1] + ref[3]); rw = torch.abs(ref[2] - ref[0]); rh = torch.abs(ref[3] - ref[1])
    fs = fw * fh; fc = fw + fh; fsc = fs / fc.clamp(min=1e-12)
    rs = rw * rh; rc = rw + rh; rsc = rs / rc.clamp(min=1e-12)
    return {"size": fs, "circum": fc, "size_circum": fsc, "size_diff": torch.abs(fs - rs), "circum_diff": torch.abs(fc - rc), "size_circum_diff": torch.abs(fsc - rsc), "x_loss": torch.abs(fx - rx), "y_loss": torch.abs(fy - ry), "w_loss": torch.abs(fw - rw), "h_loss": torch.abs(fh - rh)}


def unto_o_fields(num_classes: int) -> list[str]:
    return BASE_COLUMNS + ["prob_sum"] + [f"prob_{i}" for i in range(num_classes)] + ["final_score", "size", "size_diff", "circum", "circum_diff", "size_circum", "size_circum_diff", "x_loss", "y_loss", "w_loss", "h_loss", "obj_loss", "cls_loss"]


def build_unto_o_rows(pack: ForwardPack, rows: list[dict[str, Any]], sample_idx: int, num_classes: int) -> list[dict[str, Any]]:
    if pack.raw_flat is None or pack.raw_anchor_priors is None:
        raise RuntimeError("UnTO-O requires raw layers and anchor priors.")
    device = pack.raw_prediction.device
    pred = pack.raw_prediction[sample_idx].float()
    raw = pack.raw_flat[sample_idx]
    anchors = pack.raw_anchor_priors[sample_idx] if pack.raw_anchor_priors.ndim >= 3 else pack.raw_anchor_priors
    anchor_xyxy = _xywh_to_xyxy_tensor(anchors.to(dtype=pred.dtype, device=device))
    out = []
    for row in rows:
        raw_idx = int(row["raw_pred_idx"])
        pred_row = pred[raw_idx]; raw_row = raw[raw_idx]
        probs = pred_row[5:].detach().float() if pred_row.shape[0] > 5 else torch.zeros((0,), device=device)
        box = _xywh_to_xyxy_tensor(pred[raw_idx : raw_idx + 1, :4]).view(4).detach().float()
        shape = shape_features(box, anchor_xyxy[raw_idx])
        obj_loss = _objectness_loss_tensor(raw_row[4], torch.full_like(raw_row[4], 0.5), mode="bcewithlogits", direction="pred_to_target", reduction="sum")
        cls_logits = raw_row[5:]
        cls_loss = _class_loss_tensor(cls_logits, torch.full_like(cls_logits, 1.0 / float(cls_logits.numel())), mode="kl", direction="pred_to_target", reduction="sum") if cls_logits.numel() else torch.zeros((), device=device)
        new = dict(row)
        new["prob_sum"] = to_float(probs.sum() if probs.numel() else torch.zeros((), device=device))
        for i in range(num_classes):
            new[f"prob_{i}"] = to_float(probs[i]) if i < probs.shape[0] else 0.0
        new.update({"final_score": row["score"], "obj_loss": to_float(obj_loss), "cls_loss": to_float(cls_loss)})
        new.update({k: to_float(v) for k, v in shape.items()})
        out.append(new)
    return out


def gradient_fields(layers: list[str], reductions: list[str]) -> list[str]:
    fields = BASE_COLUMNS[:]
    for target in ["bbox_loss", "cls_loss", "obj_loss"]:
        for layer in layers:
            for red in reductions:
                fields.append(f"{target}_{layer}_{red}")
    return fields


def build_gradient_rows(detector, pack: ForwardPack, rows_by_sample: dict[int, list[dict[str, Any]]], method: str, layers: list[str], reductions: list[str], raw_conf: float, nms_iou: float, max_rows: int | None, count: int) -> tuple[list[dict[str, Any]], int]:
    if pack.raw_flat is None or pack.raw_anchor_priors is None:
        raise RuntimeError("Gradient features require raw layers and anchor priors.")
    pseudo = "cand" if method == "gradscore" else "uniform"
    params = [resolve_layer_parameter(detector.model, name) for name in layers]
    original = [bool(p.requires_grad) for p in params]
    for p in params:
        p.requires_grad_(True)
    out: list[dict[str, Any]] = []
    try:
        for sample_idx, rows in rows_by_sample.items():
            pred = pack.raw_prediction[sample_idx].float()
            logits = pack.raw_logits[sample_idx].float() if pack.raw_logits is not None else pred[:, 5:].float()
            raw = pack.raw_flat[sample_idx]
            anchors = pack.raw_anchor_priors[sample_idx] if pack.raw_anchor_priors.ndim >= 3 else pack.raw_anchor_priors
            cache = build_yolo_candidate_cache(pred.detach(), raw_conf) if pseudo == "cand" else None
            for row in rows:
                if max_rows is not None and count >= int(max_rows):
                    continue
                raw_idx = int(row["raw_pred_idx"])
                candidate_mask = None
                if cache is not None:
                    candidate_mask, _ = yolo_candidate_mask_from_cache(cache, raw_idx, nms_iou)
                new = dict(row)
                for target in ["bbox_loss", "cls_loss", "obj_loss"]:
                    scalar = build_layer_target_scalar_bbox(
                        target, pred, logits, raw, raw_idx, nms_iou, pseudo_gt=pseudo,
                        anchor_xywh=anchors[raw_idx] if raw_idx < int(anchors.shape[0]) else None,
                        cand_score_threshold=raw_conf, bbox_loss="box_l1", cls_loss="kl", obj_loss="bcewithlogits",
                        bbox_direction="pred_to_target", cls_direction="pred_to_target", obj_direction="pred_to_target",
                        candidate_mask=candidate_mask,
                    )
                    grads = [None for _ in params] if scalar is None else torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
                    for layer, grad in zip(layers, grads):
                        formatted = format_gradient_output(grad, reductions, map_reduction="none")
                        for red in reductions:
                            new[f"{target}_{layer}_{red}"] = to_float(formatted.get(red, 0.0)) if isinstance(formatted, dict) else 0.0
                    if scalar is not None:
                        del scalar, grads
                out.append(new)
                count += 1
        detector.zero_grad(set_to_none=True)
    finally:
        for p, req in zip(params, original):
            p.requires_grad_(req)
    return out, count


def build_feature_csvs(args: argparse.Namespace, config: dict[str, Any], coco: COCO, records: list[ImageRecord], methods: list[str], out_dir: Path) -> None:
    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    cat_name_to_id = {str(c["name"]): int(c["id"]) for c in coco.dataset["categories"]}
    gt_map = gt_by_image_class(coco, [r.image_id for r in records])
    files = {m: out_dir / f"{m}.csv" for m in ["gt", "score", "mc_dropout", "ensemble", "meta_detect", "unto_o", "gradscore", "unto_g"]}
    for p in files.values():
        if p.exists():
            p.unlink()
    ensemble_detectors = None
    if "ensemble" in methods:
        weights = config.get("output", {}).get("ensemble", {}).get("weights", [])
        if not isinstance(weights, list) or not weights:
            raise ValueError("Ensemble requested but config output.ensemble.weights is empty.")
        ensemble_detectors = [build_detector(config, model_weight=str(w))[0] for w in weights]
    grad_counts = {"gradscore": 0, "unto_g": 0}
    need_grad = any(m in methods for m in ["gradscore", "unto_g"])
    for batch in tqdm(list(chunks(records, args.batch_size)), desc="Raw feature generation", unit="batch"):
        detector.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(need_grad):
            pack = forward_batch(detector, batch, device, requires_grad=need_grad)
        rows_by_sample: dict[int, list[dict[str, Any]]] = {}
        for sample_idx, record in enumerate(batch):
            rows = raw_rows_for_sample(detector, record, sample_idx, pack, args.raw_conf, args.max_nms_candidates, cat_name_to_id, gt_map)
            rows_by_sample[sample_idx] = rows
            write_rows(files["gt"], rows, base_fields())
            if "score" in methods:
                write_rows(files["score"], score_rows(rows), score_fields())
            if "meta_detect" in methods:
                write_rows(files["meta_detect"], build_meta_detect_rows(pack, rows, sample_idx, num_classes, args.nms_iou), meta_detect_fields(num_classes))
            if "unto_o" in methods:
                write_rows(files["unto_o"], build_unto_o_rows(pack, rows, sample_idx, num_classes), unto_o_fields(num_classes))
        if "mc_dropout" in methods:
            cached = _forward_yolov5_features_to_head(detector.model, pack.infer_batch.detach())
            feature_runs = []
            handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, args.mc_dropout_rate)
            try:
                with torch.no_grad():
                    for _ in range(args.mc_runs):
                        output = _forward_yolov5_head_from_cache(detector, cached)
                        raw_prediction = output[0] if isinstance(output, (tuple, list)) else output
                        feature_runs.append(_mc_feature_tensor(detector, raw_prediction).detach())
            finally:
                for h in handles:
                    h.remove()
            stack = torch.stack(feature_runs, dim=0); mean = stack.mean(dim=0); std = stack.std(dim=0, unbiased=False)
            for sample_idx, rows in rows_by_sample.items():
                write_rows(files["mc_dropout"], rows_from_mean_std(rows, mean, std, sample_idx, num_classes), mean_std_fields(num_classes))
            del cached, feature_runs, stack, mean, std
        if "ensemble" in methods and ensemble_detectors is not None:
            sum_feat = None; sq_sum = None; count = 0
            with torch.no_grad():
                for ens in ensemble_detectors:
                    output = ens.model(pack.infer_batch.detach(), augment=False)
                    raw_prediction = output[0] if isinstance(output, (tuple, list)) else output
                    feat = raw_feature_tensor(ens, raw_prediction)
                    sum_feat = feat.clone() if sum_feat is None else sum_feat + feat
                    sq_sum = feat.square() if sq_sum is None else sq_sum + feat.square()
                    count += 1
            mean = sum_feat / float(count); std = (sq_sum / float(count) - mean.square()).clamp(min=0.0).sqrt()
            for sample_idx, rows in rows_by_sample.items():
                write_rows(files["ensemble"], rows_from_mean_std(rows, mean, std, sample_idx, num_classes), mean_std_fields(num_classes))
            del sum_feat, sq_sum, mean, std
        for method in ["gradscore", "unto_g"]:
            if method in methods:
                grad_rows, grad_counts[method] = build_gradient_rows(detector, pack, rows_by_sample, method, args.gradient_layers, args.gradient_reductions, args.raw_conf, args.nms_iou, args.gradient_max_rows, grad_counts[method])
                write_rows(files[method], grad_rows, gradient_fields(args.gradient_layers, args.gradient_reductions))
        del pack, rows_by_sample
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if "md_unto_g" in methods:
        combine_csv(out_dir / "meta_detect.csv", out_dir / "unto_g.csv", out_dir / "md_unto_g.csv")


def combine_csv(left_path: Path, right_path: Path, out_path: Path) -> None:
    left = pd.read_csv(left_path); right = pd.read_csv(right_path)
    merged = left.merge(right, on=["image_id", "raw_pred_idx"], suffixes=("__md", "__ug"), how="inner")
    out = pd.DataFrame()
    for col in BASE_COLUMNS:
        out[col] = merged[col] if col in merged.columns else merged[f"{col}__md"]
    for col in merged.columns:
        if col in {"image_id", "raw_pred_idx"} or col in BASE_COLUMNS:
            continue
        if any(col == f"{base}__md" or col == f"{base}__ug" for base in BASE_COLUMNS):
            continue
        out[col] = merged[col]
    out.to_csv(out_path, index=False)


def feature_columns(df: pd.DataFrame, method: str) -> list[str]:
    if method == "score":
        return ["score_indicator"] if "score_indicator" in df.columns else ["score"]
    if method in {"mc_dropout", "ensemble"}:
        cols = [c for c in df.columns if c.endswith("_std")]
        return cols if cols else [c for c in df.columns if c not in FEATURE_META_COLUMNS]
    return [c for c in df.columns if c not in FEATURE_META_COLUMNS]


def load_training_df(out_dir: Path, method: str) -> tuple[pd.DataFrame, list[str]]:
    gt = pd.read_csv(out_dir / "gt.csv")
    feat = pd.read_csv(out_dir / f"{method}.csv")
    df = feat.merge(gt[["image_id", "raw_pred_idx"] + LABEL_COLUMNS], on=["image_id", "raw_pred_idx"], how="inner")
    cols = feature_columns(df, method)
    if not cols:
        raise ValueError(f"No features for {method}")
    return df, cols


def metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "fpr95": float("nan")}
    return {"auroc": float(roc_auc_score(y_true, y_score)), "ap": float(average_precision_score(y_true, y_score)), "fpr95": float(compute_fpr_at_tpr(y_true, y_score))}


def train_oof(args: argparse.Namespace, out_dir: Path, methods: list[str]) -> None:
    for method in [m for m in methods if (out_dir / f"{m}.csv").is_file()]:
        df, cols = load_training_df(out_dir, method)
        method_dir = out_dir / "meta_classifiers" / method
        models_dir = method_dir / "models"; results_dir = method_dir / "results"
        models_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
        x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        y = df["tp_iou50"].astype(int).to_numpy(); groups = df["image_id"].astype(int).to_numpy()
        n_splits = min(int(args.num_folds), len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed) if StratifiedGroupKFold is not None and len(np.unique(y)) > 1 else GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(x, y, groups=groups)
        oof = np.full(len(df), np.nan, dtype=np.float32); rows = []
        for fold, (tr, te) in enumerate(tqdm(list(split_iter), desc=f"Meta classifier {method}", unit="fold")):
            est = build_estimator(args.meta_model, device=args.meta_device, random_seed=args.random_seed + fold)
            est.fit(x[tr], y[tr]); pred = est.predict_proba(x[te])[:, 1]
            oof[te] = pred.astype(np.float32); rows.append({"row_type": "split", "split_index": fold, **metrics(y[te], pred)})
            joblib.dump(est, models_dir / f"model_{fold}.joblib")
        mean = {"row_type": "mean", "split_index": -1}; std = {"row_type": "std", "split_index": -1}
        for key in ["auroc", "ap", "fpr95"]:
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            mean[key] = float(np.mean(vals)) if vals else float("nan"); std[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        pd.DataFrame(rows + [mean, std]).to_csv(results_dir / "evaluation_results.csv", index=False)
        eval_df = df[[c for c in BASE_COLUMNS if c in df.columns]].copy(); eval_df.insert(0, "row_index", np.arange(len(df)))
        eval_df["y_test"] = y; eval_df["y_pred"] = oof; eval_df.to_csv(results_dir / "eval_data_oof.csv", index=False)
        prob = df[["image_id", "raw_pred_idx", "coco_xmin", "coco_ymin", "coco_xmax", "coco_ymax", "category_id", "pred_class", "score"]].copy()
        prob["tp_probability"] = oof; prob.to_csv(method_dir / "raw_tp_probability.csv", index=False)
        with open(method_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"method": method, "num_rows": int(len(df)), "features": cols, "oof_metrics": metrics(y, oof)}, f, indent=2)


def nms_postprocess(df: pd.DataFrame, threshold: float, nms_iou: float, max_det: int) -> pd.DataFrame:
    keep = df[df["tp_probability"] >= float(threshold)].copy()
    if keep.empty:
        return keep
    parts = []
    for _, image_df in keep.groupby("image_id", sort=False):
        image_parts = []
        for _, cls_df in image_df.groupby("category_id", sort=False):
            boxes = torch.tensor(cls_df[["coco_xmin", "coco_ymin", "coco_xmax", "coco_ymax"]].to_numpy(dtype=np.float32))
            scores = torch.tensor(cls_df["tp_probability"].to_numpy(dtype=np.float32))
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if bool(valid.any()):
                valid_idx = np.flatnonzero(valid.numpy())
                idx = torchvision.ops.nms(boxes[valid], scores[valid], float(nms_iou)).cpu().numpy()
                image_parts.append(cls_df.iloc[valid_idx[idx]])
        if image_parts:
            parts.append(pd.concat(image_parts).sort_values("tp_probability", ascending=False).head(int(max_det)))
    return pd.concat(parts, ignore_index=True) if parts else keep.iloc[0:0].copy()


def coco_map(coco: COCO, image_ids: list[int], df: pd.DataFrame) -> dict[str, float]:
    results = []
    for r in df.itertuples(index=False):
        w = max(0.0, float(r.coco_xmax) - float(r.coco_xmin)); h = max(0.0, float(r.coco_ymax) - float(r.coco_ymin))
        if w > 0 and h > 0 and int(r.category_id) > 0:
            results.append({"image_id": int(r.image_id), "category_id": int(r.category_id), "bbox": [float(r.coco_xmin), float(r.coco_ymin), w, h], "score": float(r.tp_probability)})
    if not results:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0}
    dt = coco.loadRes(results); ev = COCOeval(coco, dt, "bbox"); ev.params.imgIds = list(map(int, image_ids))
    ev.params.iouThrs = np.asarray(IOU_THRESHOLDS, dtype=np.float64)
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"map": float(ev.stats[0] * 100.0), "ap50": float(ev.stats[1] * 100.0), "ap75": float(ev.stats[2] * 100.0)}


def compute_map_curves(args: argparse.Namespace, out_dir: Path, coco: COCO, image_ids: list[int], methods: list[str], thresholds: list[float]) -> None:
    gt = pd.read_csv(out_dir / "gt.csv")
    original = gt[["image_id", "raw_pred_idx", "coco_xmin", "coco_ymin", "coco_xmax", "coco_ymax", "category_id", "pred_class", "score"]].copy()
    original["tp_probability"] = original["score"]
    original_stats = coco_map(coco, image_ids, nms_postprocess(original, 0.0, args.nms_iou, args.max_det))
    points = []; summaries = []; post_rows = []
    for method in methods:
        p = out_dir / "meta_classifiers" / method / "raw_tp_probability.csv"
        if not p.is_file():
            continue
        prob = pd.read_csv(p)
        for threshold in thresholds:
            post = nms_postprocess(prob, threshold, args.nms_iou, args.max_det)
            stats = coco_map(coco, image_ids, post)
            points.append({"method": method, "threshold": threshold, **stats})
            summaries.append({"method": method, "threshold": threshold, "input_raw_detections": int(len(prob)), "pre_nms_detections": int((prob["tp_probability"] >= threshold).sum()), "post_nms_detections": int(len(post)), **stats, "original_score_map": original_stats["map"], "original_score_ap50": original_stats["ap50"], "original_score_ap75": original_stats["ap75"]})
            tmp = post.copy(); tmp.insert(0, "threshold", threshold); tmp.insert(0, "method", method); post_rows.append(tmp)
    pd.DataFrame(points).to_csv(out_dir / "map_vs_threshold_points.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "map_vs_threshold_summary.csv", index=False)
    if post_rows:
        pd.concat(post_rows, ignore_index=True).to_csv(out_dir / "map_vs_threshold_postprocessed_detections.csv", index=False)


def main() -> None:
    args = parse_args(); methods = normalize_methods(args.methods); thresholds = parse_thresholds(args.thresholds)
    config = configure_raw(load_yaml(resolve_path(args.config)), args.raw_conf, args.nms_iou)
    out_dir = resolve_path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(Path(args.annotation).resolve()))
    image_ids = load_sample_ids(resolve_path(args.sample_images), args.num_images, args.limit_images)
    records = resolve_images(coco, image_ids, config, Path(args.annotation).resolve(), args.image_root)
    pd.DataFrame({"image_id": image_ids}).to_csv(out_dir / f"sample_images_{len(image_ids)}.csv", index=False)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "methods": methods, "image_ids": image_ids}, f, indent=2)
    if not args.skip_feature_generation:
        build_feature_csvs(args, config, coco, records, methods, out_dir)
    if not args.skip_training:
        train_oof(args, out_dir, methods)
    if not args.skip_map:
        compute_map_curves(args, out_dir, coco, image_ids, methods, thresholds)
    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
