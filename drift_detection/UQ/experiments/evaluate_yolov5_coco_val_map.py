from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_DETECTORS_ROOT = REPO_ROOT / "object_detectors"
for _path in (REPO_ROOT, OBJECT_DETECTORS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from commands.predict.common import _resolve_nms_logits  # noqa: E402
from commands.utils.predict_utils import build_detector, preprocess_with_letterbox  # noqa: E402

IOU_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.50, 0.96, 0.05)]


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pure YOLOv5 on COCO val2017 with COCO mAP.")
    parser.add_argument(
        "--config",
        default="object_detectors/runs/yolov5/predict/coco/06-30-2026_14;31_score/used_config.yaml",
    )
    parser.add_argument("--annotation", default="D:/SEONGJIN/datasets/COCO/annotations/instances_val2017.json")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output-dir", default="documents/figures/figure 3/yolov5_val2017_map_check")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--coco-max-dets", default="1,10,100")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--auto-letterbox", action="store_true")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--category-map", choices=["name", "coco80"], default="name")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_detector(config: dict[str, Any], conf_thres: float, nms_iou: float, max_det: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    model_cfg = cfg.setdefault("model", {})
    model_cfg["confidence_threshold"] = float(conf_thres)
    model_cfg["iou_threshold"] = float(nms_iou)
    model_cfg["max_det"] = int(max_det)
    return cfg


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def coco80_to_coco91_class() -> list[int]:
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    ]


def image_root_candidates(config: dict[str, Any], annotation: Path, explicit: str) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        roots.append(resolve_path(explicit))

    coco_cfg = config.get("dataset", {}).get("coco", {})
    root = coco_cfg.get("root", "")
    image_dir = coco_cfg.get("image_dir", "")
    val_split = coco_cfg.get("val_split", "val2017")
    if root:
        base = Path(root)
        if image_dir:
            roots.extend([base / image_dir / val_split, base / image_dir])
        roots.append(base / val_split)

    roots.extend(
        [
            annotation.parent.parent / "val2017",
            annotation.parent.parent / "images" / "val2017",
            Path("D:/SEONGJIN/datasets/COCO/val2017"),
            Path("D:/SEONGJIN/datasets/COCO/images/val2017"),
            Path("D:/DataDrift/datasets/COCO/val2017"),
            Path("D:/DataDrift/datasets/COCO/images/val2017"),
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for root_path in roots:
        key = str(root_path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root_path)
    return unique


def selected_image_ids(coco: COCO, num_images: int | None, limit_images: int | None) -> list[int]:
    ids = sorted(int(x) for x in coco.getImgIds())
    n = len(ids)
    if num_images is not None:
        n = min(n, int(num_images))
    if limit_images is not None:
        n = min(n, int(limit_images))
    return ids[:n]


def resolve_images(
    coco: COCO,
    image_ids: list[int],
    config: dict[str, Any],
    annotation: Path,
    image_root: str,
) -> list[ImageRecord]:
    roots = image_root_candidates(config, annotation, image_root)
    records: list[ImageRecord] = []
    for image_id in image_ids:
        info = coco.imgs[int(image_id)]
        found = None
        for root in roots:
            candidate = root / info["file_name"]
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(f"Could not find {info['file_name']} in: {'; '.join(map(str, roots))}")
        records.append(
            ImageRecord(
                image_id=int(image_id),
                file_name=str(info["file_name"]),
                width=int(info["width"]),
                height=int(info["height"]),
                path=found,
            )
        )
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


def restore_box(
    box: torch.Tensor,
    ratio: tuple[float, float],
    pad: tuple[float, float],
    width: int,
    height: int,
) -> list[float]:
    rw, rh = ratio
    pw, ph = pad
    x1 = min(max((float(box[0]) - pw) / rw, 0.0), float(width))
    y1 = min(max((float(box[1]) - ph) / rh, 0.0), float(height))
    x2 = min(max((float(box[2]) - pw) / rw, 0.0), float(width))
    y2 = min(max((float(box[3]) - ph) / rh, 0.0), float(height))
    return [x1, y1, x2, y2]


def class_name(detector, cls_idx: int) -> str:
    if isinstance(detector.names, dict):
        return str(detector.names.get(cls_idx, cls_idx))
    if isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
        return str(detector.names[cls_idx])
    return str(cls_idx)


def predict_batch(
    detector,
    records: list[ImageRecord],
    device: torch.device,
    cat_name_to_id: dict[str, int],
    conf_thres: float,
    nms_iou: float,
    max_det: int,
    auto_letterbox: bool,
    agnostic_nms: bool,
    category_map: str,
) -> list[dict[str, Any]]:
    infer_parts, ratios, pads = [], [], []
    for record in records:
        infer, ratio, pad, _ = preprocess_with_letterbox(
            detector,
            load_image(record.path),
            device,
            requires_grad=False,
            auto=bool(auto_letterbox),
        )
        infer_parts.append(infer)
        ratios.append(tuple(float(v) for v in ratio))
        pads.append(tuple(float(v) for v in pad))

    infer_batch = torch.cat(infer_parts, dim=0)
    with torch.no_grad():
        model_output = detector.model(infer_batch, augment=False)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
        nms_logits = _resolve_nms_logits(
            raw_prediction,
            raw_logits,
            num_classes_hint=len(detector.names) if detector.names is not None else 80,
        )
        selected_preds, _selected_logits, _selected_objectness, _selected_indices = detector.non_max_suppression(
            prediction=raw_prediction,
            logits=nms_logits,
            conf_thres=float(conf_thres),
            iou_thres=float(nms_iou),
            classes=None,
            agnostic=bool(agnostic_nms),
            max_det=int(max_det),
            return_indices=True,
        )

    results: list[dict[str, Any]] = []
    for sample_idx, record in enumerate(records):
        detections = selected_preds[sample_idx] if sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
        for det in detections:
            cls_idx = int(det[5].detach().cpu().item()) if det.shape[0] > 5 else 0
            if category_map == "coco80":
                category_id = coco80_to_coco91_class()[cls_idx] if 0 <= cls_idx < 80 else None
            else:
                name = class_name(detector, cls_idx)
                category_id = cat_name_to_id.get(name)
            if category_id is None:
                continue
            x1, y1, x2, y2 = restore_box(det[:4].detach().cpu(), ratios[sample_idx], pads[sample_idx], record.width, record.height)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue
            results.append(
                {
                    "image_id": int(record.image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(det[4].detach().cpu().item()),
                }
            )
    return results


def evaluate_coco(coco: COCO, image_ids: list[int], predictions: list[dict[str, Any]], max_dets: list[int]) -> dict[str, float]:
    if not predictions:
        return {name: 0.0 for name in metric_names()}
    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.imgIds = list(map(int, image_ids))
    evaluator.params.iouThrs = np.asarray(IOU_THRESHOLDS, dtype=np.float64)
    evaluator.params.maxDets = list(max_dets)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {name: float(value * 100.0) for name, value in zip(metric_names(), evaluator.stats.tolist())}


def metric_names() -> list[str]:
    return [
        "AP@[.50:.95]",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
    ]


def write_sample_images(path: Path, image_ids: list[int]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id"])
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow({"image_id": int(image_id)})


def write_summary(path: Path, row: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    annotation_path = resolve_path(args.annotation)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_max_dets = parse_int_list(args.coco_max_dets)

    config = configure_detector(load_yaml(config_path), args.conf_thres, args.nms_iou, args.max_det)
    coco = COCO(str(annotation_path))
    image_ids = selected_image_ids(coco, args.num_images, args.limit_images)
    records = resolve_images(coco, image_ids, config, annotation_path, args.image_root)
    cat_name_to_id = {str(cat["name"]): int(cat["id"]) for cat in coco.dataset["categories"]}

    detector, device = build_detector(config)
    predictions: list[dict[str, Any]] = []
    for batch in tqdm(list(chunks(records, args.batch_size)), desc="YOLOv5 val2017 COCO mAP", unit="batch"):
        predictions.extend(
            predict_batch(
                detector=detector,
                records=batch,
                device=device,
                cat_name_to_id=cat_name_to_id,
                conf_thres=args.conf_thres,
                nms_iou=args.nms_iou,
                max_det=args.max_det,
                auto_letterbox=args.auto_letterbox,
                agnostic_nms=args.agnostic_nms,
                category_map=args.category_map,
            )
        )

    metrics = evaluate_coco(coco, image_ids, predictions, coco_max_dets)
    summary = {
        "num_images": len(image_ids),
        "num_detections": len(predictions),
        "conf_thres": float(args.conf_thres),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "coco_max_dets": ",".join(map(str, coco_max_dets)),
        "auto_letterbox": bool(args.auto_letterbox),
        "agnostic_nms": bool(args.agnostic_nms),
        "category_map": str(args.category_map),
        "config": str(config_path),
        "annotation": str(annotation_path),
        **metrics,
    }

    write_sample_images(output_dir / f"sample_images_{len(image_ids)}.csv", image_ids)
    with open(output_dir / "coco_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary(output_dir / "summary.csv", summary)

    print("\nYOLOv5 COCO val2017 mAP sanity check")
    print(
        f"images={len(image_ids)} detections={len(predictions)} "
        f"conf={args.conf_thres} nms_iou={args.nms_iou} max_det={args.max_det} "
        f"coco_max_dets={coco_max_dets} auto_letterbox={bool(args.auto_letterbox)} "
        f"agnostic_nms={bool(args.agnostic_nms)} category_map={args.category_map}"
    )
    for key in metric_names():
        print(f"{key}: {summary[key]:.3f}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
