import csv
import json
import math
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataloaders.faster_rcnn import create_dataloader as create_faster_rcnn_dataloader
from dataloaders.fcos import create_dataloader as create_fcos_dataloader
from dataloaders.yolov10 import create_dataloader as create_yolov10_dataloader
from dataloaders.yolov5 import create_dataloader as create_yolov5_dataloader


def create_dataloader(config, split="train"):
    model_type = str(config.get("model", {}).get("type", "yolov5")).strip().lower().replace("-", "_")
    if model_type in {"yolo", "yolov5"}:
        return create_yolov5_dataloader(config, split=split)
    if model_type == "yolov10":
        return create_yolov10_dataloader(config, split=split)
    if model_type == "fcos":
        return create_fcos_dataloader(config, split=split)
    if model_type in {"faster_rcnn", "fasterrcnn", "frcnn"}:
        return create_faster_rcnn_dataloader(config, split=split)
    raise ValueError(f"Unsupported dataloader model.type: {model_type}")


from commands.utils.predict_utils import (
    assign_tp_to_predictions,
    analyze_prediction_error_types,
    _box_iou_1vN_tensor,
    build_layer_target_scalar_bbox,
    build_detector,
    enable_forced_mc_dropout_on_yolov5_head,
    collect_bbox_layer_grads_per_target,
    collect_faster_rcnn_roi_layer_grads_per_target,
    draw_predictions,
    get_fn_gt_indices,
    get_prediction_class_probs,
    get_selected_prediction_class_probs,
    get_pre_nms_keep_indices,
    has_fn_for_image,
    load_gt_category_maps,
    map_boxes_to_letterbox,
    map_grad_tensor_to_numbers,
    parse_output_config,
    preprocess_with_letterbox,
    expand_layer_names,
    resolve_layer_parameter,
)


def _xywh_to_xyxy_tensor(xywh: torch.Tensor) -> torch.Tensor:
    out = xywh.clone()
    out[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    out[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    out[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    out[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return out


def _build_summary(total_images, fn_images, output_csv):
    return {
        "total_images": total_images,
        "fn_images": fn_images,
        "fn_ratio": (fn_images / total_images) if total_images else 0.0,
        "output_csv": str(output_csv),
    }


def _as_image_list(images):
    if isinstance(images, list):
        return images
    return [images[i] for i in range(images.shape[0])]


def _prepare_infer_batch(detector, images, device, auto=False):
    image_list = _as_image_list(images)
    if bool(getattr(detector, "is_faster_rcnn", False)):
        infer_batch = image_list
        ratios = [(1.0, 1.0) for _ in image_list]
        pads = [(0.0, 0.0) for _ in image_list]
        return infer_batch, ratios, pads, None

    if bool(getattr(detector, "is_fcos", False)):
        infer_batch = image_list
        ratios = [detector.get_resize_ratio(img) for img in image_list]
        pads = [(0.0, 0.0) for _ in image_list]
        return infer_batch, ratios, pads, None

    infer_tensors = []
    ratios = []
    pads = []
    resized_chws = []
    for img in image_list:
        infer_tensor, ratio, pad, resized_chw = preprocess_with_letterbox(
            detector, img, device, requires_grad=False, auto=auto
        )
        infer_tensors.append(infer_tensor)
        ratios.append(ratio)
        pads.append(pad)
        resized_chws.append(resized_chw)
    infer_batch = torch.cat(infer_tensors, dim=0)
    return infer_batch, ratios, pads, resized_chws


def _resolve_detector_nms_kwargs(detector):
    max_det = getattr(detector, "max_det", 300)
    return {
        "conf_thres": float(getattr(detector, "conf_thresh", getattr(detector, "confidence", 0.25))),
        "iou_thres": float(getattr(detector, "iou_thresh", 0.45)),
        "classes": getattr(detector, "filter_classes", None),
        "agnostic": bool(getattr(detector, "agnostic_nms", getattr(detector, "agnostic", False))),
        "max_det": int(max_det) if max_det is not None else None,
    }


def _resolve_nms_logits(raw_prediction, raw_logits, num_classes_hint=80):
    if raw_logits is not None:
        return raw_logits
    if raw_prediction is None:
        return None
    if isinstance(raw_prediction, list):
        return None
    if raw_prediction.ndim == 3 and raw_prediction.shape[2] > 5:
        return raw_prediction[:, :, 5:]
    b = int(raw_prediction.shape[0]) if raw_prediction.ndim >= 1 else 0
    n = int(raw_prediction.shape[1]) if raw_prediction.ndim >= 2 else 0
    return torch.zeros((b, n, int(num_classes_hint)), dtype=raw_prediction.dtype, device=raw_prediction.device)


def _resolve_gt_class_names(target, catid_to_name):
    gt_names = target.get("gt_class_names")
    if gt_names is not None:
        return [str(v) for v in gt_names]
    gt_labels_tensor = target["labels"]
    return [catid_to_name.get(int(label), "__unknown__") for label in gt_labels_tensor.tolist()]


def _vector_from_grad_value(grad_value):
    if isinstance(grad_value, torch.Tensor):
        if grad_value.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = grad_value.detach().float().reshape(-1).cpu().numpy()
        return np.abs(arr.astype(np.float32, copy=False))
    if isinstance(grad_value, list):
        if len(grad_value) == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = np.asarray(grad_value, dtype=np.float32).reshape(-1)
        return np.abs(arr)
    if isinstance(grad_value, (int, float)):
        return np.asarray([abs(float(grad_value))], dtype=np.float32)
    if isinstance(grad_value, dict):

        vals = []
        for k in sorted(grad_value.keys()):
            try:
                vals.append(float(grad_value[k]))
            except Exception:
                continue
        if not vals:
            return np.zeros((0,), dtype=np.float32)
        return np.abs(np.asarray(vals, dtype=np.float32).reshape(-1))
    return np.zeros((0,), dtype=np.float32)


def _mc_dropout_single_csv_writer(write_queue, output_csv, fieldnames):
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            batch_rows = write_queue.get()
            if batch_rows is None:
                break
            if batch_rows:
                writer.writerows(batch_rows)


def _sync_timing_device(device=None):
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            dev = torch.device(device)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)


class StageTimingProfiler:
    def __init__(self, run_dir, uncertainty, unit, stages=None, device=None):
        self.run_dir = Path(run_dir)
        self.uncertainty = str(uncertainty)
        self.unit = str(unit)
        self.stages = list(stages or [])
        self.device = device
        self.records = []
        self._batch_idx = 0
        self.timing_dir = self.run_dir / "timing"
        self.csv_path = self.timing_dir / f"{self.uncertainty}_timing.csv"
        self.json_path = self.timing_dir / f"{self.uncertainty}_timing.json"
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames = None

    def start(self, device=None):
        _sync_timing_device(self.device if device is None else device)
        return time.perf_counter()

    def elapsed(self, start_t, device=None):
        _sync_timing_device(self.device if device is None else device)
        return max(0.0, float(time.perf_counter() - start_t))

    def record(self, num_images, num_predictions, stage_seconds):
        row = {
            "batch_idx": self._batch_idx,
            "num_images": int(max(0, num_images)),
            "num_predictions": int(max(0, num_predictions)),
        }
        for name in self.stages:
            row[name] = float(stage_seconds.get(name, 0.0))
        for name, value in stage_seconds.items():
            if name not in row:
                row[name] = float(value)
                if name not in self.stages:
                    self.stages.append(name)
        row["total_sec"] = float(sum(row.get(name, 0.0) for name in self.stages))
        self.records.append(row)
        self._write_csv_row(row)
        self._batch_idx += 1

    def _current_fieldnames(self):
        return ["batch_idx", "num_images", "num_predictions", *self.stages, "total_sec"]

    def _open_csv(self):
        self.timing_dir.mkdir(parents=True, exist_ok=True)
        self._fieldnames = self._current_fieldnames()
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None

    def _rewrite_csv(self):
        self._close_csv()
        self._open_csv()
        for record in self.records:
            self._csv_writer.writerow(record)
        self._csv_file.flush()

    def _write_csv_row(self, row):
        fieldnames = self._current_fieldnames()
        if self._csv_writer is None:
            self._open_csv()
        elif fieldnames != self._fieldnames:
            self._rewrite_csv()
            return
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def save(self):
        self.timing_dir.mkdir(parents=True, exist_ok=True)
        if self._csv_writer is None:
            self._open_csv()
            for record in self.records:
                self._csv_writer.writerow(record)
        elif self._csv_file is not None:
            self._csv_file.flush()
        total_batches = len(self.records)
        total_images = int(sum(r["num_images"] for r in self.records))
        total_predictions = int(sum(r["num_predictions"] for r in self.records))
        stage_totals = {name: float(sum(r.get(name, 0.0) for r in self.records)) for name in self.stages}
        total_sec = float(sum(stage_totals.values()))

        summary = {
            "uncertainty": self.uncertainty,
            "unit": self.unit,
            "stages": list(self.stages),
            "total_batches": total_batches,
            "total_images": total_images,
            "total_predictions": total_predictions,
            "stage_total_sec": stage_totals,
            "total_elapsed_sec": total_sec,
            "mean_stage_sec_per_batch": {
                name: (value / total_batches) if total_batches > 0 else 0.0
                for name, value in stage_totals.items()
            },
            "mean_stage_ms_per_image": {
                name: ((value / total_images) * 1000.0) if total_images > 0 else 0.0
                for name, value in stage_totals.items()
            },
            "mean_stage_ms_per_prediction": {
                name: ((value / total_predictions) * 1000.0) if total_predictions > 0 else 0.0
                for name, value in stage_totals.items()
            },
            "batch_csv": str(self.csv_path),
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self._close_csv()
        return self.csv_path, self.json_path

__all__ = [name for name in globals().keys() if not name.startswith("__")]
