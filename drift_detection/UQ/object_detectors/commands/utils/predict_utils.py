import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import boxes as box_ops

from dataloaders.core.class_names import DATASET_CLASS_NAMES
from models.fcos import FCOSTorchObjectDetector
from models.faster_rcnn import FasterRCNNTorchObjectDetector
from models.yolov5 import YOLOV5TorchObjectDetector
from models.yolov10 import YOLOV10TorchObjectDetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent


def resolve_project_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists() or path.parts[:1] == ("object_detectors",):
        return repo_candidate
    return (PROJECT_ROOT / path).resolve()


def _sync_timing_device(device=None):
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def get_prediction_class_probs(detector, prediction):
    start = 6 if bool(getattr(detector, "has_faster_rcnn_label_column", False)) else 5
    return prediction[..., start:]


def get_selected_prediction_class_probs(detector, prediction, indices):
    if int(indices.shape[0]) == 0:
        start = 6 if bool(getattr(detector, "has_faster_rcnn_label_column", False)) else 5
        num_classes = max(0, int(prediction.shape[-1]) - start)
        return torch.zeros((0, num_classes), dtype=prediction.dtype, device=prediction.device)
    start = 6 if bool(getattr(detector, "has_faster_rcnn_label_column", False)) else 5
    return prediction[indices, start:]


def _start_timing(device=None):
    _sync_timing_device(device)
    return time.perf_counter()


def _add_elapsed_timing(timing_accumulator, stage_name, start_time, device=None):
    _sync_timing_device(device)
    if timing_accumulator is not None:
        timing_accumulator[stage_name] = float(timing_accumulator.get(stage_name, 0.0)) + (
            time.perf_counter() - start_time
        )


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_count_or_inf(value, *, key_name):
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw == "inf":
            return math.inf
        if raw == "":
            raise ValueError(f"{key_name} must be an integer >= 0 or 'inf'.")
        try:
            parsed = int(raw)
        except Exception as e:
            raise ValueError(f"{key_name} must be an integer >= 0 or 'inf'.") from e
    elif value is None:
        raise ValueError(f"{key_name} must be an integer >= 0 or 'inf'.")
    else:
        parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{key_name} must be >= 0 or 'inf'.")
    return parsed


def get_dataset_cfg(config):
    dataset_root_cfg = config["dataset"]
    used_raw = dataset_root_cfg["used_dataset"]
    if isinstance(used_raw, str):
        used_list = [used_raw.strip().lower()]
    elif isinstance(used_raw, (list, tuple)):
        used_list = [str(v).strip().lower() for v in used_raw if str(v).strip()]
    else:
        raise ValueError("dataset.used_dataset must be a string or list of strings.")
    if not used_list:
        raise ValueError("dataset.used_dataset is empty.")
    for used_dataset in used_list:
        if used_dataset not in dataset_root_cfg:
            raise ValueError(f"dataset.{used_dataset} is missing in config.")
    if len(used_list) == 1:
        used_dataset = used_list[0]
        return used_dataset, dataset_root_cfg[used_dataset]
    return "__multi__", {name: dataset_root_cfg[name] for name in used_list}


def get_annotation_path(config, split):
    used_dataset, dataset_cfg = get_dataset_cfg(config)
    if used_dataset != "coco":
        return None
    root = Path(dataset_cfg["root"])
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    ann_dir = dataset_cfg.get("annotation_dir")
    ann_name = dataset_cfg.get(f"{split}_annotation_file")
    if not ann_dir or not ann_name:
        return None
    return root / ann_dir / ann_name


def load_coco_category_maps(annotation_path):
    if annotation_path is None:
        return {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(c["id"]): c["name"] for c in payload.get("categories", [])}


def load_gt_category_maps(config, split):
    used_dataset, dataset_cfg = get_dataset_cfg(config)
    if used_dataset == "coco":
        annotation_path = get_annotation_path(config, split)
        return load_coco_category_maps(annotation_path)
    return {}


def _resolve_detector_class_names(config):
    model_cfg = config.get("model", {})
    configured_names = model_cfg.get("class_names")
    if configured_names:
        if isinstance(configured_names, str):
            return [name.strip() for name in configured_names.split(",") if name.strip()]
        return [str(v) for v in configured_names]

    used_dataset, _dataset_cfg = get_dataset_cfg(config)
    if used_dataset in DATASET_CLASS_NAMES:
        return list(DATASET_CLASS_NAMES[used_dataset])
    if used_dataset == "__multi__":
        dataset_root_cfg = config.get("dataset", {})
        used_raw = dataset_root_cfg.get("used_dataset", [])
        if isinstance(used_raw, str):
            names = [used_raw.strip().lower()]
        else:
            names = [str(v).strip().lower() for v in used_raw if str(v).strip()]
        if names and all(name in DATASET_CLASS_NAMES for name in names):
            first = list(DATASET_CLASS_NAMES[names[0]])
            if all(list(DATASET_CLASS_NAMES[name]) == first for name in names):
                return first
            raise ValueError("Mixed datasets with different class spaces are not supported.")
    return None


def build_detector(config, model_weight=None):
    torch.backends.cudnn.benchmark = False
    model_cfg = config["model"]
    device_str = model_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    confidence = model_cfg.get("confidence_threshold", 0.4)
    iou_thresh = model_cfg.get("iou_threshold", 0.45)

    model_type = str(model_cfg.get("type", "yolov5")).strip().lower()
    weight_source = model_weight if model_weight is not None else model_cfg.get("weights", "")
    if isinstance(weight_source, (list, tuple)):
        if len(weight_source) == 0:
            raise ValueError("model.weights is empty.")
        weight_source = weight_source[0]
    if model_type in {"yolov10", "yolo_v10"} and not str(weight_source).strip():
        raise ValueError("YOLOv10 predict requires model.weights. Refusing to run a random model.")
    weight_path = None
    if str(weight_source).strip():
        weight_path = Path(weight_source)
        if not weight_path.is_absolute():
            weight_path = resolve_project_path(weight_path)
    img_size = model_cfg.get("img_size", 640)
    img_size_tuple = (int(img_size), int(img_size)) if isinstance(img_size, int) else tuple(img_size)
    if model_type in {"yolov5", "yolo", "yolo_v5"}:
        detector = YOLOV5TorchObjectDetector(
            model_weight=str(weight_path),
            device=device,
            img_size=img_size_tuple,
            names=_resolve_detector_class_names(config),
            mode="eval",
            confidence=confidence,
            iou_thresh=iou_thresh,
        )
    elif model_type in {"yolov10", "yolo_v10"}:
        detector = YOLOV10TorchObjectDetector(
            model_weight=str(weight_path) if weight_path is not None else None,
            device=device,
            img_size=img_size_tuple,
            names=_resolve_detector_class_names(config),
            mode="eval",
            confidence=confidence,
            max_det=int(model_cfg.get("max_det", 300)),
        )
    elif model_type in {"faster_rcnn", "faster-rcnn", "frcnn"}:
        detector = FasterRCNNTorchObjectDetector(
            model_weight=str(weight_path) if weight_path is not None else None,
            device=device,
            names=_resolve_detector_class_names(config),
            mode="eval",
            confidence=confidence,
            iou_thresh=iou_thresh,
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    elif model_type in {"fcos"}:
        detector = FCOSTorchObjectDetector(
            model_weight=str(weight_path) if weight_path is not None else None,
            device=device,
            names=_resolve_detector_class_names(config),
            mode="eval",
            confidence=confidence,
            iou_thresh=iou_thresh,
        )
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")
    if model_type in {"yolov5", "yolo", "yolo_v5", "yolov10", "yolo_v10"}:
        detector.eval()
    else:
        detector.eval().to(device)
    return detector, device


def resolve_module_by_name(model, layer_name):
    node = model
    for token in layer_name.split("."):
        if token.isdigit():
            idx = int(token)
            if hasattr(node, "__getitem__"):
                node = node[idx]
            elif token in node._modules:
                node = node._modules[token]
            else:
                raise ValueError(f"Cannot index '{token}' in layer path '{layer_name}'")
            continue

        if hasattr(node, token):
            node = getattr(node, token)
            continue
        if hasattr(node, "_modules") and token in node._modules:
            node = node._modules[token]
            continue
        raise ValueError(f"Layer token '{token}' not found in path '{layer_name}'")

    return node


def resolve_layer_parameter(model, layer_name):
    module = resolve_module_by_name(model, layer_name)
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        return module.weight

    for param in module.parameters():
        if isinstance(param, torch.Tensor):
            return param
    raise ValueError(f"Layer '{layer_name}' has no parameters.")


def expand_layer_names(model, layer_names):
    resolved = []
    for name in normalize_to_list(layer_names):
        token = str(name).strip()
        if not token:
            continue
        if token not in resolved:
            resolved.append(token)
    return resolved


def parse_output_config(output_cfg):
    def as_dict(v):
        return v if isinstance(v, dict) else {}

    def as_int(v, d):
        try:
            return int(v)
        except Exception:
            return d

    def as_float(v, d):
        try:
            return float(v)
        except Exception:
            return d

    uncertainty = str(output_cfg.get("uncertainty", "gt")).strip().lower()
    if not uncertainty:
        uncertainty = "gt"
    supported = {
        "deterministic",
        "gt",
        "score",
        "class_probability",
        "entropy",
        "energy",
        "mc_dropout",
        "ensemble",
        "meta_detect",
        "null_detect",
        "layer_grad",
    }
    if uncertainty not in supported:
        raise ValueError(f"Unsupported uncertainty: {uncertainty}")

    active = as_dict(output_cfg.get(uncertainty, {}))
    save_csv_cfg = active.get("save_csv", {})
    save_image_cfg = as_dict(active.get("save_image", {}))
    save_csv = as_dict(save_csv_cfg)
    save_csv_enabled = bool(save_csv.get("enabled", bool(save_csv_cfg) if isinstance(save_csv_cfg, bool) else False))
    unit = "bbox"
    pre_nms = False
    pre_nms_ratio = 1.0

    gt_cfg = active if uncertainty == "gt" else {}
    meta_detect_cfg = active if uncertainty == "meta_detect" else {}
    null_detect_cfg = active if uncertainty == "null_detect" else {}
    mc_dropout_cfg = active if uncertainty == "mc_dropout" else {}
    layer_grad_cfg = active if uncertainty == "layer_grad" else {}

    gt_iou_match_threshold = as_float(gt_cfg.get("iou_match_threshold", 0.5), 0.5)
    meta_detect_score_threshold = as_float(meta_detect_cfg.get("score_threshold", 0.0), 0.0)
    meta_detect_iou_threshold = as_float(meta_detect_cfg.get("iou_threshold", 0.45), 0.45)
    mc_num_runs = as_int(mc_dropout_cfg.get("num_runs", 30), 30)
    mc_dropout_rate = as_float(mc_dropout_cfg.get("dropout_rate", 0.5), 0.5)

    target_values = []
    target_layers = []
    layer_target_values = []
    layer_target_layers = []
    layer_target_layer_map = {}
    layer_map_reduction = "none"
    layer_gradient_reduction = []
    layer_pseudo_gt = "cand"
    layer_cand_score_threshold = 0.01
    layer_bbox_loss = "box_l1"
    layer_cls_loss = "bcewithlogits"
    layer_obj_loss = "bcewithlogits"
    layer_bbox_direction = "pred_to_target"
    layer_cls_direction = "pred_to_target"
    layer_obj_direction = "pred_to_target"
    layer_roi_cand_enabled = True
    layer_roi_cand_scalar = []
    layer_roi_cand_score_threshold = 0.01
    layer_roi_bbox_loss = "l1"
    layer_roi_cls_loss = "bcewithlogits"
    layer_roi_bbox_direction = "pred_to_target"
    layer_roi_cls_direction = "pred_to_target"
    layer_rpn_cand_enabled = False
    layer_rpn_cand_scalar = []
    layer_rpn_cand_obj_threshold = 0.0
    layer_rpn_bbox_loss = "offset_l1"
    layer_rpn_obj_loss = "bcewithlogits"
    layer_rpn_bbox_direction = "pred_to_target"
    layer_rpn_obj_direction = "pred_to_target"
    layer_roi_null_enabled = True
    layer_roi_null_scalar = []
    layer_roi_null_bbox_loss = "l1"
    layer_roi_null_cls_loss = "bcewithlogits"
    layer_roi_null_bbox_direction = "pred_to_target"
    layer_roi_null_cls_direction = "pred_to_target"
    layer_rpn_null_enabled = True
    layer_rpn_null_scalar = []
    layer_rpn_null_bbox_loss = "offset_l1"
    layer_rpn_null_obj_loss = "bcewithlogits"
    layer_rpn_null_bbox_direction = "pred_to_target"
    layer_rpn_null_obj_direction = "pred_to_target"
    layer_frcnn_null_scalar = []
    layer_null_scalar = []
    layer_null_bbox_loss = "l1"
    layer_null_cls_loss = "bcewithlogits"
    layer_null_obj_loss = "bcewithlogits"
    layer_null_bbox_direction = "pred_to_target"
    layer_null_cls_direction = "pred_to_target"
    layer_null_obj_direction = "pred_to_target"

    loss_aliases = {
        "bce": "bcewithlogits",
        "bce_with_logits": "bcewithlogits",
        "bcewithlogits": "bcewithlogits",
        "l1": "l1",
        "l1_loss": "l1",
        "l2": "l2",
        "l2_loss": "l2",
        "mse": "l2",
        "kl": "kl",
        "kl_div": "kl",
        "kl_divergence": "kl",
        "ce": "ce",
        "cross_entropy": "ce",
        "abs": "abs_diff",
        "abs_diff": "abs_diff",
        "absolute_diff": "abs_diff",
        "signed": "signed_diff",
        "signed_diff": "signed_diff",
    }
    direction_aliases = {
        "pred_to_target": "pred_to_target",
        "prediction_to_target": "pred_to_target",
        "target": "pred_to_target",
        "target_to_pred": "target_to_pred",
        "target_to_prediction": "target_to_pred",
        "reverse": "target_to_pred",
    }

    def normalize_loss_option(raw, default, supported, key_name):
        normalized = loss_aliases.get(str(raw if raw is not None else default).strip().lower().replace("-", "_"))
        if normalized not in supported:
            raise ValueError(f"Unsupported {key_name}: {raw}. Supported values: {', '.join(sorted(supported))}.")
        return normalized

    def normalize_yolo_bbox_loss_option(raw, default, key_name, allow_offset=True):
        key = str(raw if raw is not None else default).strip().lower().replace("-", "_")
        aliases = {
            "box_l1": "box_l1",
            "box_l1_loss": "box_l1",
            "l1": "box_l1",
            "l1_loss": "box_l1",
            "box_l2": "box_l2",
            "box_l2_loss": "box_l2",
            "l2": "box_l2",
            "l2_loss": "box_l2",
            "mse": "box_l2",
            "offset_l1": "offset_l1",
            "offset_l1_loss": "offset_l1",
            "offset_l2": "offset_l2",
            "offset_l2_loss": "offset_l2",
            "offset_mse": "offset_l2",
        }
        normalized = aliases.get(key)
        supported = {"box_l1", "box_l2", "offset_l1", "offset_l2"} if allow_offset else {"box_l1", "box_l2"}
        if normalized not in supported:
            raise ValueError(f"Unsupported {key_name}: {raw}. Supported values: {', '.join(sorted(supported))}.")
        return normalized

    def normalize_faster_rcnn_bbox_loss_option(raw, default, key_name):
        return normalize_yolo_bbox_loss_option(raw, default, key_name, allow_offset=True)

    def normalize_direction_option(raw, key_name):
        normalized = direction_aliases.get(
            str(raw if raw is not None else "pred_to_target").strip().lower().replace("-", "_")
        )
        if normalized is None:
            raise ValueError(f"Unsupported {key_name}: {raw}. Supported values: pred_to_target, target_to_pred.")
        return normalized

    def validate_loss_directions(bbox_direction, cls_loss, obj_loss, cls_direction, obj_direction):
        if cls_direction == "target_to_pred" and cls_loss in {"bcewithlogits", "ce"}:
            raise ValueError("cls_direction=target_to_pred is only supported when cls_loss=kl.")
        if obj_direction == "target_to_pred" and obj_loss != "signed_diff":
            raise ValueError("obj_direction=target_to_pred is only supported when obj_loss=signed_diff.")

    if uncertainty == "layer_grad":
        g = as_dict(layer_grad_cfg.get("gradient", {}))
        cand_target_cfg = as_dict(g.get("cand_target", layer_grad_cfg.get("cand_target", {})))
        roi_cand_cfg = as_dict(cand_target_cfg.get("roi", cand_target_cfg.get("roi_cand", {})))
        rpn_cand_cfg = as_dict(cand_target_cfg.get("rpn", cand_target_cfg.get("rpn_cand", {})))
        null_target_cfg = as_dict(g.get("null_target", layer_grad_cfg.get("null_target", {})))
        roi_null_cfg = as_dict(null_target_cfg.get("roi", null_target_cfg.get("roi_null", {})))
        rpn_null_cfg = as_dict(null_target_cfg.get("rpn", null_target_cfg.get("rpn_null", {})))
        unified_roi_cfg = as_dict(g.get("roi", {}))
        unified_rpn_cfg = as_dict(g.get("rpn", {}))
        t = g.get("target", "cand_target")
        t_policy = str(t).strip().lower() if t is not None else "null_target"
        if t_policy in {"null_target", "null"}:
            null_scalar_probe = [str(v).strip().lower() for v in normalize_to_list(null_target_cfg.get("scalar", []))]
            has_faster_rcnn_null_cfg = bool(rpn_null_cfg or roi_null_cfg or any(str(v).startswith(("rpn_", "roi_")) for v in null_scalar_probe))
            layer_pseudo_gt = "frcnn_null" if has_faster_rcnn_null_cfg else "uniform"
        elif t_policy in {"delta_target", "delta"}:
            layer_pseudo_gt = "delta"
        elif t_policy in {"cand_target", "cand"}:
            layer_pseudo_gt = "cand"
        else:
            raise ValueError("Unsupported layer_grad.gradient.target. Supported values: cand_target, null_target, delta_target.")
        if layer_pseudo_gt == "frcnn_null":
            raw_scalar_cfg = null_target_cfg.get(
                "scalar",
                g.get("scalar", ["rpn_obj_loss", "rpn_bbox_loss", "roi_cls_loss", "roi_bbox_loss"]),
            )
        elif layer_pseudo_gt == "uniform":
            raw_scalar_cfg = null_target_cfg.get("scalar", g.get("scalar", ["bbox_loss", "obj_loss", "cls_loss"]))
        else:
            raw_scalar_cfg = cand_target_cfg.get("scalar", g.get("scalar", ["loss"]))
        layer_target_values = [v.lower() for v in normalize_to_list(raw_scalar_cfg)]
        if "loss" in layer_target_values:
            exp = []
            for v in layer_target_values:
                if v != "loss":
                    exp.append(v)
                elif layer_pseudo_gt == "frcnn_null":
                    exp.extend(["rpn_obj_loss", "rpn_bbox_loss", "roi_cls_loss", "roi_bbox_loss"])
                else:
                    exp.extend(["obj_loss", "cls_loss", "bbox_loss"])
            layer_target_values = list(dict.fromkeys(exp))
        if layer_pseudo_gt == "delta":
            unsupported_delta_values = [v for v in layer_target_values if v not in {"bbox_loss", "obj_loss", "cls_loss"}]
            if unsupported_delta_values:
                raise ValueError(
                    "YOLOv5 layer_grad delta_target supports only loss scalars: "
                    "bbox_loss, obj_loss, cls_loss, or loss."
                )

        nested_layer_cfg = {}
        active_rpn_cfg = unified_rpn_cfg or (
            rpn_null_cfg if layer_pseudo_gt == "frcnn_null" else rpn_null_cfg if layer_pseudo_gt == "uniform" else rpn_cand_cfg
        )
        active_roi_cfg = unified_roi_cfg or (
            roi_null_cfg if layer_pseudo_gt == "frcnn_null" else roi_null_cfg if layer_pseudo_gt == "uniform" else roi_cand_cfg
        )
        if layer_pseudo_gt == "frcnn_null":
            raw_layer_cfg = null_target_cfg.get("layer", g.get("layer", []))
        elif layer_pseudo_gt == "uniform":
            raw_layer_cfg = null_target_cfg.get("layer", g.get("layer", []))
        else:
            raw_layer_cfg = cand_target_cfg.get("layer", None)
            if raw_layer_cfg is None:
                for target_prefix, cand_cfg, allowed_targets in (
                    ("rpn", active_rpn_cfg, {"obj_loss", "bbox_loss"}),
                    ("roi", active_roi_cfg, {"cls_loss", "bbox_loss"}),
                ):
                    raw_nested_layers = cand_cfg.get("layer", {})
                    if not isinstance(raw_nested_layers, dict):
                        continue
                    for target_name, layer_names in raw_nested_layers.items():
                        target_key = str(target_name).strip().lower()
                        prefixed_key = f"{target_prefix}_{target_key}" if target_key in allowed_targets else target_key
                        nested_layer_cfg[prefixed_key] = layer_names
                raw_layer_cfg = nested_layer_cfg if nested_layer_cfg else g.get("layer", [])
        if isinstance(raw_layer_cfg, dict):
            layer_target_layer_map = {}
            layer_target_layers = []
            for target_name, layer_names in raw_layer_cfg.items():
                target_key = str(target_name).strip().lower()
                layers = normalize_to_list(layer_names)
                layer_target_layer_map[target_key] = layers
                for layer_name in layers:
                    if layer_name not in layer_target_layers:
                        layer_target_layers.append(layer_name)
        else:
            layer_target_layers = normalize_to_list(raw_layer_cfg)
        reduction_aliases = {
            "1-norm": "l1_norm",
            "1_norm": "l1_norm",
            "l1": "l1_norm",
            "l1_norm": "l1_norm",
            "2-norm": "l2_norm",
            "2_norm": "l2_norm",
            "l2": "l2_norm",
            "l2_norm": "l2_norm",
            "min": "min",
            "max": "max",
            "mean": "mean",
            "std": "std",
        }
        layer_gradient_reduction = []
        for reduction_name in normalize_to_list(g.get("reduction", [])):
            key = str(reduction_name).strip().lower().replace("-", "_")
            normalized = reduction_aliases.get(key)
            if normalized is None:
                raise ValueError(
                    "Unsupported layer_grad.gradient.reduction value "
                    f"'{reduction_name}'. Supported values: l1_norm, l2_norm, min, max, mean, std."
                )
            if normalized not in layer_gradient_reduction:
                layer_gradient_reduction.append(normalized)
        layer_cand_score_threshold = as_float(g.get("cand_score_threshold", 0.01), 0.01)

        if cand_target_cfg:
            layer_roi_cand_enabled = bool(roi_cand_cfg.get("enabled", True))
            layer_rpn_cand_enabled = bool(rpn_cand_cfg.get("enabled", False))
        if unified_roi_cfg:
            layer_roi_cand_enabled = True
        if unified_rpn_cfg:
            layer_rpn_cand_enabled = True
        layer_roi_cand_scalar = [
            f"roi_{v}" if v in {"bbox_loss", "cls_loss"} else v
            for v in [str(v).strip().lower() for v in normalize_to_list(roi_cand_cfg.get("scalar", []))]
        ]
        layer_rpn_cand_scalar = [
            f"rpn_{v}" if v in {"bbox_loss", "obj_loss"} else v
            for v in [str(v).strip().lower() for v in normalize_to_list(rpn_cand_cfg.get("scalar", []))]
        ]
        layer_roi_cand_score_threshold = as_float(
            active_roi_cfg.get("cand_score_threshold", g.get("cand_score_threshold", layer_cand_score_threshold)),
            layer_cand_score_threshold,
        )
        layer_rpn_cand_obj_threshold = as_float(
            active_rpn_cfg.get("cand_obj_threshold", active_rpn_cfg.get("cand_score_threshold", 0.0)),
            0.0,
        )
        if null_target_cfg:
            layer_roi_null_enabled = bool(roi_null_cfg.get("enabled", True))
            layer_rpn_null_enabled = bool(rpn_null_cfg.get("enabled", True))
        if unified_roi_cfg:
            layer_roi_null_enabled = True
        if unified_rpn_cfg:
            layer_rpn_null_enabled = True
        layer_roi_null_scalar = [
            f"roi_{v}" if v in {"bbox_loss", "cls_loss"} else v
            for v in [str(v).strip().lower() for v in normalize_to_list(roi_null_cfg.get("scalar", []))]
        ]
        layer_rpn_null_scalar = [
            f"rpn_{v}" if v in {"bbox_loss", "obj_loss"} else v
            for v in [str(v).strip().lower() for v in normalize_to_list(rpn_null_cfg.get("scalar", []))]
        ]
        layer_frcnn_null_scalar = [
            str(v).strip().lower()
            for v in normalize_to_list(
                null_target_cfg.get("scalar", ["rpn_obj_loss", "rpn_bbox_loss", "roi_cls_loss", "roi_bbox_loss"])
            )
        ]
        if "loss" in layer_frcnn_null_scalar:
            expanded_frcnn_null_scalar = []
            for value in layer_frcnn_null_scalar:
                if value == "loss":
                    expanded_frcnn_null_scalar.extend(["rpn_obj_loss", "rpn_bbox_loss", "roi_cls_loss", "roi_bbox_loss"])
                else:
                    expanded_frcnn_null_scalar.append(value)
            layer_frcnn_null_scalar = list(dict.fromkeys(expanded_frcnn_null_scalar))
        layer_bbox_loss = normalize_yolo_bbox_loss_option(
            g.get("bbox_loss", "box_l1"),
            "box_l1",
            "layer_grad.gradient.bbox_loss",
            allow_offset=True,
        )
        layer_cls_loss = normalize_loss_option(
            g.get("cls_loss", "bcewithlogits"),
            "bcewithlogits",
            {"bcewithlogits", "kl"},
            "layer_grad.gradient.cls_loss",
        )
        layer_obj_loss = normalize_loss_option(
            g.get("obj_loss", "bcewithlogits"),
            "bcewithlogits",
            {"bcewithlogits", "abs_diff", "signed_diff"},
            "layer_grad.gradient.obj_loss",
        )
        layer_bbox_direction = normalize_direction_option(g.get("bbox_direction", "pred_to_target"), "layer_grad.gradient.bbox_direction")
        layer_cls_direction = normalize_direction_option(g.get("cls_direction", "pred_to_target"), "layer_grad.gradient.cls_direction")
        layer_obj_direction = normalize_direction_option(g.get("obj_direction", "pred_to_target"), "layer_grad.gradient.obj_direction")
        validate_loss_directions(layer_bbox_direction, layer_cls_loss, layer_obj_loss, layer_cls_direction, layer_obj_direction)

        layer_roi_bbox_loss = normalize_faster_rcnn_bbox_loss_option(
            active_roi_cfg.get("bbox_loss", g.get("bbox_loss", "l1")),
            "box_l1",
            "layer_grad.gradient.roi.bbox_loss",
        )
        layer_roi_cls_loss = normalize_loss_option(
            active_roi_cfg.get("cls_loss", g.get("cls_loss", "bcewithlogits")),
            "bcewithlogits",
            {"bcewithlogits", "kl"},
            "layer_grad.gradient.roi.cls_loss",
        )
        layer_roi_bbox_direction = normalize_direction_option(
            active_roi_cfg.get("bbox_direction", g.get("bbox_direction", "pred_to_target")),
            "layer_grad.gradient.roi.bbox_direction",
        )
        layer_roi_cls_direction = normalize_direction_option(
            active_roi_cfg.get("cls_direction", g.get("cls_direction", "pred_to_target")),
            "layer_grad.gradient.roi.cls_direction",
        )
        validate_loss_directions(
            layer_roi_bbox_direction,
            layer_roi_cls_loss,
            "bcewithlogits",
            layer_roi_cls_direction,
            "pred_to_target",
        )

        layer_rpn_bbox_loss = normalize_faster_rcnn_bbox_loss_option(
            active_rpn_cfg.get("bbox_loss", "offset_l1"),
            "offset_l1",
            "layer_grad.gradient.rpn.bbox_loss",
        )
        if not layer_rpn_bbox_loss.startswith("offset_"):
            raise ValueError("Faster R-CNN layer_grad.gradient.rpn.bbox_loss supports only offset_l1 or offset_l2.")
        layer_rpn_obj_loss = normalize_loss_option(
            active_rpn_cfg.get("obj_loss", "bcewithlogits"),
            "bcewithlogits",
            {"bcewithlogits", "abs_diff", "signed_diff"},
            "layer_grad.gradient.rpn.obj_loss",
        )
        layer_rpn_bbox_direction = normalize_direction_option(
            active_rpn_cfg.get("bbox_direction", "pred_to_target"),
            "layer_grad.gradient.rpn.bbox_direction",
        )
        layer_rpn_obj_direction = normalize_direction_option(
            active_rpn_cfg.get("obj_direction", "pred_to_target"),
            "layer_grad.gradient.rpn.obj_direction",
        )

        layer_roi_null_bbox_loss = normalize_faster_rcnn_bbox_loss_option(
            active_roi_cfg.get("bbox_loss", "l1"),
            "box_l1",
            "layer_grad.gradient.roi.bbox_loss",
        )
        layer_roi_null_cls_loss = normalize_loss_option(
            active_roi_cfg.get("cls_loss", "bcewithlogits"),
            "bcewithlogits",
            {"bcewithlogits", "kl"},
            "layer_grad.gradient.roi.cls_loss",
        )
        layer_roi_null_bbox_direction = normalize_direction_option(
            active_roi_cfg.get("bbox_direction", "pred_to_target"),
            "layer_grad.gradient.roi.bbox_direction",
        )
        layer_roi_null_cls_direction = normalize_direction_option(
            active_roi_cfg.get("cls_direction", "pred_to_target"),
            "layer_grad.gradient.roi.cls_direction",
        )
        validate_loss_directions(
            layer_roi_null_bbox_direction,
            layer_roi_null_cls_loss,
            "bcewithlogits",
            layer_roi_null_cls_direction,
            "pred_to_target",
        )

        layer_rpn_null_bbox_loss = normalize_faster_rcnn_bbox_loss_option(
            active_rpn_cfg.get("bbox_loss", "offset_l1"),
            "offset_l1",
            "layer_grad.gradient.rpn.bbox_loss",
        )
        if not layer_rpn_null_bbox_loss.startswith("offset_"):
            raise ValueError("Faster R-CNN layer_grad.gradient.rpn.bbox_loss supports only offset_l1 or offset_l2.")
        layer_rpn_null_obj_loss = normalize_loss_option(
            active_rpn_cfg.get("obj_loss", "bcewithlogits"),
            "bcewithlogits",
            {"bcewithlogits", "abs_diff", "signed_diff"},
            "layer_grad.gradient.rpn.obj_loss",
        )
        layer_rpn_null_bbox_direction = normalize_direction_option(
            active_rpn_cfg.get("bbox_direction", "pred_to_target"),
            "layer_grad.gradient.rpn.bbox_direction",
        )
        layer_rpn_null_obj_direction = normalize_direction_option(
            active_rpn_cfg.get("obj_direction", "pred_to_target"),
            "layer_grad.gradient.rpn.obj_direction",
        )
        if layer_rpn_null_obj_direction == "target_to_pred" and layer_rpn_null_obj_loss != "signed_diff":
            raise ValueError("layer_grad.gradient.rpn.obj_direction=target_to_pred is only supported when obj_loss=signed_diff.")

        layer_null_scalar = [str(v).strip().lower() for v in normalize_to_list(null_target_cfg.get("scalar", []))]
        layer_null_bbox_loss = normalize_yolo_bbox_loss_option(
            null_target_cfg.get("bbox_loss", g.get("bbox_loss", "l1")),
            "box_l1",
            "layer_grad.gradient.null_target.bbox_loss",
            allow_offset=True,
        )
        layer_null_cls_loss = normalize_loss_option(
            null_target_cfg.get("cls_loss", g.get("cls_loss", "bcewithlogits")),
            "bcewithlogits",
            {"bcewithlogits", "kl"},
            "layer_grad.gradient.null_target.cls_loss",
        )
        layer_null_obj_loss = normalize_loss_option(
            null_target_cfg.get("obj_loss", g.get("obj_loss", "bcewithlogits")),
            "bcewithlogits",
            {"bcewithlogits", "abs_diff", "signed_diff"},
            "layer_grad.gradient.null_target.obj_loss",
        )
        layer_null_bbox_direction = normalize_direction_option(
            null_target_cfg.get("bbox_direction", g.get("bbox_direction", "pred_to_target")),
            "layer_grad.gradient.null_target.bbox_direction",
        )
        layer_null_cls_direction = normalize_direction_option(
            null_target_cfg.get("cls_direction", g.get("cls_direction", "pred_to_target")),
            "layer_grad.gradient.null_target.cls_direction",
        )
        layer_null_obj_direction = normalize_direction_option(
            null_target_cfg.get("obj_direction", g.get("obj_direction", "pred_to_target")),
            "layer_grad.gradient.null_target.obj_direction",
        )
        validate_loss_directions(
            layer_null_bbox_direction,
            layer_null_cls_loss,
            layer_null_obj_loss,
            layer_null_cls_direction,
            layer_null_obj_direction,
        )

    null_bbox_loss = normalize_yolo_bbox_loss_option(
        null_detect_cfg.get("bbox_loss", "box_l1"),
        "box_l1",
        "null_detect.bbox_loss",
        allow_offset=True,
    )
    null_cls_loss = normalize_loss_option(
        null_detect_cfg.get("cls_loss", "bcewithlogits"),
        "bcewithlogits",
        {"bcewithlogits", "kl"},
        "null_detect.cls_loss",
    )
    null_obj_loss = normalize_loss_option(
        null_detect_cfg.get("obj_loss", "bcewithlogits"),
        "bcewithlogits",
        {"bcewithlogits", "abs_diff", "signed_diff"},
        "null_detect.obj_loss",
    )
    null_bbox_direction = normalize_direction_option(null_detect_cfg.get("bbox_direction", "pred_to_target"), "null_detect.bbox_direction")
    null_cls_direction = normalize_direction_option(null_detect_cfg.get("cls_direction", "pred_to_target"), "null_detect.cls_direction")
    null_obj_direction = normalize_direction_option(null_detect_cfg.get("obj_direction", "pred_to_target"), "null_detect.obj_direction")
    validate_loss_directions(null_bbox_direction, null_cls_loss, null_obj_loss, null_cls_direction, null_obj_direction)
    null_feature_set = str(null_detect_cfg.get("feature_set", "full")).strip().lower().replace("-", "_")
    if null_feature_set not in {"full", "losses_only"}:
        raise ValueError("Unsupported null_detect.feature_set. Supported values: full, losses_only.")

    save_image_enabled = bool(save_image_cfg.get("enabled", bool(save_image_cfg)))
    gt_image_step = as_int(save_image_cfg.get("step", 1), 1)
    gt_image_max_num = as_int(save_image_cfg.get("max_num", 1), 1)

    return {
        "save_csv_enabled": save_csv_enabled,
        "uncertainty": uncertainty,
        "pre_nms": pre_nms,
        "pre_nms_ratio": float(pre_nms_ratio),
        "unit": unit,
        "gt_iou_match_threshold": gt_iou_match_threshold,
        "meta_detect_score_threshold": meta_detect_score_threshold,
        "meta_detect_iou_threshold": meta_detect_iou_threshold,
        "mc_num_runs": mc_num_runs,
        "mc_dropout_rate": mc_dropout_rate,
        "target_values": target_values,
        "target_layers": target_layers,
        "layer_target_values": layer_target_values,
        "layer_target_layers": layer_target_layers,
        "layer_target_layer_map": layer_target_layer_map,
        "layer_map_reduction": layer_map_reduction,
        "layer_gradient_reduction": layer_gradient_reduction,
        "layer_pseudo_gt": layer_pseudo_gt,
        "layer_cand_score_threshold": float(layer_cand_score_threshold),
        "layer_bbox_loss": layer_bbox_loss,
        "layer_cls_loss": layer_cls_loss,
        "layer_obj_loss": layer_obj_loss,
        "layer_bbox_direction": layer_bbox_direction,
        "layer_cls_direction": layer_cls_direction,
        "layer_obj_direction": layer_obj_direction,
        "layer_roi_cand_enabled": layer_roi_cand_enabled,
        "layer_roi_cand_scalar": layer_roi_cand_scalar,
        "layer_roi_cand_score_threshold": float(layer_roi_cand_score_threshold),
        "layer_roi_bbox_loss": layer_roi_bbox_loss,
        "layer_roi_cls_loss": layer_roi_cls_loss,
        "layer_roi_bbox_direction": layer_roi_bbox_direction,
        "layer_roi_cls_direction": layer_roi_cls_direction,
        "layer_rpn_cand_enabled": layer_rpn_cand_enabled,
        "layer_rpn_cand_scalar": layer_rpn_cand_scalar,
        "layer_rpn_cand_obj_threshold": float(layer_rpn_cand_obj_threshold),
        "layer_rpn_bbox_loss": layer_rpn_bbox_loss,
        "layer_rpn_obj_loss": layer_rpn_obj_loss,
        "layer_rpn_bbox_direction": layer_rpn_bbox_direction,
        "layer_rpn_obj_direction": layer_rpn_obj_direction,
        "layer_roi_null_enabled": layer_roi_null_enabled,
        "layer_roi_null_scalar": layer_roi_null_scalar,
        "layer_roi_null_bbox_loss": layer_roi_null_bbox_loss,
        "layer_roi_null_cls_loss": layer_roi_null_cls_loss,
        "layer_roi_null_bbox_direction": layer_roi_null_bbox_direction,
        "layer_roi_null_cls_direction": layer_roi_null_cls_direction,
        "layer_rpn_null_enabled": layer_rpn_null_enabled,
        "layer_rpn_null_scalar": layer_rpn_null_scalar,
        "layer_rpn_null_bbox_loss": layer_rpn_null_bbox_loss,
        "layer_rpn_null_obj_loss": layer_rpn_null_obj_loss,
        "layer_rpn_null_bbox_direction": layer_rpn_null_bbox_direction,
        "layer_rpn_null_obj_direction": layer_rpn_null_obj_direction,
        "layer_frcnn_null_scalar": layer_frcnn_null_scalar,
        "layer_null_scalar": layer_null_scalar,
        "layer_null_bbox_loss": layer_null_bbox_loss,
        "layer_null_cls_loss": layer_null_cls_loss,
        "layer_null_obj_loss": layer_null_obj_loss,
        "layer_null_bbox_direction": layer_null_bbox_direction,
        "layer_null_cls_direction": layer_null_cls_direction,
        "layer_null_obj_direction": layer_null_obj_direction,
        "null_detect_bbox_loss": null_bbox_loss,
        "null_detect_cls_loss": null_cls_loss,
        "null_detect_obj_loss": null_obj_loss,
        "null_detect_bbox_direction": null_bbox_direction,
        "null_detect_cls_direction": null_cls_direction,
        "null_detect_obj_direction": null_obj_direction,
        "null_detect_feature_set": null_feature_set,
        "layer_ref_mode": "none",
        "layer_ref_type": "none",
        "layer_ref_prototype_mode": "none",
        "layer_ref_subspace_mode": "none",
        "layer_ref_subspace_centering": "centered",
        "layer_ref_subspace_rank_mode": "energy",
        "layer_ref_subspace_energy_threshold": 0.95,
        "layer_ref_subspace_k": 10,
        "layer_ref_subspace_max_samples": 1000,
        "layer_disc_separation_score": "effect_size",
        "layer_disc_topk": 3,
        "layer_disc_fn_non_fn_map_root": "",
        "layer_ref_map_root": "",
        "save_image_enabled": save_image_enabled,
        "save_image_gt_step": gt_image_step,
        "save_image_gt_max_num": gt_image_max_num,
    }


def build_target_scalar_pre_nms(target_value, raw_prediction, raw_logits):
    if target_value == "obj":
        if raw_prediction is None or raw_prediction.numel() == 0:
            return None

        return raw_prediction[..., 4].sum()

    if target_value == "cls":
        if raw_logits is None or raw_logits.numel() == 0:
            return None

        return raw_logits.max(dim=-1).values.sum()

    raise ValueError(f"Unsupported target_value: {target_value}")


def get_pre_nms_keep_indices(pred_row, logit_row=None, pre_nms_ratio=1.0):
    if pred_row is None or pred_row.numel() == 0:
        device = pred_row.device if isinstance(pred_row, torch.Tensor) else "cpu"
        return torch.zeros((0,), dtype=torch.long, device=device)
    n = int(pred_row.shape[0])
    ratio = float(pre_nms_ratio)
    if ratio <= 0.0:
        return torch.zeros((0,), dtype=torch.long, device=pred_row.device)
    if ratio >= 1.0:
        return torch.arange(n, device=pred_row.device, dtype=torch.long)

    obj = pred_row[:, 4] if pred_row.shape[1] > 4 else torch.ones((n,), device=pred_row.device)
    if logit_row is not None and isinstance(logit_row, torch.Tensor) and logit_row.numel() > 0:
        cls_prob = torch.sigmoid(logit_row)
        cls_max = cls_prob.max(dim=1).values if cls_prob.ndim == 2 and cls_prob.shape[1] > 0 else torch.ones_like(obj)
    else:
        cls_scores = pred_row[:, 5:] if pred_row.shape[1] > 5 else None
        cls_max = cls_scores.max(dim=1).values if cls_scores is not None and cls_scores.numel() > 0 else torch.ones_like(obj)
    score = obj * cls_max

    keep = int(np.ceil(n * ratio))
    keep = max(0, min(n, keep))
    if keep == 0:
        return torch.zeros((0,), dtype=torch.long, device=pred_row.device)
    if keep == n:
        return torch.arange(n, device=pred_row.device, dtype=torch.long)
    top_idx = torch.topk(score, k=keep, largest=True, sorted=False).indices
    top_idx, _ = torch.sort(top_idx)
    return top_idx


def collect_gradients_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    layer_buffer,
    pre_nms=True,
    pre_nms_ratio=1.0,
):
    grad_stats = {}
    for target_value in target_values:
        detector.zero_grad(set_to_none=True)
        layer_buffer.clear()

        grad_input = input_tensor.detach().requires_grad_(True)
        model_output = detector.model(grad_input, augment=False)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None

        target_scalar = None
        if target_value in {"obj", "cls"}:
            if pre_nms:
                pred_img = raw_prediction[0] if raw_prediction is not None and raw_prediction.ndim == 3 else None
                logit_img = raw_logits[0] if raw_logits is not None and raw_logits.ndim == 3 else None
                keep_idx = get_pre_nms_keep_indices(pred_img, logit_img, pre_nms_ratio=pre_nms_ratio)
                if int(keep_idx.shape[0]) > 0:
                    if target_value == "obj":
                        target_scalar = pred_img[keep_idx, 4].sum()
                    else:
                        logits_for_cls = logit_img if logit_img is not None else pred_img[:, 5:]
                        if logits_for_cls is not None and logits_for_cls.numel() > 0:
                            target_scalar = logits_for_cls[keep_idx].max(dim=1).values.sum()
            else:
                with torch.no_grad():
                    max_det = getattr(detector, "max_det", 300)
                    _selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                        prediction=raw_prediction,
                        logits=raw_logits,
                        conf_thres=float(getattr(detector, "conf_thresh", getattr(detector, "confidence", 0.25))),
                        iou_thres=float(getattr(detector, "iou_thresh", 0.45)),
                        classes=getattr(detector, "filter_classes", None),
                        agnostic=bool(getattr(detector, "agnostic_nms", getattr(detector, "agnostic", False))),
                        max_det=int(max_det) if max_det is not None else None,
                        return_indices=True,
                    )
                    raw_keep_indices = (
                        selected_indices[0]
                        if selected_indices
                        else torch.zeros((0,), dtype=torch.long, device=grad_input.device)
                    )
                if int(raw_keep_indices.shape[0]) > 0:
                    pred_img = raw_prediction[0]
                    logit_img = raw_logits[0] if raw_logits is not None else pred_img[:, 5:]
                    if target_value == "obj":
                        target_scalar = pred_img[raw_keep_indices, 4].sum()
                    else:
                        if logit_img is not None and logit_img.numel() > 0:
                            target_scalar = logit_img[raw_keep_indices].max(dim=1).values.sum()
        else:

            with torch.no_grad():
                max_det = getattr(detector, "max_det", 300)
                _selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    prediction=raw_prediction,
                    logits=raw_logits,
                    conf_thres=float(getattr(detector, "conf_thresh", getattr(detector, "confidence", 0.25))),
                    iou_thres=float(getattr(detector, "iou_thresh", 0.45)),
                    classes=getattr(detector, "filter_classes", None),
                    agnostic=bool(getattr(detector, "agnostic_nms", getattr(detector, "agnostic", False))),
                    max_det=int(max_det) if max_det is not None else None,
                    return_indices=True,
                )
                raw_keep_indices = (
                    selected_indices[0]
                    if selected_indices
                    else torch.zeros((0,), dtype=torch.long, device=grad_input.device)
                )

            pred_img = raw_prediction[0]
            iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
            loss_terms = []
            for bbox_idx in range(int(raw_keep_indices.shape[0])):
                raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
                losses = build_pseudo_label_losses_for_candidates(
                    pred_img=pred_img,
                    raw_idx=raw_idx,
                    iou_threshold=iou_threshold,
                )
                if losses is not None and target_value in losses:
                    loss_terms.append(losses[target_value])
            if loss_terms:
                target_scalar = torch.stack(loss_terms).mean()

        if target_scalar is None:
            for layer_name in target_layers:
                grad_stats[f"{target_value}_{layer_name}"] = []
            if grad_input.grad is not None:
                grad_input.grad = None
            del grad_input, model_output, raw_prediction, raw_logits
            layer_buffer.clear()
            continue

        target_scalar.backward()
        layer_stats = list(layer_buffer.gradients["value"])
        layer_stats.reverse()

        for layer_idx, layer_name in enumerate(target_layers):
            key = f"{target_value}_{layer_name}"
            grad_stats[key] = layer_stats[layer_idx] if layer_idx < len(layer_stats) else []

        if grad_input.grad is not None:
            grad_input.grad = None
        del grad_input, model_output, raw_prediction, raw_logits, target_scalar, layer_stats
        detector.zero_grad(set_to_none=True)
        layer_buffer.clear()
    return grad_stats


def map_grad_tensor_to_numbers(v):
    if v is None:
        return {"l1_norm": 0.0, "l2_norm": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    v = v.detach().float().reshape(-1)
    if v.numel() == 0:
        return {"l1_norm": 0.0, "l2_norm": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "l1_norm": torch.norm(v, p=1),
        "l2_norm": torch.norm(v, p=2),
        "min": v.min(),
        "max": v.max(),
        "mean": torch.mean(v),
        "std": torch.std(v, unbiased=False),
    }


def configure_mc_dropout(model: torch.nn.Module, dropout_rate: float) -> int:
    model.eval()
    count = 0
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.p = float(dropout_rate)
            module.train()
            count += 1
    return count


def enable_forced_mc_dropout_on_yolov5_head(model: torch.nn.Module, dropout_rate: float):

    detect_module = None
    for module in model.modules():
        if hasattr(module, "m") and hasattr(module, "nc") and hasattr(module, "na"):
            try:
                if isinstance(module.m, nn.ModuleList) and len(module.m) > 0 and isinstance(module.m[0], nn.Conv2d):
                    detect_module = module
            except Exception:
                continue

    if detect_module is None:
        return []

    handles = []
    p = float(dropout_rate)
    for conv in detect_module.m:
        def _pre_hook(_module, inputs, p_drop=p):
            if not inputs:
                return inputs
            x = inputs[0]
            x = F.dropout(x, p=p_drop, training=True)
            if len(inputs) == 1:
                return (x,)
            return (x,) + tuple(inputs[1:])

        handles.append(conv.register_forward_pre_hook(_pre_hook))
    return handles


def format_gradient_output(grad_tensor, vector_reduction, map_reduction="none"):
    if grad_tensor is None:
        return []
    grad_tensor = grad_tensor.detach().float()
    map_mode = str(map_reduction).strip().lower()
    if map_mode not in {"none", "energy"}:
        raise ValueError("layer_grad.map_reduction must be 'none' or 'energy'.")

    if map_mode == "energy":
        if grad_tensor.ndim == 0:
            reduced = grad_tensor.abs().reshape(1)
        elif grad_tensor.ndim == 1:
            reduced = grad_tensor.abs()
        else:
            first_dim = int(grad_tensor.shape[0])
            reduced = grad_tensor.reshape(first_dim, -1).abs().mean(dim=1)
    else:
        reduced = grad_tensor.reshape(-1)

    if not vector_reduction:
        return reduced.detach()
    stats = map_grad_tensor_to_numbers(reduced)
    return {k: stats[k] for k in vector_reduction}


def zero_grad_numbers():
    return {
        "l1_norm": 0.0,
        "l2_norm": 0.0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "std": 0.0,
    }


def build_pseudo_label_losses(pred_row):
    eps = 1e-6
    obj_prob = pred_row[4].clamp(eps, 1.0 - eps)
    cls_prob = pred_row[5:].clamp(eps, 1.0 - eps)
    cls_idx = int(torch.argmax(cls_prob.detach()).item())
    cls_target = torch.zeros_like(cls_prob)
    cls_target[cls_idx] = 1.0

    obj_loss = F.binary_cross_entropy(obj_prob, torch.ones_like(obj_prob))
    cls_loss = F.binary_cross_entropy(cls_prob, cls_target)
    return {
        "obj_loss": obj_loss,
        "cls_loss": cls_loss,
        "loss": obj_loss + cls_loss,
    }


def _xywh_to_xyxy_tensor(xywh: torch.Tensor) -> torch.Tensor:
    out = xywh.clone()
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


def _box_iou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def _box_iou_1vN_tensor(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=box.device)
    if box.ndim > 1:
        box = box.reshape(-1, 4)[0]
    lt = torch.max(box[:2], boxes[:, :2])
    rb = torch.min(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-12)


def _flatten_raw_prediction_layers(pred_layers):
    if not isinstance(pred_layers, list):
        return None
    flat = []
    for layer_pred in pred_layers:
        if not isinstance(layer_pred, torch.Tensor) or layer_pred.ndim != 5:
            return None
        flat.append(layer_pred.reshape(layer_pred.shape[0], -1, layer_pred.shape[-1]))
    if not flat:
        return None
    return torch.cat(flat, dim=1)


def _plain_iou_xywh_tensor(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    box1_xyxy = _xywh_to_xyxy_tensor(box1.reshape(-1, 4))
    box2_xyxy = _xywh_to_xyxy_tensor(box2.reshape(-1, 4))
    if box2_xyxy.shape[0] == 1 and box1_xyxy.shape[0] > 1:
        box2_xyxy = box2_xyxy.expand(box1_xyxy.shape[0], -1)
    return _box_iou_1vN_tensor(box2_xyxy[0], box1_xyxy)


def _apply_direction(pred_value: torch.Tensor, target_value: torch.Tensor, direction: str):
    if direction == "target_to_pred":
        return target_value, pred_value
    return pred_value, target_value


def _bbox_loss_xywh_tensor(
    pred_xywh: torch.Tensor,
    target_xywh: torch.Tensor,
    mode: str = "box_l1",
    reduction: str = "mean",
    direction: str = "pred_to_target",
):
    mode = str(mode).strip().lower()
    if mode == "l1":
        mode = "box_l1"
    elif mode in {"l2", "mse"}:
        mode = "box_l2"
    target_xywh = target_xywh.to(dtype=pred_xywh.dtype, device=pred_xywh.device)
    left, right = _apply_direction(pred_xywh, target_xywh, direction)

    if mode == "box_l1":
        loss = torch.abs(left - right)
    elif mode == "box_l2":
        loss = torch.square(left - right)
    else:
        raise ValueError("YOLO bbox_loss must be one of: box_l1, box_l2")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def _yolo_offset_loss_tensor(
    raw_xywh: torch.Tensor,
    target_raw_xywh: torch.Tensor = None,
    mode: str = "offset_l1",
    reduction: str = "mean",
    direction: str = "pred_to_target",
):
    mode = str(mode).strip().lower()
    if mode not in {"offset_l1", "offset_l2"}:
        raise ValueError("YOLO offset bbox_loss must be one of: offset_l1, offset_l2")
    target = torch.zeros_like(raw_xywh) if target_raw_xywh is None else target_raw_xywh.to(
        dtype=raw_xywh.dtype,
        device=raw_xywh.device,
    )
    left, right = _apply_direction(raw_xywh, target, direction)
    loss = torch.abs(left - right) if mode == "offset_l1" else torch.square(left - right)
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def _bbox_loss_xyxy_tensor(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    mode: str = "l1",
    reduction: str = "sum",
    direction: str = "pred_to_target",
):
    mode = str(mode).strip().lower()
    if mode == "l1":
        mode = "box_l1"
    elif mode in {"l2", "mse"}:
        mode = "box_l2"
    target_xyxy = target_xyxy.to(dtype=pred_xyxy.dtype, device=pred_xyxy.device)
    left, right = _apply_direction(pred_xyxy, target_xyxy, direction)
    if mode == "box_l1":
        loss = torch.abs(left - right)
    elif mode == "box_l2":
        loss = torch.square(left - right)
    else:
        raise ValueError("Faster R-CNN box bbox_loss must be one of: box_l1, box_l2")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def _box_coder_encode_single(box_coder, reference_boxes: torch.Tensor, proposals: torch.Tensor):
    encoded = box_coder.encode([reference_boxes], [proposals])
    if isinstance(encoded, (list, tuple)):
        return encoded[0]
    return encoded


def _select_faster_rcnn_roi_deltas(
    box_regression: torch.Tensor,
    proposal_indices: torch.Tensor,
    labels_internal: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_offset: int = 0,
):
    if box_regression is None or box_regression.numel() == 0:
        return None
    if proposal_indices is None or labels_internal is None:
        return None
    if row_indices.numel() == 0:
        return box_regression.new_zeros((0, 4))
    row_indices = row_indices.to(device=proposal_indices.device, dtype=torch.long)
    if int(row_indices.max().detach().cpu().item()) >= proposal_indices.shape[0]:
        return None
    num_classes = int(box_regression.shape[-1] // 4)
    proposal_idx = proposal_indices[row_indices].to(device=box_regression.device, dtype=torch.long) + int(proposal_offset)
    labels = labels_internal[row_indices].to(device=box_regression.device, dtype=torch.long)
    valid = (
        (proposal_idx >= 0)
        & (proposal_idx < box_regression.shape[0])
        & (labels >= 0)
        & (labels < num_classes)
    )
    if not bool(valid.all()):
        return None
    deltas = box_regression[proposal_idx].view(-1, num_classes, 4)
    return deltas[torch.arange(deltas.shape[0], device=box_regression.device), labels]


def _delta_loss_tensor(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    mode: str = "l1",
    reduction: str = "sum",
    direction: str = "pred_to_target",
):
    mode = str(mode).strip().lower()
    if mode == "l1":
        mode = "offset_l1"
    elif mode in {"l2", "mse"}:
        mode = "offset_l2"
    target_delta = target_delta.to(dtype=pred_delta.dtype, device=pred_delta.device)
    left, right = _apply_direction(pred_delta, target_delta, direction)
    if mode == "offset_l1":
        loss = torch.abs(left - right)
    elif mode == "offset_l2":
        loss = torch.square(left - right)
    else:
        raise ValueError("Faster R-CNN offset bbox_loss must be one of: offset_l1, offset_l2")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def _objectness_loss_tensor(
    raw_obj: torch.Tensor,
    target_prob: torch.Tensor,
    mode: str = "bcewithlogits",
    direction: str = "pred_to_target",
    reduction: str = "sum",
):
    target_prob = target_prob.to(dtype=raw_obj.dtype, device=raw_obj.device)
    mode = str(mode).strip().lower()
    if mode == "bcewithlogits":
        if direction != "pred_to_target":
            raise ValueError("obj_direction=target_to_pred is not supported when obj_loss=bcewithlogits.")
        loss = F.binary_cross_entropy_with_logits(raw_obj, target_prob, reduction="none")
    elif mode in {"abs_diff", "signed_diff"}:
        if direction == "target_to_pred" and mode != "signed_diff":
            raise ValueError("obj_direction=target_to_pred is only supported when obj_loss=signed_diff.")
        pred_prob = torch.sigmoid(raw_obj)
        left, right = _apply_direction(pred_prob, target_prob, direction)
        loss = torch.abs(left - right) if mode == "abs_diff" else (left - right)
    else:
        raise ValueError("obj_loss must be one of: bcewithlogits, abs_diff, signed_diff")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def _smooth_distribution(target_prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target_prob = target_prob.clamp(min=eps)
    return target_prob / target_prob.sum(dim=-1, keepdim=True).clamp(min=eps)


def _class_loss_tensor(
    raw_cls_logits: torch.Tensor,
    target_prob: torch.Tensor,
    class_idx: int = None,
    mode: str = "bcewithlogits",
    direction: str = "pred_to_target",
    reduction: str = "sum",
):
    if raw_cls_logits.ndim == 1:
        raw_cls_logits = raw_cls_logits.view(1, -1)
    if target_prob.ndim == 1:
        target_prob = target_prob.view(1, -1)
    target_prob = target_prob.to(dtype=raw_cls_logits.dtype, device=raw_cls_logits.device)
    if target_prob.shape[0] == 1 and raw_cls_logits.shape[0] > 1:
        target_prob = target_prob.expand(raw_cls_logits.shape[0], -1)

    mode = str(mode).strip().lower()
    if mode == "bcewithlogits":
        if direction != "pred_to_target":
            raise ValueError("cls_direction=target_to_pred is not supported when cls_loss=bcewithlogits.")
        loss = F.binary_cross_entropy_with_logits(raw_cls_logits, target_prob, reduction="none").sum(dim=-1)
    elif mode == "ce":
        if direction != "pred_to_target":
            raise ValueError("cls_direction=target_to_pred is not supported when cls_loss=ce.")
        if class_idx is None:
            loss = -(target_prob * F.log_softmax(raw_cls_logits, dim=-1)).sum(dim=-1)
        else:
            class_target = torch.full(
                (raw_cls_logits.shape[0],),
                int(class_idx),
                dtype=torch.long,
                device=raw_cls_logits.device,
            )
            loss = F.cross_entropy(raw_cls_logits, class_target, reduction="none")
    elif mode == "kl":
        pred_log_prob = F.log_softmax(raw_cls_logits, dim=-1)
        if direction == "target_to_pred":
            pred_prob = pred_log_prob.exp()
            target_log_prob = _smooth_distribution(target_prob).log()
            loss = (pred_prob * (pred_log_prob - target_log_prob)).sum(dim=-1)
        else:
            safe_target = target_prob.clamp(min=1e-12)
            loss = torch.where(
                target_prob > 0,
                target_prob * (safe_target.log() - pred_log_prob),
                torch.zeros_like(target_prob),
            ).sum(dim=-1)
    else:
        raise ValueError("cls_loss must be one of: bcewithlogits, kl, ce")

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


@dataclass
class YoloCandidateCache:
    raw_xyxy: torch.Tensor
    raw_score: torch.Tensor
    raw_cls: torch.Tensor
    score_mask: torch.Tensor
    class_to_indices: dict


def build_yolo_candidate_cache(pred_img: torch.Tensor, score_threshold: float):
    with torch.no_grad():
        raw_xyxy = _xywh_to_xyxy_tensor(pred_img[:, :4].detach())
        cls_probs = pred_img[:, 5:].detach()
        raw_cls = (
            torch.argmax(cls_probs, dim=1)
            if cls_probs.numel()
            else torch.zeros((pred_img.shape[0],), dtype=torch.long, device=pred_img.device)
        )
        obj = pred_img[:, 4].detach()
        cls_max = cls_probs.max(dim=1).values if cls_probs.numel() else torch.ones_like(obj)
        raw_score = obj * cls_max
        score_mask = raw_score >= float(score_threshold)
        class_to_indices = {}
        if bool(score_mask.any()):
            for cls_id in torch.unique(raw_cls[score_mask]).detach().cpu().tolist():
                cls_id = int(cls_id)
                class_to_indices[cls_id] = torch.nonzero(score_mask & (raw_cls == cls_id), as_tuple=False).flatten()
        return YoloCandidateCache(
            raw_xyxy=raw_xyxy,
            raw_score=raw_score,
            raw_cls=raw_cls,
            score_mask=score_mask,
            class_to_indices=class_to_indices,
        )


def yolo_candidate_mask_from_cache(cache: YoloCandidateCache, raw_idx: int, iou_threshold: float):
    if raw_idx >= int(cache.raw_xyxy.shape[0]):
        return None, None
    pseudo_cls = int(cache.raw_cls[raw_idx].detach().cpu().item())
    candidate_indices = cache.class_to_indices.get(pseudo_cls)
    ious = torch.zeros((cache.raw_xyxy.shape[0],), dtype=cache.raw_xyxy.dtype, device=cache.raw_xyxy.device)
    candidate_mask = torch.zeros((cache.raw_xyxy.shape[0],), dtype=torch.bool, device=cache.raw_xyxy.device)
    if candidate_indices is None or int(candidate_indices.shape[0]) <= 0:
        return candidate_mask, ious
    candidate_ious = _box_iou_1vN_tensor(cache.raw_xyxy[raw_idx].view(1, 4), cache.raw_xyxy[candidate_indices])
    ious[candidate_indices] = candidate_ious
    keep_indices = candidate_indices[candidate_ious > float(iou_threshold)]
    if keep_indices.numel() > 0:
        candidate_mask[keep_indices] = True
    return candidate_mask, ious


def build_yolo_candidate_mask_for_pseudo(
    pred_img: torch.Tensor,
    raw_idx: int,
    iou_threshold: float,
    score_threshold: float = 0.01,
):
    if raw_idx >= pred_img.shape[0]:
        return None

    cache = build_yolo_candidate_cache(pred_img, score_threshold)
    candidate_mask, _ious = yolo_candidate_mask_from_cache(cache, raw_idx, iou_threshold)
    return candidate_mask


def build_pseudo_label_losses_for_candidates(
    pred_img: torch.Tensor,
    raw_idx: int,
    iou_threshold: float,
    score_threshold: float = 0.01,
    bbox_loss: str = "offset_l1",
    cls_loss: str = "bcewithlogits",
    obj_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    cls_direction: str = "pred_to_target",
    obj_direction: str = "pred_to_target",
    raw_img: torch.Tensor = None,
    requested_losses=None,
    candidate_mask: torch.Tensor = None,
    timing_accumulator=None,
    timing_device=None,
):
    if raw_idx >= pred_img.shape[0]:
        return None

    requested_losses = set(requested_losses or ["bbox_loss", "obj_loss", "cls_loss"])
    need_bbox = "bbox_loss" in requested_losses
    need_obj = "obj_loss" in requested_losses
    need_cls = "cls_loss" in requested_losses

    pseudo_row = pred_img[raw_idx].detach()
    pseudo_cls = None
    if candidate_mask is None:
        t_candidate = _start_timing(timing_device)
        candidate_mask = build_yolo_candidate_mask_for_pseudo(
            pred_img=pred_img,
            raw_idx=raw_idx,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        _add_elapsed_timing(timing_accumulator, "candidate_search_sec", t_candidate, timing_device)
    if candidate_mask is None:
        return None

    t_loss = _start_timing(timing_device)
    use_offset_bbox = need_bbox and str(bbox_loss).strip().lower().startswith("offset_")
    if raw_img is not None and raw_img.shape[0] == pred_img.shape[0]:
        raw_bbox = raw_img[:, :4] if use_offset_bbox else None
        raw_obj = raw_img[:, 4] if need_obj else None
        raw_cls = raw_img[:, 5:] if need_cls else None
    else:
        raw_bbox = None
        eps = 1e-6
        raw_obj = torch.logit(pred_img[:, 4].clamp(eps, 1.0 - eps)) if need_obj else None
        raw_cls = torch.logit(pred_img[:, 5:].clamp(eps, 1.0 - eps)) if need_cls else None

    losses = {}
    if need_bbox:
        if use_offset_bbox:
            if raw_bbox is None:
                raise ValueError("YOLO cand_target offset bbox_loss requires raw YOLO head outputs.")
            candidate_raw_bbox = raw_bbox[candidate_mask]
            pseudo_raw_bbox_target = raw_bbox[raw_idx].detach().view(1, 4).expand(candidate_raw_bbox.shape[0], -1)
            losses["bbox_loss"] = _yolo_offset_loss_tensor(
                candidate_raw_bbox,
                target_raw_xywh=pseudo_raw_bbox_target,
                mode=bbox_loss,
                reduction="sum",
                direction=bbox_direction,
            )
        else:
            candidate_boxes = pred_img[candidate_mask, :4]
            pseudo_box_target = pseudo_row[:4].view(1, 4).expand(candidate_boxes.shape[0], -1)
            losses["bbox_loss"] = _bbox_loss_xywh_tensor(
                candidate_boxes,
                pseudo_box_target,
                mode=bbox_loss,
                reduction="sum",
                direction=bbox_direction,
            )

    if need_obj:
        candidate_raw_obj = raw_obj[candidate_mask]
        obj_target = pseudo_row[4].detach().to(dtype=candidate_raw_obj.dtype, device=candidate_raw_obj.device)
        obj_target = obj_target.clamp(min=0.0, max=1.0).expand_as(candidate_raw_obj)
        losses["obj_loss"] = _objectness_loss_tensor(
            candidate_raw_obj,
            obj_target,
            mode=obj_loss,
            direction=obj_direction,
            reduction="sum",
        )

    if need_cls:
        pseudo_cls = int(torch.argmax(pseudo_row[5:]).item())
        candidate_raw_cls = raw_cls[candidate_mask]
        cls_target = torch.zeros_like(candidate_raw_cls)
        cls_target[:, pseudo_cls] = 1.0
        losses["cls_loss"] = _class_loss_tensor(
            candidate_raw_cls,
            cls_target,
            class_idx=pseudo_cls,
            mode=cls_loss,
            direction=cls_direction,
            reduction="sum",
        )
    _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
    return losses


def build_faster_rcnn_roi_candidate_losses(
    roi_box_coder,
    pred_img: torch.Tensor,
    logit_img: torch.Tensor,
    raw_idx: int,
    iou_threshold: float,
    box_regression: torch.Tensor = None,
    proposals_xyxy: torch.Tensor = None,
    proposal_indices_img: torch.Tensor = None,
    labels_internal_img: torch.Tensor = None,
    proposal_offset: int = 0,
    score_threshold: float = 0.01,
    bbox_loss: str = "box_l1",
    cls_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    cls_direction: str = "pred_to_target",
    timing_accumulator=None,
    timing_device=None,
):
    if raw_idx >= pred_img.shape[0]:
        return None
    if logit_img is None or logit_img.numel() == 0:
        return None

    t_candidate = _start_timing(timing_device)
    with torch.no_grad():
        from commands.predict.faster_rcnn.candidates import (
            build_faster_rcnn_roi_candidate_cache,
            faster_rcnn_roi_candidate_mask_from_cache,
        )

        pseudo_row = pred_img[raw_idx].detach()
        pseudo_cls = int(pseudo_row[5].detach().long().item())
        pseudo_box_xyxy = _xywh_to_xyxy_tensor(pseudo_row[:4].view(1, 4))
        candidate_cache = build_faster_rcnn_roi_candidate_cache(pred_img, logit_img, detach=True)
        candidate_mask, _ious = faster_rcnn_roi_candidate_mask_from_cache(
            candidate_cache,
            pseudo_box_xyxy,
            pseudo_cls,
            score_threshold,
            iou_threshold,
        )
    _add_elapsed_timing(timing_accumulator, "candidate_search_sec", t_candidate, timing_device)

    t_loss = _start_timing(timing_device)
    if str(bbox_loss).strip().lower().startswith("offset_"):
        candidate_indices = torch.where(candidate_mask)[0]
        candidate_deltas = _select_faster_rcnn_roi_deltas(
            box_regression,
            proposal_indices_img,
            labels_internal_img,
            candidate_indices,
            proposal_offset=proposal_offset,
        )
        pseudo_delta = _select_faster_rcnn_roi_deltas(
            box_regression,
            proposal_indices_img,
            labels_internal_img,
            torch.tensor([raw_idx], dtype=torch.long, device=candidate_indices.device),
            proposal_offset=proposal_offset,
        )
        if candidate_deltas is None or pseudo_delta is None:
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return None
        if proposals_xyxy is None:
            pseudo_delta = pseudo_delta.detach().expand(candidate_deltas.shape[0], -1)
        else:
            proposal_idx = proposal_indices_img[candidate_indices].to(device=proposals_xyxy.device, dtype=torch.long)
            valid = (proposal_idx >= 0) & (proposal_idx < proposals_xyxy.shape[0])
            if not bool(valid.all()):
                _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
                return None
            candidate_proposals = proposals_xyxy[proposal_idx].detach()
            pseudo_box_target = _xywh_to_xyxy_tensor(pred_img[raw_idx, :4].view(1, 4)).detach()
            pseudo_box_target = pseudo_box_target.expand(candidate_proposals.shape[0], -1)
            pseudo_delta = _box_coder_encode_single(
                roi_box_coder,
                pseudo_box_target,
                candidate_proposals,
            )
        bbox_loss_value = _delta_loss_tensor(
            candidate_deltas,
            pseudo_delta,
            mode=bbox_loss,
            reduction="sum",
            direction=bbox_direction,
        )
    else:
        candidate_boxes_xyxy = _xywh_to_xyxy_tensor(pred_img[candidate_mask, :4])
        pseudo_box_target = _xywh_to_xyxy_tensor(pred_img[raw_idx, :4].view(1, 4)).detach()
        pseudo_box_target = pseudo_box_target.expand(candidate_boxes_xyxy.shape[0], -1)
        bbox_loss_value = _bbox_loss_xyxy_tensor(
            candidate_boxes_xyxy,
            pseudo_box_target,
            mode=bbox_loss,
            reduction="sum",
            direction=bbox_direction,
        )

    candidate_logits = logit_img[candidate_mask]
    cls_target = torch.zeros_like(candidate_logits)
    if 0 <= pseudo_cls < cls_target.shape[1]:
        cls_target[:, pseudo_cls] = 1.0
    cls_loss_value = _class_loss_tensor(
        candidate_logits,
        cls_target,
        class_idx=pseudo_cls,
        mode=cls_loss,
        direction=cls_direction,
        reduction="sum",
    )
    _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
    return {
        "roi_bbox_loss": bbox_loss_value,
        "roi_cls_loss": cls_loss_value,
    }


def _concat_rpn_prediction_layers(objectness, pred_bbox_deltas):
    objectness_flat = []
    bbox_deltas_flat = []
    for obj_per_level, bbox_per_level in zip(objectness, pred_bbox_deltas):
        n, a, h, w = obj_per_level.shape
        obj_per_level = obj_per_level.permute(0, 2, 3, 1).reshape(n, -1)
        bbox_per_level = bbox_per_level.view(n, -1, 4, h, w)
        bbox_per_level = bbox_per_level.permute(0, 3, 4, 1, 2).reshape(n, -1, 4)
        objectness_flat.append(obj_per_level)
        bbox_deltas_flat.append(bbox_per_level)
    return torch.cat(objectness_flat, dim=1), torch.cat(bbox_deltas_flat, dim=1)


def _filter_rpn_proposals_with_indices(rpn, proposals, objectness, image_shapes, num_anchors_per_level):
    num_images = proposals.shape[0]
    device = proposals.device
    objectness = objectness.detach().reshape(num_images, -1)
    raw_indices = torch.arange(objectness.shape[1], device=device).reshape(1, -1).expand_as(objectness)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device)
        for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0).reshape(1, -1).expand_as(objectness)

    top_n_idx = rpn._get_top_n_idx(objectness, num_anchors_per_level)
    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]
    raw_indices = raw_indices[batch_idx, top_n_idx]

    objectness_prob = torch.sigmoid(objectness)
    final_boxes = []
    final_scores = []
    final_raw_indices = []
    for boxes, scores, lvl, raw_idx, img_shape in zip(proposals, objectness_prob, levels, raw_indices, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

        keep = box_ops.remove_small_boxes(boxes, rpn.min_size)
        boxes, scores, lvl, raw_idx = boxes[keep], scores[keep], lvl[keep], raw_idx[keep]

        keep = torch.where(scores >= rpn.score_thresh)[0]
        boxes, scores, lvl, raw_idx = boxes[keep], scores[keep], lvl[keep], raw_idx[keep]

        keep = box_ops.batched_nms(boxes, scores, lvl, rpn.nms_thresh)
        keep = keep[: rpn.post_nms_top_n()]
        final_boxes.append(boxes[keep])
        final_scores.append(scores[keep])
        final_raw_indices.append(raw_idx[keep])

    return final_boxes, final_scores, final_raw_indices


def _resize_boxes_xyxy_tensor(boxes, from_size, to_size):
    if boxes.numel() == 0:
        return boxes
    from_h, from_w = float(from_size[0]), float(from_size[1])
    to_h, to_w = float(to_size[0]), float(to_size[1])
    ratio_h = to_h / max(from_h, 1.0)
    ratio_w = to_w / max(from_w, 1.0)
    xmin, ymin, xmax, ymax = boxes.unbind(dim=1)
    return torch.stack((xmin * ratio_w, ymin * ratio_h, xmax * ratio_w, ymax * ratio_h), dim=1)


def build_faster_rcnn_rpn_candidate_losses(
    rpn_box_coder,
    rpn_bbox_deltas: torch.Tensor,
    rpn_anchors: torch.Tensor,
    rpn_search_boxes_xyxy: torch.Tensor,
    rpn_objectness_logits: torch.Tensor,
    source_proposal_xyxy: torch.Tensor,
    source_obj_logit: torch.Tensor,
    from_size,
    to_size,
    iou_threshold: float,
    source_bbox_delta: torch.Tensor = None,
    final_box_xyxy: torch.Tensor = None,
    obj_threshold: float = 0.0,
    bbox_loss: str = "offset_l1",
    obj_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    obj_direction: str = "pred_to_target",
    timing_accumulator=None,
    timing_device=None,
):
    if rpn_search_boxes_xyxy is None or rpn_search_boxes_xyxy.numel() == 0:
        return None
    if rpn_objectness_logits is None or rpn_objectness_logits.numel() == 0:
        return None
    if source_proposal_xyxy is None and final_box_xyxy is None:
        return None
    if not str(bbox_loss).strip().lower().startswith("offset_"):
        raise ValueError("Faster R-CNN RPN bbox_loss supports only offset_l1 or offset_l2.")
    if obj_direction == "target_to_pred" and obj_loss != "signed_diff":
        raise ValueError("Faster R-CNN RPN candidate obj_direction=target_to_pred is only supported when obj_loss=signed_diff.")

    t_candidate = _start_timing(timing_device)
    with torch.no_grad():
        from commands.predict.faster_rcnn.candidates import (
            build_faster_rcnn_rpn_candidate_cache,
            faster_rcnn_rpn_candidate_mask_from_cache,
        )

        target_box = final_box_xyxy if final_box_xyxy is not None else source_proposal_xyxy
        source_proposal = target_box.detach().view(1, 4)
        candidate_cache = build_faster_rcnn_rpn_candidate_cache(
            rpn_search_boxes_xyxy,
            rpn_objectness_logits,
            rpn_anchors,
            rpn_bbox_deltas,
            detach=True,
        )
        candidate_mask, _ious = faster_rcnn_rpn_candidate_mask_from_cache(
            candidate_cache,
            source_proposal,
            obj_threshold,
            iou_threshold,
        )
    _add_elapsed_timing(timing_accumulator, "candidate_search_sec", t_candidate, timing_device)

    if not bool(candidate_mask.any()):
        return None

    t_loss = _start_timing(timing_device)
    selected_deltas = rpn_bbox_deltas[candidate_mask]
    selected_anchors = rpn_anchors[candidate_mask]
    target_box_in_rpn_scale = _resize_boxes_xyxy_tensor(source_proposal, to_size, from_size)
    target_boxes = target_box_in_rpn_scale.expand(selected_anchors.shape[0], -1)
    target_delta = _box_coder_encode_single(
        rpn_box_coder,
        target_boxes,
        selected_anchors.detach(),
    )
    bbox_loss_value = _delta_loss_tensor(
        selected_deltas,
        target_delta.detach(),
        mode=bbox_loss,
        reduction="sum",
        direction=bbox_direction,
    )

    selected_logits = rpn_objectness_logits.reshape(-1)[candidate_mask]
    obj_target = torch.ones_like(selected_logits)
    obj_loss_value = _objectness_loss_tensor(
        selected_logits,
        obj_target,
        mode=obj_loss,
        direction=obj_direction,
        reduction="sum",
    )
    _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
    return {
        "rpn_bbox_loss": bbox_loss_value,
        "rpn_obj_loss": obj_loss_value,
    }


def build_faster_rcnn_null_losses_by_stage(
    rpn_box_coder,
    roi_box_coder,
    rpn_bbox_deltas: torch.Tensor,
    rpn_anchors: torch.Tensor,
    box_regression: torch.Tensor,
    pred_img: torch.Tensor,
    logit_img: torch.Tensor,
    raw_idx: int,
    labels_internal_img: torch.Tensor,
    proposal_indices_img: torch.Tensor,
    proposals_xyxy: torch.Tensor,
    proposal_to_rpn_raw_idx: torch.Tensor,
    rpn_objectness_logits: torch.Tensor,
    final_box_xyxy: torch.Tensor,
    from_size,
    to_size,
    proposal_offset: int = 0,
    rpn_bbox_loss: str = "offset_l1",
    rpn_obj_loss: str = "bcewithlogits",
    roi_bbox_loss: str = "box_l1",
    roi_cls_loss: str = "bcewithlogits",
    rpn_bbox_direction: str = "pred_to_target",
    rpn_obj_direction: str = "pred_to_target",
    roi_bbox_direction: str = "pred_to_target",
    roi_cls_direction: str = "pred_to_target",
    timing_accumulator=None,
    timing_device=None,
):
    if raw_idx >= pred_img.shape[0]:
        return None
    if logit_img is None or logit_img.numel() == 0 or raw_idx >= logit_img.shape[0]:
        return None
    if box_regression is None or box_regression.numel() == 0:
        return None
    if labels_internal_img is None or raw_idx >= labels_internal_img.shape[0]:
        return None
    if proposal_indices_img is None or raw_idx >= proposal_indices_img.shape[0]:
        return None
    if proposal_to_rpn_raw_idx is None or proposal_to_rpn_raw_idx.numel() == 0:
        return None
    if rpn_obj_direction == "target_to_pred" and rpn_obj_loss != "signed_diff":
        raise ValueError("Faster R-CNN null_target rpn.obj_direction=target_to_pred is only supported when obj_loss=signed_diff.")

    t_loss = _start_timing(timing_device)
    proposal_idx = int(proposal_indices_img[raw_idx].detach().cpu().item())
    if proposal_idx < 0 or proposal_idx >= proposals_xyxy.shape[0] or proposal_idx >= proposal_to_rpn_raw_idx.shape[0]:
        _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
        return None
    rpn_raw_idx = int(proposal_to_rpn_raw_idx[proposal_idx].detach().cpu().item())
    if rpn_raw_idx < 0 or rpn_raw_idx >= rpn_objectness_logits.reshape(-1).shape[0]:
        _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
        return None
    if rpn_raw_idx >= rpn_bbox_deltas.shape[0] or rpn_raw_idx >= rpn_anchors.shape[0]:
        _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
        return None
    label_internal = int(labels_internal_img[raw_idx].detach().cpu().item())
    num_classes = int(box_regression.shape[-1] // 4)
    if label_internal < 0 or label_internal >= num_classes:
        _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
        return None
    final_box_target = final_box_xyxy.detach().to(device=pred_img.device).view(1, 4)
    final_box_target_in_rpn_scale = _resize_boxes_xyxy_tensor(final_box_target, to_size, from_size)
    selected_deltas = rpn_bbox_deltas[rpn_raw_idx].view(1, 4)
    selected_anchor = rpn_anchors[rpn_raw_idx].view(1, 4)
    if not str(rpn_bbox_loss).strip().lower().startswith("offset_"):
        raise ValueError("Faster R-CNN RPN null_target bbox_loss supports only offset_l1 or offset_l2.")
    rpn_target_delta = _box_coder_encode_single(
        rpn_box_coder,
        final_box_target_in_rpn_scale,
        selected_anchor.detach(),
    ).view(1, 4)
    rpn_bbox_loss_value = _delta_loss_tensor(
        selected_deltas,
        rpn_target_delta.detach(),
        mode=rpn_bbox_loss,
        reduction="sum",
        direction=rpn_bbox_direction,
    )

    selected_obj_logit = rpn_objectness_logits.reshape(-1)[rpn_raw_idx]
    obj_target = torch.full_like(selected_obj_logit, 0.5)
    rpn_obj_loss_value = _objectness_loss_tensor(
        selected_obj_logit,
        obj_target,
        mode=rpn_obj_loss,
        direction=rpn_obj_direction,
        reduction="sum",
    )

    source_proposal = proposals_xyxy[proposal_idx].detach().view(1, 4)
    if str(roi_bbox_loss).strip().lower().startswith("offset_"):
        roi_delta = _select_faster_rcnn_roi_deltas(
            box_regression,
            proposal_indices_img,
            labels_internal_img,
            torch.tensor([raw_idx], dtype=torch.long, device=pred_img.device),
            proposal_offset=proposal_offset,
        )
        if roi_delta is None:
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return None
        roi_bbox_loss_value = _delta_loss_tensor(
            roi_delta,
            torch.zeros_like(roi_delta),
            mode=roi_bbox_loss,
            reduction="sum",
            direction=roi_bbox_direction,
        )
    else:
        final_box_from_roi = _xywh_to_xyxy_tensor(pred_img[raw_idx, :4].view(1, 4))
        roi_bbox_loss_value = _bbox_loss_xyxy_tensor(
            final_box_from_roi,
            source_proposal,
            mode=roi_bbox_loss,
            reduction="sum",
            direction=roi_bbox_direction,
        )

    cls_logits = logit_img[raw_idx]
    cls_target_value = (
        0.5
        if str(roi_cls_loss).strip().lower() == "bcewithlogits"
        else 1.0 / float(cls_logits.numel())
    )
    cls_target = torch.full_like(cls_logits, cls_target_value)
    roi_cls_loss_value = _class_loss_tensor(
        cls_logits,
        cls_target,
        class_idx=None,
        mode=roi_cls_loss,
        direction=roi_cls_direction,
        reduction="sum",
    )
    _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
    return {
        "rpn_bbox_loss": rpn_bbox_loss_value,
        "rpn_obj_loss": rpn_obj_loss_value,
        "roi_bbox_loss": roi_bbox_loss_value,
        "roi_cls_loss": roi_cls_loss_value,
    }


def build_layer_target_scalar_bbox(
    target_value,
    pred_img,
    logit_img,
    raw_img,
    raw_idx,
    iou_threshold,
    pseudo_gt="cand",
    anchor_xywh=None,
    cand_score_threshold=0.01,
    bbox_loss: str = "box_l1",
    cls_loss: str = "bcewithlogits",
    obj_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    cls_direction: str = "pred_to_target",
    obj_direction: str = "pred_to_target",
    candidate_mask: torch.Tensor = None,
    timing_accumulator=None,
    timing_device=None,
):
    if target_value == "obj":
        if raw_idx >= pred_img.shape[0]:
            return None
        return pred_img[raw_idx, 4]
    if target_value == "cls":
        if raw_idx >= logit_img.shape[0]:
            return None
        return logit_img[raw_idx].max()

    if pseudo_gt == "uniform":
        if raw_idx >= pred_img.shape[0]:
            return None
        t_loss = _start_timing(timing_device)
        pred_row = pred_img[raw_idx]
        raw_row = raw_img[raw_idx] if raw_img is not None and raw_idx < raw_img.shape[0] else None
        if target_value == "obj_loss":
            if raw_row is not None:
                raw_obj = raw_row[4]
            else:
                raw_obj = torch.logit(pred_row[4].clamp(1e-6, 1.0 - 1e-6))
            obj_target = torch.full_like(raw_obj, 0.5)
            loss = _objectness_loss_tensor(raw_obj, obj_target, mode=obj_loss, direction=obj_direction, reduction="sum")
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return loss
        if target_value == "cls_loss":
            if raw_row is not None:
                cls_logits = raw_row[5:]
            else:
                cls_logits = logit_img[raw_idx] if (logit_img is not None and raw_idx < logit_img.shape[0]) else pred_row[5:]
            if cls_logits.numel() == 0:
                _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
                return None
            cls_target_value = (
                0.5
                if str(cls_loss).strip().lower() == "bcewithlogits"
                else 1.0 / float(cls_logits.numel())
            )
            uniform_target = torch.full_like(cls_logits, cls_target_value)
            loss = _class_loss_tensor(
                cls_logits,
                uniform_target,
                class_idx=None,
                mode=cls_loss,
                direction=cls_direction,
                reduction="sum",
            )
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return loss
        if target_value == "bbox_loss":
            if anchor_xywh is None:
                _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
                return None
            if str(bbox_loss).strip().lower().startswith("offset_"):
                if raw_row is None:
                    _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
                    return None
                loss = _yolo_offset_loss_tensor(
                    raw_row[:4],
                    mode=bbox_loss,
                    reduction="mean",
                    direction=bbox_direction,
                )
            else:
                pred_xywh = pred_row[:4]
                anchor_xywh = anchor_xywh.to(dtype=pred_xywh.dtype, device=pred_xywh.device)
                loss = _bbox_loss_xywh_tensor(
                    pred_xywh,
                    anchor_xywh,
                    mode=bbox_loss,
                    reduction="mean",
                    direction=bbox_direction,
                )
            _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
            return loss
        _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss, timing_device)
        return None

    losses = build_pseudo_label_losses_for_candidates(
        pred_img=pred_img,
        raw_idx=raw_idx,
        iou_threshold=iou_threshold,
        score_threshold=cand_score_threshold,
        bbox_loss=bbox_loss,
        cls_loss=cls_loss,
        obj_loss=obj_loss,
        bbox_direction=bbox_direction,
        cls_direction=cls_direction,
        obj_direction=obj_direction,
        raw_img=raw_img,
        requested_losses=[target_value],
        candidate_mask=candidate_mask,
        timing_accumulator=timing_accumulator,
        timing_device=timing_device,
    )
    if losses is not None and target_value in losses:
        return losses[target_value]
    return None


def collect_faster_rcnn_candidate_layer_grads_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    target_layer_map=None,
    map_reduction="none",
    vector_reduction=None,
    target_mode="cand",
    roi_cand_enabled=True,
    roi_cand_score_threshold=0.01,
    roi_bbox_loss: str = "l1",
    roi_cls_loss: str = "bcewithlogits",
    roi_bbox_direction: str = "pred_to_target",
    roi_cls_direction: str = "pred_to_target",
    rpn_cand_enabled=False,
    rpn_cand_obj_threshold=0.0,
    rpn_bbox_loss: str = "offset_l1",
    rpn_obj_loss: str = "bcewithlogits",
    rpn_bbox_direction: str = "pred_to_target",
    rpn_obj_direction: str = "pred_to_target",
    roi_null_enabled=True,
    roi_null_bbox_loss: str = "l1",
    roi_null_cls_loss: str = "bcewithlogits",
    roi_null_bbox_direction: str = "pred_to_target",
    roi_null_cls_direction: str = "pred_to_target",
    rpn_null_enabled=True,
    rpn_null_bbox_loss: str = "offset_l1",
    rpn_null_obj_loss: str = "bcewithlogits",
    rpn_null_bbox_direction: str = "pred_to_target",
    rpn_null_obj_direction: str = "pred_to_target",
    rpn_null_obj_target: float = 0.5,
    null_bbox_loss: str = "l1",
    null_cls_loss: str = "bcewithlogits",
    null_obj_loss: str = "bcewithlogits",
    null_bbox_direction: str = "pred_to_target",
    null_cls_direction: str = "pred_to_target",
    null_obj_direction: str = "pred_to_target",
    timing_accumulator=None,
    timing_device=None,
):
    target_mode = str(target_mode).strip().lower()
    if target_mode not in {"cand", "frcnn_null"}:
        raise ValueError("Faster R-CNN layer_grad target_mode must be cand or null.")
    roi_bbox_loss = str(roi_bbox_loss).strip().lower()
    rpn_bbox_loss = str(rpn_bbox_loss).strip().lower()
    roi_null_bbox_loss = str(roi_null_bbox_loss).strip().lower()
    rpn_null_bbox_loss = str(rpn_null_bbox_loss).strip().lower()
    bbox_supported = {"box_l1", "box_l2", "offset_l1", "offset_l2", "l1", "l2"}
    if (
        roi_bbox_loss not in bbox_supported
        or rpn_bbox_loss not in bbox_supported
        or roi_null_bbox_loss not in bbox_supported
        or rpn_null_bbox_loss not in bbox_supported
    ):
        raise ValueError("Faster R-CNN layer_grad bbox_loss supports box_l1, box_l2, offset_l1, offset_l2.")
    if not rpn_bbox_loss.startswith("offset_") or not rpn_null_bbox_loss.startswith("offset_"):
        raise ValueError("Faster R-CNN RPN layer_grad bbox_loss supports only offset_l1 or offset_l2.")

    if not isinstance(input_tensor, list):
        input_images = [img for img in input_tensor] if isinstance(input_tensor, torch.Tensor) else list(input_tensor)
    else:
        input_images = input_tensor
    target_layer_map = target_layer_map or {target_value: list(target_layers) for target_value in target_values}
    ordered_layer_names = []
    for target_value in target_values:
        for layer_name in target_layer_map.get(target_value, []):
            if layer_name not in ordered_layer_names:
                ordered_layer_names.append(layer_name)
    layer_params_by_name = {
        layer_name: resolve_layer_parameter(detector.model, layer_name)
        for layer_name in ordered_layer_names
    }
    layer_params = list(layer_params_by_name.values())

    model = detector.detector_model
    model_params = list(model.parameters())
    original_model_requires_grad = [bool(p.requires_grad) for p in model_params]
    for param in model_params:
        param.requires_grad_(False)
    for param in layer_params:
        param.requires_grad_(True)

    was_training = model.training
    model.eval()

    image_list = [
        img.to(detector.device, non_blocking=True) if img.device != detector.device else img
        for img in input_images
    ]
    original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in image_list]
    transformed_images, _targets = model.transform(image_list, None)
    t_detector = _start_timing(timing_device)
    features = model.backbone(transformed_images.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    features_list = list(features.values())
    rpn_objectness_list, rpn_bbox_delta_list = model.rpn.head(features_list)
    num_anchors_per_level = [
        int(obj_per_level.shape[1] * obj_per_level.shape[2] * obj_per_level.shape[3])
        for obj_per_level in rpn_objectness_list
    ]
    rpn_anchors = model.rpn.anchor_generator(transformed_images, features_list)
    rpn_objectness_flat, rpn_bbox_deltas_flat = _concat_rpn_prediction_layers(
        rpn_objectness_list,
        rpn_bbox_delta_list,
    )
    rpn_decoded_for_roi = model.rpn.box_coder.decode(rpn_bbox_deltas_flat.detach(), rpn_anchors).view(
        len(image_list), -1, 4
    )
    proposals, _proposal_scores, proposal_to_rpn_raw_indices = _filter_rpn_proposals_with_indices(
        model.rpn,
        rpn_decoded_for_roi,
        rpn_objectness_flat.detach(),
        transformed_images.image_sizes,
        num_anchors_per_level,
    )
    proposal_offsets = []
    running_proposal_offset = 0
    for proposal_img in proposals:
        proposal_offsets.append(running_proposal_offset)
        running_proposal_offset += int(proposal_img.shape[0])
    roi_heads = model.roi_heads
    box_features = roi_heads.box_roi_pool(features, proposals, transformed_images.image_sizes)
    box_features = roi_heads.box_head(box_features)
    class_logits, box_regression = roi_heads.box_predictor(box_features)
    with detector.temporary_roi_score_threshold(0.0):
        detections = detector._pre_nms_detections_with_logits(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=proposals,
            image_shapes=transformed_images.image_sizes,
        )
    detections = model.transform.postprocess(detections, transformed_images.image_sizes, original_image_sizes)
    proposal_indices_by_img = []
    labels_internal_by_img = []
    for det_img in detections:
        proposal_indices_img = None
        labels_internal_img = None
        if "proposal_indices" in det_img:
            with torch.no_grad():
                labels_internal_all = det_img["labels"].to(detector.device)
                _labels_out, valid = detector._map_labels_to_output(labels_internal_all)
                valid &= det_img["scores"].to(detector.device) > 0.0
            proposal_indices_img = det_img["proposal_indices"].to(detector.device)[valid]
            labels_internal_img = labels_internal_all[valid]
        proposal_indices_by_img.append(proposal_indices_img)
        labels_internal_by_img.append(labels_internal_img)
    raw_prediction, raw_logits = detector._detections_to_contract(
        detections,
        detector.device,
        include_class_features=True,
    )
    with torch.no_grad():
        detached_prediction = [p.detach().clone() for p in raw_prediction]
        detached_logits = [l.detach().clone() for l in raw_logits] if raw_logits is not None else None
        max_det = getattr(detector, "max_det", 300)
        selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
            prediction=detached_prediction,
            logits=detached_logits,
            conf_thres=float(getattr(detector, "conf_thresh", getattr(detector, "confidence", 0.25))),
            iou_thres=float(getattr(detector, "iou_thresh", 0.45)),
            classes=getattr(detector, "filter_classes", None),
            agnostic=bool(getattr(detector, "agnostic_nms", getattr(detector, "agnostic", False))),
            max_det=int(max_det) if max_det is not None else None,
            return_indices=True,
        )
    _add_elapsed_timing(timing_accumulator, "detector_inference_sec", t_detector, timing_device)

    rows = []
    iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
    try:
        for image_idx in range(len(image_list)):
            det = (
                selected_preds[image_idx]
                if selected_preds and image_idx < len(selected_preds)
                else torch.zeros((0, 6), device=detector.device)
            )
            raw_keep_indices = (
                selected_indices[image_idx]
                if selected_indices and image_idx < len(selected_indices)
                else torch.zeros((0,), dtype=torch.long, device=detector.device)
            )
            pred_img = raw_prediction[image_idx]
            logit_img = raw_logits[image_idx] if raw_logits is not None else None
            rpn_search_boxes_img = _resize_boxes_xyxy_tensor(
                rpn_decoded_for_roi[image_idx],
                transformed_images.image_sizes[image_idx],
                original_image_sizes[image_idx],
            )
            rpn_objectness_img = rpn_objectness_flat[image_idx]
            rpn_bbox_deltas_img = rpn_bbox_deltas_flat[image_idx]
            rpn_anchors_img = rpn_anchors[image_idx]
            proposals_img = _resize_boxes_xyxy_tensor(
                proposals[image_idx],
                transformed_images.image_sizes[image_idx],
                original_image_sizes[image_idx],
            )
            proposal_to_rpn_raw_idx_img = (
                proposal_to_rpn_raw_indices[image_idx]
                if proposal_to_rpn_raw_indices and image_idx < len(proposal_to_rpn_raw_indices)
                else None
            )
            proposal_offset_img = proposal_offsets[image_idx] if image_idx < len(proposal_offsets) else 0
            proposal_indices_img = proposal_indices_by_img[image_idx] if image_idx < len(proposal_indices_by_img) else None
            labels_internal_img = labels_internal_by_img[image_idx] if image_idx < len(labels_internal_by_img) else None

            for bbox_idx in range(int(det.shape[0])):
                if bbox_idx >= int(raw_keep_indices.shape[0]):
                    raise RuntimeError(
                        "Faster R-CNN layer_grad selected_indices is shorter than selected predictions. "
                        f"image_idx={image_idx}, pred_idx={bbox_idx}, indices={int(raw_keep_indices.shape[0])}"
                    )
                raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
                grad_stats = {}
                roi_target_scalar = None
                rpn_target_scalar = None
                frcnn_null_target_scalar = None
                for target_value in target_values:
                    if target_value not in {"roi_bbox_loss", "roi_cls_loss", "rpn_bbox_loss", "rpn_obj_loss", "bbox_loss", "obj_loss", "cls_loss"}:
                        continue
                    roi_enabled = roi_cand_enabled if target_mode == "cand" else roi_null_enabled
                    rpn_enabled = rpn_cand_enabled if target_mode == "cand" else rpn_null_enabled
                    if target_value.startswith("roi_") and not roi_enabled:
                        continue
                    if target_value.startswith("rpn_") and not rpn_enabled:
                        continue
                    layers_for_target = list(target_layer_map.get(target_value, []))
                    params_for_target = [layer_params_by_name[layer_name] for layer_name in layers_for_target]
                    if target_mode == "frcnn_null":
                        if frcnn_null_target_scalar is None:
                            frcnn_null_target_scalar = build_faster_rcnn_null_losses_by_stage(
                                rpn_box_coder=model.rpn.box_coder,
                                roi_box_coder=roi_heads.box_coder,
                                rpn_bbox_deltas=rpn_bbox_deltas_img,
                                rpn_anchors=rpn_anchors_img,
                                box_regression=box_regression,
                                pred_img=pred_img,
                                logit_img=logit_img,
                                raw_idx=raw_idx,
                                labels_internal_img=labels_internal_img,
                                proposal_indices_img=proposal_indices_img,
                                proposal_offset=proposal_offset_img,
                                proposals_xyxy=proposals_img,
                                proposal_to_rpn_raw_idx=proposal_to_rpn_raw_idx_img,
                                rpn_objectness_logits=rpn_objectness_img,
                                final_box_xyxy=det[bbox_idx, :4],
                                from_size=transformed_images.image_sizes[image_idx],
                                to_size=original_image_sizes[image_idx],
                                rpn_bbox_loss=rpn_null_bbox_loss,
                                rpn_obj_loss=rpn_null_obj_loss,
                                roi_bbox_loss=roi_null_bbox_loss,
                                roi_cls_loss=roi_null_cls_loss,
                                rpn_bbox_direction=rpn_null_bbox_direction,
                                rpn_obj_direction=rpn_null_obj_direction,
                                roi_bbox_direction=roi_null_bbox_direction,
                                roi_cls_direction=roi_null_cls_direction,
                                timing_accumulator=timing_accumulator,
                                timing_device=timing_device,
                            )
                        target_scalar = frcnn_null_target_scalar
                    elif target_value.startswith("roi_"):
                        if roi_target_scalar is None:
                            if target_mode == "cand":
                                roi_target_scalar = build_faster_rcnn_roi_candidate_losses(
                                    roi_box_coder=roi_heads.box_coder,
                                    pred_img=pred_img,
                                    logit_img=logit_img,
                                    raw_idx=raw_idx,
                                    iou_threshold=iou_threshold,
                                    box_regression=box_regression,
                                    proposals_xyxy=proposals_img,
                                    proposal_indices_img=proposal_indices_img,
                                    labels_internal_img=labels_internal_img,
                                    proposal_offset=proposal_offset_img,
                                    score_threshold=roi_cand_score_threshold,
                                    bbox_loss=roi_bbox_loss,
                                    cls_loss=roi_cls_loss,
                                    bbox_direction=roi_bbox_direction,
                                    cls_direction=roi_cls_direction,
                                    timing_accumulator=timing_accumulator,
                                    timing_device=timing_device,
                                )
                        target_scalar = roi_target_scalar
                    else:
                        if rpn_target_scalar is None:
                            if target_mode == "cand":
                                source_proposal_xyxy = None
                                source_obj_logit = None
                                source_bbox_delta = None
                                if proposal_indices_img is not None and raw_idx < proposal_indices_img.shape[0]:
                                    proposal_idx = int(proposal_indices_img[raw_idx].detach().cpu().item())
                                    if 0 <= proposal_idx < proposals_img.shape[0]:
                                        source_proposal_xyxy = proposals_img[proposal_idx]
                                    if (
                                        proposal_to_rpn_raw_idx_img is not None
                                        and 0 <= proposal_idx < proposal_to_rpn_raw_idx_img.shape[0]
                                    ):
                                        source_rpn_raw_idx = int(proposal_to_rpn_raw_idx_img[proposal_idx].detach().cpu().item())
                                        flat_rpn_obj = rpn_objectness_img.reshape(-1)
                                        if 0 <= source_rpn_raw_idx < flat_rpn_obj.shape[0]:
                                            source_obj_logit = flat_rpn_obj[source_rpn_raw_idx]
                                        if 0 <= source_rpn_raw_idx < rpn_bbox_deltas_img.shape[0]:
                                            source_bbox_delta = rpn_bbox_deltas_img[source_rpn_raw_idx]
                                rpn_target_scalar = build_faster_rcnn_rpn_candidate_losses(
                                    rpn_box_coder=model.rpn.box_coder,
                                    rpn_bbox_deltas=rpn_bbox_deltas_img,
                                    rpn_anchors=rpn_anchors_img,
                                    rpn_search_boxes_xyxy=rpn_search_boxes_img,
                                    rpn_objectness_logits=rpn_objectness_img,
                                    source_proposal_xyxy=source_proposal_xyxy,
                                    source_obj_logit=source_obj_logit,
                                    source_bbox_delta=source_bbox_delta,
                                    final_box_xyxy=det[bbox_idx, :4],
                                    from_size=transformed_images.image_sizes[image_idx],
                                    to_size=original_image_sizes[image_idx],
                                    iou_threshold=iou_threshold,
                                    obj_threshold=rpn_cand_obj_threshold,
                                    bbox_loss=rpn_bbox_loss,
                                    obj_loss=rpn_obj_loss,
                                    bbox_direction=rpn_bbox_direction,
                                    obj_direction=rpn_obj_direction,
                                    timing_accumulator=timing_accumulator,
                                    timing_device=timing_device,
                                )
                        target_scalar = rpn_target_scalar
                    if target_scalar is None or target_value not in target_scalar:
                        for layer_name in layers_for_target:
                            grad_stats[f"{target_value}_{layer_name}"] = (
                                {metric: 0.0 for metric in vector_reduction} if vector_reduction else []
                            )
                        continue

                    scalar = target_scalar[target_value]
                    t_backprop = _start_timing(timing_device)
                    grads = torch.autograd.grad(
                        scalar,
                        params_for_target,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    _add_elapsed_timing(timing_accumulator, "backpropagation_sec", t_backprop, timing_device)

                    t_feature = _start_timing(timing_device)
                    for layer_idx, layer_name in enumerate(layers_for_target):
                        key = f"{target_value}_{layer_name}"
                        grad_stats[key] = format_gradient_output(
                            grads[layer_idx],
                            vector_reduction=vector_reduction,
                            map_reduction=map_reduction,
                        )
                    _add_elapsed_timing(timing_accumulator, "feature_compute_sec", t_feature, timing_device)
                    del scalar, grads

                cls_idx = int(det[bbox_idx, 5].detach().cpu().item())
                rows.append(
                    {
                        "sample_idx": image_idx,
                        "pred_idx": bbox_idx,
                        "raw_pred_idx": raw_idx,
                        "xmin": float(det[bbox_idx, 0].detach().cpu().item()),
                        "ymin": float(det[bbox_idx, 1].detach().cpu().item()),
                        "xmax": float(det[bbox_idx, 2].detach().cpu().item()),
                        "ymax": float(det[bbox_idx, 3].detach().cpu().item()),
                        "score": float(det[bbox_idx, 4].detach().cpu().item()),
                        "pred_class": detector.names[cls_idx] if detector.names is not None else cls_idx,
                        "grad_stats": grad_stats,
                    }
                )
    finally:
        for param, req_grad in zip(model_params, original_model_requires_grad):
            param.requires_grad_(req_grad)
        detector.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        del raw_prediction, raw_logits

    return rows


collect_faster_rcnn_roi_layer_grads_per_target = collect_faster_rcnn_candidate_layer_grads_per_target


def collect_bbox_layer_grads_per_target(
    detector,
    input_tensor,
    target_values,
    target_layers,
    map_reduction="none",
    vector_reduction=None,
    pseudo_gt="cand",
    cand_score_threshold=0.01,
    bbox_loss: str = "box_l1",
    cls_loss: str = "bcewithlogits",
    obj_loss: str = "bcewithlogits",
    bbox_direction: str = "pred_to_target",
    cls_direction: str = "pred_to_target",
    obj_direction: str = "pred_to_target",
    timing_accumulator=None,
    timing_device=None,
):
    layer_params = [resolve_layer_parameter(detector.model, layer_name) for layer_name in target_layers]
    original_requires_grad = [bool(p.requires_grad) for p in layer_params]
    for param in layer_params:
        param.requires_grad_(True)

    t_detector = _start_timing(timing_device)
    model_output = detector.model(input_tensor.detach(), augment=False)
    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
    raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
    pred_layers = model_output[2] if isinstance(model_output, (tuple, list)) and len(model_output) > 2 and isinstance(model_output[2], list) else None
    raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
    _add_elapsed_timing(timing_accumulator, "detector_inference_sec", t_detector, timing_device)
    with torch.no_grad():


        max_det = getattr(detector, "max_det", 300)
        t_detector = _start_timing(timing_device)
        selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
            prediction=raw_prediction,
            logits=raw_logits,
            conf_thres=float(getattr(detector, "conf_thresh", getattr(detector, "confidence", 0.25))),
            iou_thres=float(getattr(detector, "iou_thresh", 0.45)),
            classes=getattr(detector, "filter_classes", None),
            agnostic=bool(getattr(detector, "agnostic_nms", getattr(detector, "agnostic", False))),
            max_det=int(max_det) if max_det is not None else None,
            return_indices=True,
        )
        _add_elapsed_timing(timing_accumulator, "detector_inference_sec", t_detector, timing_device)

    t_loss_prep = _start_timing(timing_device)
    raw_flat = _flatten_raw_prediction_layers(pred_layers)
    _add_elapsed_timing(timing_accumulator, "loss_compute_sec", t_loss_prep, timing_device)

    def add_zero_target_stats(grad_stats, target_value, raw_zero_tensor=False):
        for layer_idx, layer_name in enumerate(target_layers):
            key = f"{target_value}_{layer_name}"
            if vector_reduction:
                grad_stats[key] = {metric: 0.0 for metric in vector_reduction}
            elif not raw_zero_tensor:
                grad_stats[key] = []
            else:
                grad_stats[key] = format_gradient_output(
                    torch.zeros_like(layer_params[layer_idx]),
                    vector_reduction=vector_reduction,
                    map_reduction=map_reduction,
                )

    rows = []
    batch_size = int(raw_prediction.shape[0]) if raw_prediction.ndim >= 3 else 1
    iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
    try:
        for sample_idx in range(batch_size):
            det = (
                selected_preds[sample_idx]
                if selected_preds and sample_idx < len(selected_preds)
                else torch.zeros((0, 6), device=input_tensor.device)
            )
            raw_keep_indices = (
                selected_indices[sample_idx]
                if selected_indices and sample_idx < len(selected_indices)
                else torch.zeros((0,), dtype=torch.long, device=input_tensor.device)
            )
            pred_img = raw_prediction[sample_idx]
            logit_img = raw_logits[sample_idx] if raw_logits is not None else pred_img[:, 5:]
            raw_img = raw_flat[sample_idx] if raw_flat is not None and raw_flat.ndim == 3 else None
            anchor_img = (
                raw_anchor_priors[sample_idx]
                if raw_anchor_priors is not None and raw_anchor_priors.ndim >= 3
                else raw_anchor_priors
                if raw_anchor_priors is not None and raw_anchor_priors.ndim == 2 and batch_size == 1
                else None
            )
            candidate_cache = None
            needs_candidate_cache = pseudo_gt != "uniform" and any(
                value in {"bbox_loss", "obj_loss", "cls_loss"} for value in target_values
            )
            if needs_candidate_cache:
                t_candidate = _start_timing(timing_device)
                candidate_cache = build_yolo_candidate_cache(pred_img, cand_score_threshold)
                _add_elapsed_timing(timing_accumulator, "candidate_search_sec", t_candidate, timing_device)

            for bbox_idx in range(int(det.shape[0])):
                if bbox_idx >= int(raw_keep_indices.shape[0]):
                    raise RuntimeError(
                        "YOLO layer_grad selected_indices is shorter than selected predictions. "
                        f"sample_idx={sample_idx}, pred_idx={bbox_idx}, indices={int(raw_keep_indices.shape[0])}"
                    )
                raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
                candidate_mask = None
                if candidate_cache is not None:
                    t_candidate = _start_timing(timing_device)
                    candidate_mask, _ious = yolo_candidate_mask_from_cache(candidate_cache, raw_idx, iou_threshold)
                    _add_elapsed_timing(timing_accumulator, "candidate_search_sec", t_candidate, timing_device)
                grad_stats = {}
                for target_value in target_values:
                    anchor_row = anchor_img[raw_idx] if (anchor_img is not None and raw_idx < anchor_img.shape[0]) else None

                    if pseudo_gt == "delta":
                        candidate_count = (
                            int(candidate_mask.sum().detach().cpu().item())
                            if candidate_mask is not None
                            else 0
                        )
                        if candidate_count <= 0:
                            add_zero_target_stats(grad_stats, target_value, raw_zero_tensor=True)
                            continue
                        cand_scalar = build_layer_target_scalar_bbox(
                            target_value=target_value,
                            pred_img=pred_img,
                            logit_img=logit_img,
                            raw_img=raw_img,
                            raw_idx=raw_idx,
                            iou_threshold=iou_threshold,
                            pseudo_gt="cand",
                            anchor_xywh=anchor_row,
                            cand_score_threshold=cand_score_threshold,
                            bbox_loss=bbox_loss,
                            cls_loss=cls_loss,
                            obj_loss=obj_loss,
                            bbox_direction=bbox_direction,
                            cls_direction=cls_direction,
                            obj_direction=obj_direction,
                            candidate_mask=candidate_mask,
                            timing_accumulator=timing_accumulator,
                            timing_device=timing_device,
                        )
                        ref_scalar = build_layer_target_scalar_bbox(
                            target_value=target_value,
                            pred_img=pred_img,
                            logit_img=logit_img,
                            raw_img=raw_img,
                            raw_idx=raw_idx,
                            iou_threshold=iou_threshold,
                            pseudo_gt="uniform",
                            anchor_xywh=anchor_row,
                            cand_score_threshold=cand_score_threshold,
                            bbox_loss=bbox_loss,
                            cls_loss=cls_loss,
                            obj_loss=obj_loss,
                            bbox_direction=bbox_direction,
                            cls_direction=cls_direction,
                            obj_direction=obj_direction,
                            candidate_mask=None,
                            timing_accumulator=timing_accumulator,
                            timing_device=timing_device,
                        )
                        target_scalar = cand_scalar
                        if ref_scalar is not None:
                            ref_term = float(candidate_count) * ref_scalar
                            target_scalar = -ref_term if target_scalar is None else target_scalar - ref_term
                    else:
                        target_scalar = build_layer_target_scalar_bbox(
                            target_value=target_value,
                            pred_img=pred_img,
                            logit_img=logit_img,
                            raw_img=raw_img,
                            raw_idx=raw_idx,
                            iou_threshold=iou_threshold,
                            pseudo_gt=pseudo_gt,
                            anchor_xywh=anchor_row,
                            cand_score_threshold=cand_score_threshold,
                            bbox_loss=bbox_loss,
                            cls_loss=cls_loss,
                            obj_loss=obj_loss,
                            bbox_direction=bbox_direction,
                            cls_direction=cls_direction,
                            obj_direction=obj_direction,
                            candidate_mask=candidate_mask,
                            timing_accumulator=timing_accumulator,
                            timing_device=timing_device,
                        )

                    if target_scalar is None:
                        add_zero_target_stats(
                            grad_stats,
                            target_value,
                            raw_zero_tensor=(pseudo_gt == "delta"),
                        )
                        continue

                    t_backprop = _start_timing(timing_device)
                    grads = torch.autograd.grad(
                        target_scalar,
                        layer_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    _add_elapsed_timing(timing_accumulator, "backpropagation_sec", t_backprop, timing_device)
                    t_feature = _start_timing(timing_device)
                    for layer_idx, layer_name in enumerate(target_layers):
                        key = f"{target_value}_{layer_name}"
                        grad_tensor = grads[layer_idx]
                        grad_stats[key] = format_gradient_output(
                            grad_tensor,
                            vector_reduction=vector_reduction,
                            map_reduction=map_reduction,
                        )
                    _add_elapsed_timing(timing_accumulator, "feature_compute_sec", t_feature, timing_device)

                    del target_scalar, grads

                cls_idx = int(det[bbox_idx, 5].detach().cpu().item())
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "pred_idx": bbox_idx,
                        "raw_pred_idx": raw_idx,
                        "xmin": float(det[bbox_idx, 0].detach().cpu().item()),
                        "ymin": float(det[bbox_idx, 1].detach().cpu().item()),
                        "xmax": float(det[bbox_idx, 2].detach().cpu().item()),
                        "ymax": float(det[bbox_idx, 3].detach().cpu().item()),
                        "score": float(det[bbox_idx, 4].detach().cpu().item()),
                        "pred_class": detector.names[cls_idx] if detector.names is not None else cls_idx,
                        "grad_stats": grad_stats,
                    }
                )
    finally:
        for param, req_grad in zip(layer_params, original_requires_grad):
            param.requires_grad_(req_grad)
        del model_output, raw_prediction, raw_logits, raw_anchor_priors, raw_flat, selected_preds, selected_indices

    return rows


def preprocess_with_letterbox(detector, image_tensor, device, requires_grad=True, auto=True):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)



    resized, ratio, pad = detector.yolo_resize(image_np, new_shape=detector.img_size, auto=auto)
    resized = resized.transpose((2, 0, 1))
    resized = np.ascontiguousarray(resized)
    input_tensor = torch.from_numpy(resized).float().unsqueeze(0).to(device) / 255.0
    input_tensor.requires_grad_(requires_grad)
    return input_tensor, ratio, pad, resized


def map_boxes_to_letterbox(boxes_tensor, ratio, pad):
    if boxes_tensor.numel() == 0:
        return []
    ratio_w, ratio_h = ratio
    pad_w, pad_h = pad
    boxes = boxes_tensor.clone().float()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_w + pad_w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_h + pad_h
    return boxes.tolist()


def normalize_class_name(name) -> str:
    return str(name).strip().lower()


def classes_match(name_a, name_b) -> bool:
    return normalize_class_name(name_a) == normalize_class_name(name_b)


def draw_predictions(image_chw, boxes, labels, scores):

    image = np.transpose(image_chw, (1, 2, 0)).copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            image,
            f"{label}:{float(score):.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def has_fn_for_image(gt_boxes, gt_class_names, pred_boxes, pred_class_names, iou_match_threshold):
    matched_pred_indices = set()
    for gt_box, gt_name in zip(gt_boxes, gt_class_names):
        found_match = False
        for pred_idx, (pred_box, pred_name) in enumerate(zip(pred_boxes, pred_class_names)):
            if pred_idx in matched_pred_indices:
                continue
            if not classes_match(gt_name, pred_name):
                continue
            if box_iou_xyxy(gt_box, pred_box) >= iou_match_threshold:
                matched_pred_indices.add(pred_idx)
                found_match = True
                break
        if not found_match:
            return 1
    return 0


def get_fn_gt_indices(gt_boxes, gt_class_names, pred_boxes, pred_class_names, iou_match_threshold):
    matched_pred_indices = set()
    fn_gt_indices = []
    for gt_idx, (gt_box, gt_name) in enumerate(zip(gt_boxes, gt_class_names)):
        found_match = False
        for pred_idx, (pred_box, pred_name) in enumerate(zip(pred_boxes, pred_class_names)):
            if pred_idx in matched_pred_indices:
                continue
            if not classes_match(gt_name, pred_name):
                continue
            if box_iou_xyxy(gt_box, pred_box) >= iou_match_threshold:
                matched_pred_indices.add(pred_idx)
                found_match = True
                break
        if not found_match:
            fn_gt_indices.append(gt_idx)
    return fn_gt_indices


def assign_tp_to_predictions(
    gt_boxes,
    gt_class_names,
    pred_boxes,
    pred_class_names,
    pred_scores,
    iou_match_threshold,
):
    n_pred = len(pred_boxes)
    matched_gt_indices = set()
    tp_flags = [0] * n_pred
    best_ious = [0.0] * n_pred

    sorted_indices = list(range(n_pred))
    if pred_scores is not None:
        sorted_indices.sort(key=lambda i: float(pred_scores[i]), reverse=True)

    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_name = pred_class_names[pred_idx]

        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx, (gt_box, gt_name) in enumerate(zip(gt_boxes, gt_class_names)):
            if not classes_match(gt_name, pred_name):
                continue
            iou = box_iou_xyxy(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        best_ious[pred_idx] = float(best_iou)
        if best_gt_idx >= 0 and best_gt_idx not in matched_gt_indices and best_iou >= iou_match_threshold:
            tp_flags[pred_idx] = 1
            matched_gt_indices.add(best_gt_idx)

    return tp_flags, best_ious


def analyze_prediction_error_types(
    gt_boxes,
    gt_class_names,
    pred_boxes,
    pred_class_names,
    pred_scores,
    iou_match_threshold,
    background_iou_threshold=0.1,
):
    n_pred = len(pred_boxes)
    tp_flags = [0] * n_pred
    best_same_class_ious = [0.0] * n_pred
    best_any_class_ious = [0.0] * n_pred
    best_same_class_gt_indices = [-1] * n_pred
    best_any_class_gt_indices = [-1] * n_pred
    best_same_class_gt_classes = [""] * n_pred
    best_any_class_gt_classes = [""] * n_pred
    matched_gt_indices = [-1] * n_pred
    duplicate_flags = [0] * n_pred
    error_types = ["localization_error"] * n_pred

    sorted_indices = list(range(n_pred))
    if pred_scores is not None:
        sorted_indices.sort(key=lambda i: float(pred_scores[i]), reverse=True)

    claimed_gt_indices = set()
    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_name = pred_class_names[pred_idx]

        best_same_iou = 0.0
        best_same_idx = -1
        best_any_iou = 0.0
        best_any_idx = -1
        for gt_idx, (gt_box, gt_name) in enumerate(zip(gt_boxes, gt_class_names)):
            iou = box_iou_xyxy(gt_box, pred_box)
            if iou > best_any_iou:
                best_any_iou = iou
                best_any_idx = gt_idx
            if classes_match(gt_name, pred_name) and iou > best_same_iou:
                best_same_iou = iou
                best_same_idx = gt_idx

        best_same_class_ious[pred_idx] = float(best_same_iou)
        best_any_class_ious[pred_idx] = float(best_any_iou)
        best_same_class_gt_indices[pred_idx] = int(best_same_idx)
        best_any_class_gt_indices[pred_idx] = int(best_any_idx)
        if best_same_idx >= 0:
            best_same_class_gt_classes[pred_idx] = str(gt_class_names[best_same_idx])
        if best_any_idx >= 0:
            best_any_class_gt_classes[pred_idx] = str(gt_class_names[best_any_idx])

        if best_same_idx >= 0 and best_same_iou >= iou_match_threshold:
            matched_gt_indices[pred_idx] = int(best_same_idx)
            if best_same_idx not in claimed_gt_indices:
                tp_flags[pred_idx] = 1
                error_types[pred_idx] = "tp"
                claimed_gt_indices.add(best_same_idx)
            else:
                duplicate_flags[pred_idx] = 1
                error_types[pred_idx] = "localization_error"
        elif best_any_idx >= 0 and best_any_iou >= iou_match_threshold:
            error_types[pred_idx] = "classification_error"
        elif best_same_idx >= 0 and best_same_iou >= background_iou_threshold:
            error_types[pred_idx] = "localization_error"
        else:
            error_types[pred_idx] = "localization_error"

    rows = []
    for pred_idx in range(n_pred):
        error_type = error_types[pred_idx]
        rows.append(
            {
                "tp": int(tp_flags[pred_idx]),
                "max_iou": float(best_same_class_ious[pred_idx]),
                "gt_iou": float(best_same_class_ious[pred_idx]),
                "error_type": error_type,
                "best_same_class_iou": float(best_same_class_ious[pred_idx]),
                "best_any_class_iou": float(best_any_class_ious[pred_idx]),
                "best_same_class_gt_idx": int(best_same_class_gt_indices[pred_idx]),
                "best_any_class_gt_idx": int(best_any_class_gt_indices[pred_idx]),
                "best_same_class_gt_class": best_same_class_gt_classes[pred_idx],
                "best_any_class_gt_class": best_any_class_gt_classes[pred_idx],
                "matched_gt_idx": int(matched_gt_indices[pred_idx]),
                "is_duplicate": int(duplicate_flags[pred_idx]),
                "is_localization_error": int(error_type == "localization_error"),
                "is_classification_error": int(error_type == "classification_error"),
            }
        )
    return rows


