import os
import random
from collections import OrderedDict
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.core.class_names import DATASET_CLASS_NAMES


def coco80_to_coco91_class():
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    ]


@lru_cache(maxsize=1)
def coco91_to_80_pairs():
    return tuple((int(cat_id), int(i)) for i, cat_id in enumerate(coco80_to_coco91_class()))


_COCO91_TO_80_TENSOR_CACHE = {}


def coco91_to_80_lookup(device):
    device = torch.device(device)
    key = str(device)
    if key not in _COCO91_TO_80_TENSOR_CACHE:
        max_cat_id = max(cat_id for cat_id, _idx in coco91_to_80_pairs())
        lookup = torch.full((max_cat_id + 1,), -1, dtype=torch.long, device=device)
        for cat_id, idx in coco91_to_80_pairs():
            lookup[cat_id] = idx
        _COCO91_TO_80_TENSOR_CACHE[key] = lookup
    return _COCO91_TO_80_TENSOR_CACHE[key]


def map_coco91_to_80(labels, offset=0):
    lookup = coco91_to_80_lookup(labels.device)
    valid_id = (labels >= 0) & (labels < int(lookup.numel()))
    mapped = torch.full_like(labels, -1)
    if valid_id.any():
        mapped[valid_id] = lookup[labels[valid_id]]
    keep = mapped >= 0
    return mapped[keep] + int(offset), keep


def _preprocess_cache_key(target, out_h, out_w):
    path = target.get("path") if isinstance(target, dict) else None
    if not path:
        return None
    return (str(path), int(out_h), int(out_w))


def _get_preprocess_cache(preprocess_cache, key):
    if preprocess_cache is None or key is None:
        return None
    cached = preprocess_cache.get(key)
    if cached is not None:
        preprocess_cache.move_to_end(key)
    return cached


def _put_preprocess_cache(preprocess_cache, preprocess_cache_size, key, tensor, ratio, pad):
    if preprocess_cache is None or key is None or preprocess_cache_size <= 0:
        return
    preprocess_cache[key] = (tensor.detach().cpu(), ratio, pad)
    preprocess_cache.move_to_end(key)
    while len(preprocess_cache) > preprocess_cache_size:
        preprocess_cache.popitem(last=False)


def make_preprocess_cache(preprocess_cache_size):
    return OrderedDict() if int(preprocess_cache_size) > 0 else None


def prepare_yolo_batch(images, targets, img_size, device, preprocess_cache=None, preprocess_cache_size=0):
    if isinstance(img_size, int):
        out_h = out_w = int(img_size)
    else:
        out_h, out_w = int(img_size[0]), int(img_size[1])
    if images and all(int(img.shape[0]) == 3 and int(img.shape[1]) == out_h and int(img.shape[2]) == out_w for img in images):
        infer_batch = torch.stack(images, dim=0).to(device=device, non_blocking=True)
        return infer_batch, [(1.0, 1.0)] * len(images), [(0.0, 0.0)] * len(images)
    infer_tensors = []
    ratios = []
    pads = []
    pad_value = float(114.0 / 255.0)
    for img, target in zip(images, targets):
        key = _preprocess_cache_key(target, out_h, out_w)
        cached = _get_preprocess_cache(preprocess_cache, key)
        if cached is not None:
            cached_tensor, cached_ratio, cached_pad = cached
            infer_tensors.append(cached_tensor)
            ratios.append(cached_ratio)
            pads.append(cached_pad)
            continue
        c, h, w = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
        if c != 3:
            raise ValueError(f"Expected 3-channel image tensor, got shape={tuple(img.shape)}")
        scale = min(float(out_h) / float(h), float(out_w) / float(w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        pad_h = out_h - new_h
        pad_w = out_w - new_w
        top = int(pad_h // 2)
        bottom = int(pad_h - top)
        left = int(pad_w // 2)
        right = int(pad_w - left)
        resized = F.pad(resized, (left, right, top, bottom), value=pad_value)
        infer_tensors.append(resized)
        ratio = (scale, scale)
        pad = (float(left), float(top))
        ratios.append(ratio)
        pads.append(pad)
        _put_preprocess_cache(preprocess_cache, preprocess_cache_size, key, resized, ratio, pad)
    return torch.stack(infer_tensors, dim=0).to(device=device, non_blocking=True), ratios, pads


def to_yolo_targets(targets, ratios, pads, img_h, img_w, device):
    rows = []
    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None or boxes.numel() == 0:
            continue
        ratio_w, ratio_h = ratios[batch_idx]
        pad_w, pad_h = pads[batch_idx]
        b = boxes.to(device=device, dtype=torch.float32, non_blocking=True).clone()
        b[:, [0, 2]] = b[:, [0, 2]] * ratio_w + pad_w
        b[:, [1, 3]] = b[:, [1, 3]] * ratio_h + pad_h
        cls = labels.to(device=device, dtype=torch.long, non_blocking=True).clone()
        if str(target.get("dataset_name", "")).lower() == "coco":
            mapped, keep = map_coco91_to_80(cls, offset=0)
            if not keep.any():
                continue
            b = b[keep]
            cls = mapped.to(dtype=torch.long)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        valid = (w > 0.0) & (h > 0.0)
        if not valid.any():
            continue
        x1, y1, w, h, cls = x1[valid], y1[valid], w[valid], h[valid], cls[valid]
        xc = (x1 + w * 0.5) / float(img_w)
        yc = (y1 + h * 0.5) / float(img_h)
        wn = w / float(img_w)
        hn = h / float(img_h)
        batch_col = torch.full((xc.shape[0],), int(batch_idx), dtype=torch.float32, device=device)
        rows.append(torch.stack([batch_col, cls.float(), xc.float(), yc.float(), wn.float(), hn.float()], dim=1))
    if not rows:
        return torch.zeros((0, 6), dtype=torch.float32, device=device)
    return torch.cat(rows, dim=0).to(dtype=torch.float32)


def amp_enabled(config, device):
    return bool(config.get("training", {}).get("amp", True)) and torch.device(device).type == "cuda"


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    device_type = torch.device(device).type
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(device, enabled):
    enabled = bool(enabled) and torch.device(device).type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(torch.device(device).type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_matching_state_dict(model, state_dict):
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(filtered, strict=False)


def set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def resolve_device(device_str):
    device_str = str(device_str or "cuda").strip().lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def active_dataset_names(config):
    raw = config.get("dataset", {}).get("used_dataset", "")
    if isinstance(raw, str):
        return [raw.strip().lower()]
    if isinstance(raw, (list, tuple)):
        return [str(v).strip().lower() for v in raw if str(v).strip()]
    return []


def resolve_train_class_names(config):
    model_cfg = config.get("model", {})
    configured_names = model_cfg.get("class_names")
    if configured_names:
        if isinstance(configured_names, str):
            return [name.strip() for name in configured_names.split(",") if name.strip()]
        return [str(v) for v in configured_names]

    names = active_dataset_names(config)
    if names and all(name in DATASET_CLASS_NAMES for name in names):
        first = list(DATASET_CLASS_NAMES[names[0]])
        if all(list(DATASET_CLASS_NAMES[name]) == first for name in names):
            return first
        raise ValueError("Mixed datasets with different class spaces are not supported.")
    if any(name in DATASET_CLASS_NAMES for name in names):
        raise ValueError("Mixed custom datasets with non-matching class spaces are not supported.")
    return None


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_total_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def training_options(config, device):
    training_cfg = config.get("training", {})
    return {
        "amp": bool(amp_enabled(config, device)),
        "freeze_feature_extractor": bool(training_cfg.get("freeze_feature_extractor", False)),
        "val_interval": max(1, int(training_cfg.get("val_interval", 1))),
        "save_last_interval": max(1, int(training_cfg.get("save_last_interval", 1))),
        "save_best": bool(training_cfg.get("save_best", True)),
        "save_optimizer": bool(training_cfg.get("save_optimizer", True)),
        "log_timing": bool(training_cfg.get("log_timing", False)),
        "grad_clip_norm": max(0.0, float(training_cfg.get("grad_clip_norm", 0.0))),
        "preprocess_cache_size": max(0, int(training_cfg.get("preprocess_cache_size", 0))),
    }


def validate_training_config(config):
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 10))
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    if epochs <= 0:
        raise ValueError("training.epochs must be >= 1.")
    if lr <= 0:
        raise ValueError("training.lr must be > 0.")
    if weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0.")
    return epochs, lr, weight_decay


def merge_epoch_timing(history_row, timing):
    if not timing:
        return history_row
    history_row.update({key: float(value) for key, value in timing.items()})
    return history_row
