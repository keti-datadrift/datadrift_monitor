import hashlib
import math
from bisect import bisect_right
from pathlib import Path

from PIL import Image
from torch.utils.data import ConcatDataset, Subset
import yaml

from dataloaders.datasets.coco import COCODataset
from dataloaders.datasets.openimages import OpenImagesDataset
from dataloaders.datasets.road import (
    BDD100KDataset,
    CityscapesDetectionDataset,
    FoggyCityscapesDetectionDataset,
    KITTIDataset,
)
from dataloaders.datasets.voc import VOCDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mode(config):
    mode = str(config.get("mode", "train")).lower()
    valid_modes = {"train", "test", "predict"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of {sorted(valid_modes)}")
    return mode


def resolve_dataset_path(root, value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)


def normalize_dataset_names(dataset_cfg):
    raw = dataset_cfg["used_dataset"]
    if isinstance(raw, str):
        names = [raw.strip().lower()]
    elif isinstance(raw, (list, tuple)):
        names = [str(v).strip().lower() for v in raw if str(v).strip()]
    else:
        raise ValueError("dataset.used_dataset must be a string or list of strings.")
    if not names:
        raise ValueError("dataset.used_dataset is empty.")
    return names


def build_single_dataset(name, dataset_cfg, root, split_key, img_size):
    if name == "coco":
        coco_split = dataset_cfg.get(f"{split_key}_split", split_key)
        ann_file_key = f"{split_key}_annotation_file"
        ann_file = dataset_cfg.get(ann_file_key)
        ann_path = None
        if ann_file:
            ann_path = str(Path(root) / dataset_cfg["annotation_dir"] / ann_file)
        image_dir = str(Path(root) / dataset_cfg["image_dir"] / coco_split)
        if not Path(image_dir).is_dir():
            fallback_dir = Path(root) / coco_split
            if fallback_dir.is_dir():
                image_dir = str(fallback_dir)
        return COCODataset(root=root, split=coco_split, image_dir=image_dir, annotation_file=ann_path, img_size=img_size)

    if name in {"voc", "pascal_voc"}:
        voc_split = dataset_cfg.get(f"{split_key}_split", split_key)
        return VOCDataset(root=root, split=voc_split, img_size=img_size)

    if name in {"openimages", "open_images", "oid"}:
        oi_split = dataset_cfg.get(f"{split_key}_split", split_key)
        return OpenImagesDataset(
            root=root,
            split=oi_split,
            img_size=img_size,
            min_gt_boxes=int(dataset_cfg.get("min_gt_boxes", 0)),
        )

    if name == "kitti":
        kitti_split = dataset_cfg.get(f"{split_key}_split", split_key)
        return KITTIDataset(
            root=root,
            split=kitti_split,
            img_size=img_size,
            image_dir=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_image_dir") or dataset_cfg.get("image_dir")),
            label_dir=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_label_dir") or dataset_cfg.get("label_dir")),
            split_file=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_split_file") or dataset_cfg.get("split_file")),
            trainval_split_ratio=float(dataset_cfg.get("trainval_split_ratio", 0.8)),
        )

    if name in {"bdd100k", "bdd"}:
        bdd_split = dataset_cfg.get(f"{split_key}_split", split_key)
        return BDD100KDataset(
            root=root,
            split=bdd_split,
            img_size=img_size,
            image_dir=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_image_dir") or dataset_cfg.get("image_dir")),
            annotation_file=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_annotation_file") or dataset_cfg.get("annotation_file")),
        )

    if name == "cityscapes":
        city_split = dataset_cfg.get(f"{split_key}_split", split_key)
        return CityscapesDetectionDataset(
            root=root,
            split=city_split,
            img_size=img_size,
            image_dir=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_image_dir") or dataset_cfg.get("image_dir")),
            annotation_dir=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_annotation_dir") or dataset_cfg.get("annotation_dir")),
        )

    if name in {"foggy_cityscapes", "foggy_city"}:
        foggy_split = dataset_cfg.get(f"{split_key}_split", split_key)
        return FoggyCityscapesDetectionDataset(
            root=root,
            split=foggy_split,
            img_size=img_size,
            image_dir=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_image_dir") or dataset_cfg.get("image_dir")),
            annotation_dir=resolve_dataset_path(root, dataset_cfg.get(f"{split_key}_annotation_dir") or dataset_cfg.get("annotation_dir")),
        )

    raise ValueError(f"Unsupported dataset name: {name}")


def sample_key(dataset, index, dataset_name, split_key):
    if hasattr(dataset, "images") and index < len(getattr(dataset, "images")):
        sample_id = str(dataset.images[index])
    elif hasattr(dataset, "samples") and index < len(getattr(dataset, "samples")):
        sample = dataset.samples[index]
        sample_id = str(sample.get("image_path", index)) if isinstance(sample, dict) else str(index)
    else:
        sample_id = str(index)
    return f"{dataset_name}|{split_key}|{sample_id}"


def apply_used_ratio(dataset, dataset_cfg, dataset_name, split_key):
    ratio = float(dataset_cfg.get("used_ratio", 1.0))
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"dataset.used_ratio for '{dataset_name}' must be in (0, 1].")
    n = len(dataset)
    if ratio >= 1.0 or n == 0:
        return dataset

    keep = max(1, int(math.ceil(n * ratio)))
    ranked = []
    for idx in range(n):
        key = sample_key(dataset, idx, dataset_name, split_key)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        ranked.append((digest, idx))
    selected = sorted(idx for _digest, idx in sorted(ranked)[:keep])
    return Subset(dataset, selected)


def build_dataset(config, split="train"):
    root_dataset_cfg = config["dataset"]
    names = normalize_dataset_names(root_dataset_cfg)
    used_ratios = root_dataset_cfg.get("used_ratio", 1.0)

    if isinstance(split, (list, tuple)):
        split_keys = [str(v).strip() for v in split if str(v).strip()]
    else:
        split_keys = [str(split).strip()] * len(names)

    if len(split_keys) != len(names):
        raise ValueError(f"Length mismatch: used_dataset has {len(names)} entries but split has {len(split_keys)} entries.")
    if isinstance(used_ratios, (list, tuple)):
        ratio_values = [float(v) for v in used_ratios]
    else:
        ratio_values = [float(used_ratios)] * len(names)
    if len(ratio_values) != len(names):
        raise ValueError(
            f"Length mismatch: used_dataset has {len(names)} entries but dataset.used_ratio has {len(ratio_values)} entries."
        )

    datasets = []
    img_size = config.get("model", {}).get("img_size", 640)
    for name, split_key, used_ratio in zip(names, split_keys, ratio_values):
        if name not in root_dataset_cfg:
            raise ValueError(f"dataset.used_dataset includes '{name}' but dataset.{name} is not defined.")
        dataset_cfg = dict(root_dataset_cfg[name])
        dataset_cfg["used_ratio"] = used_ratio

        root_path = Path(dataset_cfg["root"])
        if not root_path.is_absolute():
            root_path = (PROJECT_ROOT / root_path).resolve()
        dataset = build_single_dataset(name, dataset_cfg, str(root_path), split_key, img_size)
        datasets.append(apply_used_ratio(dataset, dataset_cfg, name, split_key))

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def dataset_image_path(dataset, index):
    if isinstance(dataset, Subset):
        return dataset_image_path(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        sample_idx = index if dataset_idx == 0 else index - dataset.cumulative_sizes[dataset_idx - 1]
        return dataset_image_path(dataset.datasets[dataset_idx], sample_idx)
    if hasattr(dataset, "images") and index < len(getattr(dataset, "images")):
        return str(dataset.images[index])
    if hasattr(dataset, "samples") and index < len(getattr(dataset, "samples")):
        sample = dataset.samples[index]
        if isinstance(sample, dict):
            return str(sample.get("image_path", ""))
    return ""


def image_aspect_ratio(path):
    try:
        with Image.open(path) as image:
            width, height = image.size
        if height <= 0:
            return 1.0
        return float(width) / float(height)
    except Exception:
        return 1.0


def dataset_aspect_ratio(dataset, index):
    if isinstance(dataset, Subset):
        return dataset_aspect_ratio(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        sample_idx = index if dataset_idx == 0 else index - dataset.cumulative_sizes[dataset_idx - 1]
        return dataset_aspect_ratio(dataset.datasets[dataset_idx], sample_idx)
    getter = getattr(dataset, "get_aspect_ratio", None)
    if callable(getter):
        ratio = getter(index)
        if ratio is not None:
            try:
                ratio = float(ratio)
            except Exception:
                ratio = None
            if ratio is not None and ratio > 0:
                return ratio
    return image_aspect_ratio(dataset_image_path(dataset, index))


def sort_dataset_by_aspect_ratio(dataset):
    ranked = []
    for idx in range(len(dataset)):
        ranked.append((dataset_aspect_ratio(dataset, idx), idx))
    return Subset(dataset, [idx for _ratio, idx in sorted(ranked)])
