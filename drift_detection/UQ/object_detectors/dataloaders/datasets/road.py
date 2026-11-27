import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from dataloaders.core.class_names import (
    bdd100k_names,
    cityscapes_names,
    foggy_cityscapes_names,
    kitti_names,
)
from dataloaders.core.io import read_image_as_rgb


def _image_tensor(path):
    image = read_image_as_rgb(str(path))
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def _empty_target(image_id, image_path, dataset_name):
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
        "path": str(image_path),
        "dataset_name": dataset_name,
        "gt_class_names": [],
    }


def _target(image_id, image_path, dataset_name, boxes, labels, gt_class_names):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
        "path": str(image_path),
        "dataset_name": dataset_name,
        "gt_class_names": gt_class_names,
    }


def _read_split_ids(split_file):
    if not split_file or not Path(split_file).is_file():
        return None
    ids = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                ids.append(Path(token).stem)
    return ids


def _list_images_recursive(root):
    root = Path(root)
    if not root.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


class KITTIDataset(Dataset):
    """
    Supports the common KITTI object layout:
    - <root>/training/image_2/*.png
    - <root>/training/label_2/*.txt
    - optional <root>/ImageSets/<split>.txt
    """

    def __init__(
        self,
        root,
        split="train",
        img_size=640,
        image_dir=None,
        label_dir=None,
        split_file=None,
        trainval_split_ratio=0.8,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        split_name = str(split).strip().lower()
        if image_dir:
            self.image_dir = Path(image_dir)
        elif split_name in {"test", "testing"}:
            self.image_dir = self.root / "testing" / "image_2"
        else:
            self.image_dir = self.root / "training" / "image_2"
        if label_dir:
            self.label_dir = Path(label_dir)
        elif split_name in {"test", "testing"}:
            self.label_dir = self.root / "testing" / "label_2"
        else:
            self.label_dir = self.root / "training" / "label_2"
        self.split_file = Path(split_file) if split_file else self.root / "ImageSets" / f"{split}.txt"
        self.trainval_split_ratio = float(trainval_split_ratio)
        self.class_names = list(kitti_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.aliases = {
            "car": "car",
            "van": "van",
            "truck": "truck",
            "pedestrian": "pedestrian",
            "person_sitting": "person_sitting",
            "cyclist": "cyclist",
            "tram": "tram",
            "misc": "misc",
        }
        self.images = self._resolve_images()

    def _resolve_images(self):
        split_ids = _read_split_ids(self.split_file)
        if split_ids is not None:
            paths = []
            for stem in split_ids:
                for ext in (".png", ".jpg", ".jpeg"):
                    candidate = self.image_dir / f"{stem}{ext}"
                    if candidate.is_file():
                        paths.append(str(candidate))
                        break
            return paths
        images = _list_images_recursive(self.image_dir)
        split_name = str(self.split).strip().lower()
        if split_name not in {"train", "val"} or not images:
            return images
        if not (0.0 < self.trainval_split_ratio < 1.0):
            return images
        cut = max(1, min(len(images) - 1, int(round(len(images) * self.trainval_split_ratio))))
        if split_name == "train":
            return images[:cut]
        return images[cut:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = Path(self.images[index])
        image = _image_tensor(image_path)
        ann_path = self.label_dir / f"{image_path.stem}.txt"
        boxes, labels, gt_class_names = [], [], []
        if ann_path.is_file():
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    raw_name = parts[0].strip().lower()
                    if raw_name == "dontcare":
                        continue
                    label_name = self.aliases.get(raw_name)
                    if label_name is None:
                        continue
                    xmin, ymin, xmax, ymax = [float(v) for v in parts[4:8]]
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.class_to_idx[label_name])
                    gt_class_names.append(label_name)
        return image, _target(index, image_path, "kitti", boxes, labels, gt_class_names)


class BDD100KDataset(Dataset):
    """
    Supports BDD100K detection JSON layouts such as:
    - <root>/images/100k/<split>/*.jpg
    - <root>/labels/det_20/det_<split>.json
    """

    def __init__(self, root, split="train", img_size=640, image_dir=None, annotation_file=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.image_dir = Path(image_dir) if image_dir else self._find_image_dir(split)
        self.annotation_file = Path(annotation_file) if annotation_file else self._find_annotation_file(split)
        self.class_names = list(bdd100k_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.images = _list_images_recursive(self.image_dir)
        self.annotations_by_name = {}
        if self.annotation_file and self.annotation_file.is_file():
            self._load_annotations()

    def _find_annotation_file(self, split):
        candidates = [
            self.root / "bdd100k_labels_release" / "bdd100k" / "labels" / f"bdd100k_labels_images_{split}.json",
            self.root / "labels" / "det_20" / f"det_{split}.json",
            self.root / "labels" / f"det_{split}.json",
            self.root / "labels" / f"bdd100k_labels_images_{split}.json",
            self.root / "labels" / "det_20" / f"bdd100k_labels_images_{split}.json",
            self.root / "labels" / "det_20" / f"{split}.json",
            self.root / "labels" / f"{split}.json",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return candidates[0]

    def _find_image_dir(self, split):
        candidates = [
            self.root / "bdd100k" / "bdd100k" / "images" / "100k" / split,
            self.root / "images" / "100k" / split,
            self.root / "bdd100k" / "bdd100k" / "images" / "10k" / split,
            self.root / "images" / "10k" / split,
        ]
        for path in candidates:
            if path.is_dir():
                return path
        return candidates[0]

    def _load_annotations(self):
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("frames") or payload.get("images") or payload.get("annotations") or []
        for frame in payload:
            name = frame.get("name") or frame.get("file_name") or frame.get("image")
            if not name:
                continue
            self.annotations_by_name[os.path.basename(str(name))] = frame.get("labels", [])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = Path(self.images[index])
        image = _image_tensor(image_path)
        anns = self.annotations_by_name.get(image_path.name, [])
        boxes, labels, gt_class_names = [], [], []
        for ann in anns:
            category = str(ann.get("category", "")).strip().lower()
            if category not in self.class_to_idx:
                continue
            box = ann.get("box2d") or ann.get("bbox")
            if not box:
                continue
            if isinstance(box, dict):
                xmin = float(box.get("x1", box.get("xmin", 0.0)))
                ymin = float(box.get("y1", box.get("ymin", 0.0)))
                xmax = float(box.get("x2", box.get("xmax", 0.0)))
                ymax = float(box.get("y2", box.get("ymax", 0.0)))
            else:
                xmin, ymin, xmax, ymax = [float(v) for v in box[:4]]
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[category])
            gt_class_names.append(category)
        return image, _target(index, image_path, "bdd100k", boxes, labels, gt_class_names)


class CityscapesDetectionDataset(Dataset):
    """
    Converts Cityscapes polygon instance annotations to detection boxes.
    For Foggy Cityscapes, pass the foggy image root and the original Cityscapes gtFine root.
    """

    def __init__(
        self,
        root,
        split="train",
        img_size=640,
        image_dir=None,
        annotation_dir=None,
        dataset_name="cityscapes",
        class_names=None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.image_dir = Path(image_dir) if image_dir else self.root / "leftImg8bit" / split
        self.annotation_dir = Path(annotation_dir) if annotation_dir else self.root / "gtFine" / split
        self.class_names = list(class_names or cityscapes_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.images = _list_images_recursive(self.image_dir)

    def __len__(self):
        return len(self.images)

    def _annotation_path(self, image_path):
        stem = Path(image_path).stem
        if "_leftImg8bit_foggy" in stem:
            stem = stem.split("_leftImg8bit_foggy", 1)[0] + "_gtFine_polygons"
        elif stem.endswith("_leftImg8bit"):
            stem = stem[:-len("_leftImg8bit")] + "_gtFine_polygons"
        else:
            stem = stem + "_gtFine_polygons"
        city = Path(image_path).parent.name
        return self.annotation_dir / city / f"{stem}.json"

    @staticmethod
    def _normalize_label(label):
        label = str(label).strip().lower()
        if label.endswith(" group"):
            label = label[:-len(" group")]
        return label

    def __getitem__(self, index):
        image_path = Path(self.images[index])
        image = _image_tensor(image_path)
        ann_path = self._annotation_path(image_path)
        boxes, labels, gt_class_names = [], [], []
        if ann_path.is_file():
            with open(ann_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for obj in payload.get("objects", []):
                label_name = self._normalize_label(obj.get("label", ""))
                if label_name not in self.class_to_idx:
                    continue
                polygon = obj.get("polygon") or []
                if not polygon:
                    continue
                xs = [float(p[0]) for p in polygon]
                ys = [float(p[1]) for p in polygon]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[label_name])
                gt_class_names.append(label_name)
        return image, _target(index, image_path, self.dataset_name, boxes, labels, gt_class_names)


class FoggyCityscapesDetectionDataset(CityscapesDetectionDataset):
    def __init__(self, root, split="train", img_size=640, image_dir=None, annotation_dir=None):
        root_path = Path(root)
        if image_dir is None:
            for candidate in (
                root_path / "leftImg8bit_foggy" / split,
                root_path / "leftImg8bit" / split,
            ):
                if candidate.is_dir():
                    image_dir = candidate
                    break
            if image_dir is None:
                image_dir = root_path / "leftImg8bit_foggy" / split
        super().__init__(
            root=root,
            split=split,
            img_size=img_size,
            image_dir=image_dir,
            annotation_dir=annotation_dir or root_path / "gtFine" / split,
            dataset_name="foggy_cityscapes",
            class_names=foggy_cityscapes_names,
        )
