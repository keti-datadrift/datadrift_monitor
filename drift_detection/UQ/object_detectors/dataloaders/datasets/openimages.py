from pathlib import Path

import torch
from torch.utils.data import Dataset

from dataloaders.core.io import read_image_as_rgb


class OpenImagesDataset(Dataset):
    """
    Expected layout (from root):
    - <root>/<split>/<class_name>/*.jpg
    - <root>/<split>/<class_name>/Label/*.txt

    Label txt line format:
    - class_name x1 y1 x2 y2
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root, split="validation", img_size=640, min_gt_boxes=0):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.min_gt_boxes = max(0, int(min_gt_boxes))

        split_dir = Path(root) / split
        if not split_dir.is_dir():
            split_dir = Path(root)
        self.split_dir = split_dir

        self.class_to_idx = {}
        self.idx_to_class = []

        self.images = self._resolve_images()
        self.samples = self._build_samples()

    def _ensure_class_index(self, class_name):
        if class_name not in self.class_to_idx:
            self.class_to_idx[class_name] = len(self.class_to_idx)
            self.idx_to_class.append(class_name)
        return self.class_to_idx[class_name]

    def _resolve_images(self):
        image_paths = []
        if not self.split_dir.is_dir():
            return image_paths

        for class_dir in sorted(self.split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if class_dir.name.lower() == "label":
                continue
            for item in sorted(class_dir.iterdir()):
                if not item.is_file():
                    continue
                if item.suffix.lower() in self.IMAGE_EXTENSIONS:
                    image_paths.append(str(item))
        return image_paths

    def _read_label_file(self, label_path):
        boxes = []
        labels = []
        class_names = []
        if not label_path.is_file():
            return boxes, labels, class_names

        with open(label_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_name = parts[0]
                try:
                    x1, y1, x2, y2 = map(float, parts[1:5])
                except ValueError:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                labels.append(self._ensure_class_index(class_name))
                boxes.append([x1, y1, x2, y2])
                class_names.append(class_name)
        return boxes, labels, class_names

    def _build_samples(self):
        samples = []
        for image_path in self.images:
            img_path = Path(image_path)
            label_path = img_path.parent / "Label" / f"{img_path.stem}.txt"
            boxes, labels, class_names = self._read_label_file(label_path)
            if len(boxes) < self.min_gt_boxes:
                continue
            samples.append(
                {
                    "image_path": image_path,
                    "boxes": boxes,
                    "labels": labels,
                    "gt_class_names": class_names,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample["image_path"]
        image = read_image_as_rgb(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes = sample["boxes"]
        labels = sample["labels"]
        gt_class_names = sample["gt_class_names"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "path": image_path,
            "dataset_name": "openimages",
            "gt_class_names": gt_class_names,
        }
        return image, target
