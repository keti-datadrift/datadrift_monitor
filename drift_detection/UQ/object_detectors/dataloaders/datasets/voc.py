import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

from dataloaders.core.class_names import pascal_voc_names
from dataloaders.core.io import list_image_files, read_image_as_rgb


class VOCDataset(Dataset):
    """
    Expected VOC layout:
    - <root>/JPEGImages/*.jpg
    - <root>/Annotations/*.xml
    - <root>/ImageSets/Main/<split>.txt
    """

    def __init__(self, root, split="train", img_size=640):
        self.root = root
        self.split = split
        self.img_size = img_size

        self.image_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")
        self.split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        self.class_names = pascal_voc_names[1:]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.images = self._resolve_images()
        self.annotations = self._load_annotations()

    def _resolve_images(self):
        if os.path.isfile(self.split_file):
            image_paths = []
            with open(self.split_file, "r", encoding="utf-8") as f:
                for line in f:
                    stem = line.strip().split()[0]
                    image_paths.append(os.path.join(self.image_dir, f"{stem}.jpg"))
            return [p for p in image_paths if os.path.isfile(p)]
        return list_image_files(self.image_dir)

    def _annotation_path(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.annotation_dir, f"{stem}.xml")

    def _empty_annotation(self):
        return {"boxes": [], "labels": [], "gt_class_names": []}

    def _parse_annotation(self, ann_path):
        if not os.path.isfile(ann_path):
            return self._empty_annotation()

        try:
            root = ET.parse(ann_path).getroot()
        except ET.ParseError:
            return self._empty_annotation()

        boxes = []
        labels = []
        gt_class_names = []
        for obj in root.findall("object"):
            label_name = obj.findtext("name", default="").strip()
            if label_name not in self.class_to_idx:
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.findtext("xmin", default="0"))
                ymin = float(bbox.findtext("ymin", default="0"))
                xmax = float(bbox.findtext("xmax", default="0"))
                ymax = float(bbox.findtext("ymax", default="0"))
            except (TypeError, ValueError):
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[label_name])
            gt_class_names.append(label_name)
        return {"boxes": boxes, "labels": labels, "gt_class_names": gt_class_names}

    def _load_annotations(self):
        return [self._parse_annotation(self._annotation_path(image_path)) for image_path in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image_as_rgb(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        annotation = self.annotations[index]
        boxes = annotation["boxes"]
        labels = annotation["labels"]
        gt_class_names = list(annotation["gt_class_names"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "path": image_path,
            "dataset_name": "voc",
            "gt_class_names": gt_class_names,
        }
        return image, target
