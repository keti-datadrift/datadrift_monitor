"""VOC active-learning data preparation helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from methods.common.io import read_json, write_json
from tools.common.dataset_pools import ensure_seeded_initial_splits


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)
VOC_CLASS_TO_ID = {name: index for index, name in enumerate(VOC_CLASSES)}


def _read_split_ids(voc_year_dir: Path, split: str) -> List[str]:
    split_file = voc_year_dir / 'ImageSets' / 'Main' / ('%s.txt' % split)
    if not split_file.exists():
        raise FileNotFoundError('VOC split file does not exist: %s' % split_file)
    return [
        line.strip().split()[0]
        for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _parse_object(obj: ET.Element) -> Tuple[int, List[int], int]:
    name = obj.findtext('name')
    if name not in VOC_CLASS_TO_ID:
        raise ValueError('Unsupported VOC class name: %s' % name)
    difficult = int(obj.findtext('difficult', default='0'))
    box = obj.find('bndbox')
    if box is None:
        raise ValueError('VOC object is missing bndbox')

    # Match the PPAL/MMDetection converter: VOC 1-based coordinates become
    # 0-based x1/y1/x2/y2 before converting to COCO xywh.
    x1 = int(box.findtext('xmin')) - 1
    y1 = int(box.findtext('ymin')) - 1
    x2 = int(box.findtext('xmax')) - 1
    y2 = int(box.findtext('ymax')) - 1
    return VOC_CLASS_TO_ID[name], [x1, y1, x2, y2], difficult


def _segmentation_from_xyxy(bbox: Sequence[int]) -> List[List[int]]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return [[x1, y1, x1, y2, x2, y2, x2, y1]]


def _annotation_from_xyxy(
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox_xyxy: Sequence[int],
    difficult: int,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    return {
        'segmentation': _segmentation_from_xyxy(bbox_xyxy),
        'area': int(width * height),
        'ignore': 0,
        'iscrowd': 1 if difficult else 0,
        'image_id': int(image_id),
        'bbox': [x1, y1, width, height],
        'category_id': int(category_id),
        'id': int(annotation_id),
    }


def _parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[int, List[int], int]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    if size is None:
        raise ValueError('VOC XML is missing size: %s' % xml_path)
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    objects = [_parse_object(obj) for obj in root.findall('object')]
    return width, height, objects


def _categories() -> List[Dict[str, Any]]:
    return [
        {'supercategory': 'none', 'id': int(index), 'name': name}
        for index, name in enumerate(VOC_CLASSES)
    ]


def build_voc0712_oracle(vocdevkit: Path, split: str = 'trainval') -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    annotation_id = 0
    image_id = 0

    for year in ('2007', '2012'):
        year_dir = Path(vocdevkit) / ('VOC%s' % year)
        if not year_dir.exists():
            raise FileNotFoundError('Expected VOC year directory: %s' % year_dir)

        for sample_id in _read_split_ids(year_dir, split):
            xml_path = year_dir / 'Annotations' / ('%s.xml' % sample_id)
            width, height, objects = _parse_voc_xml(xml_path)
            images.append({
                'id': int(image_id),
                'file_name': 'VOC%s/JPEGImages/%s.jpg' % (year, sample_id),
                'height': int(height),
                'width': int(width),
            })
            for category_id, bbox_xyxy, difficult in objects:
                annotations.append(_annotation_from_xyxy(
                    annotation_id,
                    image_id,
                    category_id,
                    bbox_xyxy,
                    difficult,
                ))
                annotation_id += 1
            image_id += 1

    return {
        'images': images,
        'type': 'instance',
        'categories': _categories(),
        'annotations': annotations,
    }


def ensure_voc_active_learning(
    voc_cfg: Dict[str, object],
    root: Path,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    vocdevkit = Path(str(voc_cfg.get('vocdevkit', 'data/VOCdevkit')))
    oracle_output = Path(str(voc_cfg.get('oracle_output', 'data/VOC0712/annotations/trainval_0712.json')))
    split_output_dir = Path(str(voc_cfg.get('split_output_dir', 'data/active_learning/voc')))
    if not vocdevkit.is_absolute():
        vocdevkit = Path(root) / vocdevkit
    if not oracle_output.is_absolute():
        oracle_output = Path(root) / oracle_output
    if not split_output_dir.is_absolute():
        split_output_dir = Path(root) / split_output_dir

    n_labeled = int(voc_cfg.get('n_labeled', 827))
    split = str(voc_cfg.get('split', 'trainval'))
    dataset_prefix = str(voc_cfg.get('dataset_prefix', 'voc'))

    oracle = (
        read_json(oracle_output)
        if oracle_output.exists()
        else build_voc0712_oracle(vocdevkit, split=split)
    )
    oracle_created = not oracle_output.exists()
    if oracle_created:
        write_json(oracle_output, oracle)

    split_paths = ensure_seeded_initial_splits(
        oracle,
        output_dir=split_output_dir,
        dataset_prefix=dataset_prefix,
        n_labeled=n_labeled,
        seeds=[0] if seeds is None else seeds,
    )
    pools_created = any(
        record.get('status') == 'created' for record in split_paths
    )
    return {
        'component': 'dataset',
        'type': 'voc0712',
        'status': 'ready',
        'action': 'created' if oracle_created or pools_created else 'kept',
        'oracle_path': str(oracle_output),
        'split_paths': split_paths,
    }
