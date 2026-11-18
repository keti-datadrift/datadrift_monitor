import json
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'visualizers' / 'runs' / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_single_dog_detection_candidates"
COCO_IMG_DIR = Path(r'D:/DataDrift/datasets/COCO/train2017')
COCO_ANN = Path(r'D:/DataDrift/datasets/COCO/annotations/instances_train2017.json')
COCO_TP = REPO_ROOT / r'object_detectors/runs/yolov5/predict/coco/06-15-2026_18;54_gt/tp.csv'
VOC_ROOT = Path(r'D:/DataDrift/datasets/VOC/VOCdevkit/VOC2012')
VOC_TP = REPO_ROOT / r'object_detectors/runs/yolov5/predict/voc/06-14-2026_13;36_gt/tp.csv'


def load_coco_single_dog_ids():
    with COCO_ANN.open('r', encoding='utf-8') as f:
        data = json.load(f)
    dog_cat_id = next(c['id'] for c in data['categories'] if c['name'] == 'dog')
    img_meta = {im['id']: im for im in data['images']}
    anns_by_image = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            continue
        anns_by_image.setdefault(ann['image_id'], []).append(ann)
    result = {}
    for image_id, anns in anns_by_image.items():
        if len(anns) == 1 and anns[0]['category_id'] == dog_cat_id:
            result[image_id] = img_meta[image_id]
    return result


def coco_candidates():
    single_dog = load_coco_single_dog_ids()
    if not single_dog:
        return pd.DataFrame()
    usecols = ['image_id', 'image_path', 'pred_idx', 'raw_pred_idx', 'xmin', 'ymin', 'xmax', 'ymax', 'score', 'pred_class', 'max_iou', 'gt_iou', 'tp', 'error_type']
    df = pd.read_csv(COCO_TP, usecols=usecols)
    df = df[(df['image_id'].isin(single_dog.keys())) & (df['tp'] == 1) & (df['pred_class'] == 'dog')].copy()
    if df.empty:
        return df
    df['dataset'] = 'coco'
    df['local_image_path'] = df['image_id'].map(lambda x: str(COCO_IMG_DIR / single_dog[int(x)]['file_name']))
    df['image_width'] = df['image_id'].map(lambda x: single_dog[int(x)]['width'])
    df['image_height'] = df['image_id'].map(lambda x: single_dog[int(x)]['height'])
    df['gt_object_count'] = 1
    df['rank_score'] = df['score'].astype(float) + df['gt_iou'].astype(float)
    return df.sort_values(['rank_score', 'score', 'gt_iou'], ascending=False)


def voc_annotation_objects(xml_path):
    root = ET.parse(xml_path).getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.findtext('name')
        if name:
            objects.append(name.strip().lower())
    return objects


def load_voc_single_dog_ids():
    result = {}
    ann_dir = VOC_ROOT / 'Annotations'
    img_dir = VOC_ROOT / 'JPEGImages'
    for xml_path in ann_dir.glob('*.xml'):
        objects = voc_annotation_objects(xml_path)
        if len(objects) == 1 and objects[0] == 'dog':
            image_path = img_dir / f'{xml_path.stem}.jpg'
            if image_path.exists():
                result[xml_path.stem] = image_path
    return result


def voc_candidates():
    single_dog = load_voc_single_dog_ids()
    if not single_dog or not VOC_TP.exists():
        return pd.DataFrame()
    usecols = ['image_id', 'image_path', 'pred_idx', 'raw_pred_idx', 'xmin', 'ymin', 'xmax', 'ymax', 'score', 'pred_class', 'max_iou', 'gt_iou', 'tp', 'error_type']
    df = pd.read_csv(VOC_TP, usecols=usecols)
    df['image_id_str'] = df['image_path'].map(lambda value: Path(str(value)).stem)
    df = df[(df['image_id_str'].isin(single_dog.keys())) & (df['tp'] == 1) & (df['pred_class'] == 'dog')].copy()
    if df.empty:
        return df
    df['dataset'] = 'voc'
    df['local_image_path'] = df['image_id_str'].map(lambda x: str(single_dog[x]))
    df['image_width'] = ''
    df['image_height'] = ''
    df['gt_object_count'] = 1
    df['rank_score'] = df['score'].astype(float) + df['gt_iou'].astype(float)
    return df.sort_values(['rank_score', 'score', 'gt_iou'], ascending=False)


def copy_images(df):
    image_dir = OUT_DIR / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for _, row in df.head(30).iterrows():
        src = Path(row['local_image_path'])
        if not src.exists():
            continue
        dst = image_dir / f"{row['dataset']}_{src.name}"
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for loader in (coco_candidates, voc_candidates):
        df = loader()
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError('No single-dog TP candidates found.')
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(['rank_score', 'dataset', 'score', 'gt_iou'], ascending=[False, True, False, False]).reset_index(drop=True)
    all_df.insert(0, 'rank', range(1, len(all_df) + 1))
    all_df.to_csv(OUT_DIR / 'single_dog_candidates.csv', index=False)
    copied = copy_images(all_df)
    lines = ['# Single Dog Detection Candidates', '']
    for row in all_df.head(20).to_dict('records'):
        lines.append(
            '- rank {rank}: dataset={dataset}, image_id={image_id}, path=`{path}`, score={score:.3f}, iou={iou:.3f}'.format(
                rank=int(row['rank']),
                dataset=row['dataset'],
                image_id=row['image_id'],
                path=row['local_image_path'],
                score=float(row['score']),
                iou=float(row['gt_iou']),
            )
        )
    (OUT_DIR / 'recommended_single_dog.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('candidates', len(all_df))
    print(all_df[['rank', 'dataset', 'image_id', 'local_image_path', 'score', 'gt_iou', 'xmin', 'ymin', 'xmax', 'ymax']].head(20).to_string(index=False))
    print('copied', len(copied))
    print('saved', OUT_DIR)


if __name__ == '__main__':
    main()
