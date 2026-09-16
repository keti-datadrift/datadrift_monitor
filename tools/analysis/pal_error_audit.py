"""Retrospective PAL diagnostics using saved detections and test exports.

This tool never trains a detector or changes acquisition. Unlabeled GT is used
only for retrospective measurement. Requires tidecv==1.0.1 in addition to the
runtime environment. Run from the repository root.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from pycocotools import mask as mask_utils
from sklearn.metrics import average_precision_score, roc_auc_score
from tidecv import Data, TIDE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.common.matching import match_detections_to_ground_truth
from methods.pal.inference import load_detection_records
from methods.pal.lius import score_unlabeled_detections, train_classwise_models


ERRORS = ('Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss')
DETECTION_TYPES = ('TP',) + ERRORS[:-1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert,
                                     allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def auc(target, score):
    return float(roc_auc_score(target, score)) if len(set(target)) == 2 else None


def binary_metrics(target, probability):
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not len(target):
        return dict(n=0)
    bins = []
    ece = 0.0
    for low in np.arange(10) / 10:
        take = ((probability >= low) &
                (probability < low + .1 if low < .9 else probability <= 1))
        if not take.any():
            continue
        observed, predicted = target[take].mean(), probability[take].mean()
        ece += take.mean() * abs(observed - predicted)
        bins.append(dict(low=float(low), n=int(take.sum()),
                         observed=float(observed), predicted=float(predicted)))
    return dict(n=len(target), fp_fraction=float(target.mean()),
                auroc=auc(target, probability),
                ap=float(average_precision_score(target, probability))
                if target.sum() else None,
                brier=float(np.mean((target - probability) ** 2)),
                ece=float(ece), calibration_bins=bins)


def make_tide(pool, detections, max_dets=200, threshold=.5):
    """Use TIDE errors with explicit VOC difficult filtering, not crowd IoA.

    The VOC evaluator ignores a detection when its highest-IoU same-class
    annotation is difficult and reaches the evaluation threshold. Apply that
    rule before TIDE and exclude difficult GT. TIDE's own matching/oracles and
    101-point AP are otherwise unchanged; this is not native VOC07 AP.
    """
    gt = Data('ground_truth', max_dets=max_dets)
    pred = Data('predictions', max_dets=max_dets)
    annotations = defaultdict(list)
    for cat in pool['categories']:
        gt.add_class(cat['id'], cat['name'])
        pred.add_class(cat['id'], cat['name'])
    for image in pool['images']:
        gt.add_image(image['id'], str(image.get('file_name', image['id'])))
    for ann in pool['annotations']:
        annotations[(ann['image_id'], ann['category_id'])].append(ann)
        if not (ann.get('iscrowd', 0) or ann.get('ignore', 0)):
            gt.add_ground_truth(ann['image_id'], ann['category_id'], ann['bbox'])
    ignored = []
    for index, det in enumerate(detections):
        group = annotations[(det['image_id'], det['category_id'])]
        if any(a.get('iscrowd', 0) or a.get('ignore', 0) for a in group):
            overlap = mask_utils.iou([det['bbox']], [a['bbox'] for a in group],
                                     [False] * len(group))[0]
            best = int(np.argmax(overlap))
            if overlap[best] >= threshold and (group[best].get('iscrowd', 0) or
                                               group[best].get('ignore', 0)):
                ignored.append(index)
                continue
        pred.add_detection(det['image_id'], det['category_id'],
                           float(det['score']), det['bbox'])
        pred.annotations[-1]['source_index'] = index
    engine = TIDE(pos_threshold=threshold)
    run = engine.evaluate(gt, pred)
    labels = np.full(len(detections), 'Ignored', dtype='<U8')
    for item in pred.annotations:
        if 'used' not in item:
            raise ValueError('TIDE max_dets dropped an input detection')
        if item['used']:
            labels[item['source_index']] = 'TP'
    for error in run.errors:
        if hasattr(error, 'pred'):
            labels[error.pred['source_index']] = error.short_name
    if int(np.sum(labels == 'Ignored')) != len(ignored):
        raise ValueError('Unclassified detections in TIDE result')
    return run, labels, ignored


def per_image_counts(run, labels, detections):
    rows = {image_id: Counter() for image_id in run.gt.images}
    for truth in run.gt.annotations:
        rows[truth['image']]['GT'] += 1
    for det, label in zip(detections, labels):
        rows[det['image_id']][str(label)] += 1
    for errors in run.false_negatives.values():
        for truth in errors:
            rows[truth['image']]['FN'] += 1
    for error in run.errors:
        if error.short_name == 'Miss':
            rows[error.gt['image']]['Miss'] += 1
    return rows


def analyze_predictors(source, destination):
    oracle = read_json(ROOT / 'data/VOC0712/annotations/trainval_0712.json')
    by_image = defaultdict(list)
    for ann in oracle['annotations']:
        by_image[ann['image_id']].append(ann)
    summary_rows, type_rows, class_rows, batch_rows, calibration_rows = [], [], [], [], []
    for index in range(1, 7):
        print('Predictor round %d started' % index, flush=True)
        folder = source / ('round_%02d' % index)
        out = destination / ('round_%02d' % index)
        out.mkdir(exist_ok=True)
        diagnostic = read_json(folder / 'pal_diagnostics.json')
        labeled = read_json(diagnostic['inputs']['labeled_pool_json'])
        unlabeled = read_json(diagnostic['inputs']['unlabeled_pool_json'])
        labeled_det = load_detection_records(folder / 'pal_labeled_detections.bbox.json')
        detections = load_detection_records(folder / 'pal_unlabeled_detections.bbox.json')
        labeled_ids = {x['id'] for x in labeled['images']}
        unlabeled_ids = {x['id'] for x in unlabeled['images']}
        assert not labeled_ids.intersection(unlabeled_ids)
        assert all(d['image_id'] in labeled_ids for d in labeled_det)
        assert all(d['image_id'] in unlabeled_ids for d in detections)
        matched = match_detections_to_ground_truth(labeled_det, labeled)
        models = train_classwise_models(matched)
        scored = score_unlabeled_detections(detections, models)
        assert len(matched) == diagnostic['matched_detection_count']
        assert len(scored) == diagnostic['scored_detection_count']
        # Only this oracle copy is ever used to evaluate the unlabeled pool.
        pool = dict(images=unlabeled['images'], categories=oracle['categories'],
                    annotations=[a for iid in unlabeled_ids for a in by_image[iid]])
        for j, det in enumerate(detections):
            det['_audit_index'] = j
        original_target = np.zeros(len(detections), dtype=int)
        for det in match_detections_to_ground_truth(detections, pool):
            original_target[det['_audit_index']] = 1 - det['target']
        probability = np.array([1 - d['tp_probability'] for d in scored])
        entropy = np.array([d['lius_score'] for d in scored])
        confidence = np.array([d['score'] for d in detections])
        classes = np.array([d['category_id'] for d in detections])
        run, labels, ignored = make_tide(pool, detections)
        valid = labels != 'Ignored'
        target = (labels != 'TP').astype(int)
        native_metrics = binary_metrics(original_target, probability)
        metrics = binary_metrics(target[valid], probability[valid])
        for row in metrics['calibration_bins']:
            calibration_rows.append(dict(round=index, **row))
        summary = dict(round=index, pool_images=len(unlabeled_ids),
                       detections=len(detections), ignored=len(ignored),
                       **{k: v for k, v in metrics.items() if k != 'calibration_bins'})
        summary.update(native_pal_auroc=native_metrics['auroc'],
                       confidence_auroc=auc(target[valid], 1 - confidence[valid]),
                       target_disagreement=int(np.sum(original_target[valid] != target[valid])))
        per_class_auc = []
        fallback_detections = 0
        for cat in oracle['categories']:
            cid = cat['id']
            fit = [d for d in matched if d['category_id'] == cid]
            model = models.get(cid)
            take = (classes == cid) & valid
            values = binary_metrics(target[take], probability[take])
            prior_fp = 1 - np.mean([d['target'] for d in fit]) if fit else .5
            row = dict(round=index, category=cat['name'], labeled_n=len(fit),
                       labeled_tp=sum(d['target'] for d in fit),
                       fallback=model.fallback_reason if model else 'absent_class',
                       **{k: v for k, v in values.items() if k != 'calibration_bins'})
            row['prior_brier'] = float(np.mean((target[take] - prior_fp) ** 2)) if take.any() else None
            class_rows.append(row)
            if values.get('auroc') is not None:
                per_class_auc.append(values['auroc'])
            if model is None or model.used_fallback:
                fallback_detections += int((classes == cid).sum())
        summary['macro_auroc'] = float(np.mean(per_class_auc))
        summary['fallback_detections'] = fallback_detections
        summary_rows.append(summary)
        top_masks = {}
        ordered = np.flatnonzero(valid)[np.argsort(-entropy[valid], kind='stable')]
        for percent in (5, 10, 20):
            mask = np.zeros(len(labels), dtype=bool)
            mask[ordered[:int(np.ceil(len(ordered) * percent / 100))]] = True
            top_masks[percent] = mask
        for kind in DETECTION_TYPES:
            take = labels == kind
            pair = (labels == 'TP') | take
            row = dict(round=index, type=kind, n=int(take.sum()))
            if take.any():
                row.update(q_fp_mean=float(probability[take].mean()),
                           q_fp_median=float(np.median(probability[take])),
                           q_fp_p10=float(np.quantile(probability[take], .1)),
                           q_fp_p90=float(np.quantile(probability[take], .9)),
                           entropy_mean=float(entropy[take].mean()),
                           entropy_median=float(np.median(entropy[take])),
                           q_fp_ge_half=float(np.mean(probability[take] >= .5)),
                           auc_vs_tp=auc(take[pair].astype(int), probability[pair]) if kind != 'TP' else None)
                subtype_class_aucs = []
                subtype_class_coverages = []
                for cid in sorted(set(classes)):
                    subset = pair & (classes == cid)
                    value = auc(take[subset].astype(int), probability[subset]) if kind != 'TP' else None
                    if value is not None:
                        subtype_class_aucs.append(value)
                    subtype = take & (classes == cid)
                    if subtype.any():
                        subtype_class_coverages.append(float(np.mean(top_masks[10][subtype])))
                row['macro_auc_vs_tp'] = float(np.mean(subtype_class_aucs)) if subtype_class_aucs else None
                row['macro_top10_coverage'] = float(np.mean(subtype_class_coverages))
                for percent, mask in top_masks.items():
                    row['top%d_coverage' % percent] = float(np.mean(mask[take]))
                    row['top%d_composition' % percent] = float(np.mean(take[mask]))
            type_rows.append(row)
        selected = set(diagnostic['selected_image_ids'])
        assert len(selected) == 414 and selected.issubset(unlabeled_ids)
        new_pool = read_json(folder / 'annotations/new_labeled.json')
        assert {x['id'] for x in new_pool['images']} - labeled_ids == selected
        image_counts = per_image_counts(run, labels, detections)
        for kind in DETECTION_TYPES + ('Miss', 'FN', 'GT'):
            total = sum(c[kind] for c in image_counts.values())
            chosen = sum(image_counts[iid][kind] for iid in selected)
            gt_total = sum(c['GT'] for c in image_counts.values())
            gt_chosen = sum(image_counts[iid]['GT'] for iid in selected)
            batch_rows.append(dict(round=index, type=kind, pool_count=total,
                                   selected_count=chosen, selected_images=len(selected),
                                   pool_images=len(unlabeled_ids),
                                   pool_per_image=total / len(unlabeled_ids),
                                   selected_per_image=chosen / len(selected),
                                   count_enrichment=(chosen / len(selected)) / (total / len(unlabeled_ids)) if total else None,
                                   pool_per_gt=total / gt_total,
                                   selected_per_gt=chosen / gt_chosen,
                                   gt_normalized_enrichment=(chosen / gt_chosen) / (total / gt_total) if total else None))
        np.savez_compressed(out / 'predictor_detections.npz',
                            image_id=[d['image_id'] for d in detections],
                            category_id=classes, score=confidence, q_fp=probability,
                            entropy=entropy, error_type=labels,
                            original_pal_fp=original_target)
        write_csv(out / 'acquisition_image_errors.csv',
                  [dict(image_id=iid, selected=iid in selected, **dict(counts))
                   for iid, counts in image_counts.items()])
        write_json(out / 'predictor_summary.json', dict(summary=summary,
                   native_target_metrics=native_metrics, tide_target_metrics=metrics))
        write_csv(destination / 'predictor_rounds.csv', summary_rows)
        write_csv(destination / 'predictor_types.csv', type_rows)
        write_csv(destination / 'predictor_classes.csv', class_rows)
        write_csv(destination / 'predictor_calibration.csv', calibration_rows)
        write_csv(destination / 'selected_batch_errors.csv', batch_rows)
        print('Predictor round %d: AUC %.4f, macro %.4f, fallback %d' %
              (index, metrics['auroc'], summary['macro_auroc'], fallback_detections), flush=True)


def test_dataset():
    from mmcv import Config
    from mmdet.datasets import build_dataset
    import mmdet.alod.datasets  # Register project datasets.
    config = Config.fromfile('configs/alod_mmdet/retinanet_voc_base.py')
    specification = dict(config.data.test)
    specification.pop('samples_per_gpu', None)
    specification.update(ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
                         img_prefix='data/VOCdevkit/VOC2007/', test_mode=True)
    return build_dataset(specification)


def test_pool_and_detections(dataset, results):
    def xywh(box):
        # XMLDataset uses zero-based inclusive boxes; native eval adds one.
        return [float(box[0]), float(box[1]), float(box[2] - box[0] + 1),
                float(box[3] - box[1] + 1)]
    pool = dict(images=[], annotations=[],
                categories=[dict(id=i, name=name) for i, name in enumerate(dataset.CLASSES)])
    detections = []
    assert len(results) == len(dataset) == 4952
    for iid, result in enumerate(results):
        pool['images'].append(dict(id=iid, file_name=dataset.data_infos[iid]['filename']))
        info = dataset.get_ann_info(iid)
        for suffix, ignored in (('', False), ('_ignore', True)):
            for box, label in zip(info['bboxes' + suffix], info['labels' + suffix]):
                pool['annotations'].append(dict(id=len(pool['annotations']), image_id=iid,
                    category_id=int(label), bbox=xywh(box), iscrowd=int(ignored)))
        for cid, boxes in enumerate(result):
            for box in boxes:
                detections.append(dict(image_id=iid, category_id=cid,
                                       bbox=xywh(box), score=float(box[4])))
    return pool, detections


def analyze_test(source, destination, wait_for_exports=False):
    from mmdet.core import eval_map
    dataset = test_dataset()
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    rows, class_rows, size_rows = [], [], []
    for index in range(1, 8):
        folder = destination / ('round_%02d' % index)
        if wait_for_exports:
            deadline = time.monotonic() + 1800
            if not (folder / 'inference_complete.json').exists():
                print('Waiting for test export round %d' % index, flush=True)
            while not (folder / 'inference_complete.json').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Test export did not complete: %s' % folder)
                time.sleep(5)
        if not (folder / 'inference_complete.json').exists():
            raise FileNotFoundError('Complete test export is missing: %s' % folder)
        print('Test error decomposition round %d started' % index, flush=True)
        with (folder / 'test_predictions.pkl').open('rb') as stream:
            results = pickle.load(stream)
        native50, native50class = eval_map(results, annotations, dataset='voc07',
            iou_thr=.5, logger='silent', nproc=2, use_legacy_coordinate=True)
        native75, _ = eval_map(results, annotations, dataset='voc07',
            iou_thr=.75, logger='silent', nproc=2, use_legacy_coordinate=True)
        original_eval = read_json(sorted((source / ('round_%02d' % index)).glob('eval_*.json'))[-1])
        original_map = original_eval['metric']['mAP']
        if abs(native50 - original_map) > .0005:
            raise ValueError('Re-exported AP50 does not reproduce round %d: %.8f vs %.8f' %
                             (index, native50, original_map))
        pool, detections = test_pool_and_detections(dataset, results)
        run, labels, ignored = make_tide(pool, detections, max_dets=100)
        delta = {}
        class_delta = defaultdict(dict)
        for error_type in TIDE._error_types:
            fixed = run.fix_errors(lambda error: isinstance(error, error_type))
            delta[error_type.short_name] = max(fixed.get_mAP() - run.ap, 0)
            for cid in range(len(dataset.CLASSES)):
                class_delta[cid][error_type.short_name] = max(
                    fixed.objs[cid].get_ap() - run.ap_data.objs[cid].get_ap(), 0)
        counts = Counter(str(x) for x in labels)
        misses = [e.gt for e in run.errors if e.short_name == 'Miss']
        false_negatives = [g for group in run.false_negatives.values() for g in group]
        num_gt = len(run.gt.annotations)
        row = dict(round=index, native_ap50=float(native50 * 100),
                   native_ap75=float(native75 * 100), original_ap50=original_map * 100,
                   reproduction_difference_pp=(native50 - original_map) * 100,
                   tide_ap50=run.ap, detections=len(detections), ignored=len(ignored),
                   gt=num_gt, tp=counts['TP'], fn=len(false_negatives), miss=len(misses),
                   recall=counts['TP'] / num_gt)
        for kind in ERRORS:
            row['dap_' + kind] = delta[kind]
            row['count_' + kind] = len(misses) if kind == 'Miss' else counts[kind]
        high = np.array([d['score'] >= .3 for d in detections])
        for kind in DETECTION_TYPES:
            row['score03_' + kind] = int(np.sum(high & (labels == kind)))
        row['score03_recall'] = row['score03_TP'] / num_gt
        assert counts['TP'] + len(false_negatives) == num_gt
        assert counts['TP'] + sum(counts[k] for k in ERRORS[:-1]) + len(ignored) == len(detections)
        rows.append(row)
        for cid, name in enumerate(dataset.CLASSES):
            class_gt = [g for g in run.gt.annotations if g['class'] == cid]
            class_fn = [g for g in false_negatives if g['class'] == cid]
            class_miss = [g for g in misses if g['class'] == cid]
            class_rows.append(dict(round=index, category=name, gt=len(class_gt),
                fn=len(class_fn), miss=len(class_miss), recall=1 - len(class_fn) / len(class_gt),
                native_ap50=float(native50class[cid]['ap'] * 100),
                **{'dap_' + k: v for k, v in class_delta[cid].items()}))
        fn_ids = {g['_id'] for g in false_negatives}
        miss_ids = {g['_id'] for g in misses}
        for name, low, high_area in (('small', 0, 32 ** 2),
                                     ('medium', 32 ** 2, 96 ** 2),
                                     ('large', 96 ** 2, float('inf'))):
            group = [g for g in run.gt.annotations if low <= g['bbox'][2] * g['bbox'][3] < high_area]
            size_rows.append(dict(round=index, size=name, gt=len(group),
                fn=sum(g['_id'] in fn_ids for g in group),
                miss=sum(g['_id'] in miss_ids for g in group),
                recall=1 - sum(g['_id'] in fn_ids for g in group) / len(group)))
        write_json(folder / 'test_error_summary.json', row)
        write_csv(destination / 'test_rounds.csv', rows)
        write_csv(destination / 'test_classes.csv', class_rows)
        write_csv(destination / 'test_sizes.csv', size_rows)
        print('Test round %d: native AP50 %.3f, AP75 %.3f, TIDE %.3f, dAP %s' %
              (index, native50 * 100, native75 * 100, run.ap, delta), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('predictor', 'test'), required=True)
    parser.add_argument('--wait-for-exports', action='store_true',
                        help='Wait up to 30 minutes per round for concurrent inference')
    args = parser.parse_args()
    source, destination = args.run.resolve(), args.output.resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Analysis output must be outside the source run')
    destination.mkdir(parents=True, exist_ok=True)
    if args.stage == 'predictor':
        analyze_predictors(source, destination)
    else:
        analyze_test(source, destination, args.wait_for_exports)


if __name__ == '__main__':
    main()
