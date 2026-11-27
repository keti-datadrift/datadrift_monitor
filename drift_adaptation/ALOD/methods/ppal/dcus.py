from collections import OrderedDict
from pathlib import Path

import numpy as np

from methods.common.coco_pool import image_ids, read_coco_json
from methods.ppal.base import BaseALSampler
from methods.ppal.inference import (
    class_quality_checkpoint_path,
    load_class_quality,
    load_uncertainty_detections,
)


eps = 1e-10


class DCUSSampler(BaseALSampler):
    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        score_thr,
        class_weight_ub,
        class_weight_alpha,
        dataset_type,
    ):
        super(DCUSSampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.score_thr = score_thr
        self.class_weight_ub = class_weight_ub
        self.class_weight_alpha = class_weight_alpha

    def _get_classwise_weight(self, class_qualities):
        reverse_q = 1 - class_qualities
        b = np.exp(1. / self.class_weight_alpha) - 1
        _weights = 1 + self.class_weight_alpha * np.log(b * reverse_q + 1) * self.class_weight_ub

        class_weights = dict()
        for i in range(min(len(_weights), len(self.CLASSES))):
            cid = self.class_name2id[self.CLASSES[i]]
            class_weights[cid] = float(_weights[i])
        return class_weights

    def al_acquisition(self, result_json, last_label_path):
        checkpoint_path = class_quality_checkpoint_path(result_json)
        class_qualities = load_class_quality(checkpoint_path)
        class_weights = self._get_classwise_weight(class_qualities)
        results = load_uncertainty_detections(result_json)

        last_labeled_data = read_coco_json(Path(last_label_path))
        last_labeled_img_ids = image_ids(last_labeled_data)

        image_hit = dict()
        for img_id in self.oracle_data.keys():
            image_hit[img_id] = 0
        for img_id in last_labeled_img_ids:
            image_hit[img_id] = 1

        image_uncertainties = OrderedDict()
        for img_id in self.oracle_data.keys():
            if image_hit[img_id] == 0:
                image_uncertainties[img_id] = [0.]

        valid_detection_count = 0
        per_class_detection_counts = OrderedDict()
        for res in results:
            img_id = res['image_id']
            img_size = (self.oracle_data[img_id]['image']['width'], self.oracle_data[img_id]['image']['height'])
            if not self.is_box_valid(res['bbox'], img_size):
                continue
            if res['score'] < self.score_thr:
                continue
            uncertainty = float(res['cls_uncertainty'])
            label = res['category_id']
            if label not in class_weights:
                continue
            image_uncertainties[img_id].append(uncertainty * class_weights[label])
            per_class_detection_counts[label] = per_class_detection_counts.get(label, 0) + 1
            valid_detection_count += 1

        for img_id in image_uncertainties.keys():
            _img_uncertainties = np.array(image_uncertainties[img_id])
            image_uncertainties[img_id] = _img_uncertainties.sum()

        img_ids = []
        merged_img_uncertainties = []
        for k, v in image_uncertainties.items():
            img_ids.append(k)
            merged_img_uncertainties.append(v)
        img_ids = np.array(img_ids)
        merged_img_uncertainties = np.array(merged_img_uncertainties)

        inds_sort = np.argsort(-1. * merged_img_uncertainties)
        sampled_inds = inds_sort[:self.n_images]
        sampled_img_ids = img_ids[sampled_inds].tolist()
        candidate_records = []
        for rank, index in enumerate(sampled_inds.tolist(), start=1):
            score = float(merged_img_uncertainties[index])
            candidate_records.append({
                'image_id': img_ids[index].item() if hasattr(img_ids[index], 'item') else img_ids[index],
                'rank': rank,
                'score': score,
                'source': 'dcus',
                'components': {
                    'dcus_score': score,
                    'weighted_uncertainty': score,
                },
                'metadata': {},
            })

        metrics = {
            'class_quality_checkpoint': str(checkpoint_path),
            'score_threshold': float(self.score_thr),
            'class_weight_upper_bound': float(self.class_weight_ub),
            'class_quality_alpha': float(self.class_weight_alpha),
            'raw_detection_count': len(results),
            'valid_detection_count': valid_detection_count,
            'class_weights': class_weights,
            'per_class_detection_counts': dict(per_class_detection_counts),
            'last_labeled_count': len(last_labeled_img_ids),
            'uncertainty_pool_count': len(sampled_img_ids),
        }
        return sampled_img_ids, candidate_records, metrics

    def al_round(self, result_path, last_label_path):
        self.round += 1
        self.latest_labeled = last_label_path

        sampled_img_ids, candidate_records, metrics = self.al_acquisition(result_path, last_label_path)
        return {
            'selected_image_ids': sampled_img_ids,
            'candidate_records': candidate_records,
            'selected_count': len(sampled_img_ids),
            'metrics': metrics,
        }
