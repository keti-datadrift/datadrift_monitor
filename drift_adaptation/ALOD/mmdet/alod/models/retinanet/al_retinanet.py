import torch
import numpy as np

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.alod.models.utils import (
    bbox2result_with_epal_features,
    bbox2result_with_pre_nms_count,
    bbox2result_with_uncertainty,
)

@DETECTORS.register_module()
class ALRetinaNet(RetinaNet):

    def __init__(self,
                 **kwargs):
        super(ALRetinaNet, self).__init__(**kwargs)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.
        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = []
        for result in results_list:
            if isinstance(result, dict):
                epal_keys = {
                    'det_bboxes', 'det_labels', 'epal_features',
                    'epal_miss_features'
                }
                if epal_keys.issubset(result):
                    bbox_results.append(
                        bbox2result_with_epal_features(
                            result['det_bboxes'], result['det_labels'],
                            result['epal_features'],
                            result['epal_miss_features']))
                    continue
                required_keys = {'det_bboxes', 'det_labels', 'pre_nms_counts'}
                if not required_keys.issubset(result):
                    raise ValueError('Unsupported ALRetinaNet dict result keys: %r' % sorted(result.keys()))
                bbox_results.append(
                    bbox2result_with_pre_nms_count(
                        result['det_bboxes'], result['det_labels'],
                        result['pre_nms_counts'], self.bbox_head.num_classes,
                        class_scores=result.get('class_scores')))
            elif len(result) == 4:
                det_bboxes, det_labels, cls_uncertainties, box_uncertainties = result
                bbox_results.append(
                    bbox2result_with_uncertainty(
                        det_bboxes, det_labels, cls_uncertainties,
                        box_uncertainties, self.bbox_head.num_classes))
            elif len(result) == 3:
                det_bboxes, det_labels, pre_nms_counts = result
                bbox_results.append(
                    bbox2result_with_pre_nms_count(
                        det_bboxes, det_labels, pre_nms_counts,
                        self.bbox_head.num_classes))
            elif len(result) == 2:
                det_bboxes, det_labels = result
                bbox_results.append(
                    bbox2result(
                        det_bboxes, det_labels, self.bbox_head.num_classes))
            else:
                raise ValueError('Unsupported ALRetinaNet test result: %r' % (result, ))
        return bbox_results
