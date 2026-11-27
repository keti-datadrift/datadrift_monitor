import torch

from mmcv.ops import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.utils import filter_scores_and_topk
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_head import RetinaHead


@HEADS.register_module()
class RetinaHeadPAL(RetinaHead):
    def __init__(self, **kwargs):
        super(RetinaHeadPAL, self).__init__(**kwargs)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_class_scores = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                class_scores = cls_score.sigmoid()
            else:
                class_scores = cls_score.softmax(-1)[:, :-1]

            results = filter_scores_and_topk(
                class_scores, cfg.score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    priors=priors,
                    class_scores=class_scores))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            class_scores = filtered_results['class_scores']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_class_scores.append(class_scores)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       mlvl_class_scores,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           mlvl_class_scores,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)
        assert len(mlvl_scores) == len(mlvl_class_scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_class_scores = torch.cat(mlvl_class_scores)

        if mlvl_score_factors is not None:
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                pre_nms_counts = torch.zeros_like(mlvl_scores)
                return dict(
                    det_bboxes=det_bboxes,
                    det_labels=mlvl_labels,
                    pre_nms_counts=pre_nms_counts,
                    class_scores=mlvl_class_scores)

            det_bboxes, keep_idxs = batched_nms(
                mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
            keep_idxs = keep_idxs[:cfg.max_per_img]
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs]
            det_class_scores = mlvl_class_scores[keep_idxs]
            pre_nms_counts = self._count_pre_nms_boxes(
                mlvl_bboxes, mlvl_labels, det_bboxes[:, :4], det_labels, cfg)
            return dict(
                det_bboxes=det_bboxes,
                det_labels=det_labels,
                pre_nms_counts=pre_nms_counts,
                class_scores=det_class_scores)
        else:
            raise NotImplementedError(
                'PAL LIUS export requires NMS so pre_nms_count can be '
                'assigned to final detections.')

    def _count_pre_nms_boxes(self, candidate_bboxes, candidate_labels,
                             det_bboxes, det_labels, cfg):
        if det_bboxes.numel() == 0 or candidate_bboxes.numel() == 0:
            return det_bboxes.new_zeros((det_bboxes.shape[0], ))

        iou_thr = float(cfg.get('pre_nms_count_iou_thr', 0.5))
        overlaps = bbox_overlaps(det_bboxes, candidate_bboxes)
        label_matches = det_labels[:, None] == candidate_labels[None, :]
        counts = ((overlaps >= iou_thr) & label_matches).sum(dim=1)
        return counts.to(dtype=det_bboxes.dtype)
