# Copyright (c) OpenMMLab. All rights reserved.
"""Shared RetinaNet feature-export head utilities."""

from __future__ import annotations

import numpy as np
import torch

from mmcv.ops import batched_nms
from mmcv.runner import force_fp32, get_dist_info

from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.models.dense_heads.retina_head import RetinaHead


def padded_queue_length(total_images, world_size):
    total_images = int(total_images)
    remainder = total_images % world_size
    return total_images if remainder == 0 else total_images + world_size - remainder


def image_feature_from_levels(mlvl_feats):
    pooled = []
    for feat in mlvl_feats:
        pooled.append(feat.float().mean(dim=(1, 2)))
    return torch.stack(pooled, dim=0).mean(dim=0)


def image_id_tensor_from_meta(img_meta, device, owner_name):
    if 'image_id' not in img_meta:
        raise KeyError(
            '%s requires image_id in img_meta. Add AddImageIdToMeta to the '
            'test pipeline and include image_id in Collect.meta_keys.'
            % owner_name)
    return torch.tensor([[int(img_meta['image_id'])]], dtype=torch.int, device=device)


def unique_valid_indices(image_ids, total_images):
    keep = []
    seen = set()
    for index, image_id in enumerate(np.asarray(image_ids).reshape(-1).tolist()):
        image_id = int(image_id)
        if image_id in seen:
            continue
        seen.add(image_id)
        keep.append(index)
        if len(keep) >= int(total_images):
            break
    return np.asarray(keep, dtype=np.int64)


class RetinaFeatureExportBase(RetinaHead):
    """Common RetinaNet inference path for ALOD feature exporters."""

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        cls_score, bbox_pred, cls_feat = self.forward(feats)
        results_list = self.get_bboxes(
            cls_score, bbox_pred, fpn_feats=cls_feat,
            img_metas=img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   fpn_feats=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            with_score_factors = False
        else:
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            fpn_feats_list = select_single_mlvl(fpn_feats, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, score_factor_list,
                fpn_feats_list, mlvl_priors, img_meta, cfg, rescale,
                with_nms, **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           fpn_feats_list,
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
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None

        for cls_score, bbox_pred, score_factor, priors in zip(
                cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(
                scores, 0, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(
            mlvl_scores, mlvl_labels, mlvl_bboxes, fpn_feats_list, img_meta,
            cfg, rescale, with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           mlvl_feats,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)
        img_shape = img_meta['img_shape']

        mlvl_bboxes_unscale = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes = mlvl_bboxes_unscale / mlvl_bboxes_unscale.new_tensor(img_meta['scale_factor'])
        else:
            raise NotImplementedError

        lvl_inds = torch.cat([torch.zeros_like(x) + i for (i, x) in enumerate(mlvl_scores)])
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if not with_nms:
            raise NotImplementedError

        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
        det_bboxes = det_bboxes[:cfg.max_per_img]
        det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
        cls_scores = det_bboxes[:, -1]
        cls_uncertainties = -1 * (
            cls_scores * torch.log(cls_scores + 1e-10)
            + (1 - cls_scores) * torch.log((1 - cls_scores) + 1e-10))
        box_uncertainties = torch.zeros_like(cls_uncertainties)

        rank, world_size = get_dist_info()
        self.collect_export_features(
            mlvl_feats=mlvl_feats,
            det_bboxes=det_bboxes,
            det_labels=det_labels,
            cls_scores=cls_scores,
            keep_idxs=keep_idxs,
            lvl_inds=lvl_inds,
            mlvl_bboxes_unscale=mlvl_bboxes_unscale,
            img_meta=img_meta,
            img_shape=img_shape,
            cfg=cfg,
        )
        self.current_images += world_size

        if self.current_images >= self.total_images:
            torch.cuda.empty_cache()
            if rank == 0:
                self.export_features()
            else:
                torch.cuda.synchronize()

        return det_bboxes, det_labels, cls_uncertainties, box_uncertainties

    def collect_export_features(self, **kwargs):
        raise NotImplementedError

    def export_features(self):
        raise NotImplementedError
