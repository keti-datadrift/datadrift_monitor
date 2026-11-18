# Copyright (c) OpenMMLab. All rights reserved.
import json

import numpy as np
import torch

from mmcv.runner import get_dist_info

from mmdet.models.builder import HEADS

from mmdet.alod.models.retinanet.feature_export_utils import (
    RetinaFeatureExportBase,
    image_id_tensor_from_meta,
    padded_queue_length,
    unique_valid_indices,
)
from mmdet.alod.models.utils import concat_all_gather, get_inter_feats


@HEADS.register_module()
class RetinaDetectionFeatureExportHead(RetinaFeatureExportBase):
    """Export PPAL-style detection-level RetinaNet classification-tower features."""

    def __init__(self, total_images, max_det, feat_dim, output_path, **kwargs):
        super(RetinaDetectionFeatureExportHead, self).__init__(**kwargs)

        _, world_size = get_dist_info()
        self.total_images = int(total_images)
        self.queue_length = padded_queue_length(self.total_images, world_size)
        self.current_images = 0
        self.max_det = int(max_det)
        self.feat_dim = int(feat_dim)
        self.output_path = output_path

        self.register_buffer("image_id_queue", torch.zeros((self.queue_length, 1), dtype=torch.int) - 1)
        self.register_buffer("det_label_queue", torch.zeros((self.queue_length, self.max_det), dtype=torch.long) - 1)
        self.register_buffer("det_score_queue", torch.zeros((self.queue_length, self.max_det)))
        self.register_buffer("det_feat_queue", torch.zeros((self.queue_length, self.max_det, self.feat_dim)))
        self.register_buffer("det_valid_queue", torch.zeros((self.queue_length, self.max_det), dtype=torch.bool))

    def _pad_detection_values(self, det_labels, det_scores, det_feats):
        n_det = min(int(det_labels.numel()), self.max_det)
        padded_labels = det_labels.new_full((self.max_det, ), -1)
        padded_scores = det_scores.new_zeros((self.max_det, ))
        padded_feats = det_feats.new_zeros((self.max_det, self.feat_dim))
        padded_valid = torch.zeros((self.max_det, ), dtype=torch.bool, device=det_labels.device)
        if n_det > 0:
            padded_labels[:n_det] = det_labels[:n_det].to(dtype=torch.long)
            padded_scores[:n_det] = det_scores[:n_det]
            padded_feats[:n_det] = det_feats[:n_det]
            padded_valid[:n_det] = True
        return padded_labels, padded_scores, padded_feats, padded_valid

    def collect_export_features(
        self,
        mlvl_feats,
        det_labels,
        cls_scores,
        keep_idxs,
        lvl_inds,
        mlvl_bboxes_unscale,
        img_meta,
        img_shape,
        cfg,
        **kwargs
    ):
        _, world_size = get_dist_info()
        image_id = image_id_tensor_from_meta(
            img_meta,
            self.image_id_queue.device,
            'RetinaDetectionFeatureExportHead',
        )
        det_lvl_inds = lvl_inds[keep_idxs][:cfg.max_per_img]
        det_unscale_bboxes = mlvl_bboxes_unscale[keep_idxs][:cfg.max_per_img]
        det_feats = get_inter_feats(mlvl_feats, det_lvl_inds, det_unscale_bboxes, img_shape)

        collected_img_ids = concat_all_gather(image_id.reshape(1, 1))
        self.image_id_queue[self.current_images:(self.current_images + world_size)] = collected_img_ids

        padded_labels, padded_scores, padded_feats, padded_valid = self._pad_detection_values(
            det_labels, cls_scores, det_feats)
        collected_det_labels = concat_all_gather(padded_labels.reshape(1, self.max_det).contiguous())
        collected_det_scores = concat_all_gather(padded_scores.reshape(1, self.max_det).contiguous())
        collected_det_feats = concat_all_gather(padded_feats.reshape(1, self.max_det, self.feat_dim).contiguous())
        collected_det_valid = concat_all_gather(padded_valid.reshape(1, self.max_det).contiguous())
        self.det_label_queue[self.current_images:(self.current_images + world_size)] = collected_det_labels
        self.det_score_queue[self.current_images:(self.current_images + world_size)] = collected_det_scores
        self.det_feat_queue[self.current_images:(self.current_images + world_size)] = collected_det_feats
        self.det_valid_queue[self.current_images:(self.current_images + world_size)] = collected_det_valid

    def export_features(self):
        valid_inds = (self.image_id_queue >= 0).reshape(-1)
        image_ids = self.image_id_queue[valid_inds].detach().cpu().numpy()
        det_labels = self.det_label_queue[valid_inds].detach().cpu().numpy()
        det_scores = self.det_score_queue[valid_inds].detach().cpu().numpy()
        det_features = self.det_feat_queue[valid_inds].detach().cpu().numpy()
        det_valid = self.det_valid_queue[valid_inds].detach().cpu().numpy()

        keep = unique_valid_indices(image_ids, self.total_images)
        image_ids = image_ids[keep]
        det_labels = det_labels[keep]
        det_scores = det_scores[keep]
        det_features = det_features[keep]
        det_valid = det_valid[keep]

        metadata = dict(
            schema='alod_detection_feature_artifact',
            schema_version=1,
            total_images=int(self.total_images),
            exported_images=int(image_ids.shape[0]),
            max_det=int(self.max_det),
            feat_dim=int(self.feat_dim),
        )
        np.savez_compressed(
            self.output_path,
            image_ids=image_ids.reshape(-1),
            det_labels=det_labels.astype(np.int64, copy=False),
            det_scores=det_scores.astype(np.float32, copy=False),
            det_features=det_features.astype(np.float32, copy=False),
            det_valid=det_valid.astype(np.bool_, copy=False),
            metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
        )
        return
