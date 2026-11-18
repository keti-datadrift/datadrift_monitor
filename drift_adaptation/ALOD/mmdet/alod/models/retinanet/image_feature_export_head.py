# Copyright (c) OpenMMLab. All rights reserved.
import json

import numpy as np
import torch

from mmcv.runner import get_dist_info

from mmdet.models.builder import HEADS

from mmdet.alod.models.retinanet.feature_export_utils import (
    RetinaFeatureExportBase,
    image_feature_from_levels,
    image_id_tensor_from_meta,
    padded_queue_length,
    unique_valid_indices,
)
from mmdet.alod.models.utils import concat_all_gather


@HEADS.register_module()
class RetinaImageFeatureExportHead(RetinaFeatureExportBase):
    """Export one image-level RetinaNet classification-tower feature per image."""

    def __init__(self, total_images, feat_dim, output_path, **kwargs):
        super(RetinaImageFeatureExportHead, self).__init__(**kwargs)

        _, world_size = get_dist_info()
        self.total_images = int(total_images)
        self.queue_length = padded_queue_length(self.total_images, world_size)
        self.current_images = 0
        self.feat_dim = int(feat_dim)
        self.output_path = output_path

        self.register_buffer("image_id_queue", torch.zeros((self.queue_length, 1), dtype=torch.int) - 1)
        self.register_buffer("image_feature_queue", torch.zeros((self.queue_length, self.feat_dim)))

    def collect_export_features(self, mlvl_feats, img_meta, **kwargs):
        _, world_size = get_dist_info()
        image_id = image_id_tensor_from_meta(
            img_meta,
            self.image_id_queue.device,
            'RetinaImageFeatureExportHead',
        )
        image_feature = image_feature_from_levels(mlvl_feats)

        collected_img_ids = concat_all_gather(image_id.reshape(1, 1))
        collected_image_features = concat_all_gather(
            image_feature.reshape(1, self.feat_dim).contiguous())
        self.image_id_queue[self.current_images:(self.current_images + world_size)] = collected_img_ids
        self.image_feature_queue[self.current_images:(self.current_images + world_size)] = collected_image_features

    def export_features(self):
        valid_inds = (self.image_id_queue >= 0).reshape(-1)
        image_ids = self.image_id_queue[valid_inds].detach().cpu().numpy()
        image_features = self.image_feature_queue[valid_inds].detach().cpu().numpy()

        keep = unique_valid_indices(image_ids, self.total_images)
        image_ids = image_ids[keep]
        image_features = image_features[keep]

        metadata = dict(
            schema='alod_image_feature_artifact',
            schema_version=1,
            total_images=int(self.total_images),
            exported_images=int(image_ids.shape[0]),
            feat_dim=int(self.feat_dim),
        )
        np.savez_compressed(
            self.output_path,
            image_ids=image_ids.reshape(-1),
            image_features=image_features.astype(np.float32, copy=False),
            metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
        )
        return
