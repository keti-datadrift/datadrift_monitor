import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import images_to_levels, multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_head import RetinaHead


@HEADS.register_module()
class RetinaHeadMIAL(RetinaHead):
    """RetinaNet head for MIAL/MI-AOD style active learning.

    The head follows the MI-AOD RetinaNet structure with two classifier
    branches, one box regression branch, and one MIL branch. Training phase is
    controlled explicitly by ``set_mial_phase`` from
    ``tools/internal/train_mial_detector.py``.
    """

    def __init__(self, mial_lambda=0.5, **kwargs):
        self.mial_lambda = float(mial_lambda)
        self.mial_phase = 'det'
        self.mial_unlabeled = False
        super(RetinaHeadMIAL, self).__init__(**kwargs)
        self.loss_imgcls = nn.BCELoss()

    def set_mial_phase(self, phase, unlabeled=False):
        if phase not in ('det', 'min', 'max'):
            raise ValueError('Unsupported MIAL phase: %s' % phase)
        self.mial_phase = phase
        self.mial_unlabeled = bool(unlabeled)

    def _init_layers(self):
        self.f_1_convs = nn.ModuleList()
        self.f_2_convs = nn.ModuleList()
        self.f_r_convs = nn.ModuleList()
        self.f_mil_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.f_1_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.f_2_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.f_r_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.f_mil_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.f_1_retina = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.f_2_retina = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.f_r_retina = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            3,
            padding=1)
        self.f_mil_retina = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)

    def init_weights(self):
        for convs in (
                self.f_1_convs, self.f_2_convs, self.f_r_convs,
                self.f_mil_convs):
            for module in convs:
                normal_init(module.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.f_1_retina, std=0.01, bias=bias_cls)
        normal_init(self.f_2_retina, std=0.01, bias=bias_cls)
        normal_init(self.f_mil_retina, std=0.01, bias=bias_cls)
        normal_init(self.f_r_retina, std=0.01)

    def forward_single(self, x):
        f_1_feat = x
        f_2_feat = x
        f_r_feat = x
        f_mil_feat = x
        for conv in self.f_1_convs:
            f_1_feat = conv(f_1_feat)
        for conv in self.f_2_convs:
            f_2_feat = conv(f_2_feat)
        for conv in self.f_r_convs:
            f_r_feat = conv(f_r_feat)
        for conv in self.f_mil_convs:
            f_mil_feat = conv(f_mil_feat)

        y_f_1 = self.f_1_retina(f_1_feat)
        y_f_2 = self.f_2_retina(f_2_feat)
        y_f_r = self.f_r_retina(f_r_feat)
        y_f_mil = self.f_mil_retina(f_mil_feat)
        y_cls_term = ((y_f_1 + y_f_2) / 2).detach()
        y_f_mil = y_f_mil.permute(0, 2, 3, 1).reshape(
            y_f_1.shape[0], -1, self.cls_out_channels)
        y_cls_term = y_cls_term.permute(0, 2, 3, 1).reshape(
            y_f_1.shape[0], -1, self.cls_out_channels)
        y_head_cls = (
            F.softmax(y_f_mil, dim=2)
            * F.softmax(y_cls_term.sigmoid().max(2, keepdim=True)[0], dim=1)
        )
        return y_f_1, y_f_2, y_f_r, y_head_cls

    def _loss_single(self, cls_score, bbox_pred, anchors, labels,
                     label_weights, bbox_targets, bbox_weights,
                     num_total_samples):
        return self.loss_single(
            cls_score,
            bbox_pred,
            anchors,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_samples)

    def _target_inputs(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                       img_metas, gt_bboxes_ignore):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for anchors in anchor_list:
            concat_anchor_list.append(torch.cat(anchors))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        return (all_anchor_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_samples)

    def _det_losses_for_branch(self, cls_scores, bbox_preds, gt_bboxes,
                               gt_labels, img_metas, gt_bboxes_ignore=None):
        target_inputs = self._target_inputs(
            cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore)
        if target_inputs is None:
            return None
        (all_anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_samples) = target_inputs
        losses_cls, losses_bbox = multi_apply(
            self._loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return losses_cls, losses_bbox

    def _zero_loss(self, reference):
        return reference.sum() * 0.0

    def _image_label_loss(self, mil_scores, gt_labels):
        if not gt_labels:
            return self._zero_loss(mil_scores[0])
        image_scores = mil_scores[0].new_zeros(
            (len(gt_labels), self.cls_out_channels))
        image_targets = mil_scores[0].new_zeros(
            (len(gt_labels), self.cls_out_channels))
        for img_idx, labels in enumerate(gt_labels):
            if labels.numel() > 0:
                image_targets[img_idx, labels.long()] = 1.0
        for level_scores in mil_scores:
            image_scores = torch.max(image_scores, level_scores.sum(1))
        image_scores = image_scores.clamp(1e-5, 1.0 - 1e-5)
        return self.loss_imgcls(image_scores, image_targets)

    def _pseudo_image_label_loss(self, cls_scores_pair, mil_scores):
        batch_size = mil_scores[0].shape[0]
        image_scores = mil_scores[0].new_zeros(
            (batch_size, self.cls_out_channels))
        pseudo = mil_scores[0].new_zeros((batch_size, self.cls_out_channels))
        with torch.no_grad():
            for scores_1, scores_2 in zip(cls_scores_pair[0],
                                          cls_scores_pair[1]):
                y_f_i = (
                    scores_1.permute(0, 2, 3, 1).reshape(
                        batch_size, -1, self.cls_out_channels).sigmoid()
                    + scores_2.permute(0, 2, 3, 1).reshape(
                        batch_size, -1, self.cls_out_channels).sigmoid()
                ) / 2
                pseudo = torch.max(pseudo, y_f_i.max(1)[0])
            pseudo = (pseudo >= 0.5).to(dtype=image_scores.dtype)
        for level_scores in mil_scores:
            image_scores = torch.max(image_scores, level_scores.sum(1))
        image_scores = image_scores.clamp(1e-5, 1.0 - 1e-5)
        if (pseudo.sum(1) == 0).any():
            return self._zero_loss(image_scores)
        return self.loss_imgcls(image_scores, pseudo.detach())

    def _wave_loss_single(self, scores_1, scores_2, mil_scores, maximize=False):
        scores_1 = scores_1.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).sigmoid()
        scores_2 = scores_2.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).sigmoid()
        weights = mil_scores.detach().reshape(-1, self.cls_out_channels)
        discrepancy = (scores_1 - scores_2).abs()
        if maximize:
            discrepancy = 1.0 - discrepancy
        return (discrepancy * weights).mean(dim=1).sum() * self.mial_lambda

    def _wave_losses(self, cls_scores_pair, mil_scores, maximize=False):
        return [
            self._wave_loss_single(scores_1, scores_2, level_mil, maximize)
            for scores_1, scores_2, level_mil in zip(
                cls_scores_pair[0], cls_scores_pair[1], mil_scores)
        ]

    @force_fp32(apply_to=('cls_scores_1', 'cls_scores_2', 'bbox_preds'))
    def loss(self,
             cls_scores_1,
             cls_scores_2,
             bbox_preds,
             mil_scores,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        if self.mial_phase == 'det':
            branch_1 = self._det_losses_for_branch(
                cls_scores_1, bbox_preds, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore)
            branch_2 = self._det_losses_for_branch(
                cls_scores_2, bbox_preds, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore)
            if branch_1 is None or branch_2 is None:
                return None
            losses_cls = [
                (left + right) / 2 for left, right in zip(branch_1[0], branch_2[0])
            ]
            losses_bbox = [
                (left + right) / 2 for left, right in zip(branch_1[1], branch_2[1])
            ]
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_mial_imgcls=[self._image_label_loss(mil_scores, gt_labels)])

        if self.mial_unlabeled:
            pseudo_loss = self._pseudo_image_label_loss(
                (cls_scores_1, cls_scores_2), mil_scores)
            if self.mial_phase == 'min':
                return dict(
                    loss_mial_wave_min=self._wave_losses(
                        (cls_scores_1, cls_scores_2), mil_scores,
                        maximize=False),
                    loss_mial_imgcls=[pseudo_loss])
            return dict(
                loss_mial_wave_max=self._wave_losses(
                    (cls_scores_1, cls_scores_2), mil_scores,
                    maximize=True))

        branch_1 = self._det_losses_for_branch(
            cls_scores_1, bbox_preds, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore)
        branch_2 = self._det_losses_for_branch(
            cls_scores_2, bbox_preds, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore)
        if branch_1 is None or branch_2 is None:
            return None
        losses_cls = [
            (left + right) / 2 for left, right in zip(branch_1[0], branch_2[0])
        ]
        losses_bbox = [
            (left + right) / 2 for left, right in zip(branch_1[1], branch_2[1])
        ]
        if self.mial_phase == 'max':
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox)
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_mial_imgcls=[self._image_label_loss(mil_scores, gt_labels)])

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        proposal_list = self.get_bboxes(
            outs[0], outs[2], img_metas=img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        outs = self(feats)
        return self.get_bboxes(outs[0], outs[2], img_metas=img_metas,
                               rescale=rescale)

    def image_uncertainty(self, feats, topk=10000):
        cls_scores_1, cls_scores_2, _, _ = self(feats)
        batch_size = cls_scores_1[0].shape[0]
        records = []
        for img_idx in range(batch_size):
            per_level = []
            for scores_1, scores_2 in zip(cls_scores_1, cls_scores_2):
                s1 = scores_1[img_idx].permute(1, 2, 0).reshape(
                    -1, self.cls_out_channels).sigmoid()
                s2 = scores_2[img_idx].permute(1, 2, 0).reshape(
                    -1, self.cls_out_channels).sigmoid()
                per_level.append((s1 - s2).pow(2).mean(dim=1))
            instance_scores = torch.cat(per_level, dim=0)
            effective_topk = min(int(topk), int(instance_scores.numel()))
            if effective_topk <= 0:
                score = instance_scores.new_tensor(0.0)
            else:
                score = instance_scores.topk(effective_topk).values.mean()
            records.append(dict(
                score=score,
                instance_count=int(instance_scores.numel()),
                effective_topk=effective_topk))
        return records
