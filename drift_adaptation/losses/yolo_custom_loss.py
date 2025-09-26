import torch
import torch.nn as nn
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.metrics import bbox_iou


class SampleWiseDetectionLoss:
    """Compute YOLOv8 detection loss and return per-sample loss as well."""

    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.device = device
        self.hyp = h
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.stride = m.stride
        self.use_dfl = m.reg_max > 1

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        batch_size = feats[0].shape[0]

        # Flatten features
        reshaped = [xi.view(batch_size, self.no, -1) for xi in feats]
        concat_feats = torch.cat(reshaped, dim=2)
        pred_distri, pred_scores = concat_feats.split((self.reg_max * 4, self.nc), dim=1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predicted bboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Assign targets
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Normalization factor (batch-level)
        global_norm = target_scores.sum().clamp(min=1.0)
        
        # Raw classification loss per-anchor per-class
        cls_loss_raw = self.bce(pred_scores, target_scores.to(dtype))  # (B, N, C)
        cls_loss_raw = cls_loss_raw.sum(dim=(1,2)) / global_norm * self.hyp.cls

        # Compute per-sample (per-image) losses
        sample_losses = torch.zeros(batch_size, device=self.device)

        # Bbox & DFL loss per image
        for i in range(batch_size):
            if fg_mask[i].sum():
                # call BboxLoss with full anchor_points & masks
                iou_i, dfl_i = self.bbox_loss(
                    pred_distri[i],
                    pred_bboxes[i],
                    anchor_points,
                    (target_bboxes[i] / stride_tensor),
                    target_scores[i],
                    global_norm,
                    fg_mask[i].squeeze(-1)
                )
                # apply gains
                box_i = iou_i * self.hyp.box
                dfl_i = dfl_i * self.hyp.dfl
            else:
                box_i = dfl_i = torch.tensor(0.0, device=self.device)
            sample_losses[i] = box_i + dfl_i + cls_loss_raw[i]        

        return sample_losses.sum(), sample_losses

class BboxLoss(nn.Module):
    """Compute IoU and DFL loss, returning per-anchor values for sample-wise aggregation."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.reg_max = reg_max

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask
    ):
        # 1) IoU loss (CIoU)
        weight = target_scores[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], 
            target_bboxes[fg_mask], 
            xywh=False, 
            CIoU=True
        )
        loss_iou = ((1.0 - iou).unsqueeze(-1) * weight).sum() / target_scores_sum

        # 2) DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, 
                target_bboxes, 
                self.dfl_loss.reg_max - 1
            )

            pd = pred_dist[fg_mask].view(-1, self.reg_max)
            td = target_ltrb[fg_mask]
            loss_dfl_per_anchor = self.dfl_loss(pd, td)
            loss_dfl = (loss_dfl_per_anchor.unsqueeze(-1) * weight).sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.new_tensor(0.0)

        return loss_iou, loss_dfl
