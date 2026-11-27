import math

import torch


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor):
    return math.ceil(float(x) / divisor) * divisor


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.split([2, 2], dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def bbox_iou(box1, box2, xywh=True, CIoU=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
            + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
        ) / 4
        v = (4 / math.pi**2) * ((w2 / (h2 + eps)).atan() - (w1 / (h1 + eps)).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
    return iou


def v10postprocess_with_indices(preds, max_det, nc):
    boxes, scores = preds.split([4, nc], dim=-1)
    num_preds = preds.shape[1]
    max_det = min(int(max_det), int(num_preds))
    max_scores = scores.amax(dim=-1)
    _max_scores, first_index = torch.topk(max_scores, max_det, dim=-1)
    gathered_boxes = torch.gather(boxes, dim=1, index=first_index.unsqueeze(-1).repeat(1, 1, 4))
    gathered_scores = torch.gather(scores, dim=1, index=first_index.unsqueeze(-1).repeat(1, 1, nc))
    final_scores, second_index = torch.topk(gathered_scores.flatten(1), max_det, dim=-1)
    labels = second_index % nc
    local_index = second_index // nc
    final_boxes = gathered_boxes.gather(dim=1, index=local_index.unsqueeze(-1).repeat(1, 1, 4))
    raw_box_indices = first_index.gather(dim=1, index=local_index)
    raw_indices = raw_box_indices * int(nc) + labels
    return final_boxes, final_scores, labels, raw_indices


__all__ = [
    "autopad",
    "bbox2dist",
    "bbox_iou",
    "dist2bbox",
    "make_anchors",
    "make_divisible",
    "v10postprocess_with_indices",
    "xywh2xyxy",
]
