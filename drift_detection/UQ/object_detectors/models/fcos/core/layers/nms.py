
import torch
from torchvision.ops import batched_nms, nms as _torchvision_nms


def nms(boxes, scores, nms_thresh):
    return _torchvision_nms(boxes, scores, nms_thresh)


def ml_nms(boxes, scores, labels, nms_thresh):
    return batched_nms(boxes, scores, labels.to(dtype=torch.long), nms_thresh)
