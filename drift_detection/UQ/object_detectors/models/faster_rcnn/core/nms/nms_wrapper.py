import torch
from torchvision.ops import nms as torchvision_nms


def nms(dets, thresh, force_cpu=False):
    """NMS compatible with the original faster-rcnn.pytorch API.

    The vendored reference implementation used custom C/CUDA extensions through
    torch.utils.ffi, which is not available in modern PyTorch. Keep the same
    call shape, but delegate to torchvision's maintained operator.
    """
    if dets.shape[0] == 0:
        return dets.new_empty((0,), dtype=torch.long)

    boxes = dets[:, :4]
    scores = dets[:, 4]
    if force_cpu:
        keep = torchvision_nms(boxes.cpu(), scores.cpu(), thresh)
        return keep.to(dets.device)
    return torchvision_nms(boxes, scores, thresh)
