from __future__ import absolute_import

from torchvision.ops import nms as torchvision_nms


def nms_gpu(dets, thresh):
    return torchvision_nms(dets[:, :4], dets[:, 4], thresh)
