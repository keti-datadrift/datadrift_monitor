import torch
import torch.nn.functional as F
import numpy as np

def _distributed_ready():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Reference: MoCo v2
    """
    if not _distributed_ready():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def concat_all_sum(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not _distributed_ready():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.stack(tensors_gather, dim=-1).sum(dim=-1)
    return output


def get_inter_feats(lvl_feats, lvl_inds, boxes, img_shape):

    img_w, img_h = img_shape[:2]

    cx = ((0.5 * (boxes[:, 0] + boxes[:, 2]) / img_w) - 0.5) * 2
    cy = ((0.5 * (boxes[:, 1] + boxes[:, 3]) / img_h) - 0.5) * 2

    coor = torch.stack((cx, cy), dim=-1) # [n_det, 2]
    ret_feats = coor.new_full((coor.shape[0], lvl_feats[0].shape[0]), 0.)

    for l in range(len(lvl_feats)):
        mask_l = lvl_inds == l
        if mask_l.sum() == 0:
            continue

        feat_l = lvl_feats[l][None, :, :, :]   # [1, C, H, W]
        coor_l = coor[mask_l][None, None, :, :]  # [1, 1, n_det_lvl, 2]
        inter_feat = F.grid_sample(feat_l, coor_l, mode='bilinear')  # [1, C, 1, n_det_lvl]
        inter_feat = inter_feat.squeeze(dim=0).squeeze(dim=1).transpose(0, 1)  # [n_det_lvl, C]
        ret_feats[mask_l] = inter_feat

    return ret_feats


def bbox2result_with_uncertainty(bboxes, labels, cls_uncertainties, box_uncertainties, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            cls_uncertainties = cls_uncertainties.detach().cpu().numpy()
            box_uncertainties = box_uncertainties.detach().cpu().numpy()
            bboxes = np.concatenate((bboxes, cls_uncertainties.reshape(-1, 1), box_uncertainties.reshape(-1, 1)), axis=1)
        return [bboxes[labels == i, :] for i in range(num_classes)]


def bbox2result_with_pre_nms_count(
        bboxes, labels, pre_nms_counts, num_classes, class_scores=None):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(pre_nms_counts, torch.Tensor):
            pre_nms_counts = pre_nms_counts.detach().cpu().numpy()
        if class_scores is not None and isinstance(class_scores, torch.Tensor):
            class_scores = class_scores.detach().cpu().numpy()
        bboxes = np.asarray(bboxes)
        labels = np.asarray(labels)
        pre_nms_counts = np.asarray(pre_nms_counts)
        if class_scores is not None:
            class_scores = np.asarray(class_scores)
        columns = [bboxes, pre_nms_counts.reshape(-1, 1)]
        if class_scores is not None:
            columns.append(class_scores)
        bboxes = np.concatenate(columns, axis=1)
        return [bboxes[labels == i, :] for i in range(num_classes)]


def bbox2result_with_epal_features(
        bboxes, labels, epal_features, epal_miss_features):
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    converted_features = {}
    for key, value in epal_features.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        converted_features[key] = np.asarray(value)

    if isinstance(epal_miss_features, torch.Tensor):
        epal_miss_features = epal_miss_features.detach().cpu().numpy()

    return dict(
        det_bboxes=np.asarray(bboxes),
        det_labels=np.asarray(labels),
        epal_features=converted_features,
        epal_miss_features=np.asarray(epal_miss_features))
