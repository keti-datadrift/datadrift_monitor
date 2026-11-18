import torch

from commands.utils.predict_utils import _resize_boxes_xyxy_tensor


def stats_tensor(values, device):
    if values is None or values.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero, zero, zero
    x = values.detach().float().reshape(-1)
    return torch.min(x), torch.max(x), torch.mean(x), torch.std(x, unbiased=False)


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def selected_probs_from_cache(cache, raw_pred_idx, num_classes, device):
    if 0 <= int(raw_pred_idx) < int(cache.probs.shape[0]) and cache.probs.numel() > 0:
        probs = cache.probs[int(raw_pred_idx)].detach().float()
    else:
        probs = torch.zeros((0,), dtype=torch.float32, device=device)
    out = torch.zeros((max(0, int(num_classes)),), dtype=torch.float32, device=device)
    n = min(int(out.shape[0]), int(probs.shape[0]))
    if n > 0:
        out[:n] = probs[:n].to(device=device)
    return out


def roi_feature_vector_from_cache(cache, raw_idx, class_count, device):
    raw_idx = int(raw_idx)
    if raw_idx < 0 or raw_idx >= int(cache.raw_xyxy.shape[0]):
        return None
    bbox_score = torch.cat([cache.raw_xyxy[raw_idx].detach().float(), cache.scores[raw_idx : raw_idx + 1].detach().float()])
    probs = torch.zeros((int(class_count),), dtype=torch.float32, device=device)
    if cache.probs.numel() > 0:
        raw_probs = cache.probs[raw_idx].detach().float()
        n = min(int(class_count), int(raw_probs.shape[0]))
        if n > 0:
            probs[:n] = raw_probs[:n].to(device=device)
    return torch.cat([bbox_score.to(device=device), probs], dim=0)


def _xyxy_to_shape(box):
    box = box.detach().float().view(4)
    x = 0.5 * (box[0] + box[2])
    y = 0.5 * (box[1] + box[3])
    w = torch.abs(box[2] - box[0])
    h = torch.abs(box[3] - box[1])
    size = w * h
    circum = w + h
    return x, y, w, h, size, circum, size / circum.clamp(min=1e-12)


def zero_shape_diff_features(prefix, device):
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return {
        f"{prefix}_size_diff": zero,
        f"{prefix}_circum_diff": zero,
        f"{prefix}_size_circum_diff": zero,
        f"{prefix}_x_loss": zero,
        f"{prefix}_y_loss": zero,
        f"{prefix}_w_loss": zero,
        f"{prefix}_h_loss": zero,
    }


def shape_diff_features(final_xyxy, reference_xyxy, prefix, device):
    if reference_xyxy is None:
        return zero_shape_diff_features(prefix, device)
    final_x, final_y, final_w, final_h, final_size, final_circum, final_size_circum = _xyxy_to_shape(
        final_xyxy.to(device=device)
    )
    ref_x, ref_y, ref_w, ref_h, ref_size, ref_circum, ref_size_circum = _xyxy_to_shape(
        reference_xyxy.to(device=device)
    )
    return {
        f"{prefix}_size_diff": torch.abs(final_size - ref_size),
        f"{prefix}_circum_diff": torch.abs(final_circum - ref_circum),
        f"{prefix}_size_circum_diff": torch.abs(final_size_circum - ref_size_circum),
        f"{prefix}_x_loss": torch.abs(final_x - ref_x),
        f"{prefix}_y_loss": torch.abs(final_y - ref_y),
        f"{prefix}_w_loss": torch.abs(final_w - ref_w),
        f"{prefix}_h_loss": torch.abs(final_h - ref_h),
    }


def faster_rcnn_null_reference_boxes(
    raw_pred_idx,
    proposal_indices_img,
    proposal_to_rpn_raw_idx,
    proposals_xyxy,
    rpn_anchors,
    from_size,
    to_size,
    device,
):
    raw_pred_idx = int(raw_pred_idx)
    if proposal_indices_img is None or raw_pred_idx < 0 or raw_pred_idx >= int(proposal_indices_img.shape[0]):
        return None, None, None
    proposal_idx = int(proposal_indices_img[raw_pred_idx].detach().cpu().item())
    if proposal_idx < 0 or proposal_idx >= int(proposals_xyxy.shape[0]):
        return None, None, None
    roi_reference = proposals_xyxy[proposal_idx].detach().float().to(device=device)
    if proposal_to_rpn_raw_idx is None or proposal_idx >= int(proposal_to_rpn_raw_idx.shape[0]):
        return None, roi_reference, None
    rpn_raw_idx = int(proposal_to_rpn_raw_idx[proposal_idx].detach().cpu().item())
    if rpn_raw_idx < 0 or rpn_raw_idx >= int(rpn_anchors.shape[0]):
        return None, roi_reference, None
    rpn_reference = _resize_boxes_xyxy_tensor(
        rpn_anchors[rpn_raw_idx].detach().view(1, 4),
        from_size,
        to_size,
    )[0].float().to(device=device)
    return rpn_reference, roi_reference, rpn_raw_idx


def build_meta_feature_values(candidate_boxes, candidate_scores, candidate_ious, final_xyxy, device):
    x = 0.5 * (candidate_boxes[:, 0] + candidate_boxes[:, 2])
    y = 0.5 * (candidate_boxes[:, 1] + candidate_boxes[:, 3])
    w = torch.abs(candidate_boxes[:, 2] - candidate_boxes[:, 0])
    h = torch.abs(candidate_boxes[:, 3] - candidate_boxes[:, 1])
    size_vals = w * h
    circum_vals = w + h
    size_circum_vals = size_vals / circum_vals.clamp(min=1e-12)
    iou_pb = torch.where(candidate_ious == 1.0, torch.zeros_like(candidate_ious), candidate_ious)
    iou_pb_pos = iou_pb[iou_pb > 0]

    fx1, fy1, fx2, fy2 = final_xyxy.detach().float().unbind()
    fw = torch.abs(fx2 - fx1)
    fh = torch.abs(fy2 - fy1)
    fsize = fw * fh
    fcircum = fw + fh
    fsize_circum = fsize / fcircum.clamp(min=1e-12)

    x_min, x_max, x_mean, x_std = stats_tensor(x, device)
    y_min, y_max, y_mean, y_std = stats_tensor(y, device)
    w_min, w_max, w_mean, w_std = stats_tensor(w, device)
    h_min, h_max, h_mean, h_std = stats_tensor(h, device)
    size_min, size_max, size_mean, size_std = stats_tensor(size_vals, device)
    circum_min, circum_max, circum_mean, circum_std = stats_tensor(circum_vals, device)
    size_circum_min, size_circum_max, size_circum_mean, size_circum_std = stats_tensor(size_circum_vals, device)
    score_min, score_max, score_mean, score_std = stats_tensor(candidate_scores, device)
    iou_pb_min, iou_pb_max, iou_pb_mean, iou_pb_std = stats_tensor(iou_pb_pos, device)

    return {
        "num_candidate_boxes": float(candidate_boxes.shape[0]),
        "x_min": x_min,
        "x_max": x_max,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_min": y_min,
        "y_max": y_max,
        "y_mean": y_mean,
        "y_std": y_std,
        "w_min": w_min,
        "w_max": w_max,
        "w_mean": w_mean,
        "w_std": w_std,
        "h_min": h_min,
        "h_max": h_max,
        "h_mean": h_mean,
        "h_std": h_std,
        "size": fsize,
        "size_min": size_min,
        "size_max": size_max,
        "size_mean": size_mean,
        "size_std": size_std,
        "circum": fcircum,
        "circum_min": circum_min,
        "circum_max": circum_max,
        "circum_mean": circum_mean,
        "circum_std": circum_std,
        "size_circum": fsize_circum,
        "size_circum_min": size_circum_min,
        "size_circum_max": size_circum_max,
        "size_circum_mean": size_circum_mean,
        "size_circum_std": size_circum_std,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "score_std": score_std,
        "iou_pb_min": iou_pb_min,
        "iou_pb_max": iou_pb_max,
        "iou_pb_mean": iou_pb_mean,
        "iou_pb_std": iou_pb_std,
    }


def meta_feature_names():
    return [
        "num_candidate_boxes",
        "x_min",
        "x_max",
        "x_mean",
        "x_std",
        "y_min",
        "y_max",
        "y_mean",
        "y_std",
        "w_min",
        "w_max",
        "w_mean",
        "w_std",
        "h_min",
        "h_max",
        "h_mean",
        "h_std",
        "size",
        "size_min",
        "size_max",
        "size_mean",
        "size_std",
        "circum",
        "circum_min",
        "circum_max",
        "circum_mean",
        "circum_std",
        "size_circum",
        "size_circum_min",
        "size_circum_max",
        "size_circum_mean",
        "size_circum_std",
        "score_min",
        "score_max",
        "score_mean",
        "score_std",
        "iou_pb_min",
        "iou_pb_max",
        "iou_pb_mean",
        "iou_pb_std",
    ]


__all__ = [
    "build_meta_feature_values",
    "faster_rcnn_null_reference_boxes",
    "meta_feature_names",
    "roi_feature_vector_from_cache",
    "selected_probs_from_cache",
    "shape_diff_features",
    "stats_tensor",
    "tensor_to_float",
    "zero_shape_diff_features",
]
