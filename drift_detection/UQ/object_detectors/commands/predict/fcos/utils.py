from dataclasses import dataclass

from commands.predict.common import *
from commands.predict.fcos.common import select_fcos_post_nms, unpack_fcos_model_output


@dataclass
class FcosForwardNMSResult:
    infer_batch: object
    ratios: list
    pads: list
    resized_chws: object
    processed_images: list
    raw_prediction: list
    raw_logits: object
    raw_indices: object
    selected_preds: list
    selected_logits: list
    selected_indices: list
    pre_nms_prediction: object
    detector_inference_sec: float


@dataclass
class FcosDetectionRow:
    sample_idx: int
    pred_idx: int
    raw_pred_idx: int
    box: torch.Tensor
    score: float
    cls_idx: int
    base: dict


@dataclass
class FcosCandidateCache:
    prediction: torch.Tensor
    boxes_xyxy: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor
    score_threshold: float
    levels: torch.Tensor = None
    location_indices: torch.Tensor = None
    class_indices: torch.Tensor = None


def fcos_class_name(detector, cls_idx):
    cls_idx = int(cls_idx)
    if isinstance(detector.names, dict):
        return detector.names.get(cls_idx, str(cls_idx))
    if isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
        return detector.names[cls_idx]
    return str(cls_idx)


def run_fcos_forward_nms(
    *,
    detector,
    image_list,
    device,
    timing,
    keep_pre_nms=False,
    keep_class_outputs=True,
):
    infer_batch, ratios, pads, resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
    with torch.no_grad():
        processed_images = detector.preprocess_images(infer_batch)
        t_detector = timing.start()
        model_output = detector.forward_preprocessed(
            processed_images,
            keep_pre_nms=keep_pre_nms,
            keep_class_outputs=keep_class_outputs,
        )
        raw_prediction, raw_logits, raw_indices = unpack_fcos_model_output(model_output)
        pre_nms_prediction = None
        if keep_pre_nms:
            pre_nms_prediction, _pre_nms_logits, _pre_nms_indices = detector.get_last_pre_nms_predictions()
        selected = select_fcos_post_nms(
            detector,
            raw_prediction,
            raw_logits,
            raw_indices,
        )
        selected_preds = selected[0]
        selected_logits = selected[1]
        selected_indices = selected[3]
    detector_inference_sec = timing.elapsed(t_detector)
    return FcosForwardNMSResult(
        infer_batch=infer_batch,
        ratios=ratios,
        pads=pads,
        resized_chws=resized_chws,
        processed_images=processed_images,
        raw_prediction=raw_prediction,
        raw_logits=raw_logits,
        raw_indices=raw_indices,
        selected_preds=selected_preds,
        selected_logits=selected_logits,
        selected_indices=selected_indices,
        pre_nms_prediction=pre_nms_prediction,
        detector_inference_sec=detector_inference_sec,
    )


def ensure_fcos_selected_indices(selected_indices, selected_preds, sample_idx):
    det = (
        selected_preds[sample_idx]
        if selected_preds and sample_idx < len(selected_preds)
        else torch.zeros((0, 6))
    )
    raw_keep = (
        selected_indices[sample_idx]
        if selected_indices and sample_idx < len(selected_indices)
        else torch.zeros((0,), dtype=torch.long, device=det.device)
    )
    if int(raw_keep.shape[0]) < int(det.shape[0]):
        raise RuntimeError(
            f"FCOS selected_indices length mismatch: image_idx={sample_idx}, "
            f"num_detections={int(det.shape[0])}, num_indices={int(raw_keep.shape[0])}"
        )
    return raw_keep


def iter_fcos_detection_rows(detector, targets, selected_preds, selected_indices, device):
    for sample_idx in range(len(targets)):
        target = targets[sample_idx]
        image_id = int(target["image_id"][0].item())
        image_path = target["path"]
        det = (
            selected_preds[sample_idx]
            if selected_preds and sample_idx < len(selected_preds)
            else torch.zeros((0, 6), dtype=torch.float32, device=device)
        )
        raw_keep = ensure_fcos_selected_indices(selected_indices, selected_preds, sample_idx).to(device=device)
        for pred_idx in range(int(det.shape[0])):
            raw_pred_idx = int(raw_keep[pred_idx].detach().cpu().item())
            if raw_pred_idx < 0:
                raise RuntimeError(
                    f"FCOS raw_pred_idx is negative: image_idx={sample_idx}, pred_idx={pred_idx}, raw_pred_idx={raw_pred_idx}"
                )
            box = det[pred_idx]
            cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
            base = {
                "image_id": image_id,
                "image_path": image_path,
                "pred_idx": pred_idx,
                "raw_pred_idx": raw_pred_idx,
                "xmin": float(box[0].detach().cpu().item()),
                "ymin": float(box[1].detach().cpu().item()),
                "xmax": float(box[2].detach().cpu().item()),
                "ymax": float(box[3].detach().cpu().item()),
                "score": float(box[4].detach().cpu().item()),
                "pred_class": fcos_class_name(detector, cls_idx),
            }
            yield FcosDetectionRow(
                sample_idx=sample_idx,
                pred_idx=pred_idx,
                raw_pred_idx=raw_pred_idx,
                box=box,
                score=float(box[4].detach().cpu().item()),
                cls_idx=cls_idx,
                base=base,
            )


def selected_fcos_class_logits(result, sample_idx, num_classes, device):
    det = (
        result.selected_preds[sample_idx]
        if result.selected_preds and sample_idx < len(result.selected_preds)
        else torch.zeros((0, 6), dtype=torch.float32, device=device)
    )
    if result.selected_logits and sample_idx < len(result.selected_logits):
        logits = result.selected_logits[sample_idx]
        if logits is not None and int(logits.shape[0]) == int(det.shape[0]) and int(logits.shape[-1]) > 0:
            if int(logits.shape[-1]) == int(num_classes):
                return logits.to(device=device, dtype=torch.float32)
            if int(logits.shape[-1]) > int(num_classes):
                return logits[:, :num_classes].to(device=device, dtype=torch.float32)
            return torch.nn.functional.pad(
                logits.to(device=device, dtype=torch.float32),
                (0, int(num_classes) - int(logits.shape[-1])),
            )
    if int(det.shape[0]) == 0:
        return torch.zeros((0, int(num_classes)), dtype=torch.float32, device=device)
    raise RuntimeError(
        f"FCOS selected class logits are unavailable or misaligned: "
        f"image_idx={sample_idx}, num_detections={int(det.shape[0])}"
    )


def selected_fcos_class_probs(result, sample_idx, num_classes, device):
    return torch.sigmoid(selected_fcos_class_logits(result, sample_idx, num_classes, device))


def build_fcos_candidate_cache(candidate_prediction, score_threshold):
    if candidate_prediction is None or candidate_prediction.numel() == 0:
        device = candidate_prediction.device if isinstance(candidate_prediction, torch.Tensor) else torch.device("cpu")
        empty = torch.zeros((0,), dtype=torch.float32, device=device)
        return FcosCandidateCache(
            prediction=torch.zeros((0, 6), dtype=torch.float32, device=device),
            boxes_xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
            scores=empty,
            classes=torch.zeros((0,), dtype=torch.long, device=device),
            score_threshold=float(score_threshold),
            levels=torch.zeros((0,), dtype=torch.long, device=device),
            location_indices=torch.zeros((0,), dtype=torch.long, device=device),
            class_indices=torch.zeros((0,), dtype=torch.long, device=device),
        )
    pred = candidate_prediction.detach().float()
    boxes_xyxy = _xywh_to_xyxy_tensor(pred[:, :4])
    scores = pred[:, 4].detach().float()
    classes = pred[:, 5].detach().long() if pred.shape[1] > 5 else torch.zeros_like(scores, dtype=torch.long)
    return FcosCandidateCache(
        prediction=pred,
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        classes=classes,
        score_threshold=float(score_threshold),
        levels=torch.zeros((pred.shape[0],), dtype=torch.long, device=pred.device),
        location_indices=torch.arange(pred.shape[0], dtype=torch.long, device=pred.device),
        class_indices=classes,
    )


def _flatten_fcos_level_output(tensor, image_idx):
    if tensor.ndim == 4:
        return tensor[image_idx].permute(1, 2, 0).reshape(-1, tensor.shape[1])
    return tensor


def _flatten_fcos_centerness(tensor, image_idx):
    if tensor.ndim == 4:
        return tensor[image_idx].permute(1, 2, 0).reshape(-1)
    return tensor.reshape(-1)


def _build_fcos_dense_candidate_cache_impl(model_output, image_idx, score_threshold, detach=True):
    box_cls = model_output.get("box_cls")
    box_regression = model_output.get("box_regression")
    centerness = model_output.get("centerness")
    locations = model_output.get("locations")
    if not box_cls or not box_regression or not centerness or not locations:
        device = torch.device("cpu")
        empty = torch.zeros((0,), dtype=torch.float32, device=device)
        return FcosCandidateCache(
            prediction=torch.zeros((0, 6), dtype=torch.float32, device=device),
            boxes_xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
            scores=empty,
            classes=torch.zeros((0,), dtype=torch.long, device=device),
            score_threshold=float(score_threshold),
            levels=torch.zeros((0,), dtype=torch.long, device=device),
            location_indices=torch.zeros((0,), dtype=torch.long, device=device),
            class_indices=torch.zeros((0,), dtype=torch.long, device=device),
        )

    boxes_by_level = []
    scores_by_level = []
    classes_by_level = []
    levels_by_level = []
    loc_indices_by_level = []
    class_indices_by_level = []
    predictions_by_level = []
    for level, (cls_level, reg_level, cnt_level, loc_level) in enumerate(zip(box_cls, box_regression, centerness, locations)):
        if detach:
            cls_level = cls_level.detach()
            reg_level = reg_level.detach()
            cnt_level = cnt_level.detach()
            loc_level = loc_level.detach()
        cls_logits = _flatten_fcos_level_output(cls_level, image_idx)
        reg = _flatten_fcos_level_output(reg_level, image_idx)
        cnt_logits = _flatten_fcos_centerness(cnt_level, image_idx)
        loc = loc_level.to(device=reg.device, dtype=reg.dtype)
        if detach:
            cls_logits = cls_logits.detach()
            reg = reg.detach()
            cnt_logits = cnt_logits.detach()
            loc = loc.detach()
        num_locations = int(loc.shape[0])
        num_classes = int(cls_logits.shape[1]) if cls_logits.ndim == 2 else 0
        if num_locations == 0 or num_classes == 0:
            continue

        boxes = torch.stack(
            [
                loc[:, 0] - reg[:, 0],
                loc[:, 1] - reg[:, 1],
                loc[:, 0] + reg[:, 2],
                loc[:, 1] + reg[:, 3],
            ],
            dim=1,
        )
        cls_probs = cls_logits.sigmoid()
        cnt_probs = cnt_logits.sigmoid().view(-1, 1)
        scores = torch.sqrt((cls_probs * cnt_probs).clamp(min=0.0))
        loc_idx = torch.arange(num_locations, dtype=torch.long, device=reg.device)
        cls_idx = torch.arange(num_classes, dtype=torch.long, device=reg.device)
        flat_scores = scores.reshape(-1)
        flat_classes = cls_idx.view(1, -1).expand(num_locations, num_classes).reshape(-1)
        flat_loc_idx = loc_idx.view(-1, 1).expand(num_locations, num_classes).reshape(-1)
        flat_boxes = boxes[:, None, :].expand(num_locations, num_classes, 4).reshape(-1, 4)
        flat_levels = torch.full_like(flat_loc_idx, int(level))
        xywh = flat_boxes.clone()
        xywh[:, 0] = (flat_boxes[:, 0] + flat_boxes[:, 2]) * 0.5
        xywh[:, 1] = (flat_boxes[:, 1] + flat_boxes[:, 3]) * 0.5
        xywh[:, 2] = (flat_boxes[:, 2] - flat_boxes[:, 0]).clamp(min=0.0)
        xywh[:, 3] = (flat_boxes[:, 3] - flat_boxes[:, 1]).clamp(min=0.0)
        prediction = torch.cat([xywh, flat_scores[:, None], flat_classes.to(xywh.dtype)[:, None]], dim=1)

        boxes_by_level.append(flat_boxes)
        scores_by_level.append(flat_scores)
        classes_by_level.append(flat_classes)
        levels_by_level.append(flat_levels)
        loc_indices_by_level.append(flat_loc_idx)
        class_indices_by_level.append(flat_classes)
        predictions_by_level.append(prediction)

    if not boxes_by_level:
        device = box_cls[0].device
        empty = torch.zeros((0,), dtype=torch.float32, device=device)
        return FcosCandidateCache(
            prediction=torch.zeros((0, 6), dtype=torch.float32, device=device),
            boxes_xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
            scores=empty,
            classes=torch.zeros((0,), dtype=torch.long, device=device),
            score_threshold=float(score_threshold),
            levels=torch.zeros((0,), dtype=torch.long, device=device),
            location_indices=torch.zeros((0,), dtype=torch.long, device=device),
            class_indices=torch.zeros((0,), dtype=torch.long, device=device),
        )

    prediction = torch.cat(predictions_by_level, dim=0)
    boxes_xyxy = torch.cat(boxes_by_level, dim=0)
    scores = torch.cat(scores_by_level, dim=0)
    classes = torch.cat(classes_by_level, dim=0)
    levels = torch.cat(levels_by_level, dim=0)
    location_indices = torch.cat(loc_indices_by_level, dim=0)
    class_indices = torch.cat(class_indices_by_level, dim=0)
    if detach:
        prediction = prediction.detach()
        boxes_xyxy = boxes_xyxy.detach()
        scores = scores.detach()
        classes = classes.detach()
        levels = levels.detach()
        location_indices = location_indices.detach()
        class_indices = class_indices.detach()
    return FcosCandidateCache(
        prediction=prediction.float(),
        boxes_xyxy=boxes_xyxy.float(),
        scores=scores.float(),
        classes=classes.long(),
        score_threshold=float(score_threshold),
        levels=levels.long(),
        location_indices=location_indices.long(),
        class_indices=class_indices.long(),
    )


def build_fcos_dense_candidate_cache(model_output, image_idx, score_threshold, detach=True):
    if detach:
        with torch.no_grad():
            return _build_fcos_dense_candidate_cache_impl(
                model_output,
                image_idx,
                score_threshold,
                detach=True,
            )
    return _build_fcos_dense_candidate_cache_impl(
        model_output,
        image_idx,
        score_threshold,
        detach=False,
    )


def fcos_candidate_mask_from_cache(cache, final_box_xyxy, final_cls, iou_threshold):
    final_cls = int(final_cls)
    candidate_count = int(cache.boxes_xyxy.shape[0])
    mask = torch.zeros((candidate_count,), dtype=torch.bool, device=cache.boxes_xyxy.device)
    ious = torch.zeros((candidate_count,), dtype=cache.boxes_xyxy.dtype, device=cache.boxes_xyxy.device)
    if candidate_count == 0:
        return mask, ious
    score_class_mask = (cache.scores >= float(cache.score_threshold)) & (cache.classes == final_cls)
    if bool(score_class_mask.any()):
        candidate_indices = torch.nonzero(score_class_mask, as_tuple=False).flatten()
        candidate_ious = _box_iou_1vN_tensor(final_box_xyxy.view(1, 4), cache.boxes_xyxy[candidate_indices])
        ious[candidate_indices] = candidate_ious
        mask[candidate_indices] = candidate_ious > float(iou_threshold)
    return mask, ious


__all__ = [
    "FcosForwardNMSResult",
    "FcosDetectionRow",
    "FcosCandidateCache",
    "run_fcos_forward_nms",
    "iter_fcos_detection_rows",
    "ensure_fcos_selected_indices",
    "selected_fcos_class_logits",
    "selected_fcos_class_probs",
    "build_fcos_candidate_cache",
    "build_fcos_dense_candidate_cache",
    "fcos_candidate_mask_from_cache",
    "fcos_class_name",
]
