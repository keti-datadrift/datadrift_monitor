from dataclasses import dataclass

import torch

from commands.predict.common import _resolve_nms_logits


@dataclass
class YoloForwardNMSResult:
    model_output: object
    raw_prediction: torch.Tensor
    raw_logits: torch.Tensor
    raw_layers: object
    raw_anchor_priors: torch.Tensor
    selected_preds: list
    selected_logits: list
    selected_objectness: list
    selected_indices: list
    detector_inference_sec: float


def run_yolo_forward_nms(detector, infer_batch, nms_kwargs, timing=None, num_classes_hint=80):
    t_detector = timing.start() if timing is not None else None
    model_output = detector.model(infer_batch, augment=False)
    raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
    raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
    raw_layers = model_output[2] if isinstance(model_output, (tuple, list)) and len(model_output) > 2 else None
    raw_anchor_priors = model_output[3] if isinstance(model_output, (tuple, list)) and len(model_output) > 3 else None
    nms_logits = _resolve_nms_logits(raw_prediction, raw_logits, num_classes_hint=num_classes_hint)
    selected_preds, selected_logits, selected_objectness, selected_indices = detector.non_max_suppression(
        prediction=raw_prediction,
        logits=nms_logits,
        conf_thres=nms_kwargs["conf_thres"],
        iou_thres=nms_kwargs["iou_thres"],
        classes=nms_kwargs["classes"],
        agnostic=nms_kwargs["agnostic"],
        max_det=nms_kwargs["max_det"],
        return_indices=True,
    )
    detector_inference_sec = timing.elapsed(t_detector) if timing is not None else 0.0
    return YoloForwardNMSResult(
        model_output=model_output,
        raw_prediction=raw_prediction,
        raw_logits=raw_logits,
        raw_layers=raw_layers,
        raw_anchor_priors=raw_anchor_priors,
        selected_preds=selected_preds,
        selected_logits=selected_logits,
        selected_objectness=selected_objectness,
        selected_indices=selected_indices,
        detector_inference_sec=detector_inference_sec,
    )


def yolo_pred_class_name(detector, cls_idx):
    if isinstance(detector.names, dict):
        return detector.names.get(cls_idx, str(cls_idx))
    if isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
        return detector.names[cls_idx]
    return str(cls_idx)


def iter_yolo_detection_rows(detector, targets, selected_preds, selected_indices, device):
    for sample_idx in range(len(targets)):
        target = targets[sample_idx]
        image_id = int(target["image_id"][0].item())
        image_path = target["path"]
        det = (
            selected_preds[sample_idx]
            if selected_preds and sample_idx < len(selected_preds)
            else torch.zeros((0, 6), device=device)
        )
        raw_keep = (
            selected_indices[sample_idx]
            if selected_indices and sample_idx < len(selected_indices)
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        for pred_idx, box in enumerate(det):
            if pred_idx >= int(raw_keep.shape[0]):
                raise RuntimeError(
                    "YOLO selected_indices is shorter than selected predictions. "
                    f"sample_idx={sample_idx}, pred_idx={pred_idx}, indices={int(raw_keep.shape[0])}"
                )
            raw_pred_idx = int(raw_keep[pred_idx].detach().cpu().item())
            cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
            yield {
                "sample_idx": sample_idx,
                "image_id": image_id,
                "image_path": image_path,
                "pred_idx": pred_idx,
                "raw_pred_idx": raw_pred_idx,
                "box": box,
                "base_row": {
                    "image_id": image_id,
                    "image_path": image_path,
                    "pred_idx": pred_idx,
                    "raw_pred_idx": raw_pred_idx,
                    "xmin": float(box[0].detach().cpu().item()),
                    "ymin": float(box[1].detach().cpu().item()),
                    "xmax": float(box[2].detach().cpu().item()),
                    "ymax": float(box[3].detach().cpu().item()),
                    "score": float(box[4].detach().cpu().item()),
                    "pred_class": yolo_pred_class_name(detector, cls_idx),
                },
            }


def selected_sigmoid_class_outputs(raw_prediction, sample_idx, raw_keep, num_classes, device):
    if int(raw_keep.shape[0]) <= 0 or raw_prediction[sample_idx].shape[1] <= 5:
        return torch.zeros((0, num_classes), dtype=torch.float32, device=device)
    return raw_prediction[sample_idx][raw_keep, 5:].detach().float()


def selected_class_logits(raw_logits, raw_prediction, sample_idx, raw_keep, num_classes, device):
    if raw_logits is not None:
        if int(raw_keep.shape[0]) <= 0:
            return torch.zeros((0, num_classes), dtype=torch.float32, device=device)
        return raw_logits[sample_idx][raw_keep].detach().float()
    probs = selected_sigmoid_class_outputs(raw_prediction, sample_idx, raw_keep, num_classes, device)
    return torch.logit(probs.clamp(min=1e-8, max=1.0 - 1e-8)) if probs.numel() else probs


def selected_softmax_class_probs(raw_logits, raw_prediction, sample_idx, raw_keep, num_classes, device):
    logits = selected_class_logits(raw_logits, raw_prediction, sample_idx, raw_keep, num_classes, device)
    return torch.softmax(logits, dim=-1) if logits.numel() else logits
