from dataclasses import dataclass

import torch


@dataclass
class YoloV10ForwardResult:
    model_output: object
    raw_levels: object
    decoded_prediction: torch.Tensor
    raw_logits: torch.Tensor
    selected_preds: list
    selected_indices: list
    source_points: torch.Tensor
    detector_inference_sec: float


def run_yolov10_forward(detector, infer_batch=None, timing=None, grad=False, feature_cache=None, source_points=None):
    t_detector = timing.start() if timing is not None else None
    output = (
        detector.forward_layer_grad(infer_batch, source_points=source_points)
        if grad
        else detector.forward_nms_free(infer_batch, feature_cache=feature_cache, source_points=source_points)
    )
    detector_inference_sec = timing.elapsed(t_detector) if timing is not None else 0.0
    return YoloV10ForwardResult(
        model_output=output["model_output"],
        raw_levels=output["raw_levels"],
        decoded_prediction=output["decoded_prediction"],
        raw_logits=output["raw_logits"],
        selected_preds=output["selected_preds"],
        selected_indices=output["selected_indices"],
        source_points=output["source_points"],
        detector_inference_sec=detector_inference_sec,
    )


def run_yolov10_raw_forward(detector, infer_batch=None, timing=None, feature_cache=None, source_points=None):
    t_detector = timing.start() if timing is not None else None
    output = detector.forward_raw_decoded(infer_batch, feature_cache=feature_cache, source_points=source_points)
    detector_inference_sec = timing.elapsed(t_detector) if timing is not None else 0.0
    return YoloV10ForwardResult(
        model_output=output["model_output"],
        raw_levels=output["raw_levels"],
        decoded_prediction=output["decoded_prediction"],
        raw_logits=output["raw_logits"],
        selected_preds=[],
        selected_indices=[],
        source_points=output["source_points"],
        detector_inference_sec=detector_inference_sec,
    )


__all__ = ["YoloV10ForwardResult", "run_yolov10_forward", "run_yolov10_raw_forward"]
