from dataclasses import dataclass

import torch

from commands.predict.common import _prepare_infer_batch, _resolve_detector_nms_kwargs
from commands.utils.predict_utils import (
    _concat_rpn_prediction_layers,
    _filter_rpn_proposals_with_indices,
)


@dataclass
class FasterRCNNForwardResult:
    infer_batch: list
    raw_prediction: list
    raw_logits: list | None
    selected_preds: list
    selected_logits: list
    selected_indices: list
    detector_inference_sec: float


@dataclass
class FasterRCNNIntermediateForwardResult:
    infer_batch: list
    image_list: list
    original_image_sizes: list
    transformed_images: object
    features: dict
    rpn_objectness_flat: torch.Tensor
    rpn_bbox_deltas_flat: torch.Tensor
    rpn_anchors: list
    proposals: list
    proposal_to_rpn_raw_indices: list
    proposal_offsets: list
    box_regression: torch.Tensor
    raw_prediction: list
    raw_logits: list | None
    selected_preds: list
    selected_logits: list
    selected_indices: list
    proposal_indices_by_img: list
    labels_internal_by_img: list
    detector_inference_sec: float


def run_faster_rcnn_forward(detector, image_list, device, timing, score_threshold=None):
    infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
    preprocessed = detector.preprocess_images(infer_batch)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    roi_threshold = (
        min(float(getattr(detector, "confidence", getattr(detector, "conf_thresh", 0.25))), float(score_threshold))
        if score_threshold is not None
        else None
    )
    t_detector = timing.start()
    with torch.no_grad():
        with detector.temporary_roi_score_threshold(roi_threshold):
            model_output = detector.forward_preprocessed(preprocessed)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
        selected_preds, selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
            prediction=raw_prediction,
            logits=raw_logits,
            conf_thres=nms_kwargs["conf_thres"],
            iou_thres=nms_kwargs["iou_thres"],
            classes=nms_kwargs["classes"],
            agnostic=nms_kwargs["agnostic"],
            max_det=nms_kwargs["max_det"],
            return_indices=True,
        )
    detector_inference_sec = timing.elapsed(t_detector)
    return FasterRCNNForwardResult(
        infer_batch=infer_batch,
        raw_prediction=raw_prediction,
        raw_logits=raw_logits,
        selected_preds=selected_preds,
        selected_logits=selected_logits,
        selected_indices=selected_indices,
        detector_inference_sec=detector_inference_sec,
    )


def run_faster_rcnn_intermediate_forward(detector, image_list, device, timing):
    infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
    model = detector.detector_model
    was_training = model.training
    model.eval()
    moved_images = [img.to(detector.device, non_blocking=True) if img.device != detector.device else img for img in infer_batch]
    original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in moved_images]
    transformed_images, _targets = model.transform(moved_images, None)
    t_detector = timing.start()
    try:
        with torch.no_grad():
            features = model.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            features_list = list(features.values())
            rpn_objectness_list, rpn_bbox_delta_list = model.rpn.head(features_list)
            num_anchors_per_level = [
                int(obj_per_level.shape[1] * obj_per_level.shape[2] * obj_per_level.shape[3])
                for obj_per_level in rpn_objectness_list
            ]
            rpn_anchors = model.rpn.anchor_generator(transformed_images, features_list)
            rpn_objectness_flat, rpn_bbox_deltas_flat = _concat_rpn_prediction_layers(
                rpn_objectness_list,
                rpn_bbox_delta_list,
            )
            rpn_decoded_for_roi = model.rpn.box_coder.decode(rpn_bbox_deltas_flat.detach(), rpn_anchors).view(
                len(moved_images),
                -1,
                4,
            )
            proposals, _proposal_scores, proposal_to_rpn_raw_indices = _filter_rpn_proposals_with_indices(
                model.rpn,
                rpn_decoded_for_roi,
                rpn_objectness_flat.detach(),
                transformed_images.image_sizes,
                num_anchors_per_level,
            )
            proposal_offsets = []
            running_proposal_offset = 0
            for proposal_img in proposals:
                proposal_offsets.append(running_proposal_offset)
                running_proposal_offset += int(proposal_img.shape[0])
            roi_heads = model.roi_heads
            box_features = roi_heads.box_roi_pool(features, proposals, transformed_images.image_sizes)
            box_features = roi_heads.box_head(box_features)
            class_logits, box_regression = roi_heads.box_predictor(box_features)
            detections = detector._pre_nms_detections_with_logits(
                class_logits=class_logits,
                box_regression=box_regression,
                proposals=proposals,
                image_shapes=transformed_images.image_sizes,
            )
            detections = model.transform.postprocess(detections, transformed_images.image_sizes, original_image_sizes)
            proposal_indices_by_img = []
            labels_internal_by_img = []
            for det_img in detections:
                labels_internal_all = det_img["labels"].to(detector.device)
                _labels_out, valid = detector._map_labels_to_output(labels_internal_all)
                valid &= det_img["scores"].to(detector.device) > 0.0
                proposal_indices_by_img.append(det_img["proposal_indices"].to(detector.device)[valid])
                labels_internal_by_img.append(labels_internal_all[valid])
            raw_prediction, raw_logits = detector._detections_to_contract(detections, detector.device, include_class_features=True)
            nms_kwargs = _resolve_detector_nms_kwargs(detector)
            selected_preds, selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                prediction=raw_prediction,
                logits=raw_logits,
                conf_thres=nms_kwargs["conf_thres"],
                iou_thres=nms_kwargs["iou_thres"],
                classes=nms_kwargs["classes"],
                agnostic=nms_kwargs["agnostic"],
                max_det=nms_kwargs["max_det"],
                return_indices=True,
            )
        detector_inference_sec = timing.elapsed(t_detector)
    finally:
        if was_training:
            model.train()
    return FasterRCNNIntermediateForwardResult(
        infer_batch=infer_batch,
        image_list=moved_images,
        original_image_sizes=original_image_sizes,
        transformed_images=transformed_images,
        features=features,
        rpn_objectness_flat=rpn_objectness_flat,
        rpn_bbox_deltas_flat=rpn_bbox_deltas_flat,
        rpn_anchors=rpn_anchors,
        proposals=proposals,
        proposal_to_rpn_raw_indices=proposal_to_rpn_raw_indices,
        proposal_offsets=proposal_offsets,
        box_regression=box_regression,
        raw_prediction=raw_prediction,
        raw_logits=raw_logits,
        selected_preds=selected_preds,
        selected_logits=selected_logits,
        selected_indices=selected_indices,
        proposal_indices_by_img=proposal_indices_by_img,
        labels_internal_by_img=labels_internal_by_img,
        detector_inference_sec=detector_inference_sec,
    )


__all__ = [
    "FasterRCNNForwardResult",
    "FasterRCNNIntermediateForwardResult",
    "run_faster_rcnn_forward",
    "run_faster_rcnn_intermediate_forward",
]
