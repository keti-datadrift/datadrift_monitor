from pathlib import Path
from urllib.parse import urlparse
import shutil
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import batched_nms, boxes as box_ops


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTER_RCNN_COCO_WEIGHT = (
    PROJECT_ROOT / "models" / "faster_rcnn" / "weights" / "coco" / "fasterrcnn_resnet50_fpn_coco.pth"
)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_default_coco_weight(path):
    if path is None:
        return True
    return Path(path).name == DEFAULT_FASTER_RCNN_COCO_WEIGHT.name


def _seed_torchvision_cache_from_local(local_path):
    local_path = Path(local_path)
    if not local_path.is_file():
        return
    filename = Path(urlparse(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url).path).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if cache_path.is_file():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, cache_path)


def _real_coco_categories():
    categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
    return [name for name in categories if name not in {"__background__", "N/A"}]


def _load_checkpoint_state(model, checkpoint_path, device):
    payload = _torch_load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], nn.Module):
        state_dict = payload["model"].state_dict()
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported Faster R-CNN checkpoint payload: {checkpoint_path}")

    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing = len(current) - len(filtered)
    if missing:
        print(f"[WARN] Faster R-CNN checkpoint partial load: loaded={len(filtered)}, skipped={missing}")
    model.load_state_dict(filtered, strict=False)


class _DropoutFastRCNNPredictor(nn.Module):
    def __init__(self, base_predictor, dropout_rate=0.0):
        super().__init__()
        in_features = base_predictor.cls_score.in_features
        num_classes = base_predictor.cls_score.out_features
        self.dropout = float(dropout_rate)
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)
        self.cls_score.load_state_dict(base_predictor.cls_score.state_dict())
        self.bbox_pred.load_state_dict(base_predictor.bbox_pred.state_dict())

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=True)
        return self.cls_score(x), self.bbox_pred(x)


class _ForwardProxy:
    def __init__(self, owner):
        self._owner = owner

    def __call__(self, images, augment=False, need_logits=True, **_kwargs):
        return self._owner._forward_impl(images, need_logits=need_logits)

    def __getattr__(self, name):
        return getattr(self._owner.detector_model, name)


class FasterRCNNTorchObjectDetector(nn.Module):
    def __init__(
        self,
        model_weight=None,
        device="cuda",
        names=None,
        mode="eval",
        confidence=0.25,
        iou_thresh=0.45,
        pretrained=True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.mode = str(mode)
        self.confidence = float(confidence)
        self.conf_thresh = float(confidence)
        self.iou_thresh = float(iou_thresh)
        self.max_det = None
        self.agnostic = False
        self.agnostic_nms = False
        self.is_faster_rcnn = True
        self.has_faster_rcnn_label_column = True
        self.uses_custom_faster_rcnn_forward = True

        self.names = list(names or _real_coco_categories())
        self.num_classes_no_bg = len(self.names)
        self._torchvision_categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
        self._uses_torchvision_coco_space = self.names == _real_coco_categories()
        self._output_class_ids = self._build_output_class_ids()
        self.register_buffer(
            "_internal_to_output_cpu",
            self._build_internal_to_output_tensor(),
            persistent=False,
        )
        self.num_classes_with_bg = (
            len(self._torchvision_categories)
            if self._uses_torchvision_coco_space and bool(pretrained) and _is_default_coco_weight(model_weight)
            else self.num_classes_no_bg + 1
        )

        self.detector_model = self._build_model(model_weight=model_weight, pretrained=bool(pretrained))
        self.max_det = int(self.detector_model.roi_heads.detections_per_img)
        self.model = _ForwardProxy(self)
        self.detector_model.to(self.device)
        self.detector_model.train() if self.mode == "train" else self.detector_model.eval()
        print("[INFO] Faster R-CNN model is loaded")

    def _build_output_class_ids(self):
        if not self._uses_torchvision_coco_space:
            return list(range(1, self.num_classes_no_bg + 1))
        name_to_idx = {
            str(name): idx
            for idx, name in enumerate(self._torchvision_categories)
            if name not in {"__background__", "N/A"}
        }
        return [int(name_to_idx[name]) for name in self.names]

    def _build_internal_to_output_tensor(self):
        if self._uses_torchvision_coco_space:
            mapping = torch.full((len(self._torchvision_categories),), -1, dtype=torch.long)
            for out_idx, internal_idx in enumerate(self._output_class_ids):
                mapping[int(internal_idx)] = int(out_idx)
            return mapping
        mapping = torch.full((self.num_classes_no_bg + 1,), -1, dtype=torch.long)
        if self.num_classes_no_bg > 0:
            mapping[1:] = torch.arange(self.num_classes_no_bg, dtype=torch.long)
        return mapping

    def _build_model(self, model_weight, pretrained):
        use_official_coco = pretrained and self._uses_torchvision_coco_space and _is_default_coco_weight(model_weight)
        if use_official_coco and model_weight is not None:
            _seed_torchvision_cache_from_local(model_weight)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if use_official_coco else None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            num_classes=None if weights is not None else self.num_classes_with_bg,
            box_score_thresh=0.0,
            box_nms_thresh=self.iou_thresh,
        )

        if not use_official_coco:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes_with_bg)



        if model_weight and Path(model_weight).is_file() and not use_official_coco:
            _load_checkpoint_state(model, Path(model_weight), self.device)
        elif model_weight and not Path(model_weight).is_file() and not use_official_coco:
            raise FileNotFoundError(f"Faster R-CNN weights not found: {model_weight}")

        return model

    def set_dropout_rate(self, dropout_rate):
        predictor = self.detector_model.roi_heads.box_predictor
        if not isinstance(predictor, _DropoutFastRCNNPredictor):
            try:
                param = next(predictor.parameters())
                predictor_device = param.device
                predictor_dtype = param.dtype
            except StopIteration:
                predictor_device = self.device
                predictor_dtype = torch.float32
            self.detector_model.roi_heads.box_predictor = _DropoutFastRCNNPredictor(predictor, dropout_rate=0.0)
            predictor = self.detector_model.roi_heads.box_predictor
            predictor.to(device=predictor_device, dtype=predictor_dtype)
        predictor.dropout = float(dropout_rate)

    def _map_labels_to_output(self, labels_internal):
        labels_internal = labels_internal.to(torch.long)
        internal_to_output = self._internal_to_output_cpu.to(labels_internal.device)
        valid = (labels_internal >= 0) & (labels_internal < internal_to_output.numel())
        labels = internal_to_output[labels_internal.clamp(min=0, max=internal_to_output.numel() - 1)]
        valid &= labels >= 0
        return labels, valid

    def _forward_impl(self, images, need_logits=True):
        if isinstance(images, torch.Tensor):
            image_list = [img for img in images]
        else:
            image_list = list(images)

        was_training = self.detector_model.training
        self.detector_model.eval()
        with torch.inference_mode():
            image_list = [
                img.to(self.device, non_blocking=True) if img.device != self.device else img
                for img in image_list
            ]
            if need_logits:
                detections = self._custom_inference(image_list)
            else:
                detections = self.detector_model(image_list)
        if was_training:
            self.detector_model.train()
        return self._detections_to_contract(detections, self.device, include_class_features=need_logits)

    def _custom_inference(self, image_list):
        """Run Faster R-CNN explicitly so uncertainty code can use ROI logits.

        torchvision's public eval forward only returns post-NMS boxes, labels,
        and scores.  For UQ we need the ROI-head class candidates before final
        class-wise NMS, so this path mirrors torchvision postprocessing up to
        score/small-box filtering and lets ``non_max_suppression`` apply NMS.
        """
        original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in image_list]
        images, _targets = self.detector_model.transform(image_list, None)

        features = self.detector_model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, _proposal_losses = self.detector_model.rpn(images, features, None)
        roi_heads = self.detector_model.roi_heads
        box_features = roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = roi_heads.box_head(box_features)
        class_logits, box_regression = roi_heads.box_predictor(box_features)

        detections = self._pre_nms_detections_with_logits(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=proposals,
            image_shapes=images.image_sizes,
        )
        detections = self.detector_model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections

    def preprocess_images(self, images):
        if isinstance(images, torch.Tensor):
            image_list = [img for img in images]
        else:
            image_list = list(images)

        image_list = [
            img.to(self.device, non_blocking=True) if img.device != self.device else img
            for img in image_list
        ]
        original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in image_list]
        transformed_images, _targets = self.detector_model.transform(image_list, None)
        return {
            "transformed_images": transformed_images,
            "original_image_sizes": original_image_sizes,
        }

    def forward_preprocessed(self, preprocessed):
        was_training = self.detector_model.training
        self.detector_model.eval()
        with torch.inference_mode():
            images = preprocessed["transformed_images"]
            original_image_sizes = preprocessed["original_image_sizes"]
            features = self.detector_model.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = {"0": features}

            proposals, _proposal_losses = self.detector_model.rpn(images, features, None)
            roi_heads = self.detector_model.roi_heads
            box_features = roi_heads.box_roi_pool(features, proposals, images.image_sizes)
            box_features = roi_heads.box_head(box_features)
            class_logits, box_regression = roi_heads.box_predictor(box_features)
            detections = self._pre_nms_detections_with_logits(
                class_logits=class_logits,
                box_regression=box_regression,
                proposals=proposals,
                image_shapes=images.image_sizes,
            )
            detections = self.detector_model.transform.postprocess(
                detections,
                images.image_sizes,
                original_image_sizes,
            )
        if was_training:
            self.detector_model.train()
        return self._detections_to_contract(detections, self.device, include_class_features=True)

    @contextmanager
    def temporary_roi_score_threshold(self, threshold=None):
        roi_heads = getattr(self.detector_model, "roi_heads", None)
        if roi_heads is None or threshold is None or not hasattr(roi_heads, "score_thresh"):
            yield
            return
        old_threshold = float(roi_heads.score_thresh)
        roi_heads.score_thresh = float(threshold)
        try:
            yield
        finally:
            roi_heads.score_thresh = old_threshold

    def prepare_roi_cache(self, images):
        """Cache deterministic Faster R-CNN stages before ROI prediction.

        MC dropout for this wrapper is applied only in ``roi_heads.box_predictor``.
        Reusing transformed images, backbone features, RPN proposals, and box-head
        features avoids repeating the expensive detector trunk for every MC run.
        """
        if isinstance(images, torch.Tensor):
            image_list = [img for img in images]
        else:
            image_list = list(images)

        was_training = self.detector_model.training
        self.detector_model.eval()
        with torch.inference_mode():
            image_list = [
                img.to(self.device, non_blocking=True) if img.device != self.device else img
                for img in image_list
            ]
            original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in image_list]
            transformed_images, _targets = self.detector_model.transform(image_list, None)
            features = self.detector_model.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            proposals, _proposal_losses = self.detector_model.rpn(transformed_images, features, None)
            roi_heads = self.detector_model.roi_heads
            box_features = roi_heads.box_roi_pool(features, proposals, transformed_images.image_sizes)
            box_features = roi_heads.box_head(box_features)
        if was_training:
            self.detector_model.train()

        return {
            "original_image_sizes": original_image_sizes,
            "image_sizes": transformed_images.image_sizes,
            "proposals": proposals,
            "box_features": box_features,
        }

    def forward_from_roi_cache(self, cache):
        was_training = self.detector_model.training
        self.detector_model.eval()
        with torch.inference_mode():
            roi_heads = self.detector_model.roi_heads
            class_logits, box_regression = roi_heads.box_predictor(cache["box_features"])
            detections = self._pre_nms_detections_with_logits(
                class_logits=class_logits,
                box_regression=box_regression,
                proposals=cache["proposals"],
                image_shapes=cache["image_sizes"],
            )
            detections = self.detector_model.transform.postprocess(
                detections,
                cache["image_sizes"],
                cache["original_image_sizes"],
            )
        if was_training:
            self.detector_model.train()
        return self._detections_to_contract(detections, self.device, include_class_features=True)

    def _pre_nms_detections_with_logits(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = int(class_logits.shape[-1])
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.detector_model.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_logits = class_logits.split(boxes_per_image, 0)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_detections = []
        for image_idx, (boxes, scores, logits, image_shape) in enumerate(
            zip(pred_boxes_list, pred_scores_list, pred_logits, image_shapes)
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            num_proposals = int(scores.shape[0])
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            proposal_indices = torch.arange(num_proposals, device=device).view(-1, 1).expand_as(scores)


            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            proposal_indices = proposal_indices[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            proposal_indices = proposal_indices.reshape(-1)

            score_thresh = float(self.detector_model.roi_heads.score_thresh)
            inds = torch.where(scores > score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            proposal_indices = proposal_indices[inds]

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            proposal_indices = proposal_indices[keep]

            logits = logits[proposal_indices]
            probs = pred_scores_list[image_idx][proposal_indices] if proposal_indices.numel() > 0 else scores.new_zeros((0, num_classes))

            all_detections.append({
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "class_logits": logits,
                "class_probs": probs,
                "proposal_indices": proposal_indices,
            })
        return all_detections

    def _select_output_class_columns(self, values):
        if self._uses_torchvision_coco_space and values.shape[-1] == len(self._torchvision_categories):
            return values[..., self._output_class_ids]
        if values.shape[-1] == self.num_classes_no_bg + 1:
            return values[..., 1:]
        return values[..., : self.num_classes_no_bg]

    def _detections_to_contract(self, detections, device, include_class_features=True):
        rows_by_image = []
        logits_by_image = []
        c = self.num_classes_no_bg

        for det in detections:
            boxes = det["boxes"].to(device)
            scores = det["scores"].to(device)
            labels_internal = det["labels"].to(device)
            labels, valid = self._map_labels_to_output(labels_internal)

            valid &= scores > 0.0
            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]

            xywh = boxes.clone()
            if xywh.numel():
                xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
                xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
                xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
                xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)

            if not include_class_features:
                rows_by_image.append(torch.cat([xywh, scores[:, None], labels.to(scores.dtype)[:, None]], dim=1))
                continue

            raw_logits = det.get("class_logits")
            raw_probs = det.get("class_probs")
            if raw_logits is not None:
                raw_logits = raw_logits.to(device)
                logits_full = self._select_output_class_columns(raw_logits)
                if raw_probs is not None:
                    probs_full = self._select_output_class_columns(raw_probs.to(device))
                else:
                    probs_full = F.softmax(logits_full, dim=-1)
                logits = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
                probs = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
                if int(valid.sum().item()) > 0:
                    logits = logits_full[valid]
                    probs = probs_full[valid]
            else:
                probs = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
                logits = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
            if boxes.shape[0] > 0:
                if raw_logits is None:
                    idx = torch.arange(boxes.shape[0], device=device)
                    probs[idx, labels] = scores
                    logits[idx, labels] = torch.logit(scores.clamp(min=1e-6, max=1.0 - 1e-6))

            rows = torch.cat([xywh, scores[:, None], labels.to(scores.dtype)[:, None], probs], dim=1)
            rows_by_image.append(rows)
            logits_by_image.append(logits)

        if not include_class_features:
            return rows_by_image, None

        if not rows_by_image:
            return (
                [],
                None,
            )
        return rows_by_image, logits_by_image

    def forward(self, images, augment=False):
        return self._forward_impl(images)

    @staticmethod
    def non_max_suppression(
        prediction,
        logits,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        max_det=300,
        return_indices=False,
        **_kwargs,
    ):
        outputs = []
        logits_outputs = []
        objectness_outputs = []
        index_outputs = []
        for image_idx, pred in enumerate(prediction):
            if pred.numel() == 0:
                device = pred.device
                if isinstance(logits, list) and image_idx < len(logits):
                    logit_dim = int(logits[image_idx].shape[-1])
                else:
                    logit_dim = int(logits.shape[-1]) if logits is not None and hasattr(logits, "shape") else 0
                outputs.append(torch.zeros((0, 6), dtype=pred.dtype, device=device))
                logits_outputs.append(torch.zeros((0, logit_dim), dtype=pred.dtype, device=device))
                objectness_outputs.append(torch.zeros((0, 1), dtype=pred.dtype, device=device))
                index_outputs.append(torch.zeros((0,), dtype=torch.long, device=device))
                continue

            scores = pred[:, 4]
            labels = pred[:, 5].to(torch.long)
            keep = scores > float(conf_thres)
            if classes is not None:
                class_tensor = torch.as_tensor(classes, device=pred.device, dtype=torch.long)
                keep &= (labels[:, None] == class_tensor[None]).any(dim=1)
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
            candidate_xywh = pred[keep_idx, :4]
            candidate_xyxy = candidate_xywh.clone()
            if candidate_xyxy.numel():
                candidate_xyxy[:, 0] = candidate_xywh[:, 0] - candidate_xywh[:, 2] * 0.5
                candidate_xyxy[:, 1] = candidate_xywh[:, 1] - candidate_xywh[:, 3] * 0.5
                candidate_xyxy[:, 2] = candidate_xywh[:, 0] + candidate_xywh[:, 2] * 0.5
                candidate_xyxy[:, 3] = candidate_xywh[:, 1] + candidate_xywh[:, 3] * 0.5
                nms_labels = torch.zeros_like(labels[keep_idx]) if agnostic else labels[keep_idx]
                nms_keep = batched_nms(candidate_xyxy, scores[keep_idx], nms_labels, float(iou_thres))
                if max_det is not None:
                    nms_keep = nms_keep[: int(max_det)]
                keep_idx = keep_idx[nms_keep]

            xywh = pred[keep_idx, :4]
            xyxy = xywh.clone()
            if xyxy.numel():
                xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
                xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
                xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
                xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
            det = torch.cat([xyxy, scores[keep_idx, None], labels[keep_idx, None].to(pred.dtype)], dim=1)
            outputs.append(det)
            if isinstance(logits, list):
                logits_outputs.append(logits[image_idx][keep_idx] if image_idx < len(logits) else pred[keep_idx, 6:])
            else:
                logits_outputs.append(logits[image_idx][keep_idx] if logits is not None else pred[keep_idx, 6:])
            objectness_outputs.append(torch.ones((keep_idx.shape[0], 1), dtype=pred.dtype, device=pred.device))
            index_outputs.append(keep_idx)

        if return_indices:
            return outputs, logits_outputs, objectness_outputs, index_outputs
        return outputs, logits_outputs, objectness_outputs
