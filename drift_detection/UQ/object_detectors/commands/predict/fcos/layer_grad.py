from commands.predict.common import *
from commands.predict.fcos.common import select_fcos_post_nms
from commands.predict.fcos.utils import (
    build_fcos_dense_candidate_cache,
    ensure_fcos_selected_indices,
    fcos_candidate_mask_from_cache,
    fcos_class_name,
)
from commands.utils.predict_utils import (
    _apply_direction,
    _class_loss_tensor,
    _objectness_loss_tensor,
    format_gradient_output,
)


def _safe_npz_key(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _gradient_to_np_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _norm_loss_name(value, default, allowed, aliases=None):
    aliases = aliases or {}
    key = str(value if value is not None else default).strip().lower().replace("-", "_")
    key = aliases.get(key, key)
    if key not in allowed:
        raise ValueError(f"Unsupported loss option '{value}'. Supported values: {', '.join(sorted(allowed))}.")
    return key


def _norm_direction(value):
    key = str(value if value is not None else "pred_to_target").strip().lower().replace("-", "_")
    aliases = {
        "pred_to_target": "pred_to_target",
        "prediction_to_target": "pred_to_target",
        "target": "pred_to_target",
        "target_to_pred": "target_to_pred",
        "target_to_prediction": "target_to_pred",
        "reverse": "target_to_pred",
    }
    normalized = aliases.get(key)
    if normalized is None:
        raise ValueError("direction must be pred_to_target or target_to_pred.")
    return normalized


def _parse_fcos_layer_grad_config(config):
    out = config.get("output", {})
    active = out.get("layer_grad", {}) if isinstance(out.get("layer_grad", {}), dict) else {}
    g = active.get("gradient", {}) if isinstance(active.get("gradient", {}), dict) else {}
    target = str(g.get("target", "cand_target")).strip().lower()
    if target in {"cand", "candidate"}:
        target = "cand_target"
    if target in {"null", "uniform"}:
        target = "null_target"
    if target not in {"cand_target", "null_target"}:
        raise ValueError("FCOS layer_grad.gradient.target must be cand_target or null_target.")

    scalar = [str(v).strip().lower() for v in _as_list(g.get("scalar", ["bbox_loss", "cls_loss", "cnt_loss"]))]
    if "loss" in scalar:
        expanded = []
        for value in scalar:
            expanded.extend(["bbox_loss", "cls_loss", "cnt_loss"] if value == "loss" else [value])
        scalar = expanded
    scalar = list(dict.fromkeys(scalar))
    allowed_scalar = {"bbox_loss", "cls_loss", "cnt_loss"}
    scalar = [v for v in scalar if v in allowed_scalar]
    if not scalar:
        scalar = ["bbox_loss", "cls_loss", "cnt_loss"]

    layer_cfg = g.get("layer", {})
    default_layers = {
        "bbox_loss": ["detector_model.rpn.head.bbox_pred"],
        "cls_loss": ["detector_model.rpn.head.cls_logits"],
        "cnt_loss": ["detector_model.rpn.head.centerness"],
    }
    if isinstance(layer_cfg, dict):
        target_layer_map = {
            key: _as_list(layer_cfg.get(key, default_layers[key]))
            for key in allowed_scalar
        }
    else:
        shared_layers = _as_list(layer_cfg)
        target_layer_map = {key: shared_layers or default_layers[key] for key in allowed_scalar}

    reduction_aliases = {
        "l1": "l1_norm",
        "l1_norm": "l1_norm",
        "1_norm": "l1_norm",
        "l2": "l2_norm",
        "l2_norm": "l2_norm",
        "2_norm": "l2_norm",
        "min": "min",
        "max": "max",
        "mean": "mean",
        "std": "std",
    }
    reductions = []
    for value in _as_list(g.get("reduction", [])):
        key = str(value).strip().lower().replace("-", "_")
        normalized = reduction_aliases.get(key)
        if normalized is None:
            raise ValueError("Unsupported FCOS layer_grad.gradient.reduction.")
        if normalized not in reductions:
            reductions.append(normalized)

    bbox_loss = _norm_loss_name(
        g.get("bbox_loss", "l1"),
        "l1",
        {"l1", "l2"},
        aliases={"box_l1": "l1", "box_l2": "l2"},
    )
    cls_loss = _norm_loss_name(g.get("cls_loss", "bcewithlogits"), "bcewithlogits", {"bcewithlogits", "kl"}, aliases={"bce": "bcewithlogits"})
    cnt_loss = _norm_loss_name(
        g.get("cnt_loss", "bcewithlogits"),
        "bcewithlogits",
        {"bcewithlogits", "abs_diff", "signed_diff"},
        aliases={"bce": "bcewithlogits", "abs": "abs_diff", "signed": "signed_diff"},
    )
    bbox_direction = _norm_direction(g.get("bbox_direction", "pred_to_target"))
    cls_direction = _norm_direction(g.get("cls_direction", "pred_to_target"))
    cnt_direction = _norm_direction(g.get("cnt_direction", "pred_to_target"))
    if cls_direction == "target_to_pred" and cls_loss != "kl":
        raise ValueError("FCOS cls_direction=target_to_pred is only supported when cls_loss=kl.")
    if cnt_direction == "target_to_pred" and cnt_loss != "signed_diff":
        raise ValueError("FCOS cnt_direction=target_to_pred is only supported when cnt_loss=signed_diff.")

    return {
        "save_csv": bool(active.get("save_csv", {}).get("enabled", False) if isinstance(active.get("save_csv", {}), dict) else active.get("save_csv", False)),
        "target": target,
        "scalar": scalar,
        "target_layer_map": target_layer_map,
        "reduction": reductions,
        "cand_score_threshold": float(g.get("cand_score_threshold", 0.05)),
        "cand_iou_threshold": float(g.get("cand_iou_threshold", g.get("iou_threshold", 0.45))),
        "bbox_loss": bbox_loss,
        "cls_loss": cls_loss,
        "cnt_loss": cnt_loss,
        "bbox_direction": bbox_direction,
        "cls_direction": cls_direction,
        "cnt_direction": cnt_direction,
    }


def _flatten_level_output(tensor, image_idx):
    if tensor.ndim == 4:
        return tensor[image_idx].permute(1, 2, 0).reshape(-1, tensor.shape[1])
    return tensor


def _flatten_centerness(tensor, image_idx):
    if tensor.ndim == 4:
        return tensor[image_idx].permute(1, 2, 0).reshape(-1)
    return tensor.reshape(-1)


class _FcosLossFlattenCache:
    def __init__(self, model_output, image_idx, device):
        self.box_cls = model_output["box_cls"]
        self.box_regression = model_output["box_regression"]
        self.centerness = model_output["centerness"]
        self.locations = model_output["locations"]
        self.image_idx = image_idx
        self.device = device
        self.flat_ltrb_by_level = {}
        self.flat_cls_by_level = {}
        self.flat_cnt_by_level = {}
        self.loc_by_level = {}

    def flat_ltrb(self, level):
        if level not in self.flat_ltrb_by_level:
            self.flat_ltrb_by_level[level] = _flatten_level_output(self.box_regression[level], self.image_idx)
        return self.flat_ltrb_by_level[level]

    def flat_cls(self, level):
        if level not in self.flat_cls_by_level:
            self.flat_cls_by_level[level] = _flatten_level_output(self.box_cls[level], self.image_idx)
        return self.flat_cls_by_level[level]

    def flat_cnt(self, level):
        if level not in self.flat_cnt_by_level:
            self.flat_cnt_by_level[level] = _flatten_centerness(self.centerness[level], self.image_idx)
        return self.flat_cnt_by_level[level]

    def locations_for(self, level, dtype):
        key = (level, dtype)
        if key not in self.loc_by_level:
            self.loc_by_level[key] = self.locations[level].to(device=self.device, dtype=dtype)
        return self.loc_by_level[key]


def _ltrb_target_from_box(locations_xy, box_xyxy):
    x = locations_xy[:, 0]
    y = locations_xy[:, 1]
    return torch.stack(
        [
            x - box_xyxy[0],
            y - box_xyxy[1],
            box_xyxy[2] - x,
            box_xyxy[3] - y,
        ],
        dim=1,
    ).clamp(min=0.0)


def _bbox_ltrb_loss(pred_ltrb, target_ltrb, mode="l1", direction="pred_to_target", reduction="sum"):
    left, right = _apply_direction(pred_ltrb, target_ltrb.to(device=pred_ltrb.device, dtype=pred_ltrb.dtype), direction)
    loss = torch.abs(left - right) if mode == "l1" else torch.square(left - right)
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def _source_indices_from_boxlist(boxlist, row_idx):
    level = int(boxlist.get_field("pre_nms_level")[row_idx].detach().cpu().item())
    loc_idx = int(boxlist.get_field("pre_nms_location_idx")[row_idx].detach().cpu().item())
    raw_idx = int(boxlist.get_field("pre_nms_candidate_idx")[row_idx].detach().cpu().item())
    cls_one_based = int(boxlist.get_field("labels")[row_idx].detach().cpu().item())
    return level, loc_idx, raw_idx, cls_one_based


def _prediction_class_name(detector, cls_idx):
    return fcos_class_name(detector, cls_idx)


def _resolve_fcos_candidate_sources(
    *,
    target_mode,
    model_output,
    image_idx,
    pred_idx,
    final_box,
    final_cls,
    raw_idx,
    cand_score_threshold,
    cand_iou_threshold,
    timing,
    timing_accumulator,
    candidate_sources=None,
):
    detections = model_output["detections"]
    if candidate_sources is not None:
        return candidate_sources
    resolved_sources = []
    if target_mode == "cand_target":
        return resolved_sources
    else:
        if image_idx >= len(detections) or pred_idx >= len(detections[image_idx]):
            return resolved_sources
        source_boxlist = detections[image_idx]
        level, loc_idx, _raw, _cls_one_based = _source_indices_from_boxlist(source_boxlist, pred_idx)
        resolved_sources.append((level, loc_idx))
    return resolved_sources


def _build_fcos_losses(
    *,
    target_mode,
    target_values,
    model_output,
    image_idx,
    pred_idx,
    final_box,
    final_cls,
    raw_idx,
    cand_score_threshold,
    cand_iou_threshold,
    bbox_loss,
    cls_loss,
    cnt_loss,
    bbox_direction,
    cls_direction,
    cnt_direction,
    timing,
    timing_accumulator,
    candidate_sources=None,
    flat_cache=None,
):
    box_cls = model_output["box_cls"]
    box_regression = model_output["box_regression"]
    centerness = model_output["centerness"]
    locations = model_output["locations"]
    device = final_box.device
    losses = {}

    if candidate_sources is None:
        candidate_sources = _resolve_fcos_candidate_sources(
            target_mode=target_mode,
            model_output=model_output,
            image_idx=image_idx,
            pred_idx=pred_idx,
            final_box=final_box,
            final_cls=final_cls,
            raw_idx=raw_idx,
            cand_score_threshold=cand_score_threshold,
            cand_iou_threshold=cand_iou_threshold,
            timing=timing,
            timing_accumulator=timing_accumulator,
            candidate_sources=candidate_sources,
        )

    t_loss = timing.start()
    requested = set(target_values)
    need_bbox = "bbox_loss" in requested
    need_cls = "cls_loss" in requested
    need_cnt = "cnt_loss" in requested
    need_ltrb_target = need_bbox
    bbox_terms = []
    cls_terms = []
    cnt_terms = []
    num_classes = int(box_cls[0].shape[1])
    final_cls = int(final_cls)
    if flat_cache is None:
        flat_cache = _FcosLossFlattenCache(model_output, image_idx, device)

    for level, loc_idx in candidate_sources:
        if need_bbox or need_ltrb_target:
            pred_ltrb = flat_cache.flat_ltrb(level)[loc_idx].view(1, 4)
        else:
            pred_ltrb = None

        if need_cls:
            cls_logits = flat_cache.flat_cls(level)[loc_idx].view(-1)
            if target_mode == "cand_target":
                cls_target = torch.zeros((num_classes,), dtype=box_cls[level].dtype, device=device)
                if 0 <= final_cls < num_classes:
                    cls_target[final_cls] = 1.0
            else:
                cls_target_value = (
                    0.5
                    if str(cls_loss).strip().lower() == "bcewithlogits"
                    else 1.0 / float(max(num_classes, 1))
                )
                cls_target = torch.full((num_classes,), cls_target_value, dtype=box_cls[level].dtype, device=device)

        if need_cnt:
            cnt_logit = flat_cache.flat_cnt(level)[loc_idx].view(())

        if need_ltrb_target:
            if target_mode == "cand_target":
                loc_xy = flat_cache.locations_for(level, pred_ltrb.dtype)[loc_idx].view(1, 2)
                target_ltrb = _ltrb_target_from_box(
                    loc_xy,
                    final_box.detach().to(device=device, dtype=pred_ltrb.dtype),
                )
            else:
                target_ltrb = torch.zeros_like(pred_ltrb)

        if need_cnt:
            if target_mode == "cand_target":
                cnt_target = torch.ones_like(cnt_logit)
            else:
                cnt_target = torch.full_like(cnt_logit, 0.5)

        if need_bbox:
            bbox_terms.append(_bbox_ltrb_loss(pred_ltrb, target_ltrb, mode=bbox_loss, direction=bbox_direction, reduction="sum"))
        if need_cls:
            cls_terms.append(
                _class_loss_tensor(
                    cls_logits,
                    cls_target,
                    class_idx=final_cls if target_mode == "cand_target" else None,
                    mode=cls_loss,
                    direction=cls_direction,
                    reduction="sum",
                )
            )
        if need_cnt:
            cnt_terms.append(
                _objectness_loss_tensor(
                    cnt_logit.view(1),
                    cnt_target.view(1),
                    mode=cnt_loss,
                    direction=cnt_direction,
                    reduction="sum",
                )
            )
    if bbox_terms:
        losses["bbox_loss"] = torch.stack([v.reshape(()) for v in bbox_terms]).sum()
    if cls_terms:
        losses["cls_loss"] = torch.stack([v.reshape(()) for v in cls_terms]).sum()
    if cnt_terms:
        losses["cnt_loss"] = torch.stack([v.reshape(()) for v in cnt_terms]).sum()
    timing_accumulator["loss_compute_sec"] += timing.elapsed(t_loss)
    return losses


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed_common = parse_output_config(config.get("output", {}))
    layer_cfg = _parse_fcos_layer_grad_config(config)
    if not layer_cfg["save_csv"]:
        return

    output_csv = run_dir / "layer_grad.csv"
    save_raw_gradients = not layer_cfg["reduction"]
    gradients_dir = run_dir / "gradients"
    if save_raw_gradients:
        gradients_dir.mkdir(parents=True, exist_ok=True)

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not bool(getattr(detector, "is_fcos", False)):
        raise ValueError("commands.predict.fcos.layer_grad requires model.type=fcos.")

    target_values = layer_cfg["scalar"]
    target_layer_map = {
        target_value: expand_layer_names(detector, layer_cfg["target_layer_map"].get(target_value, []))
        for target_value in target_values
    }
    layer_params = {
        target_value: [resolve_layer_parameter(detector, layer_name) for layer_name in target_layer_map[target_value]]
        for target_value in target_values
    }
    original_requires_grad = {
        id(param): bool(param.requires_grad)
        for params in layer_params.values()
        for param in params
    }
    for params in layer_params.values():
        for param in params:
            param.requires_grad_(True)

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=parsed_common.get("unit", "bbox"),
        stages=[
            "detector_inference_sec",
            "candidate_search_sec",
            "loss_compute_sec",
            "backpropagation_sec",
            "feature_compute_sec",
        ],
        device=device,
    )

    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
    ]
    for target_value in target_values:
        for layer_name in target_layer_map[target_value]:
            grad_key = f"{target_value}_{layer_name}"
            if save_raw_gradients:
                fieldnames.append(grad_key)
            else:
                fieldnames.extend(f"{grad_key}_{metric}" for metric in layer_cfg["reduction"])

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for batch_idx, (images, targets) in enumerate(tqdm(
                dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
            )):
                image_list = _as_image_list(images)
                infer_batch = _prepare_infer_batch(detector, image_list, device, auto=False)[0]
                stage_seconds = {
                    "detector_inference_sec": 0.0,
                    "candidate_search_sec": 0.0,
                    "loss_compute_sec": 0.0,
                    "backpropagation_sec": 0.0,
                    "feature_compute_sec": 0.0,
                }

                detector.zero_grad(set_to_none=True)
                fcos_preprocessed = detector.preprocess_images(infer_batch)
                t_detector = timing.start()
                model_output = detector.forward_layer_grad(fcos_preprocessed, include_post_logits=False)

                row_prediction = model_output["post_prediction"]
                row_indices = model_output["post_indices"]
                selected = select_fcos_post_nms(
                    detector,
                    row_prediction,
                    None,
                    row_indices,
                )
                selected_preds = selected[0]
                selected_indices = selected[3]
                stage_seconds["detector_inference_sec"] += timing.elapsed(t_detector)

                candidate_caches = {}
                if layer_cfg["target"] == "cand_target":
                    for sample_idx in range(len(image_list)):
                        t_candidate = timing.start()
                        candidate_caches[sample_idx] = build_fcos_dense_candidate_cache(
                            model_output,
                            sample_idx,
                            layer_cfg["cand_score_threshold"],
                            detach=True,
                        )
                        stage_seconds["candidate_search_sec"] += timing.elapsed(t_candidate)
                flat_caches = {
                    sample_idx: _FcosLossFlattenCache(model_output, sample_idx, device)
                    for sample_idx in range(len(image_list))
                }

                batch_rows = []
                batch_items = 0
                batch_grad_arrays = {}
                grad_call_index = 0
                expected_grad_calls = int(batch_items) * len(target_values)
                if save_raw_gradients:
                    npz_name = f"layer_grad_batch_{batch_idx:06d}.npz"
                    npz_rel_path = (Path("gradients") / npz_name).as_posix()
                    npz_path = gradients_dir / npz_name

                for sample_idx in range(len(image_list)):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]
                    det = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                    raw_keep = ensure_fcos_selected_indices(selected_indices, selected_preds, sample_idx).to(device=device)
                    batch_items += int(det.shape[0])
                expected_grad_calls = int(batch_items) * len(target_values)

                for sample_idx in range(len(image_list)):
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]
                    det = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                    raw_keep = ensure_fcos_selected_indices(selected_indices, selected_preds, sample_idx).to(device=device)
                    for pred_idx in range(int(det.shape[0])):
                        raw_idx = int(raw_keep[pred_idx].detach().cpu().item())
                        final_box = det[pred_idx, :4]
                        final_cls = int(det[pred_idx, 5].detach().cpu().item()) if det.shape[1] > 5 else 0
                        candidate_sources = None
                        if layer_cfg["target"] == "cand_target":
                            t_candidate = timing.start()
                            cache = candidate_caches.get(sample_idx)
                            cand_mask, _ious = fcos_candidate_mask_from_cache(
                                cache,
                                final_box.detach().float(),
                                final_cls,
                                layer_cfg["cand_iou_threshold"],
                            )
                            stage_seconds["candidate_search_sec"] += timing.elapsed(t_candidate)
                            candidate_sources = []
                            if cache is not None and cache.levels is not None and cache.location_indices is not None:
                                candidate_indices = torch.where(cand_mask)[0]
                                if int(candidate_indices.numel()) > 0:
                                    levels = cache.levels[candidate_indices].detach().cpu().tolist()
                                    loc_indices = cache.location_indices[candidate_indices].detach().cpu().tolist()
                                    candidate_sources = [(int(level), int(loc_idx)) for level, loc_idx in zip(levels, loc_indices)]
                        row = {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_idx,
                            "xmin": float(det[pred_idx, 0].detach().cpu().item()),
                            "ymin": float(det[pred_idx, 1].detach().cpu().item()),
                            "xmax": float(det[pred_idx, 2].detach().cpu().item()),
                            "ymax": float(det[pred_idx, 3].detach().cpu().item()),
                            "score": float(det[pred_idx, 4].detach().cpu().item()),
                            "pred_class": _prediction_class_name(detector, final_cls),
                        }

                        losses = _build_fcos_losses(
                            target_mode=layer_cfg["target"],
                            target_values=target_values,
                            model_output=model_output,
                            image_idx=sample_idx,
                            pred_idx=pred_idx,
                            final_box=final_box,
                            final_cls=final_cls,
                            raw_idx=raw_idx,
                            cand_score_threshold=layer_cfg["cand_score_threshold"],
                            cand_iou_threshold=layer_cfg["cand_iou_threshold"],
                            bbox_loss=layer_cfg["bbox_loss"],
                            cls_loss=layer_cfg["cls_loss"],
                            cnt_loss=layer_cfg["cnt_loss"],
                            bbox_direction=layer_cfg["bbox_direction"],
                            cls_direction=layer_cfg["cls_direction"],
                            cnt_direction=layer_cfg["cnt_direction"],
                            timing=timing,
                            timing_accumulator=stage_seconds,
                            candidate_sources=candidate_sources,
                            flat_cache=flat_caches[sample_idx],
                        )

                        for target_value in target_values:
                            params = layer_params[target_value]
                            layer_names = target_layer_map[target_value]
                            scalar = losses.get(target_value)
                            if scalar is None:
                                for layer_name in layer_names:
                                    key = f"{target_value}_{layer_name}"
                                    if save_raw_gradients:
                                        row[key] = ""
                                    else:
                                        for metric in layer_cfg["reduction"]:
                                            row[f"{key}_{metric}"] = 0.0
                                continue

                            t_backprop = timing.start()
                            grad_call_index += 1
                            grads = torch.autograd.grad(
                                scalar,
                                params,
                                retain_graph=(grad_call_index < expected_grad_calls),
                                allow_unused=True,
                            )
                            stage_seconds["backpropagation_sec"] += timing.elapsed(t_backprop)

                            t_feature = timing.start()
                            for layer_idx, layer_name in enumerate(layer_names):
                                key = f"{target_value}_{layer_name}"
                                grad_value = format_gradient_output(
                                    grads[layer_idx],
                                    vector_reduction=layer_cfg["reduction"],
                                    map_reduction="none",
                                )
                                if save_raw_gradients:
                                    array_key = f"r{len(batch_grad_arrays):06d}_{_safe_npz_key(key)}"
                                    batch_grad_arrays[array_key] = _gradient_to_np_array(grad_value)
                                    row[key] = f"{npz_rel_path}:{array_key}"
                                else:
                                    for metric in layer_cfg["reduction"]:
                                        value = grad_value.get(metric, 0.0) if isinstance(grad_value, dict) else 0.0
                                        row[f"{key}_{metric}"] = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
                                    if layer_cfg["reduction"]:
                                        del value
                                del grad_value
                            stage_seconds["feature_compute_sec"] += timing.elapsed(t_feature)
                            del grads
                            del scalar
                        batch_rows.append(row)
                        del losses, final_box

                for row in batch_rows:
                    writer.writerow(row)
                csv_file.flush()
                if save_raw_gradients and batch_grad_arrays:
                    np.savez(npz_path, **batch_grad_arrays)
                timing.record(
                    num_images=len(image_list),
                    num_predictions=batch_items,
                    stage_seconds=stage_seconds,
                )
                if batch_items:
                    del det, raw_keep
                if batch_rows:
                    del row
                del batch_rows, batch_grad_arrays, candidate_caches
                del infer_batch, fcos_preprocessed, model_output
                del row_prediction, row_indices
                del selected, selected_preds, selected_indices
                del image_list
                detector.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        for params in layer_params.values():
            for param in params:
                param.requires_grad_(original_requires_grad.get(id(param), bool(param.requires_grad)))
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    if save_raw_gradients:
        print(f"Saved gradient arrays: {gradients_dir}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_layer_grad_csv"]
