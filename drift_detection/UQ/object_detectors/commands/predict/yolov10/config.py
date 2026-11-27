def parse_yolov10_output_config(config):
    output = config.get("output", {}) if isinstance(config, dict) and "output" in config else config
    output = output if isinstance(output, dict) else {}
    uncertainty = str(output.get("uncertainty", "")).strip().lower()
    if not uncertainty:
        uncertainty = "gt"
    supported = {
        "deterministic",
        "gt",
        "score",
        "class_probability",
        "entropy",
        "energy",
        "mc_dropout",
        "ensemble",
        "meta_detect",
        "null_detect",
        "layer_grad",
    }
    if uncertainty not in supported:
        raise ValueError(f"Unsupported YOLOv10 uncertainty: {uncertainty}")

    def as_dict(value):
        return value if isinstance(value, dict) else {}

    def as_bool_save(value):
        if isinstance(value, bool):
            return value
        value = as_dict(value)
        return bool(value.get("enabled", False))

    def as_int(value, default):
        try:
            return int(value)
        except Exception:
            return default

    def as_float(value, default):
        try:
            return float(value)
        except Exception:
            return default

    def normalize_loss(raw, default, supported_values, key):
        value = str(raw if raw is not None else default).strip().lower().replace("-", "_")
        aliases = {
            "bce": "bcewithlogits",
            "bce_with_logits": "bcewithlogits",
            "bcewithlogits": "bcewithlogits",
            "kl": "kl",
            "kl_div": "kl",
            "kl_divergence": "kl",
            "l1": "l1",
            "l1_loss": "l1",
            "l2": "l2",
            "l2_loss": "l2",
            "mse": "l2",
        }.get(value)
        if value in {"box_l1", "box_l2", "offset_l1", "offset_l2", "obj_loss", "obj"}:
            raise ValueError(f"YOLOv10 does not support YOLOv5-style {key}: {raw}")
        if value not in supported_values and aliases not in supported_values:
            raise ValueError(f"Unsupported YOLOv10 {key}: {raw}. Supported values: {', '.join(sorted(supported_values))}.")
        return aliases or value

    def normalize_direction(raw, default="pred_to_target"):
        value = str(raw if raw is not None else default).strip().lower().replace("-", "_")
        aliases = {
            "pred_to_target": "pred_to_target",
            "prediction_to_target": "pred_to_target",
            "target": "pred_to_target",
            "target_to_pred": "target_to_pred",
            "target_to_prediction": "target_to_pred",
            "reverse": "target_to_pred",
        }
        if value not in aliases:
            raise ValueError(f"Unsupported YOLOv10 direction: {raw}")
        return aliases[value]

    def normalize_list(raw, default=None, *, lower=False):
        value = default if raw is None else raw
        values = value if isinstance(value, (list, tuple)) else [value]
        result = []
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            result.append(text.lower() if lower else text)
        return result

    active = as_dict(output.get(uncertainty, {}))
    parsed = {
        "uncertainty": uncertainty,
        "unit": "bbox",
        "save_csv_enabled": as_bool_save(active.get("save_csv", False)),
        "save_image_enabled": as_bool_save(active.get("save_image", False)),
        "save_image_gt_step": 1,
        "save_image_gt_max_num": 1,
        "gt_iou_match_threshold": 0.5,
        "mc_num_runs": 30,
        "mc_dropout_rate": 0.5,
        "meta_detect_score_threshold": 0.0,
        "meta_detect_iou_threshold": 0.45,
        "null_detect_cls_loss": "kl",
        "null_detect_cls_direction": "pred_to_target",
        "null_detect_feature_set": "full",
    }
    if uncertainty == "gt":
        parsed["gt_iou_match_threshold"] = as_float(active.get("iou_match_threshold", 0.5), 0.5)
        save_image_cfg = as_dict(active.get("save_image", {}))
        parsed["save_image_gt_step"] = as_int(save_image_cfg.get("step", 1), 1)
        parsed["save_image_gt_max_num"] = as_int(save_image_cfg.get("max_num", 1), 1)
    elif uncertainty == "mc_dropout":
        parsed["mc_num_runs"] = as_int(active.get("num_runs", 30), 30)
        parsed["mc_dropout_rate"] = as_float(active.get("dropout_rate", 0.5), 0.5)
    elif uncertainty == "meta_detect":
        parsed["meta_detect_score_threshold"] = as_float(active.get("score_threshold", 0.0), 0.0)
        parsed["meta_detect_iou_threshold"] = as_float(active.get("iou_threshold", 0.45), 0.45)
    elif uncertainty == "null_detect":
        for forbidden in ("obj_loss", "bbox_loss", "bbox_direction"):
            if forbidden in active:
                raise ValueError(f"YOLOv10 null_detect does not support {forbidden}.")
        parsed["null_detect_cls_loss"] = normalize_loss(active.get("cls_loss", "kl"), "kl", {"bcewithlogits", "kl"}, "null_detect.cls_loss")
        parsed["null_detect_cls_direction"] = normalize_direction(active.get("cls_direction", "pred_to_target"))
        feature_set = str(active.get("feature_set", "full")).strip().lower()
        if feature_set not in {"full", "losses_only"}:
            raise ValueError("YOLOv10 null_detect.feature_set must be full or losses_only.")
        parsed["null_detect_feature_set"] = feature_set
    elif uncertainty == "layer_grad":
        layer_cfg = as_dict(output.get("layer_grad", {}))
        grad = as_dict(layer_cfg.get("gradient", {}))
        target = str(grad.get("target", "null_target")).strip().lower()
        if target not in {"null_target", "cand_target"}:
            raise ValueError("YOLOv10 layer_grad.gradient.target must be null_target or cand_target.")
        for forbidden in ("obj_loss", "obj_direction", "bbox_direction", "layer"):
            if forbidden in grad:
                raise ValueError(f"YOLOv10 layer_grad does not support gradient.{forbidden}.")
        scalar = normalize_list(grad.get("scalar", ["bbox_loss", "cls_loss"]), lower=True)
        if "loss" in scalar:
            scalar = ["bbox_loss", "cls_loss"]
        for value in scalar:
            if value not in {"bbox_loss", "cls_loss"}:
                raise ValueError("YOLOv10 layer_grad.gradient.scalar supports bbox_loss and cls_loss only.")
        reduction = normalize_list(grad.get("reduction", ["l1_norm", "l2_norm", "min", "max", "mean", "std"]))
        if not reduction:
            raise ValueError("YOLOv10 layer_grad requires reduction metrics; raw gradient saving is not supported.")
        bbox_loss = normalize_loss(grad.get("bbox_loss", "l1"), "l1", {"l1", "l2"}, "layer_grad.gradient.bbox_loss")
        cls_loss = normalize_loss(grad.get("cls_loss", "bcewithlogits"), "bcewithlogits", {"bcewithlogits", "kl"}, "layer_grad.gradient.cls_loss")
        bbox_layers = normalize_list(grad.get("bbox_layer", ["model.23.one2one_cv2.0.2"]))
        cls_layers = normalize_list(grad.get("cls_layer", ["model.23.one2one_cv3.0.2"]))
        if "bbox_loss" in scalar and not bbox_layers:
            raise ValueError("YOLOv10 layer_grad.gradient.bbox_layer must not be empty when bbox_loss is active.")
        if "cls_loss" in scalar and not cls_layers:
            raise ValueError("YOLOv10 layer_grad.gradient.cls_layer must not be empty when cls_loss is active.")
        for layer_name in bbox_layers:
            if not layer_name.startswith("model.23.one2one_cv2."):
                raise ValueError("YOLOv10 bbox_loss layer_grad supports only one-to-one bbox head layers: model.23.one2one_cv2.*")
        for layer_name in cls_layers:
            if not layer_name.startswith("model.23.one2one_cv3."):
                raise ValueError("YOLOv10 cls_loss layer_grad supports only one-to-one cls head layers: model.23.one2one_cv3.*")
        parsed["layer_grad"] = {
            "scalar": scalar,
            "layers_by_scalar": {
                "bbox_loss": bbox_layers if "bbox_loss" in scalar else [],
                "cls_loss": cls_layers if "cls_loss" in scalar else [],
            },
            "reduction": reduction,
            "bbox_loss": bbox_loss,
            "cls_loss": cls_loss,
            "cls_direction": normalize_direction(grad.get("cls_direction", "pred_to_target")),
            "target": target,
            "cand_score_threshold": as_float(grad.get("cand_score_threshold", 0.0), 0.0),
            "cand_iou_threshold": as_float(grad.get("cand_iou_threshold", active.get("iou_threshold", 0.45)), 0.45),
        }
    return parsed


__all__ = ["parse_yolov10_output_config"]
