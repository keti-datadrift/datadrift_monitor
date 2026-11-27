import csv
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_DETECTORS_ROOT = REPO_ROOT / "object_detectors"
if str(OBJECT_DETECTORS_ROOT) not in sys.path:
    sys.path.insert(0, str(OBJECT_DETECTORS_ROOT))

from commands.predict.common import (  # noqa: E402
    StageTimingProfiler,
    _as_image_list,
    _prepare_infer_batch,
)
from commands.utils.predict_utils import (  # noqa: E402
    _add_elapsed_timing,
    _flatten_raw_prediction_layers,
    _start_timing,
    build_detector,
    build_layer_target_scalar_bbox,
    build_yolo_candidate_cache,
    expand_layer_names,
    format_gradient_output,
    parse_output_config,
    resolve_layer_parameter,
    yolo_candidate_mask_from_cache,
)
from dataloaders.yolov5 import create_dataloader  # noqa: E402

OBJECT_DETECTOR_CONFIG = r"object_detectors/configs/yolov5/predict/coco.yaml"
META_CLASSIFIER_CONFIG = (
    r"meta_models/configs/meta_classifier/train.yaml"
)
META_REGRESSOR_CONFIG = (
    r"meta_models/configs/meta_regressor/train.yaml"
)

GT_ROOT = "object_detectors/runs/yolov5/predict/coco/06-15-2026_18;54_gt"

GRID_NAME = "06-16-2026_06;46_layer_grad_grid"

RUN_LAYER_GRAD = True
RUN_META_CLASSIFIER = True
RUN_META_REGRESSOR = True
REUSE_EXISTING = True

MAX_COMBINATIONS = None


TARGETS = ["cand_target", "null_target", "delta_target"]
BBOX_LOSSES_BY_TARGET = {
    "cand_target": ["box_l1", "box_l2"],
    "null_target": ["box_l1", "box_l2"],
    "delta_target": ["box_l1", "box_l2"],
}
CLS_LOSSES = ["bcewithlogits", "kl"]
OBJ_LOSSES = ["bcewithlogits", "abs_diff"]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _dataset_name(config: dict) -> str:
    raw = config.get("dataset", {}).get("used_dataset", "unknown")
    if isinstance(raw, str):
        names = [raw.strip().lower()]
    elif isinstance(raw, (list, tuple)):
        names = [str(v).strip().lower() for v in raw if str(v).strip()]
    else:
        names = []
    if not names:
        return "unknown"
    return "-".join(
        "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
        for name in names
    )


def _model_name(config: dict) -> str:
    raw = str(config.get("model", {}).get("type", "yolov5")).strip().lower()
    return (
        "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip(
            "_"
        )
        or "yolov5"
    )


def _abbr(value: str) -> str:
    aliases = {
        "cand_target": "cand",
        "null_target": "null",
        "delta_target": "delta",
        "pred_to_target": "pred",
        "target_to_pred": "rev",
        "bcewithlogits": "bce",
        "abs_diff": "abs",
        "signed_diff": "signed",
        "bbox_loss": "bbox",
        "cls_loss": "cls",
        "obj_loss": "obj",
    }
    return aliases.get(value, value)


def _combo_slug(combo: dict) -> str:
    term = combo["term"]
    if term == "bbox_loss":
        spec = f"b-{combo['bbox_loss']}-{_abbr(combo['bbox_direction'])}"
    elif term == "cls_loss":
        spec = f"c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
    elif term == "obj_loss":
        spec = f"o-{_abbr(combo['obj_loss'])}-{_abbr(combo['obj_direction'])}"
    else:
        raise ValueError(f"Unsupported term: {term}")
    return f"layer_grad_t-{_abbr(combo['target'])}__term-{_abbr(term)}__{spec}"


def _timestamped_combo_dir(root: Path, slug: str) -> Path:
    existing = sorted(root.glob(f"??-??-????_??;??_{slug}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{slug}"


def _valid_cls_directions(cls_loss: str) -> list[str]:
    if cls_loss == "kl":
        return ["pred_to_target", "target_to_pred"]
    return ["pred_to_target"]


def _valid_bbox_directions(_bbox_loss: str) -> list[str]:
    return ["pred_to_target"]


def _valid_obj_directions(obj_loss: str) -> list[str]:
    if obj_loss in {"bcewithlogits", "abs_diff"}:
        return ["pred_to_target"]
    return ["pred_to_target", "target_to_pred"]


def iter_term_combinations():
    for target in TARGETS:
        for bbox_loss in BBOX_LOSSES_BY_TARGET[target]:
            for bbox_direction in _valid_bbox_directions(bbox_loss):
                yield {
                    "target": target,
                    "term": "bbox_loss",
                    "bbox_loss": bbox_loss,
                    "bbox_direction": bbox_direction,
                    "cls_loss": "bcewithlogits",
                    "cls_direction": "pred_to_target",
                    "obj_loss": "bcewithlogits",
                    "obj_direction": "pred_to_target",
                }

        for cls_loss in CLS_LOSSES:
            for cls_direction in _valid_cls_directions(cls_loss):
                yield {
                    "target": target,
                    "term": "cls_loss",
                    "bbox_loss": "box_l1",
                    "bbox_direction": "pred_to_target",
                    "cls_loss": cls_loss,
                    "cls_direction": cls_direction,
                    "obj_loss": "bcewithlogits",
                    "obj_direction": "pred_to_target",
                }

        for obj_loss in OBJ_LOSSES:
            for obj_direction in _valid_obj_directions(obj_loss):
                yield {
                    "target": target,
                    "term": "obj_loss",
                    "bbox_loss": "box_l1",
                    "bbox_direction": "pred_to_target",
                    "cls_loss": "bcewithlogits",
                    "cls_direction": "pred_to_target",
                    "obj_loss": obj_loss,
                    "obj_direction": obj_direction,
                }


def _term_combo_key(combo: dict) -> tuple:
    return (
        combo["target"],
        combo["term"],
        combo["bbox_loss"],
        combo["bbox_direction"],
        combo["cls_loss"],
        combo["cls_direction"],
        combo["obj_loss"],
        combo["obj_direction"],
    )


def _meta_combo_slug(combo: dict) -> str:
    return (
        f"layer_grad_t-{_abbr(combo['target'])}"
        f"__b-{combo['bbox_loss']}-{_abbr(combo['bbox_direction'])}"
        f"__c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
        f"__o-{_abbr(combo['obj_loss'])}-{_abbr(combo['obj_direction'])}"
    )


def iter_meta_combinations(term_run_dirs: dict[tuple, Path]):
    count = 0
    term_combos = list(iter_term_combinations())
    for target in TARGETS:
        bbox_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "bbox_loss"
        ]
        cls_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "cls_loss"
        ]
        obj_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "obj_loss"
        ]
        for bbox_combo in bbox_combos:
            for cls_combo in cls_combos:
                for obj_combo in obj_combos:
                    yield {
                        "target": target,
                        "bbox_loss": bbox_combo["bbox_loss"],
                        "bbox_direction": bbox_combo["bbox_direction"],
                        "cls_loss": cls_combo["cls_loss"],
                        "cls_direction": cls_combo["cls_direction"],
                        "obj_loss": obj_combo["obj_loss"],
                        "obj_direction": obj_combo["obj_direction"],
                        "input_roots": [
                            term_run_dirs[_term_combo_key(bbox_combo)],
                            term_run_dirs[_term_combo_key(cls_combo)],
                            term_run_dirs[_term_combo_key(obj_combo)],
                        ],
                    }
                    count += 1
                    if MAX_COMBINATIONS is not None and count >= int(MAX_COMBINATIONS):
                        return


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _safe_npz_key(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _gradient_to_np_array(value):
    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _scalar_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _prepare_layer_grad_config(base_config: dict, combo: dict) -> dict:
    config = deepcopy(base_config)
    config["mode"] = "predict"
    output_cfg = config.setdefault("output", {})
    output_cfg["uncertainty"] = "layer_grad"
    layer_grad_cfg = output_cfg.setdefault("layer_grad", {})
    layer_grad_cfg.setdefault("save_csv", {})["enabled"] = True
    grad_cfg = layer_grad_cfg.setdefault("gradient", {})
    grad_cfg.update(
        {
            "target": combo["target"],
            "scalar": [combo["term"]],
            "bbox_loss": combo["bbox_loss"],
            "cls_loss": combo["cls_loss"],
            "obj_loss": combo["obj_loss"],
            "bbox_direction": combo["bbox_direction"],
            "cls_direction": combo["cls_direction"],
            "obj_direction": combo["obj_direction"],
        }
    )
    return config


def _layer_grad_fieldnames(
    combo: dict, target_layers: list[str], reductions: list[str]
) -> list[str]:
    fieldnames = [
        "image_id",
        "image_path",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "score",
        "pred_class",
    ]
    for layer_name in target_layers:
        grad_key = f"{combo['term']}_{layer_name}"
        if reductions:
            fieldnames.extend(f"{grad_key}_{metric}" for metric in reductions)
        else:
            fieldnames.append(grad_key)
    return fieldnames


def _timing_stages_for_target(target: str) -> list[str]:
    stages = [
        "detector_inference_sec",
        "loss_compute_sec",
        "backpropagation_sec",
        "feature_compute_sec",
    ]
    if target in {"cand_target", "delta_target"}:
        stages.insert(1, "candidate_search_sec")
    return stages


def _empty_stage_seconds(
    target: str, detector_inference_sec: float, loss_prep_sec: float
) -> dict:
    stages = {
        "detector_inference_sec": float(detector_inference_sec),
        "loss_compute_sec": float(loss_prep_sec),
        "backpropagation_sec": 0.0,
        "feature_compute_sec": 0.0,
    }
    if target in {"cand_target", "delta_target"}:
        stages["candidate_search_sec"] = 0.0
    return stages


def _run_yolo_layer_grad_terms_once(
    base_config: dict, combo_dirs: list[tuple[dict, Path]]
) -> None:
    active = [
        (combo, run_dir)
        for combo, run_dir in combo_dirs
        if RUN_LAYER_GRAD
        and not (REUSE_EXISTING and (run_dir / "layer_grad.csv").is_file())
    ]
    if not active:
        return

    config = _prepare_layer_grad_config(base_config, active[0][0])
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    target_layers = parsed["layer_target_layers"]
    layer_map_reduction = parsed["layer_map_reduction"]
    layer_gradient_reduction = parsed["layer_gradient_reduction"]
    cand_score_threshold = float(parsed.get("layer_cand_score_threshold", 0.01))

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError(
            "Loaded 0 images. Check dataset root/image_dir/split configuration in YAML."
        )

    detector, device = build_detector(config)
    if bool(getattr(detector, "is_faster_rcnn", False)) or bool(
        getattr(detector, "is_fcos", False)
    ):
        raise ValueError(
            "grid_search_yolo_layer_grad_meta_classifier.py only supports YOLO layer_grad runs."
        )

    target_layers = expand_layer_names(detector.model, target_layers)
    layer_params = [
        resolve_layer_parameter(detector.model, layer_name)
        for layer_name in target_layers
    ]
    original_requires_grad = [bool(param.requires_grad) for param in layer_params]
    for param in layer_params:
        param.requires_grad_(True)

    handles = {}
    profilers = {}
    raw_gradient_dirs = {}
    save_raw_gradients = not layer_gradient_reduction
    try:
        for combo, run_dir in active:
            run_dir.mkdir(parents=True, exist_ok=True)
            _save_yaml(
                _prepare_layer_grad_config(base_config, combo),
                run_dir / "grid_object_detector_config.yaml",
            )
            csv_file = open(
                run_dir / "layer_grad.csv", "w", newline="", encoding="utf-8"
            )
            writer = csv.DictWriter(
                csv_file,
                fieldnames=_layer_grad_fieldnames(
                    combo, target_layers, layer_gradient_reduction
                ),
            )
            writer.writeheader()
            handles[_term_combo_key(combo)] = (combo, run_dir, csv_file, writer)
            profilers[_term_combo_key(combo)] = StageTimingProfiler(
                run_dir=run_dir,
                uncertainty="layer_grad",
                unit="bbox",
                stages=_timing_stages_for_target(combo["target"]),
                device=device,
            )
            if save_raw_gradients:
                gradients_dir = run_dir / "gradients"
                gradients_dir.mkdir(parents=True, exist_ok=True)
                raw_gradient_dirs[_term_combo_key(combo)] = gradients_dir

        for batch_idx, (images, targets) in enumerate(
            tqdm(
                dataloader,
                desc="Object Detector (predict - layer_grad multi)",
                total=len(dataloader),
            )
        ):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(
                detector, image_list, device, auto=False
            )

            t_detector = _start_timing(device)
            model_output = detector.model(infer_batch.detach(), augment=False)
            raw_prediction = (
                model_output[0]
                if isinstance(model_output, (tuple, list))
                else model_output
            )
            raw_logits = (
                model_output[1]
                if isinstance(model_output, (tuple, list)) and len(model_output) > 1
                else None
            )
            pred_layers = (
                model_output[2]
                if isinstance(model_output, (tuple, list))
                and len(model_output) > 2
                and isinstance(model_output[2], list)
                else None
            )
            raw_anchor_priors = (
                model_output[3]
                if isinstance(model_output, (tuple, list)) and len(model_output) > 3
                else None
            )
            shared_detector_timing = {"detector_inference_sec": 0.0}
            _add_elapsed_timing(
                shared_detector_timing, "detector_inference_sec", t_detector, device
            )

            with torch.no_grad():
                t_nms = _start_timing(device)
                (
                    selected_preds,
                    _selected_logits,
                    _selected_objectness,
                    selected_indices,
                ) = detector.non_max_suppression(
                    prediction=raw_prediction,
                    logits=raw_logits,
                    conf_thres=float(
                        getattr(
                            detector,
                            "conf_thresh",
                            getattr(detector, "confidence", 0.25),
                        )
                    ),
                    iou_thres=float(getattr(detector, "iou_thresh", 0.45)),
                    classes=getattr(detector, "filter_classes", None),
                    agnostic=bool(
                        getattr(
                            detector,
                            "agnostic_nms",
                            getattr(detector, "agnostic", False),
                        )
                    ),
                    max_det=(
                        int(getattr(detector, "max_det", 300))
                        if getattr(detector, "max_det", 300) is not None
                        else None
                    ),
                    return_indices=True,
                )
                _add_elapsed_timing(
                    shared_detector_timing, "detector_inference_sec", t_nms, device
                )
            detector_inference_sec = shared_detector_timing["detector_inference_sec"]

            raw_flat = _flatten_raw_prediction_layers(pred_layers)

            stage_by_key = {
                key: _empty_stage_seconds(combo["target"], detector_inference_sec, 0.0)
                for key, (combo, _run_dir, _csv_file, _writer) in handles.items()
            }
            rows_by_key = {key: [] for key in handles}
            grad_arrays_by_key = {key: {} for key in handles}
            batch_items = 0

            batch_size = int(raw_prediction.shape[0]) if raw_prediction.ndim >= 3 else 1
            iou_threshold = float(getattr(detector, "iou_thresh", 0.45))
            candidate_keys = [
                key
                for key, (
                    combo,
                    _run_dir,
                    _csv_file,
                    _writer,
                ) in handles.items()
                if combo["target"] in {"cand_target", "delta_target"}
            ]
            for sample_idx in range(batch_size):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det = (
                    selected_preds[sample_idx]
                    if selected_preds and sample_idx < len(selected_preds)
                    else torch.zeros((0, 6), device=device)
                )
                raw_keep_indices = (
                    selected_indices[sample_idx]
                    if selected_indices and sample_idx < len(selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                pred_img = raw_prediction[sample_idx]
                logit_img = (
                    raw_logits[sample_idx]
                    if raw_logits is not None
                    else pred_img[:, 5:]
                )
                raw_img = (
                    raw_flat[sample_idx]
                    if raw_flat is not None and raw_flat.ndim == 3
                    else None
                )
                anchor_img = (
                    raw_anchor_priors[sample_idx]
                    if raw_anchor_priors is not None and raw_anchor_priors.ndim >= 3
                    else (
                        raw_anchor_priors
                        if raw_anchor_priors is not None
                        and raw_anchor_priors.ndim == 2
                        and batch_size == 1
                        else None
                    )
                )
                candidate_cache = None
                if candidate_keys:
                    t_candidate = _start_timing(device)
                    candidate_cache = build_yolo_candidate_cache(
                        pred_img,
                        cand_score_threshold,
                    )
                    cache_timing = {"candidate_search_sec": 0.0}
                    _add_elapsed_timing(
                        cache_timing,
                        "candidate_search_sec",
                        t_candidate,
                        device,
                    )
                    for candidate_key in candidate_keys:
                        stage_by_key[candidate_key]["candidate_search_sec"] += cache_timing["candidate_search_sec"]

                batch_items += int(det.shape[0])
                for bbox_idx in range(int(det.shape[0])):
                    if bbox_idx >= int(raw_keep_indices.shape[0]):
                        raise RuntimeError(
                            "YOLO grid layer_grad selected_indices is shorter than selected predictions. "
                            f"sample_idx={sample_idx}, pred_idx={bbox_idx}, indices={int(raw_keep_indices.shape[0])}"
                        )
                    raw_idx = int(raw_keep_indices[bbox_idx].detach().cpu().item())
                    cls_idx = int(det[bbox_idx, 5].detach().cpu().item())
                    base_row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "pred_idx": bbox_idx,
                        "raw_pred_idx": raw_idx,
                        "xmin": float(det[bbox_idx, 0].detach().cpu().item()),
                        "ymin": float(det[bbox_idx, 1].detach().cpu().item()),
                        "xmax": float(det[bbox_idx, 2].detach().cpu().item()),
                        "ymax": float(det[bbox_idx, 3].detach().cpu().item()),
                        "score": float(det[bbox_idx, 4].detach().cpu().item()),
                        "pred_class": (
                            detector.names[cls_idx]
                            if detector.names is not None
                            else cls_idx
                        ),
                    }
                    anchor_row = (
                        anchor_img[raw_idx]
                        if (anchor_img is not None and raw_idx < anchor_img.shape[0])
                        else None
                    )
                    candidate_mask = None
                    if candidate_cache is not None:
                        t_candidate = _start_timing(device)
                        candidate_mask, _candidate_ious = yolo_candidate_mask_from_cache(
                            candidate_cache,
                            raw_idx,
                            iou_threshold,
                        )
                        candidate_timing = {"candidate_search_sec": 0.0}
                        _add_elapsed_timing(
                            candidate_timing,
                            "candidate_search_sec",
                            t_candidate,
                            device,
                        )
                        for candidate_key in candidate_keys:
                            stage_by_key[candidate_key][
                                "candidate_search_sec"
                            ] += candidate_timing["candidate_search_sec"]

                    for key, (combo, _run_dir, _csv_file, _writer) in handles.items():
                        if combo["target"] == "delta_target":
                            candidate_count = (
                                int(candidate_mask.sum().detach().cpu().item())
                                if candidate_mask is not None
                                else 0
                            )
                            if candidate_count <= 0:
                                target_scalar = None
                            else:
                                cand_scalar = build_layer_target_scalar_bbox(
                                    target_value=combo["term"],
                                    pred_img=pred_img,
                                    logit_img=logit_img,
                                    raw_img=raw_img,
                                    raw_idx=raw_idx,
                                    iou_threshold=iou_threshold,
                                    pseudo_gt="cand",
                                    anchor_xywh=anchor_row,
                                    cand_score_threshold=cand_score_threshold,
                                    bbox_loss=combo["bbox_loss"],
                                    cls_loss=combo["cls_loss"],
                                    obj_loss=combo["obj_loss"],
                                    bbox_direction=combo["bbox_direction"],
                                    cls_direction=combo["cls_direction"],
                                    obj_direction=combo["obj_direction"],
                                    candidate_mask=candidate_mask,
                                    timing_accumulator=stage_by_key[key],
                                    timing_device=device,
                                )
                                ref_scalar = build_layer_target_scalar_bbox(
                                    target_value=combo["term"],
                                    pred_img=pred_img,
                                    logit_img=logit_img,
                                    raw_img=raw_img,
                                    raw_idx=raw_idx,
                                    iou_threshold=iou_threshold,
                                    pseudo_gt="uniform",
                                    anchor_xywh=anchor_row,
                                    cand_score_threshold=cand_score_threshold,
                                    bbox_loss=combo["bbox_loss"],
                                    cls_loss=combo["cls_loss"],
                                    obj_loss=combo["obj_loss"],
                                    bbox_direction=combo["bbox_direction"],
                                    cls_direction=combo["cls_direction"],
                                    obj_direction=combo["obj_direction"],
                                    candidate_mask=None,
                                    timing_accumulator=stage_by_key[key],
                                    timing_device=device,
                                )
                                target_scalar = cand_scalar
                                if ref_scalar is not None:
                                    ref_term = float(candidate_count) * ref_scalar
                                    target_scalar = (
                                        -ref_term
                                        if target_scalar is None
                                        else target_scalar - ref_term
                                    )
                        else:
                            target_scalar = build_layer_target_scalar_bbox(
                                target_value=combo["term"],
                                pred_img=pred_img,
                                logit_img=logit_img,
                                raw_img=raw_img,
                                raw_idx=raw_idx,
                                iou_threshold=iou_threshold,
                                pseudo_gt=(
                                    "uniform"
                                    if combo["target"] == "null_target"
                                    else "cand"
                                ),
                                anchor_xywh=anchor_row,
                                cand_score_threshold=cand_score_threshold,
                                bbox_loss=combo["bbox_loss"],
                                cls_loss=combo["cls_loss"],
                                obj_loss=combo["obj_loss"],
                                bbox_direction=combo["bbox_direction"],
                                cls_direction=combo["cls_direction"],
                                obj_direction=combo["obj_direction"],
                                candidate_mask=(
                                    candidate_mask
                                    if combo["target"] == "cand_target"
                                    else None
                                ),
                                timing_accumulator=stage_by_key[key],
                                timing_device=device,
                            )

                        row = dict(base_row)
                        if target_scalar is None:
                            for layer_name in target_layers:
                                grad_name = f"{combo['term']}_{layer_name}"
                                if save_raw_gradients:
                                    if combo["target"] == "delta_target":
                                        layer_idx = target_layers.index(layer_name)
                                        grad_value = format_gradient_output(
                                            torch.zeros_like(layer_params[layer_idx]),
                                            vector_reduction=layer_gradient_reduction,
                                            map_reduction=layer_map_reduction,
                                        )
                                        array_key = (
                                            f"s{sample_idx:03d}_p{int(bbox_idx):06d}_"
                                            f"r{int(raw_idx):06d}_{_safe_npz_key(grad_name)}"
                                        )
                                        grad_arrays_by_key[key][array_key] = (
                                            _gradient_to_np_array(grad_value)
                                        )
                                        row[grad_name] = (
                                            f"gradients/layer_grad_batch_{batch_idx:06d}.npz::{array_key}"
                                        )
                                    else:
                                        row[grad_name] = ""
                                else:
                                    for metric in layer_gradient_reduction:
                                        row[f"{grad_name}_{metric}"] = 0.0
                            rows_by_key[key].append(row)
                            continue

                        t_backprop = _start_timing(device)
                        grads = torch.autograd.grad(
                            target_scalar,
                            layer_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        _add_elapsed_timing(
                            stage_by_key[key], "backpropagation_sec", t_backprop, device
                        )

                        t_feature = _start_timing(device)
                        for layer_idx, layer_name in enumerate(target_layers):
                            grad_name = f"{combo['term']}_{layer_name}"
                            grad_value = format_gradient_output(
                                grads[layer_idx],
                                vector_reduction=layer_gradient_reduction,
                                map_reduction=layer_map_reduction,
                            )
                            if save_raw_gradients:
                                array_key = (
                                    f"s{sample_idx:03d}_p{int(bbox_idx):06d}_"
                                    f"r{int(raw_idx):06d}_{_safe_npz_key(grad_name)}"
                                )
                                grad_arrays_by_key[key][array_key] = (
                                    _gradient_to_np_array(grad_value)
                                )
                                row[grad_name] = (
                                    f"gradients/layer_grad_batch_{batch_idx:06d}.npz::{array_key}"
                                )
                            else:
                                for metric in layer_gradient_reduction:
                                    value = (
                                        grad_value.get(metric, 0.0)
                                        if isinstance(grad_value, dict)
                                        else 0.0
                                    )
                                    row[f"{grad_name}_{metric}"] = _scalar_to_float(
                                        value
                                    )
                        _add_elapsed_timing(
                            stage_by_key[key], "feature_compute_sec", t_feature, device
                        )
                        rows_by_key[key].append(row)
                        del target_scalar, grads

            for key, (_combo, run_dir, csv_file, writer) in handles.items():
                if save_raw_gradients and grad_arrays_by_key[key]:
                    np.savez(
                        raw_gradient_dirs[key]
                        / f"layer_grad_batch_{batch_idx:06d}.npz",
                        **grad_arrays_by_key[key],
                    )
                writer.writerows(rows_by_key[key])
                csv_file.flush()
                profilers[key].record(
                    num_images=len(image_list),
                    num_predictions=batch_items,
                    stage_seconds=stage_by_key[key],
                )

            del infer_batch, model_output, raw_prediction, raw_logits, raw_anchor_priors
            del raw_flat, selected_preds, selected_indices
    finally:
        for param, req_grad in zip(layer_params, original_requires_grad):
            param.requires_grad_(req_grad)
        for _combo, _run_dir, csv_file, _writer in handles.values():
            csv_file.close()
        for profiler in profilers.values():
            profiler.save()
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _prepare_meta_config(
    base_config: dict, input_roots: list[Path], gt_root: str
) -> dict:
    config = deepcopy(base_config)
    config["mode"] = "train"
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["input_root"] = [str(path) for path in input_roots]
    if gt_root:
        dataset_cfg["gt_root"] = str(_resolve_path(gt_root))
    return config


def _read_mean_metrics(eval_csv: Path, metric_keys=("auroc", "ap", "fpr95", "ece", "ace")) -> dict:
    if not eval_csv.is_file():
        return {}
    with open(eval_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    mean_row = None
    for row in rows:
        first_value = next(iter(row.values()), "")
        if str(first_value).strip().lower() == "mean":
            mean_row = row
            break
    if mean_row is None and rows:
        mean_row = rows[-1]
    out = {}
    for key in metric_keys:
        try:
            out[key] = float(mean_row.get(key, "nan"))
        except Exception:
            out[key] = float("nan")
    return out


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_key(row: dict) -> tuple:
    return (
        row["bbox_loss"],
        row["bbox_direction"],
        row["cls_loss"],
        row["cls_direction"],
        row["obj_loss"],
        row["obj_direction"],
    )


def main() -> None:
    od_config_path = _resolve_path(OBJECT_DETECTOR_CONFIG)
    meta_config_path = _resolve_path(META_CLASSIFIER_CONFIG)
    meta_regressor_config_path = _resolve_path(META_REGRESSOR_CONFIG)
    od_base_config = _load_yaml(od_config_path)
    meta_base_config = _load_yaml(meta_config_path)
    meta_regressor_base_config = _load_yaml(meta_regressor_config_path)
    dataset = _dataset_name(od_base_config)
    model = _model_name(od_base_config)

    grid_name = (
        GRID_NAME.strip()
        or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_layer_grad_grid"
    )
    od_grid_root = (
        REPO_ROOT
        / "object_detectors"
        / "runs"
        / model
        / "predict"
        / dataset
        / grid_name
    )
    meta_grid_root = (
        REPO_ROOT
        / "meta_models"
        / "runs"
        / "meta_classifier"
        / model
        / "train"
        / dataset
        / grid_name
    )
    meta_regressor_grid_root = (
        REPO_ROOT
        / "meta_models"
        / "runs"
        / "meta_regressor"
        / model
        / "train"
        / dataset
        / grid_name
    )
    od_grid_root.mkdir(parents=True, exist_ok=True)
    meta_grid_root.mkdir(parents=True, exist_ok=True)
    meta_regressor_grid_root.mkdir(parents=True, exist_ok=True)

    gt_root = (
        GT_ROOT.strip()
        or str(meta_base_config.get("dataset", {}).get("gt_root", "")).strip()
    )
    if not gt_root:
        raise ValueError(
            "GT_ROOT is empty and META_CLASSIFIER_CONFIG has no dataset.gt_root."
        )

    term_run_dirs: dict[tuple, Path] = {}
    term_combo_list = list(iter_term_combinations())
    print(f"Layer-grad term runs: {len(term_combo_list)}", flush=True)
    combo_dirs = []
    for idx, combo in enumerate(term_combo_list, start=1):
        slug = _combo_slug(combo)
        print(f"[OD {idx}/{len(term_combo_list)}] {slug}", flush=True)

        layer_dir = _timestamped_combo_dir(od_grid_root, slug)
        term_run_dirs[_term_combo_key(combo)] = layer_dir
        combo_dirs.append((combo, layer_dir))

    _run_yolo_layer_grad_terms_once(od_base_config, combo_dirs)

    rows = []
    meta_combo_list = list(iter_meta_combinations(term_run_dirs))
    print(f"Meta-classifier combinations: {len(meta_combo_list)}", flush=True)
    for idx, combo in enumerate(meta_combo_list, start=1):
        slug = _meta_combo_slug(combo)
        print(f"[META {idx}/{len(meta_combo_list)}] {slug}", flush=True)

        meta_dir = _timestamped_combo_dir(meta_grid_root, slug)
        eval_csv = meta_dir / "results" / "evaluation_results.csv"

        if RUN_META_CLASSIFIER and not (REUSE_EXISTING and eval_csv.is_file()):
            meta_config = _prepare_meta_config(
                meta_base_config, combo["input_roots"], gt_root
            )
            meta_config_path_i = meta_dir / "grid_meta_classifier_config.yaml"
            _save_yaml(meta_config, meta_config_path_i)
            _run(
                [
                    sys.executable,
                    "meta_models/main.py",
                    "--config",
                    str(meta_config_path_i),
                    "--run-dir",
                    str(meta_dir),
                ]
            )

        metrics = _read_mean_metrics(eval_csv)
        rows.append(
            {
                "target": combo["target"],
                "bbox_loss": combo["bbox_loss"],
                "bbox_direction": combo["bbox_direction"],
                "cls_loss": combo["cls_loss"],
                "cls_direction": combo["cls_direction"],
                "obj_loss": combo["obj_loss"],
                "obj_direction": combo["obj_direction"],
                "auroc": metrics.get("auroc", ""),
                "ap": metrics.get("ap", ""),
                "fpr95": metrics.get("fpr95", ""),
                "ece": metrics.get("ece", ""),
                "ace": metrics.get("ace", ""),
            }
        )
        _write_results(meta_grid_root, rows)

    print(f"Saved grid results: {meta_grid_root / 'grid_results.csv'}")
    print(f"Saved pair comparison: {meta_grid_root / 'target_pair_comparison.csv'}")
    print(f"Saved null comparison: {meta_grid_root / 'better_null_results.csv'}")
    print(f"Saved three-way comparison: {meta_grid_root / 'target_three_way_comparison.csv'}")

    regressor_rows = []
    print(f"Meta-regressor combinations: {len(meta_combo_list)}", flush=True)
    for idx, combo in enumerate(meta_combo_list, start=1):
        slug = _meta_combo_slug(combo)
        print(f"[REG {idx}/{len(meta_combo_list)}] {slug}", flush=True)

        meta_regressor_dir = _timestamped_combo_dir(meta_regressor_grid_root, slug)
        eval_csv = meta_regressor_dir / "results" / "evaluation_results.csv"

        if RUN_META_REGRESSOR and not (REUSE_EXISTING and eval_csv.is_file()):
            meta_regressor_config = _prepare_meta_config(
                meta_regressor_base_config, combo["input_roots"], gt_root
            )
            meta_regressor_config_path_i = meta_regressor_dir / "grid_meta_regressor_config.yaml"
            _save_yaml(meta_regressor_config, meta_regressor_config_path_i)
            _run(
                [
                    sys.executable,
                    "meta_models/main.py",
                    "--config",
                    str(meta_regressor_config_path_i),
                    "--run-dir",
                    str(meta_regressor_dir),
                ]
            )

        metrics = _read_mean_metrics(eval_csv, ("r2",))
        regressor_rows.append(
            {
                "target": combo["target"],
                "bbox_loss": combo["bbox_loss"],
                "bbox_direction": combo["bbox_direction"],
                "cls_loss": combo["cls_loss"],
                "cls_direction": combo["cls_direction"],
                "obj_loss": combo["obj_loss"],
                "obj_direction": combo["obj_direction"],
                "r2": metrics.get("r2", ""),
            }
        )
        _write_regressor_results(meta_regressor_grid_root, regressor_rows)

    print(f"Saved regressor grid results: {meta_regressor_grid_root / 'grid_results.csv'}")
    print(
        "Saved regressor three-way comparison: "
        f"{meta_regressor_grid_root / 'target_three_way_comparison.csv'}"
    )


def _write_results(out_dir: Path, rows: list[dict]) -> None:
    fields = [
        "target",
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
        "auroc",
        "ap",
        "fpr95",
        "ece",
        "ace",
    ]
    _write_csv(out_dir / "grid_results.csv", rows, fields)

    by_key = {}
    for row in rows:
        by_key.setdefault(_comparison_key(row), {})[row["target"]] = row

    compare_rows = []
    for key, pair in by_key.items():
        if "cand_target" not in pair or "null_target" not in pair:
            continue
        cand = pair["cand_target"]
        null = pair["null_target"]
        try:
            cand_auroc = float(cand["auroc"])
            null_auroc = float(null["auroc"])
            cand_ap = float(cand["ap"])
            null_ap = float(null["ap"])
        except Exception:
            continue
        compare_rows.append(
            {
                "bbox_loss": key[0],
                "bbox_direction": key[1],
                "cls_loss": key[2],
                "cls_direction": key[3],
                "obj_loss": key[4],
                "obj_direction": key[5],
                "cand_auroc": cand_auroc,
                "null_auroc": null_auroc,
                "delta_auroc": null_auroc - cand_auroc,
                "cand_ap": cand_ap,
                "null_ap": null_ap,
                "delta_ap": null_ap - cand_ap,
                "null_better_auroc": int(null_auroc > cand_auroc),
                "null_better_ap": int(null_ap > cand_ap),
            }
        )
    compare_rows.sort(key=lambda r: (r["delta_auroc"], r["delta_ap"]), reverse=True)
    comparison_fields = [
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
        "cand_auroc",
        "null_auroc",
        "delta_auroc",
        "cand_ap",
        "null_ap",
        "delta_ap",
        "null_better_auroc",
        "null_better_ap",
    ]
    _write_csv(
        out_dir / "target_pair_comparison.csv",
        compare_rows,
        comparison_fields,
    )
    _write_csv(
        out_dir / "better_null_results.csv",
        [row for row in compare_rows if row["null_better_auroc"]],
        comparison_fields,
    )

    metrics = ["auroc", "ap", "fpr95", "ece", "ace"]
    three_way_rows = []
    for key, grouped in by_key.items():
        if not all(target in grouped for target in TARGETS):
            continue
        row = {
            "bbox_loss": key[0],
            "bbox_direction": key[1],
            "cls_loss": key[2],
            "cls_direction": key[3],
            "obj_loss": key[4],
            "obj_direction": key[5],
        }
        for target in TARGETS:
            prefix = _abbr(target)
            for metric in metrics:
                row[f"{prefix}_{metric}"] = grouped[target].get(metric, "")
        three_way_rows.append(row)
    three_way_rows.sort(
        key=lambda r: (
            r["bbox_loss"],
            r["cls_loss"],
            r["cls_direction"],
            r["obj_loss"],
            r["obj_direction"],
        )
    )
    three_way_fields = [
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
    ]
    for target in TARGETS:
        prefix = _abbr(target)
        three_way_fields.extend(f"{prefix}_{metric}" for metric in metrics)
    _write_csv(
        out_dir / "target_three_way_comparison.csv",
        three_way_rows,
        three_way_fields,
    )


def _write_regressor_results(out_dir: Path, rows: list[dict]) -> None:
    fields = [
        "target",
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
        "r2",
    ]
    _write_csv(out_dir / "grid_results.csv", rows, fields)

    by_key = {}
    for row in rows:
        by_key.setdefault(_comparison_key(row), {})[row["target"]] = row

    three_way_rows = []
    for key, grouped in by_key.items():
        if not all(target in grouped for target in TARGETS):
            continue
        row = {
            "bbox_loss": key[0],
            "bbox_direction": key[1],
            "cls_loss": key[2],
            "cls_direction": key[3],
            "obj_loss": key[4],
            "obj_direction": key[5],
        }
        for target in TARGETS:
            row[f"{_abbr(target)}_r2"] = grouped[target].get("r2", "")
        three_way_rows.append(row)
    three_way_rows.sort(
        key=lambda r: (
            r["bbox_loss"],
            r["cls_loss"],
            r["cls_direction"],
            r["obj_loss"],
            r["obj_direction"],
        )
    )
    three_way_fields = [
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
    ]
    three_way_fields.extend(f"{_abbr(target)}_r2" for target in TARGETS)
    _write_csv(
        out_dir / "target_three_way_comparison.csv",
        three_way_rows,
        three_way_fields,
    )


if __name__ == "__main__":
    main()
