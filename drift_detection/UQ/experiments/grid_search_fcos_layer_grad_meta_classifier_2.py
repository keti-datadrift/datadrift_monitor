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
from commands.predict.fcos.common import select_fcos_post_nms  # noqa: E402
from commands.predict.fcos.layer_grad import (  # noqa: E402
    _FcosLossFlattenCache,
    _build_fcos_losses,
    _gradient_to_np_array,
    _parse_fcos_layer_grad_config,
    _prediction_class_name,
    _safe_npz_key,
    _source_indices_from_boxlist,
)
from commands.predict.fcos.utils import (  # noqa: E402
    build_fcos_dense_candidate_cache,
    ensure_fcos_selected_indices,
    fcos_candidate_mask_from_cache,
)
from commands.utils.predict_utils import (  # noqa: E402
    expand_layer_names,
    format_gradient_output,
    resolve_layer_parameter,
)
from commands.utils.predict_utils import build_detector  # noqa: E402
from dataloaders.fcos import create_dataloader  # noqa: E402

OBJECT_DETECTOR_CONFIG = r"object_detectors/configs/fcos/predict/coco.yaml"
META_CLASSIFIER_CONFIG = (
    r"meta_models/configs/meta_classifier/train.yaml"
)

GT_ROOT = ""

GRID_NAME = ""

RUN_LAYER_GRAD = True
RUN_META_CLASSIFIER = True
REUSE_EXISTING = False

MAX_COMBINATIONS = None

TARGETS = ["cand_target", "null_target"]
BBOX_LOSSES = ["l1", "l2"]
CLS_LOSSES = ["bcewithlogits", "kl"]
CNT_LOSSES = ["bcewithlogits", "abs_diff"]


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
    raw = str(config.get("model", {}).get("type", "fcos")).strip().lower()
    return (
        "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip(
            "_"
        )
        or "fcos"
    )


def _abbr(value: str) -> str:
    aliases = {
        "cand_target": "cand",
        "null_target": "null",
        "pred_to_target": "pred",
        "target_to_pred": "rev",
        "bcewithlogits": "bce",
        "abs_diff": "abs",
        "signed_diff": "signed",
        "bbox_loss": "bbox",
        "cls_loss": "cls",
        "cnt_loss": "cnt",
    }
    return aliases.get(value, value)


def _valid_cls_directions(cls_loss: str) -> list[str]:
    if cls_loss == "kl":
        return ["pred_to_target", "target_to_pred"]
    return ["pred_to_target"]


def _valid_cnt_directions(_cnt_loss: str) -> list[str]:
    return ["pred_to_target"]


def _combo_slug(combo: dict) -> str:
    term = combo["term"]
    if term == "bbox_loss":
        spec = f"b-{combo['bbox_loss']}-{_abbr(combo['bbox_direction'])}"
    elif term == "cls_loss":
        spec = f"c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
    elif term == "cnt_loss":
        spec = f"cnt-{_abbr(combo['cnt_loss'])}-{_abbr(combo['cnt_direction'])}"
    else:
        raise ValueError(f"Unsupported FCOS layer_grad term: {term}")
    return f"layer_grad_t-{_abbr(combo['target'])}__term-{_abbr(term)}__{spec}"


def _timestamped_combo_dir(root: Path, slug: str) -> Path:
    existing = sorted(root.glob(f"??-??-????_??;??_{slug}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{slug}"


def _timestamped_meta_dir(root: Path, index: int, target: str) -> Path:
    suffix = f"m{int(index):03d}_{_abbr(target)}"
    existing = sorted(root.glob(f"??-??-????_??;??_{suffix}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{suffix}"


def _base_combo(target: str, term: str) -> dict:
    return {
        "target": target,
        "term": term,
        "bbox_loss": "l1",
        "bbox_direction": "pred_to_target",
        "cls_loss": "bcewithlogits",
        "cls_direction": "pred_to_target",
        "cnt_loss": "bcewithlogits",
        "cnt_direction": "pred_to_target",
    }


def iter_term_combinations():
    for target in TARGETS:
        for bbox_loss in BBOX_LOSSES:
            combo = _base_combo(target, "bbox_loss")
            combo.update({"bbox_loss": bbox_loss})
            yield combo

        for cls_loss in CLS_LOSSES:
            for cls_direction in _valid_cls_directions(cls_loss):
                combo = _base_combo(target, "cls_loss")
                combo.update({"cls_loss": cls_loss, "cls_direction": cls_direction})
                yield combo

        for cnt_loss in CNT_LOSSES:
            for cnt_direction in _valid_cnt_directions(cnt_loss):
                combo = _base_combo(target, "cnt_loss")
                combo.update({"cnt_loss": cnt_loss, "cnt_direction": cnt_direction})
                yield combo


def _term_combo_key(combo: dict) -> tuple:
    return (
        combo["target"],
        combo["term"],
        combo["bbox_loss"],
        combo["bbox_direction"],
        combo["cls_loss"],
        combo["cls_direction"],
        combo["cnt_loss"],
        combo["cnt_direction"],
    )


def _meta_combo_slug(combo: dict) -> str:
    return (
        f"layer_grad_t-{_abbr(combo['target'])}"
        f"__b-{combo['bbox_loss']}-{_abbr(combo['bbox_direction'])}"
        f"__c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
        f"__cnt-{_abbr(combo['cnt_loss'])}-{_abbr(combo['cnt_direction'])}"
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
        cnt_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "cnt_loss"
        ]
        for bbox_combo in bbox_combos:
            for cls_combo in cls_combos:
                for cnt_combo in cnt_combos:
                    yield {
                        "target": target,
                        "bbox_loss": bbox_combo["bbox_loss"],
                        "bbox_direction": bbox_combo["bbox_direction"],
                        "cls_loss": cls_combo["cls_loss"],
                        "cls_direction": cls_combo["cls_direction"],
                        "cnt_loss": cnt_combo["cnt_loss"],
                        "cnt_direction": cnt_combo["cnt_direction"],
                        "input_roots": [
                            term_run_dirs[_term_combo_key(bbox_combo)],
                            term_run_dirs[_term_combo_key(cls_combo)],
                            term_run_dirs[_term_combo_key(cnt_combo)],
                        ],
                    }
                    count += 1
                    if MAX_COMBINATIONS is not None and count >= int(MAX_COMBINATIONS):
                        return


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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
            "cnt_loss": combo["cnt_loss"],
            "bbox_direction": combo["bbox_direction"],
            "cls_direction": combo["cls_direction"],
            "cnt_direction": combo["cnt_direction"],
        }
    )
    return config


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


def _read_mean_metrics(eval_csv: Path) -> dict:
    if not eval_csv.is_file():
        return {}
    with open(eval_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mean_row = None
    for row in rows:
        first_value = next(iter(row.values()), "")
        if str(first_value).strip().lower() == "mean":
            mean_row = row
            break
    if mean_row is None and rows:
        mean_row = rows[-1]
    out = {}
    for key in ("auroc", "ap", "fpr95", "ece", "ace"):
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


def _layer_grad_fieldnames(combo: dict, target_layer_map: dict, reductions: list[str]) -> list[str]:
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
    for layer_name in target_layer_map[combo["term"]]:
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
    if target == "cand_target":
        stages.insert(1, "candidate_search_sec")
    return stages


def _empty_stage_seconds(target: str, detector_inference_sec: float) -> dict:
    stages = {
        "detector_inference_sec": float(detector_inference_sec),
        "loss_compute_sec": 0.0,
        "backpropagation_sec": 0.0,
        "feature_compute_sec": 0.0,
    }
    if target == "cand_target":
        stages["candidate_search_sec"] = 0.0
    return stages


def _scalar_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def _run_fcos_layer_grad_terms_once(
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
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not bool(getattr(detector, "is_fcos", False)):
        raise ValueError("grid_search_fcos_layer_grad_meta_classifier_2.py requires an FCOS object detector config.")

    combo_cfgs = {
        _term_combo_key(combo): _parse_fcos_layer_grad_config(
            _prepare_layer_grad_config(base_config, combo)
        )
        for combo, _run_dir in active
    }

    target_layer_map_by_key = {}
    layer_params_by_key = {}
    original_requires_grad = {}
    for combo, _run_dir in active:
        key = _term_combo_key(combo)
        layer_cfg = combo_cfgs[key]
        expanded = {
            target_value: expand_layer_names(
                detector,
                layer_cfg["target_layer_map"].get(target_value, []),
            )
            for target_value in layer_cfg["scalar"]
        }
        target_layer_map_by_key[key] = expanded
        params = [
            resolve_layer_parameter(detector, layer_name)
            for layer_name in expanded[combo["term"]]
        ]
        layer_params_by_key[key] = params
        for param in params:
            original_requires_grad[id(param)] = bool(param.requires_grad)
            param.requires_grad_(True)

    handles = {}
    profilers = {}
    raw_gradient_dirs = {}
    try:
        for combo, run_dir in active:
            key = _term_combo_key(combo)
            layer_cfg = combo_cfgs[key]
            reductions = layer_cfg["reduction"]
            run_dir.mkdir(parents=True, exist_ok=True)
            _save_yaml(
                _prepare_layer_grad_config(base_config, combo),
                run_dir / "grid_object_detector_config.yaml",
            )
            csv_file = open(run_dir / "layer_grad.csv", "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                csv_file,
                fieldnames=_layer_grad_fieldnames(
                    combo,
                    target_layer_map_by_key[key],
                    reductions,
                ),
            )
            writer.writeheader()
            handles[key] = (combo, run_dir, csv_file, writer)
            profilers[key] = StageTimingProfiler(
                run_dir=run_dir,
                uncertainty="layer_grad",
                unit="bbox",
                stages=_timing_stages_for_target(combo["target"]),
                device=device,
            )
            if not reductions:
                gradients_dir = run_dir / "gradients"
                gradients_dir.mkdir(parents=True, exist_ok=True)
                raw_gradient_dirs[key] = gradients_dir

        cand_keys = [
            key
            for key, (combo, _run_dir, _csv_file, _writer) in handles.items()
            if combo["target"] == "cand_target"
        ]

        for batch_idx, (images, targets) in enumerate(
            tqdm(
                dataloader,
                desc="Object Detector (predict - layer_grad multi v2)",
                total=len(dataloader),
            )
        ):
            image_list = _as_image_list(images)
            infer_batch = _prepare_infer_batch(detector, image_list, device, auto=False)[0]
            timing_ref = next(iter(profilers.values()))
            detector.zero_grad(set_to_none=True)
            fcos_preprocessed = detector.preprocess_images(infer_batch)
            t_detector = timing_ref.start()
            model_output = detector.forward_layer_grad(fcos_preprocessed, include_post_logits=False)
            row_prediction = model_output["post_prediction"]
            row_indices = model_output["post_indices"]
            selected = select_fcos_post_nms(detector, row_prediction, None, row_indices)
            selected_preds = selected[0]
            selected_indices = selected[3]
            detector_inference_sec = timing_ref.elapsed(t_detector)

            stage_by_key = {
                key: _empty_stage_seconds(combo["target"], detector_inference_sec)
                for key, (combo, _run_dir, _csv_file, _writer) in handles.items()
            }
            rows_by_key = {key: [] for key in handles}
            grad_arrays_by_key = {key: {} for key in handles}

            candidate_caches = {}
            if cand_keys:
                for sample_idx in range(len(image_list)):
                    t_candidate = timing_ref.start()
                    candidate_caches[sample_idx] = build_fcos_dense_candidate_cache(
                        model_output,
                        sample_idx,
                        combo_cfgs[cand_keys[0]]["cand_score_threshold"],
                        detach=True,
                    )
                    elapsed = timing_ref.elapsed(t_candidate)
                    for key in cand_keys:
                        stage_by_key[key]["candidate_search_sec"] += elapsed

            flat_caches = {
                sample_idx: _FcosLossFlattenCache(model_output, sample_idx, device)
                for sample_idx in range(len(image_list))
            }
            row_contexts = []
            batch_items = 0

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det = (
                    selected_preds[sample_idx]
                    if selected_preds and sample_idx < len(selected_preds)
                    else torch.zeros((0, 6), dtype=torch.float32, device=device)
                )
                raw_keep = ensure_fcos_selected_indices(
                    selected_indices,
                    selected_preds,
                    sample_idx,
                ).to(device=device)
                batch_items += int(det.shape[0])

                for pred_idx in range(int(det.shape[0])):
                    raw_idx = int(raw_keep[pred_idx].detach().cpu().item())
                    final_box = det[pred_idx, :4].detach()
                    final_cls = int(det[pred_idx, 5].detach().cpu().item()) if det.shape[1] > 5 else 0
                    base_row = {
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

                    null_sources = []
                    detections = model_output["detections"]
                    if sample_idx < len(detections) and pred_idx < len(detections[sample_idx]):
                        level, loc_idx, _raw, _cls_one_based = _source_indices_from_boxlist(
                            detections[sample_idx],
                            pred_idx,
                        )
                        null_sources = [(level, loc_idx)]

                    cand_sources = []
                    if cand_keys:
                        cache = candidate_caches.get(sample_idx)
                        t_candidate = timing_ref.start()
                        cand_mask, _ious = fcos_candidate_mask_from_cache(
                            cache,
                            final_box.float(),
                            final_cls,
                            combo_cfgs[cand_keys[0]]["cand_iou_threshold"],
                        )
                        elapsed = timing_ref.elapsed(t_candidate)
                        for key in cand_keys:
                            stage_by_key[key]["candidate_search_sec"] += elapsed
                        if cache is not None and cache.levels is not None and cache.location_indices is not None:
                            candidate_indices = torch.where(cand_mask)[0]
                            if int(candidate_indices.numel()) > 0:
                                levels = cache.levels[candidate_indices].detach().cpu().tolist()
                                loc_indices = cache.location_indices[candidate_indices].detach().cpu().tolist()
                                cand_sources = [
                                    (int(level), int(loc_idx))
                                    for level, loc_idx in zip(levels, loc_indices)
                                ]

                    row_contexts.append(
                        {
                            "sample_idx": sample_idx,
                            "pred_idx": pred_idx,
                            "raw_idx": raw_idx,
                            "final_box": final_box,
                            "final_cls": final_cls,
                            "base_row": base_row,
                            "cand_sources": cand_sources,
                            "null_sources": null_sources,
                        }
                    )

            expected_grad_calls = 0
            for ctx in row_contexts:
                for key, (combo, _run_dir, _csv_file, _writer) in handles.items():
                    sources = ctx["cand_sources"] if combo["target"] == "cand_target" else ctx["null_sources"]
                    if sources:
                        expected_grad_calls += 1

            grad_call_index = 0
            for ctx in row_contexts:
                for key, (combo, _run_dir, _csv_file, _writer) in handles.items():
                    layer_cfg = combo_cfgs[key]
                    target_value = combo["term"]
                    sources = ctx["cand_sources"] if combo["target"] == "cand_target" else ctx["null_sources"]
                    row = dict(ctx["base_row"])

                    losses = _build_fcos_losses(
                        target_mode=combo["target"],
                        target_values=[target_value],
                        model_output=model_output,
                        image_idx=ctx["sample_idx"],
                        pred_idx=ctx["pred_idx"],
                        final_box=ctx["final_box"],
                        final_cls=ctx["final_cls"],
                        raw_idx=ctx["raw_idx"],
                        cand_score_threshold=layer_cfg["cand_score_threshold"],
                        cand_iou_threshold=layer_cfg["cand_iou_threshold"],
                        bbox_loss=combo["bbox_loss"],
                        cls_loss=combo["cls_loss"],
                        cnt_loss=combo["cnt_loss"],
                        bbox_direction=combo["bbox_direction"],
                        cls_direction=combo["cls_direction"],
                        cnt_direction=combo["cnt_direction"],
                        timing=profilers[key],
                        timing_accumulator=stage_by_key[key],
                        candidate_sources=sources,
                        flat_cache=flat_caches[ctx["sample_idx"]],
                    )
                    scalar = losses.get(target_value)
                    layer_names = target_layer_map_by_key[key][target_value]
                    params = layer_params_by_key[key]
                    reductions = layer_cfg["reduction"]

                    if scalar is None:
                        for layer_name in layer_names:
                            grad_name = f"{target_value}_{layer_name}"
                            if reductions:
                                for metric in reductions:
                                    row[f"{grad_name}_{metric}"] = 0.0
                            else:
                                row[grad_name] = ""
                        rows_by_key[key].append(row)
                        del losses
                        continue

                    t_backprop = profilers[key].start()
                    grad_call_index += 1
                    grads = torch.autograd.grad(
                        scalar,
                        params,
                        retain_graph=(grad_call_index < expected_grad_calls),
                        allow_unused=True,
                    )
                    stage_by_key[key]["backpropagation_sec"] += profilers[key].elapsed(t_backprop)

                    t_feature = profilers[key].start()
                    for layer_idx, layer_name in enumerate(layer_names):
                        grad_name = f"{target_value}_{layer_name}"
                        grad_value = format_gradient_output(
                            grads[layer_idx],
                            vector_reduction=reductions,
                            map_reduction="none",
                        )
                        if reductions:
                            for metric in reductions:
                                value = grad_value.get(metric, 0.0) if isinstance(grad_value, dict) else 0.0
                                row[f"{grad_name}_{metric}"] = _scalar_to_float(value)
                        else:
                            array_key = (
                                f"s{ctx['sample_idx']:03d}_p{int(ctx['pred_idx']):06d}_"
                                f"r{int(ctx['raw_idx']):06d}_{_safe_npz_key(grad_name)}"
                            )
                            grad_arrays_by_key[key][array_key] = _gradient_to_np_array(grad_value)
                            row[grad_name] = f"gradients/layer_grad_batch_{batch_idx:06d}.npz::{array_key}"
                        del grad_value
                    stage_by_key[key]["feature_compute_sec"] += profilers[key].elapsed(t_feature)
                    rows_by_key[key].append(row)
                    del losses, scalar, grads

            for key, (_combo, run_dir, csv_file, writer) in handles.items():
                if grad_arrays_by_key[key]:
                    np.savez(
                        raw_gradient_dirs[key] / f"layer_grad_batch_{batch_idx:06d}.npz",
                        **grad_arrays_by_key[key],
                    )
                writer.writerows(rows_by_key[key])
                csv_file.flush()
                profilers[key].record(
                    num_images=len(image_list),
                    num_predictions=batch_items,
                    stage_seconds=stage_by_key[key],
                )

            del row_contexts, rows_by_key, grad_arrays_by_key
            del candidate_caches, flat_caches
            del infer_batch, fcos_preprocessed, model_output
            del row_prediction, row_indices, selected, selected_preds, selected_indices
            del image_list
            detector.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for params in layer_params_by_key.values():
            for param in params:
                param.requires_grad_(original_requires_grad.get(id(param), bool(param.requires_grad)))
        for _combo, _run_dir, csv_file, _writer in handles.values():
            csv_file.close()
        for profiler in profilers.values():
            profiler.save()
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _result_row(combo: dict, metrics: dict) -> dict:
    return {
        "target": combo["target"],
        "bbox_loss": combo["bbox_loss"],
        "bbox_direction": combo["bbox_direction"],
        "cls_loss": combo["cls_loss"],
        "cls_direction": combo["cls_direction"],
        "cnt_loss": combo["cnt_loss"],
        "cnt_direction": combo["cnt_direction"],
        "auroc": metrics.get("auroc", ""),
        "ap": metrics.get("ap", ""),
        "fpr95": metrics.get("fpr95", ""),
        "ece": metrics.get("ece", ""),
        "ace": metrics.get("ace", ""),
    }


def _comparison_key(row: dict) -> tuple:
    return (
        row["bbox_loss"],
        row["bbox_direction"],
        row["cls_loss"],
        row["cls_direction"],
        row["cnt_loss"],
        row["cnt_direction"],
    )


def _write_results(out_dir: Path, rows: list[dict]) -> None:
    result_fields = [
        "target",
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "cnt_loss",
        "cnt_direction",
        "auroc",
        "ap",
        "fpr95",
        "ece",
        "ace",
    ]
    _write_csv(out_dir / "grid_results.csv", rows, result_fields)

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
                "cnt_loss": key[4],
                "cnt_direction": key[5],
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
        "cnt_loss",
        "cnt_direction",
        "cand_auroc",
        "null_auroc",
        "delta_auroc",
        "cand_ap",
        "null_ap",
        "delta_ap",
        "null_better_auroc",
        "null_better_ap",
    ]
    _write_csv(out_dir / "target_pair_comparison.csv", compare_rows, comparison_fields)
    _write_csv(
        out_dir / "better_null_results.csv",
        [row for row in compare_rows if row["null_better_auroc"]],
        comparison_fields,
    )


def main() -> None:
    od_config_path = _resolve_path(OBJECT_DETECTOR_CONFIG)
    meta_config_path = _resolve_path(META_CLASSIFIER_CONFIG)
    od_base_config = _load_yaml(od_config_path)
    meta_base_config = _load_yaml(meta_config_path)
    if str(od_base_config.get("model", {}).get("type", "")).strip().lower() != "fcos":
        raise ValueError(
            "grid_search_fcos_layer_grad_meta_classifier_2.py requires an FCOS object detector config."
        )
    dataset = _dataset_name(od_base_config)
    model = _model_name(od_base_config)

    grid_name = (
        GRID_NAME.strip()
        or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_fcos_layer_grad_grid_v2"
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
        / "meta_classifier"
        / "runs"
        / model
        / "train"
        / dataset
        / grid_name
    )
    od_grid_root.mkdir(parents=True, exist_ok=True)
    meta_grid_root.mkdir(parents=True, exist_ok=True)

    gt_root = (
        GT_ROOT.strip()
        or str(meta_base_config.get("dataset", {}).get("gt_root", "")).strip()
    )
    if not gt_root:
        raise ValueError(
            "GT_ROOT is empty and META_CLASSIFIER_CONFIG has no dataset.gt_root."
        )

    term_run_dirs: dict[tuple, Path] = {}
    combo_dirs = []
    term_combo_list = list(iter_term_combinations())
    print(f"Layer-grad term runs: {len(term_combo_list)}", flush=True)
    for idx, combo in enumerate(term_combo_list, start=1):
        slug = _combo_slug(combo)
        print(f"[OD {idx}/{len(term_combo_list)}] {slug}", flush=True)
        layer_dir = _timestamped_combo_dir(od_grid_root, slug)
        term_run_dirs[_term_combo_key(combo)] = layer_dir
        combo_dirs.append((combo, layer_dir))

    _run_fcos_layer_grad_terms_once(od_base_config, combo_dirs)

    rows = []
    meta_combo_list = list(iter_meta_combinations(term_run_dirs))
    cand_count = sum(1 for combo in meta_combo_list if combo["target"] == "cand_target")
    null_count = sum(1 for combo in meta_combo_list if combo["target"] == "null_target")
    print(
        f"Meta-classifier combinations: {len(meta_combo_list)} "
        f"(cand_target={cand_count}, null_target={null_count})",
        flush=True,
    )
    for idx, combo in enumerate(meta_combo_list, start=1):
        slug = _meta_combo_slug(combo)
        print(f"[META {idx}/{len(meta_combo_list)}] {slug}", flush=True)

        meta_dir = _timestamped_meta_dir(meta_grid_root, idx, combo["target"])
        eval_csv = meta_dir / "results" / "evaluation_results.csv"
        if RUN_META_CLASSIFIER and not (REUSE_EXISTING and eval_csv.is_file()):
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_config = _prepare_meta_config(
                meta_base_config, combo["input_roots"], gt_root
            )
            meta_config_path = meta_dir / "grid_meta_classifier_config.yaml"
            _save_yaml(meta_config, meta_config_path)
            _run(
                [
                    sys.executable,
                    "meta_models/main.py",
                    "--config",
                    str(meta_config_path),
                    "--run-dir",
                    str(meta_dir),
                ]
            )

        metrics = _read_mean_metrics(eval_csv)
        rows.append(_result_row(combo, metrics))
        _write_results(meta_grid_root, rows)

    print(f"Saved grid results: {meta_grid_root / 'grid_results.csv'}")
    print(f"Saved pair comparison: {meta_grid_root / 'target_pair_comparison.csv'}")
    print(f"Saved null comparison: {meta_grid_root / 'better_null_results.csv'}")


if __name__ == "__main__":
    main()
