import csv
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_DETECTORS_ROOT = REPO_ROOT / "object_detectors"
if str(OBJECT_DETECTORS_ROOT) not in sys.path:
    sys.path.insert(0, str(OBJECT_DETECTORS_ROOT))

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch  # noqa: E402
from commands.predict.yolov10.config import parse_yolov10_output_config  # noqa: E402
from commands.predict.yolov10.features import build_yolov10_candidate_cache, yolov10_candidate_mask_from_cache  # noqa: E402
from commands.predict.yolov10.forward import run_yolov10_forward  # noqa: E402
from commands.predict.yolov10.layer_grad import _build_candidate_scalar_terms, _build_null_scalar_terms  # noqa: E402
from commands.predict.yolov10.rows import iter_yolov10_detection_rows  # noqa: E402
from commands.utils.predict_utils import (  # noqa: E402
    build_detector,
    expand_layer_names,
    map_grad_tensor_to_numbers,
    resolve_layer_parameter,
)
from dataloaders.yolov10 import create_dataloader  # noqa: E402

OBJECT_DETECTOR_CONFIG = r"object_detectors/configs/yolov10/predict/coco.yaml"
META_CLASSIFIER_CONFIG = r"meta_models/configs/meta_classifier/train.yaml"

GT_ROOT = ""
GRID_NAME = ""

RUN_LAYER_GRAD = True
RUN_META_CLASSIFIER = True
REUSE_EXISTING = False

MAX_COMBINATIONS = None

TARGETS = ["cand_target", "null_target"]
BBOX_LOSSES = ["l1", "l2"]
CLS_COMBOS = [
    ("bcewithlogits", "pred_to_target"),
    ("kl", "pred_to_target"),
    ("kl", "target_to_pred"),
]


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
    return "-".join("".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name) for name in names)


def _model_name(config: dict) -> str:
    raw = str(config.get("model", {}).get("type", "yolov10")).strip().lower()
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "yolov10"


def _abbr(value: str) -> str:
    aliases = {
        "null_target": "null",
        "pred_to_target": "pred",
        "target_to_pred": "rev",
        "bcewithlogits": "bce",
        "bbox_loss": "bbox",
        "cls_loss": "cls",
    }
    return aliases.get(value, value)


def _combo_slug(combo: dict) -> str:
    term = combo["term"]
    if term == "bbox_loss":
        spec = f"b-{combo['bbox_loss']}"
    elif term == "cls_loss":
        spec = f"c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
    else:
        raise ValueError(f"Unsupported YOLOv10 layer_grad term: {term}")
    return f"layer_grad_t-{_abbr(combo['target'])}__term-{_abbr(term)}__{spec}"


def _timestamped_combo_dir(root: Path, slug: str) -> Path:
    existing = sorted(root.glob(f"??-??-????_??;??_{slug}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{slug}"


def _timestamped_meta_dir(root: Path, slug: str) -> Path:
    existing = sorted(root.glob(f"??-??-????_??;??_{slug}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{slug}"


def iter_term_combinations():
    for target in TARGETS:
        for bbox_loss in BBOX_LOSSES:
            yield {
                "target": target,
                "term": "bbox_loss",
                "bbox_loss": bbox_loss,
                "cls_loss": "bcewithlogits",
                "cls_direction": "pred_to_target",
            }
        for cls_loss, cls_direction in CLS_COMBOS:
            yield {
                "target": target,
                "term": "cls_loss",
                "bbox_loss": "l1",
                "cls_loss": cls_loss,
                "cls_direction": cls_direction,
            }


def _term_combo_key(combo: dict) -> tuple:
    return combo["target"], combo["term"], combo["bbox_loss"], combo["cls_loss"], combo["cls_direction"]


def _meta_combo_slug(combo: dict) -> str:
    return (
        f"layer_grad_t-{_abbr(combo['target'])}"
        f"__b-{combo['bbox_loss']}"
        f"__c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
    )


def iter_meta_combinations(term_run_dirs: dict[tuple, Path]):
    count = 0
    term_combos = list(iter_term_combinations())
    for target in TARGETS:
        bbox_combos = [combo for combo in term_combos if combo["target"] == target and combo["term"] == "bbox_loss"]
        cls_combos = [combo for combo in term_combos if combo["target"] == target and combo["term"] == "cls_loss"]
        for bbox_combo in bbox_combos:
            for cls_combo in cls_combos:
                yield {
                    "target": target,
                    "bbox_loss": bbox_combo["bbox_loss"],
                    "cls_loss": cls_combo["cls_loss"],
                    "cls_direction": cls_combo["cls_direction"],
                    "bbox_term_root": term_run_dirs[_term_combo_key(bbox_combo)],
                    "cls_term_root": term_run_dirs[_term_combo_key(cls_combo)],
                    "input_roots": [
                        term_run_dirs[_term_combo_key(bbox_combo)],
                        term_run_dirs[_term_combo_key(cls_combo)],
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
    cand_score_threshold = combo.get("cand_score_threshold", grad_cfg.get("cand_score_threshold", 0.0))
    cand_iou_threshold = combo.get("cand_iou_threshold", grad_cfg.get("cand_iou_threshold", 0.45))
    grad_cfg = {key: value for key, value in grad_cfg.items() if key in {"bbox_layer", "cls_layer", "reduction"}}
    grad_cfg.update(
        {
            "target": combo["target"],
            "scalar": [combo["term"]],
            "bbox_loss": combo["bbox_loss"],
            "cls_loss": combo["cls_loss"],
            "cls_direction": combo["cls_direction"],
            "cand_score_threshold": cand_score_threshold,
            "cand_iou_threshold": cand_iou_threshold,
        }
    )
    layer_grad_cfg["gradient"] = grad_cfg
    return config


def _layer_grad_fieldnames(combo: dict, params_by_scalar: dict, reductions: list[str]) -> list[str]:
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
    scalar_name = combo["term"]
    for layer_name, _param in params_by_scalar[scalar_name]:
        safe = layer_name.replace(".", "_")
        for metric in reductions:
            fieldnames.append(f"{safe}_{scalar_name}_{metric}")
    return fieldnames


def _prepare_meta_config(base_config: dict, input_roots: list[Path], gt_root: str) -> dict:
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
        row_type = str(row.get("row_type", "")).strip().lower()
        first_value = next(iter(row.values()), "")
        if row_type == "mean" or str(first_value).strip().lower() == "mean":
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


def _scalar_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _run_yolov10_layer_grad_terms_once(base_config: dict, combo_dirs: list[tuple[dict, Path]]) -> None:
    active = [
        (combo, run_dir)
        for combo, run_dir in combo_dirs
        if RUN_LAYER_GRAD and not (REUSE_EXISTING and (run_dir / "layer_grad.csv").is_file())
    ]
    if not active:
        return

    config = _prepare_layer_grad_config(base_config, active[0][0])
    split = config.get("dataset", {}).get("split", "val")
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not bool(getattr(detector, "is_yolov10", False)):
        raise ValueError("grid_search_yolov10_layer_grad_meta_classifier.py requires a YOLOv10 object detector config.")

    parsed_by_key = {}
    params_by_key = {}
    profilers = {}
    handles = {}
    unique_params = {}
    original_requires_grad = {}
    try:
        for combo, run_dir in active:
            run_dir.mkdir(parents=True, exist_ok=True)
            term_config = _prepare_layer_grad_config(base_config, combo)
            parsed = parse_yolov10_output_config(term_config)
            layer_cfg = parsed["layer_grad"]
            key = _term_combo_key(combo)
            parsed_by_key[key] = parsed
            params_by_scalar = {}
            for scalar_name in layer_cfg["scalar"]:
                layers = expand_layer_names(detector.model, layer_cfg["layers_by_scalar"][scalar_name])
                params_by_scalar[scalar_name] = [
                    (layer_name, resolve_layer_parameter(detector.model, layer_name)) for layer_name in layers
                ]
                for _layer_name, param in params_by_scalar[scalar_name]:
                    unique_params[id(param)] = param
                    original_requires_grad.setdefault(id(param), bool(param.requires_grad))
                    param.requires_grad_(True)
            params_by_key[key] = params_by_scalar
            _save_yaml(term_config, run_dir / "grid_object_detector_config.yaml")
            csv_file = open(run_dir / "layer_grad.csv", "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                csv_file,
                fieldnames=_layer_grad_fieldnames(combo, params_by_scalar, layer_cfg["reduction"]),
            )
            writer.writeheader()
            handles[key] = combo, run_dir, csv_file, writer
            stages = ["detector_inference_sec"]
            if layer_cfg["target"] == "cand_target":
                stages.append("candidate_search_sec")
            stages.extend(["loss_compute_sec", "backpropagation_sec", "feature_compute_sec"])
            profilers[key] = StageTimingProfiler(
                run_dir=run_dir,
                uncertainty="layer_grad",
                unit=parsed["unit"],
                stages=stages,
                device=device,
            )

        first_profiler = next(iter(profilers.values()))
        for images, targets in tqdm(dataloader, desc="Object Detector (predict - yolov10 layer_grad grid)", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            forward = run_yolov10_forward(detector, infer_batch, timing=first_profiler, grad=True)
            all_items = list(iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device))
            stage_by_key = {
                key: {
                    "detector_inference_sec": forward.detector_inference_sec,
                    "loss_compute_sec": 0.0,
                    "backpropagation_sec": 0.0,
                    "feature_compute_sec": 0.0,
                }
                for key in handles
            }
            cand_keys = [key for key in handles if parsed_by_key[key]["layer_grad"]["target"] == "cand_target"]
            if cand_keys:
                for key in cand_keys:
                    stage_by_key[key]["candidate_search_sec"] = 0.0
            candidate_caches = {}
            candidate_indices_by_context = {}
            if cand_keys:
                for sample_idx in range(len(image_list)):
                    t_candidate = first_profiler.start()
                    candidate_caches[sample_idx] = build_yolov10_candidate_cache(forward, sample_idx)
                    elapsed = first_profiler.elapsed(t_candidate)
                    for key in cand_keys:
                        stage_by_key[key]["candidate_search_sec"] += elapsed
                for item_idx, item in enumerate(all_items):
                    grouped_keys = {}
                    for key in cand_keys:
                        layer_cfg = parsed_by_key[key]["layer_grad"]
                        group_key = (
                            float(layer_cfg["cand_score_threshold"]),
                            float(layer_cfg["cand_iou_threshold"]),
                        )
                        grouped_keys.setdefault(group_key, []).append(key)
                    for (score_threshold, iou_threshold), keys in grouped_keys.items():
                        cache = candidate_caches[item["sample_idx"]]
                        t_candidate = first_profiler.start()
                        cand_mask, _ious = yolov10_candidate_mask_from_cache(
                            cache,
                            item["box"][:4],
                            item["raw_class_idx"],
                            score_threshold,
                            iou_threshold,
                        )
                        elapsed = first_profiler.elapsed(t_candidate)
                        candidate_indices = torch.where(cand_mask)[0]
                        for key in keys:
                            stage_by_key[key]["candidate_search_sec"] += elapsed
                            candidate_indices_by_context[(item_idx, key)] = candidate_indices
            total_grad_calls = 0
            for item_idx, _item in enumerate(all_items):
                for key in handles:
                    layer_cfg = parsed_by_key[key]["layer_grad"]
                    if layer_cfg["target"] == "cand_target" and candidate_indices_by_context[(item_idx, key)].numel() == 0:
                        continue
                    total_grad_calls += 1
            grad_call_idx = 0
            for item_idx, item in enumerate(all_items):
                for key, (combo, _run_dir, _csv_file, writer) in handles.items():
                    layer_cfg = parsed_by_key[key]["layer_grad"]
                    scalar_name = combo["term"]
                    row = dict(item["base_row"])
                    if layer_cfg["target"] == "cand_target" and candidate_indices_by_context[(item_idx, key)].numel() == 0:
                        t_feature = profilers[key].start()
                        for layer_name, _param in params_by_key[key][scalar_name]:
                            safe = layer_name.replace(".", "_")
                            for metric in layer_cfg["reduction"]:
                                row[f"{safe}_{scalar_name}_{metric}"] = 0.0
                        stage_by_key[key]["feature_compute_sec"] += profilers[key].elapsed(t_feature)
                        writer.writerow(row)
                        continue
                    t_loss = profilers[key].start()
                    if layer_cfg["target"] == "cand_target":
                        scalar_terms = _build_candidate_scalar_terms(
                            forward,
                            item,
                            candidate_indices_by_context[(item_idx, key)],
                            layer_cfg,
                            device,
                        )
                    else:
                        scalar_terms = _build_null_scalar_terms(forward, item, layer_cfg, device)
                    if scalar_name in scalar_terms:
                        target_scalar = scalar_terms[scalar_name]
                    else:
                        raise ValueError(f"Unsupported YOLOv10 layer_grad term: {scalar_name}")
                    stage_by_key[key]["loss_compute_sec"] += profilers[key].elapsed(t_loss)

                    grad_call_idx += 1
                    params = [param for _layer_name, param in params_by_key[key][scalar_name]]
                    t_back = profilers[key].start()
                    grads = torch.autograd.grad(
                        target_scalar,
                        params,
                        retain_graph=grad_call_idx < total_grad_calls,
                    )
                    stage_by_key[key]["backpropagation_sec"] += profilers[key].elapsed(t_back)

                    t_feature = profilers[key].start()
                    for (layer_name, _param), grad in zip(params_by_key[key][scalar_name], grads):
                        safe = layer_name.replace(".", "_")
                        stats = map_grad_tensor_to_numbers(grad)
                        for metric in layer_cfg["reduction"]:
                            value = stats.get(metric, 0.0)
                            row[f"{safe}_{scalar_name}_{metric}"] = _scalar_to_float(value)
                    stage_by_key[key]["feature_compute_sec"] += profilers[key].elapsed(t_feature)
                    writer.writerow(row)
                    del target_scalar, grads

            for key, (_combo, _run_dir, csv_file, _writer) in handles.items():
                csv_file.flush()
                profilers[key].record(len(image_list), len(all_items), stage_by_key[key])
            del infer_batch, forward, all_items, candidate_caches, candidate_indices_by_context
    finally:
        for param_id, param in unique_params.items():
            param.requires_grad_(original_requires_grad[param_id])
        for _combo, _run_dir, csv_file, _writer in handles.values():
            csv_file.close()
        for profiler in profilers.values():
            profiler.save()
        del detector
        if "device" in locals() and device.type == "cuda":
            torch.cuda.empty_cache()


def _prepare_grid_roots(od_base_config: dict, meta_base_config: dict) -> tuple[Path, Path, str]:
    dataset = _dataset_name(od_base_config)
    model = _model_name(od_base_config)
    grid_name = GRID_NAME.strip() or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_yolov10_layer_grad_grid"
    od_grid_root = REPO_ROOT / "object_detectors" / "runs" / model / "predict" / dataset / grid_name
    meta_grid_root = REPO_ROOT / "meta_models" / "runs" / "meta_classifier" / model / "train" / dataset / grid_name
    od_grid_root.mkdir(parents=True, exist_ok=True)
    meta_grid_root.mkdir(parents=True, exist_ok=True)
    gt_root = GT_ROOT.strip() or str(meta_base_config.get("dataset", {}).get("gt_root", "")).strip()
    if not gt_root:
        raise ValueError("GT_ROOT is empty and META_CLASSIFIER_CONFIG has no dataset.gt_root.")
    return od_grid_root, meta_grid_root, gt_root


def _write_results(out_dir: Path, rows: list[dict]) -> None:
    fields = [
        "target",
        "bbox_loss",
        "cls_loss",
        "cls_direction",
        "auroc",
        "ap",
        "fpr95",
        "ece",
        "ace",
        "bbox_term_root",
        "cls_term_root",
        "meta_run_dir",
    ]
    _write_csv(out_dir / "grid_results.csv", rows, fields)


def main() -> None:
    od_config_path = _resolve_path(OBJECT_DETECTOR_CONFIG)
    meta_config_path = _resolve_path(META_CLASSIFIER_CONFIG)
    od_base_config = _load_yaml(od_config_path)
    meta_base_config = _load_yaml(meta_config_path)
    if str(od_base_config.get("model", {}).get("type", "")).strip().lower() != "yolov10":
        raise ValueError("grid_search_yolov10_layer_grad_meta_classifier.py requires a YOLOv10 object detector config.")

    od_grid_root, meta_grid_root, gt_root = _prepare_grid_roots(od_base_config, meta_base_config)

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

    _run_yolov10_layer_grad_terms_once(od_base_config, combo_dirs)

    rows = []
    meta_combo_list = list(iter_meta_combinations(term_run_dirs))
    print(f"Meta-classifier combinations: {len(meta_combo_list)}", flush=True)
    for idx, combo in enumerate(meta_combo_list, start=1):
        slug = _meta_combo_slug(combo)
        print(f"[META {idx}/{len(meta_combo_list)}] {slug}", flush=True)
        meta_dir = _timestamped_meta_dir(meta_grid_root, slug)
        eval_csv = meta_dir / "results" / "evaluation_results.csv"
        if RUN_META_CLASSIFIER and not (REUSE_EXISTING and eval_csv.is_file()):
            meta_config = _prepare_meta_config(meta_base_config, combo["input_roots"], gt_root)
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
        rows.append(
            {
                "target": combo["target"],
                "bbox_loss": combo["bbox_loss"],
                "cls_loss": combo["cls_loss"],
                "cls_direction": combo["cls_direction"],
                "auroc": metrics.get("auroc", ""),
                "ap": metrics.get("ap", ""),
                "fpr95": metrics.get("fpr95", ""),
                "ece": metrics.get("ece", ""),
                "ace": metrics.get("ace", ""),
                "bbox_term_root": str(combo["bbox_term_root"]),
                "cls_term_root": str(combo["cls_term_root"]),
                "meta_run_dir": str(meta_dir),
            }
        )
        _write_results(meta_grid_root, rows)

    print(f"Saved grid results: {meta_grid_root / 'grid_results.csv'}")


if __name__ == "__main__":
    main()
