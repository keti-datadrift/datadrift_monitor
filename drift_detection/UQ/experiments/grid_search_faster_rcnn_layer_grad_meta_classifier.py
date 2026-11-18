import csv
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Edit these paths before running.
OBJECT_DETECTOR_CONFIG = (
    r"object_detectors/configs/faster_rcnn/predict/coco.yaml"
)
META_CLASSIFIER_CONFIG = (
    r"meta_models/configs/meta_classifier/train.yaml"
)

# If empty, dataset.gt_root is read from META_CLASSIFIER_CONFIG.
GT_ROOT = ""

# Set to a fixed name to resume a previous grid root. Empty string creates a new timestamped root.
GRID_NAME = ""

RUN_LAYER_GRAD = True
RUN_META_CLASSIFIER = True
REUSE_EXISTING = False

# Use None for all meta-classifier combinations, or set a small int for a smoke test.
# Object detector term CSVs are still generated once for all 26 single-term settings.
MAX_COMBINATIONS = None

RPN_BBOX_LOSSES = ["offset_l1", "offset_l2"]
ROI_BBOX_LOSSES = ["box_l1", "box_l2", "offset_l1", "offset_l2"]
BBOX_DIRECTIONS = ["pred_to_target"]
OBJ_LOSSES = ["bcewithlogits", "abs_diff", "signed_diff"]
CLS_LOSSES = ["bcewithlogits", "kl"]


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
    raw = str(config.get("model", {}).get("type", "faster_rcnn")).strip().lower()
    return (
        "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
        or "faster_rcnn"
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
        "rpn_bbox_loss": "rpnb",
        "rpn_obj_loss": "rpno",
        "roi_bbox_loss": "roib",
        "roi_cls_loss": "roic",
    }
    return aliases.get(value, value)


def _valid_cls_directions(cls_loss: str) -> list[str]:
    return (
        ["pred_to_target", "target_to_pred"] if cls_loss == "kl" else ["pred_to_target"]
    )


def _valid_obj_directions(obj_loss: str) -> list[str]:
    if obj_loss == "signed_diff":
        return ["pred_to_target", "target_to_pred"]
    return ["pred_to_target"]


def _combo_slug(combo: dict) -> str:
    target = combo["target"]
    term = combo["term"]
    if term == "rpn_bbox_loss":
        spec = f"rpnb-{combo['rpn_bbox_loss']}-{_abbr(combo['rpn_bbox_direction'])}"
    elif term == "rpn_obj_loss":
        spec = (
            f"rpno-{_abbr(combo['rpn_obj_loss'])}-{_abbr(combo['rpn_obj_direction'])}"
        )
    elif term == "roi_bbox_loss":
        spec = f"roib-{combo['roi_bbox_loss']}-{_abbr(combo['roi_bbox_direction'])}"
    elif term == "roi_cls_loss":
        spec = (
            f"roic-{_abbr(combo['roi_cls_loss'])}-{_abbr(combo['roi_cls_direction'])}"
        )
    else:
        raise ValueError(f"Unsupported Faster R-CNN layer_grad term: {term}")
    return f"layer_grad_t-{_abbr(target)}__term-{_abbr(term)}__{spec}"


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
        "rpn_bbox_loss": "offset_l1",
        "rpn_bbox_direction": "pred_to_target",
        "rpn_obj_loss": "bcewithlogits",
        "rpn_obj_direction": "pred_to_target",
        "roi_bbox_loss": "box_l1",
        "roi_bbox_direction": "pred_to_target",
        "roi_cls_loss": "bcewithlogits",
        "roi_cls_direction": "pred_to_target",
    }


def iter_term_combinations():
    for target in ("cand_target", "null_target"):
        for rpn_bbox_loss in RPN_BBOX_LOSSES:
            for rpn_bbox_direction in BBOX_DIRECTIONS:
                combo = _base_combo(target, "rpn_bbox_loss")
                combo.update(
                    {
                        "rpn_bbox_loss": rpn_bbox_loss,
                        "rpn_bbox_direction": rpn_bbox_direction,
                    }
                )
                yield combo

        for rpn_obj_loss in OBJ_LOSSES:
            for rpn_obj_direction in _valid_obj_directions(rpn_obj_loss):
                combo = _base_combo(target, "rpn_obj_loss")
                combo.update(
                    {
                        "rpn_obj_loss": rpn_obj_loss,
                        "rpn_obj_direction": rpn_obj_direction,
                    }
                )
                yield combo

        for roi_bbox_loss in ROI_BBOX_LOSSES:
            for roi_bbox_direction in BBOX_DIRECTIONS:
                combo = _base_combo(target, "roi_bbox_loss")
                combo.update(
                    {
                        "roi_bbox_loss": roi_bbox_loss,
                        "roi_bbox_direction": roi_bbox_direction,
                    }
                )
                yield combo

        for roi_cls_loss in CLS_LOSSES:
            for roi_cls_direction in _valid_cls_directions(roi_cls_loss):
                combo = _base_combo(target, "roi_cls_loss")
                combo.update(
                    {
                        "roi_cls_loss": roi_cls_loss,
                        "roi_cls_direction": roi_cls_direction,
                    }
                )
                yield combo


def _term_combo_key(combo: dict) -> tuple:
    return (
        combo["target"],
        combo["term"],
        combo["rpn_bbox_loss"],
        combo["rpn_bbox_direction"],
        combo["rpn_obj_loss"],
        combo["rpn_obj_direction"],
        combo["roi_bbox_loss"],
        combo["roi_bbox_direction"],
        combo["roi_cls_loss"],
        combo["roi_cls_direction"],
    )


def _meta_combo_slug(combo: dict) -> str:
    return (
        f"layer_grad_t-{_abbr(combo['target'])}"
        f"__rpnb-{combo['rpn_bbox_loss']}-{_abbr(combo['rpn_bbox_direction'])}"
        f"__rpno-{_abbr(combo['rpn_obj_loss'])}-{_abbr(combo['rpn_obj_direction'])}"
        f"__roib-{combo['roi_bbox_loss']}-{_abbr(combo['roi_bbox_direction'])}"
        f"__roic-{_abbr(combo['roi_cls_loss'])}-{_abbr(combo['roi_cls_direction'])}"
    )


def iter_meta_combinations(term_run_dirs: dict[tuple, Path]):
    count = 0
    term_combos = list(iter_term_combinations())
    for target in ("cand_target", "null_target"):
        rpn_bbox_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "rpn_bbox_loss"
        ]
        rpn_obj_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "rpn_obj_loss"
        ]
        roi_bbox_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "roi_bbox_loss"
        ]
        roi_cls_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "roi_cls_loss"
        ]
        for rpn_bbox_combo in rpn_bbox_combos:
            for rpn_obj_combo in rpn_obj_combos:
                for roi_bbox_combo in roi_bbox_combos:
                    for roi_cls_combo in roi_cls_combos:
                        yield {
                            "target": target,
                            "rpn_bbox_loss": rpn_bbox_combo["rpn_bbox_loss"],
                            "rpn_bbox_direction": rpn_bbox_combo["rpn_bbox_direction"],
                            "rpn_obj_loss": rpn_obj_combo["rpn_obj_loss"],
                            "rpn_obj_direction": rpn_obj_combo["rpn_obj_direction"],
                            "roi_bbox_loss": roi_bbox_combo["roi_bbox_loss"],
                            "roi_bbox_direction": roi_bbox_combo["roi_bbox_direction"],
                            "roi_cls_loss": roi_cls_combo["roi_cls_loss"],
                            "roi_cls_direction": roi_cls_combo["roi_cls_direction"],
                            "input_roots": [
                                term_run_dirs[_term_combo_key(rpn_bbox_combo)],
                                term_run_dirs[_term_combo_key(rpn_obj_combo)],
                                term_run_dirs[_term_combo_key(roi_bbox_combo)],
                                term_run_dirs[_term_combo_key(roi_cls_combo)],
                            ],
                        }
                        count += 1
                        if MAX_COMBINATIONS is not None and count >= int(
                            MAX_COMBINATIONS
                        ):
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

    target = combo["target"]
    grad_cfg["target"] = target
    grad_cfg.pop("scalar", None)
    grad_cfg.pop("layer", None)
    grad_cfg.pop("rpn", None)
    grad_cfg.pop("roi", None)
    if target in {"cand_target", "null_target"}:
        block_name = "cand_target" if target == "cand_target" else "null_target"
        rpn_cfg = {
            "obj_loss": combo["rpn_obj_loss"],
            "bbox_loss": combo["rpn_bbox_loss"],
            "obj_direction": combo["rpn_obj_direction"],
            "bbox_direction": combo["rpn_bbox_direction"],
        }
        roi_cfg = {
            "cls_loss": combo["roi_cls_loss"],
            "bbox_loss": combo["roi_bbox_loss"],
            "cls_direction": combo["roi_cls_direction"],
            "bbox_direction": combo["roi_bbox_direction"],
        }
        if target == "cand_target":
            rpn_cfg["cand_obj_threshold"] = 0.0
            roi_cfg["cand_score_threshold"] = 0.0
        grad_cfg[block_name] = {
            "scalar": [combo["term"]],
            "layer": {
                "rpn_obj_loss": ["rpn.head.conv", "rpn.head.cls_logits"],
                "rpn_bbox_loss": ["rpn.head.conv", "rpn.head.bbox_pred"],
                "roi_cls_loss": [
                    "roi_heads.box_head.fc7",
                    "roi_heads.box_predictor.cls_score",
                ],
                "roi_bbox_loss": [
                    "roi_heads.box_head.fc7",
                    "roi_heads.box_predictor.bbox_pred",
                ],
            },
            "rpn": rpn_cfg,
            "roi": roi_cfg,
        }
    else:
        raise ValueError(f"Unsupported Faster R-CNN layer_grad target: {target}")
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


def _run_layer_grad_if_needed(
    od_base_config: dict,
    od_grid_root: Path,
    combo: dict,
) -> Path:
    slug = _combo_slug(combo)
    layer_dir = _timestamped_combo_dir(od_grid_root, slug)
    layer_csv = layer_dir / "layer_grad.csv"
    if RUN_LAYER_GRAD and not (REUSE_EXISTING and layer_csv.is_file()):
        od_config = _prepare_layer_grad_config(od_base_config, combo)
        od_config_path = layer_dir / "grid_object_detector_config.yaml"
        _save_yaml(od_config, od_config_path)
        _run(
            [
                sys.executable,
                "object_detectors/main.py",
                "--config",
                str(od_config_path),
                "--run-dir",
                str(layer_dir),
            ]
        )
    return layer_dir


def main() -> None:
    od_config_path = _resolve_path(OBJECT_DETECTOR_CONFIG)
    meta_config_path = _resolve_path(META_CLASSIFIER_CONFIG)
    od_base_config = _load_yaml(od_config_path)
    meta_base_config = _load_yaml(meta_config_path)
    dataset = _dataset_name(od_base_config)
    model = _model_name(od_base_config)

    grid_name = (
        GRID_NAME.strip()
        or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_faster_rcnn_layer_grad_grid"
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
    term_combo_list = list(iter_term_combinations())
    print(f"Layer-grad term runs: {len(term_combo_list)}", flush=True)
    for idx, combo in enumerate(term_combo_list, start=1):
        slug = _combo_slug(combo)
        print(f"[OD {idx}/{len(term_combo_list)}] {slug}", flush=True)
        layer_dir = _run_layer_grad_if_needed(od_base_config, od_grid_root, combo)
        term_run_dirs[_term_combo_key(combo)] = layer_dir

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


def _result_row(combo: dict, metrics: dict) -> dict:
    row = {
        "target": combo["target"],
        "rpn_bbox_loss": combo.get("rpn_bbox_loss", ""),
        "rpn_bbox_direction": combo.get("rpn_bbox_direction", ""),
        "rpn_obj_loss": combo.get("rpn_obj_loss", ""),
        "rpn_obj_direction": combo.get("rpn_obj_direction", ""),
        "roi_bbox_loss": combo.get("roi_bbox_loss", ""),
        "roi_bbox_direction": combo.get("roi_bbox_direction", ""),
        "roi_cls_loss": combo.get("roi_cls_loss", ""),
        "roi_cls_direction": combo.get("roi_cls_direction", ""),
        "auroc": metrics.get("auroc", ""),
        "ap": metrics.get("ap", ""),
        "fpr95": metrics.get("fpr95", ""),
        "ece": metrics.get("ece", ""),
        "ace": metrics.get("ace", ""),
    }
    return row


def _comparison_key(row: dict) -> tuple:
    return (
        row["rpn_bbox_loss"],
        row["rpn_bbox_direction"],
        row["rpn_obj_loss"],
        row["rpn_obj_direction"],
        row["roi_bbox_loss"],
        row["roi_bbox_direction"],
        row["roi_cls_loss"],
        row["roi_cls_direction"],
    )


def _write_results(out_dir: Path, rows: list[dict]) -> None:
    result_fields = [
        "target",
        "rpn_bbox_loss",
        "rpn_bbox_direction",
        "rpn_obj_loss",
        "rpn_obj_direction",
        "roi_bbox_loss",
        "roi_bbox_direction",
        "roi_cls_loss",
        "roi_cls_direction",
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
                "rpn_bbox_loss": key[0],
                "rpn_bbox_direction": key[1],
                "rpn_obj_loss": key[2],
                "rpn_obj_direction": key[3],
                "roi_bbox_loss": key[4],
                "roi_bbox_direction": key[5],
                "roi_cls_loss": key[6],
                "roi_cls_direction": key[7],
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
        "rpn_bbox_loss",
        "rpn_bbox_direction",
        "rpn_obj_loss",
        "rpn_obj_direction",
        "roi_bbox_loss",
        "roi_bbox_direction",
        "roi_cls_loss",
        "roi_cls_direction",
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


if __name__ == "__main__":
    main()
