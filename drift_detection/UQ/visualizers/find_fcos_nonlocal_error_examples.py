import csv
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TP_CSV = REPO_ROOT / r"object_detectors/runs/fcos/predict/coco/06-17-2026_02;48_gt/tp.csv"
UNTO_O_RESULTS = REPO_ROOT / r"meta_models/runs/meta_classifier/fcos/train/coco/06-18-2026_14;07_null_detect/results"
LOCAL_COCO = Path(r"D:/DataDrift/datasets/COCO/train2017")
OUT_DIR = REPO_ROOT / "visualizers" / "runs" / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_fcos_nonlocal_error_examples"

BG_IOU_MAX = 0.05
MAX_CANDIDATE_ROWS = 8


def key_from_row(row):
    return str(int(row["image_id"])), Path(str(row["image_path"])).name, str(int(row["raw_pred_idx"]))


def load_unto_o_probs():
    sums = defaultdict(float)
    counts = defaultdict(int)
    files = sorted(UNTO_O_RESULTS.glob("eval_data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No eval_data_*.csv files found under {UNTO_O_RESULTS}")
    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = key_from_row(row)
                sums[key] += float(row["y_pred"])
                counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}, dict(counts)


def focus_error_type(row):
    if int(row["tp"]) == 1:
        return "tp"
    if row["error_type"] == "classification_error":
        return "classification_error"
    if float(row["max_iou"]) <= BG_IOU_MAX:
        return "background_error"
    return "localization_error"


def selection_type(row):
    score = float(row["score"])
    prob = float(row["unto_o_tp_prob"])
    tp = int(row["tp"])
    focus = row["focus_error_type"]
    if tp == 1 and prob - score >= 0.25 and prob >= 0.75:
        return "tp_keep"
    if focus == "classification_error" and score - prob >= 0.18 and prob <= 0.30:
        return "classification_filter"
    if focus == "background_error" and score - prob >= 0.18 and prob <= 0.30:
        return "background_filter"
    if focus == "localization_error" and score - prob >= 0.18 and prob <= 0.30:
        return "localization_filter"
    return ""


def load_rows():
    probs, eval_counts = load_unto_o_probs()
    summary = {}
    selected = []
    missing = 0
    usecols = [
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
        "max_iou",
        "gt_iou",
        "tp",
        "error_type",
    ]
    for chunk in pd.read_csv(TP_CSV, usecols=usecols, chunksize=200000):
        for row in chunk.to_dict("records"):
            image_id = int(row["image_id"])
            stat = summary.setdefault(
                image_id,
                {
                    "image_id": image_id,
                    "local_image_path": str(LOCAL_COCO / f"{image_id:012d}.jpg"),
                    "total_rows": 0,
                    "tp_total": 0,
                    "classification_total": 0,
                    "background_total": 0,
                    "localization_total": 0,
                    "tp_keep_count": 0,
                    "classification_filter_count": 0,
                    "background_filter_count": 0,
                    "localization_filter_count": 0,
                },
            )
            stat["total_rows"] += 1
            if int(row["tp"]) == 1:
                stat["tp_total"] += 1
            focus = focus_error_type(row)
            if focus == "classification_error":
                stat["classification_total"] += 1
            elif focus == "background_error":
                stat["background_total"] += 1
            elif focus == "localization_error":
                stat["localization_total"] += 1

            key = key_from_row(row)
            prob = probs.get(key)
            if prob is None:
                missing += 1
                continue
            row = dict(row)
            row["image_id"] = image_id
            row["pred_idx"] = int(row["pred_idx"])
            row["raw_pred_idx"] = int(row["raw_pred_idx"])
            row["score"] = float(row["score"])
            row["tp"] = int(row["tp"])
            row["gt_iou"] = float(row["gt_iou"])
            row["max_iou"] = float(row["max_iou"])
            row["unto_o_tp_prob"] = float(prob)
            row["unto_o_eval_count"] = int(eval_counts[key])
            row["focus_error_type"] = focus
            row["score_prob_gap"] = row["score"] - row["unto_o_tp_prob"]
            row["prob_score_gap"] = row["unto_o_tp_prob"] - row["score"]
            row["local_image_path"] = stat["local_image_path"]
            row["selection_type"] = selection_type(row)
            if row["selection_type"]:
                stat[f"{row['selection_type']}_count"] += 1
                selected.append(row)
    return pd.DataFrame(summary.values()), pd.DataFrame(selected), missing


def trim_selected_rows(selected):
    if selected.empty:
        return selected
    selected = selected.copy()
    selected["sort_score"] = selected.apply(
        lambda row: row["prob_score_gap"] if row["selection_type"] == "tp_keep" else row["score_prob_gap"],
        axis=1,
    )
    selected = selected.sort_values(["image_id", "selection_type", "sort_score"], ascending=[True, True, False])
    selected = selected.groupby(["image_id", "selection_type"], group_keys=False).head(MAX_CANDIDATE_ROWS)
    return selected.drop(columns=["sort_score"])


def rank_images(summary):
    summary = summary.copy()
    summary["nonlocal_filter_count"] = summary["classification_filter_count"] + summary["background_filter_count"]
    summary["rank_score"] = (
        5.0 * summary["classification_filter_count"]
        + 3.0 * summary["background_filter_count"]
        + 2.0 * summary["tp_keep_count"]
        - 2.0 * summary["localization_filter_count"]
        - 0.02 * summary["localization_total"]
    )
    candidates = summary[
        (summary["nonlocal_filter_count"] >= 2)
        & (summary["tp_keep_count"] >= 1)
        & (summary["localization_filter_count"] <= 1)
    ].copy()
    candidates = candidates.sort_values(
        ["rank_score", "classification_filter_count", "background_filter_count", "tp_keep_count"],
        ascending=[False, False, False, False],
    )
    candidates.insert(0, "rank", range(1, len(candidates) + 1))
    return candidates


def copy_recommended_images(candidates):
    image_dir = OUT_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    overall = candidates.head(12)
    cls = candidates[candidates["classification_filter_count"] > 0].head(12)
    picks = pd.concat([overall, cls]).drop_duplicates("image_id")
    lines = ["# FCOS Non-Localization Error Example Candidates", ""]
    for record in picks.to_dict("records"):
        src = Path(record["local_image_path"])
        if src.exists():
            shutil.copy2(src, image_dir / src.name)
        lines.append(
            "- rank {rank}: `{path}` | tp_keep={tp}, class_filter={cls}, bg_filter={bg}, loc_filter={loc}".format(
                rank=int(record["rank"]),
                path=src,
                tp=int(record["tp_keep_count"]),
                cls=int(record["classification_filter_count"]),
                bg=int(record["background_filter_count"]),
                loc=int(record["localization_filter_count"]),
            )
        )
    (OUT_DIR / "recommended_paths.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, selected, missing = load_rows()
    selected = trim_selected_rows(selected)
    candidates = rank_images(summary)
    candidates.to_csv(OUT_DIR / "candidate_images.csv", index=False)
    selected[selected["image_id"].isin(set(candidates.head(80)["image_id"].tolist()))].to_csv(
        OUT_DIR / "candidate_detections.csv",
        index=False,
    )
    copy_recommended_images(candidates)
    print(f"missing UnTO-O rows: {missing}")
    print(f"candidate images: {len(candidates)}")
    cols = [
        "rank",
        "image_id",
        "local_image_path",
        "tp_keep_count",
        "classification_filter_count",
        "background_filter_count",
        "localization_filter_count",
        "rank_score",
    ]
    print(candidates[cols].head(25).to_string(index=False))
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
