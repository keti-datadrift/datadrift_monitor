import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TP_CSV = REPO_ROOT / r"object_detectors/runs/yolov5/predict/coco/06-15-2026_18;54_gt/tp.csv"
UNTO_O_RESULTS = REPO_ROOT / r"meta_models/runs/meta_classifier/yolov5/train/coco/06-17-2026_00;42_null_detect/results"
LOCAL_COCO = Path(r"D:/DataDrift/datasets/COCO/train2017")
OUT_DIR = REPO_ROOT / r"visualizers/runs/06-25-2026_21;00_nonlocal_error_examples"

BG_IOU_MAX = 0.05
MAX_CANDIDATE_ROWS = 8


def key_from_row(row):
    return str(int(row["image_id"])), Path(str(row["image_path"])).name, str(int(row["raw_pred_idx"]))


def load_unto_o_probs():
    sums = defaultdict(float)
    counts = defaultdict(int)
    for path in sorted(UNTO_O_RESULTS.glob("eval_data_*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = key_from_row(row)
                sums[key] += float(row["y_pred"])
                counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}, dict(counts)


def classify_focus(row):
    if int(row["tp"]) == 1:
        return "tp"
    if row["error_type"] == "classification_error":
        return "classification_error"
    if float(row["max_iou"]) <= BG_IOU_MAX:
        return "background_error"
    return "localization_error"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probs, eval_counts = load_unto_o_probs()
    rows = []
    missing = 0
    usecols = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax",
        "score", "pred_class", "max_iou", "gt_iou", "tp", "error_type",
    ]
    for chunk in pd.read_csv(TP_CSV, usecols=usecols, chunksize=200000):
        for row in chunk.to_dict("records"):
            key = key_from_row(row)
            prob = probs.get(key)
            if prob is None:
                missing += 1
                continue
            row = dict(row)
            row["unto_o_tp_prob"] = prob
            row["unto_o_eval_count"] = eval_counts[key]
            row["focus_error_type"] = classify_focus(row)
            row["score_prob_gap"] = float(row["score"]) - prob
            row["prob_score_gap"] = prob - float(row["score"])
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No merged rows found.")

    df["image_id"] = df["image_id"].astype(int)
    df["score"] = df["score"].astype(float)
    df["max_iou"] = df["max_iou"].astype(float)
    df["tp"] = df["tp"].astype(int)
    df["local_image_path"] = df["image_id"].map(lambda x: str(LOCAL_COCO / f"{x:012d}.jpg"))

    tp_keep = (df["tp"] == 1) & (df["unto_o_tp_prob"] - df["score"] >= 0.20) & (df["unto_o_tp_prob"] >= 0.65)
    class_filter = (df["focus_error_type"] == "classification_error") & (df["score"] - df["unto_o_tp_prob"] >= 0.20) & (df["unto_o_tp_prob"] <= 0.35)
    bg_filter = (df["focus_error_type"] == "background_error") & (df["score"] - df["unto_o_tp_prob"] >= 0.20) & (df["unto_o_tp_prob"] <= 0.35)
    loc_filter = (df["focus_error_type"] == "localization_error") & (df["score"] - df["unto_o_tp_prob"] >= 0.20) & (df["unto_o_tp_prob"] <= 0.35)

    df["selection_type"] = ""
    df.loc[tp_keep, "selection_type"] = "tp_keep"
    df.loc[class_filter, "selection_type"] = "classification_filter"
    df.loc[bg_filter, "selection_type"] = "background_filter"
    df.loc[loc_filter, "selection_type"] = "localization_filter"

    group = df.groupby("image_id")
    summary = group.agg(
        local_image_path=("local_image_path", "first"),
        total_rows=("raw_pred_idx", "size"),
        tp_total=("tp", "sum"),
        classification_total=("focus_error_type", lambda s: int((s == "classification_error").sum())),
        background_total=("focus_error_type", lambda s: int((s == "background_error").sum())),
        localization_total=("focus_error_type", lambda s: int((s == "localization_error").sum())),
        tp_keep_count=("selection_type", lambda s: int((s == "tp_keep").sum())),
        classification_filter_count=("selection_type", lambda s: int((s == "classification_filter").sum())),
        background_filter_count=("selection_type", lambda s: int((s == "background_filter").sum())),
        localization_filter_count=("selection_type", lambda s: int((s == "localization_filter").sum())),
    ).reset_index()
    summary["nonlocal_filter_count"] = summary["classification_filter_count"] + summary["background_filter_count"]
    summary["localization_penalty"] = summary["localization_filter_count"] + 0.05 * summary["localization_total"]
    summary["rank_score"] = (
        5.0 * summary["classification_filter_count"]
        + 3.0 * summary["background_filter_count"]
        + 2.0 * summary["tp_keep_count"]
        - 2.0 * summary["localization_filter_count"]
        - 0.03 * summary["localization_total"]
    )
    summary = summary[
        (summary["nonlocal_filter_count"] >= 2)
        & (summary["tp_keep_count"] >= 1)
        & (summary["localization_filter_count"] <= 1)
    ].copy()
    summary = summary.sort_values(
        ["rank_score", "classification_filter_count", "background_filter_count", "tp_keep_count"],
        ascending=[False, False, False, False],
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))

    top_ids = set(summary.head(50)["image_id"].tolist())
    cand = df[df["image_id"].isin(top_ids) & df["selection_type"].isin(["tp_keep", "classification_filter", "background_filter"])].copy()
    cand["sort_score"] = cand.apply(
        lambda r: r["prob_score_gap"] if r["selection_type"] == "tp_keep" else r["score_prob_gap"], axis=1
    )
    cand = cand.sort_values(["image_id", "selection_type", "sort_score"], ascending=[True, True, False])
    cand = cand.groupby(["image_id", "selection_type"], group_keys=False).head(MAX_CANDIDATE_ROWS)
    cand = cand.drop(columns=["sort_score"])

    summary.to_csv(OUT_DIR / "candidate_images.csv", index=False)
    cand.to_csv(OUT_DIR / "candidate_detections.csv", index=False)

    with open(OUT_DIR / "candidate_examples.md", "w", encoding="utf-8") as handle:
        handle.write("# Non-Localization Error Example Candidates\n\n")
        handle.write(f"Skipped rows without UnTO-O prediction: {missing}\n\n")
        for rec in summary.head(20).to_dict("records"):
            handle.write(
                f"- rank {rec['rank']}: image_id={rec['image_id']}, path=`{rec['local_image_path']}`, "
                f"class_filter={rec['classification_filter_count']}, bg_filter={rec['background_filter_count']}, "
                f"tp_keep={rec['tp_keep_count']}, loc_filter={rec['localization_filter_count']}\n"
            )
    print(f"rows merged: {len(df)}")
    print(f"candidate images: {len(summary)}")
    print(summary.head(20)[["rank", "image_id", "local_image_path", "tp_keep_count", "classification_filter_count", "background_filter_count", "localization_filter_count", "rank_score"]].to_string(index=False))
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
