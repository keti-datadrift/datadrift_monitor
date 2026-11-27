import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

TP_CSV = REPO_ROOT / r"object_detectors/runs/yolov5/predict/coco/06-15-2026_18;54_gt/tp.csv"
LOCAL_COCO_IMAGE_ROOT = Path(r"D:/DataDrift/datasets/COCO/train2017")

MAX_IMAGES = 50
MAX_DETECTIONS_PER_CASE_PER_IMAGE = 3

CASE_ORDER = [
    "high_score_tp",
    "high_score_fp",
    "high_score_bad_localization",
    "low_score_tp",
]

CASE_DESCRIPTIONS = {
    "high_score_tp": "High-score TP",
    "high_score_fp": "High-score FP",
    "high_score_bad_localization": "High-score poor localization",
    "low_score_tp": "Low-score TP",
}


def timestamp():
    return datetime.now().strftime("%m-%d-%Y_%H;%M")


def local_image_path(image_id):
    return LOCAL_COCO_IMAGE_ROOT / f"{int(image_id):012d}.jpg"


def case_labels(row):
    score = float(row["score"])
    tp = int(row["tp"])
    iou = float(row["gt_iou"])
    error_type = row.get("error_type", "")
    labels = []
    if tp == 1 and score >= 0.85 and iou >= 0.65:
        labels.append("high_score_tp")
    if tp == 0 and score >= 0.80 and error_type == "classification_error":
        labels.append("high_score_fp")
    if tp == 0 and score >= 0.65 and error_type == "localization_error" and 0.10 <= iou < 0.45:
        labels.append("high_score_bad_localization")
    if tp == 1 and 0.25 <= score <= 0.40 and iou >= 0.45:
        labels.append("low_score_tp")
    return labels


def row_score(case_type, row):
    score = float(row["score"])
    iou = float(row["gt_iou"])
    if case_type == "high_score_tp":
        return (score, iou)
    if case_type == "high_score_fp":
        return (score, 1.0 - iou)
    if case_type == "high_score_bad_localization":
        return (score, iou)
    if case_type == "low_score_tp":
        return (-score, iou)
    return (0.0, 0.0)


def compact_row(row, case_type):
    keys = [
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
    out = {key: row.get(key, "") for key in keys}
    out["case_type"] = case_type
    out["case_label"] = CASE_DESCRIPTIONS[case_type]
    out["local_image_path"] = str(local_image_path(row["image_id"]))
    return out


def collect_candidates():
    image_stats = defaultdict(Counter)
    image_rows = defaultdict(lambda: defaultdict(list))
    missing_paths = set()

    with open(TP_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            stats = image_stats[image_id]
            stats["total_rows"] += 1
            if int(row["tp"]) == 1:
                stats["tp_total"] += 1
            else:
                stats["fp_total"] += 1

            path = local_image_path(image_id)
            if not path.exists():
                missing_paths.add(str(path))
                continue

            for case_type in case_labels(row):
                stats[case_type] += 1
                bucket = image_rows[image_id][case_type]
                bucket.append(compact_row(row, case_type))
                bucket.sort(key=lambda item: row_score(case_type, item), reverse=True)
                del bucket[MAX_DETECTIONS_PER_CASE_PER_IMAGE:]

    return image_stats, image_rows, missing_paths


def image_rank_tuple(image_id, stats):
    case_type_count = sum(1 for case_type in CASE_ORDER if stats[case_type] > 0)
    case_row_count = sum(stats[case_type] for case_type in CASE_ORDER)
    strict_fp = stats["high_score_fp"]
    low_tp = stats["low_score_tp"]
    bad_loc = stats["high_score_bad_localization"]
    high_tp = stats["high_score_tp"]
    complexity_penalty = max(0, int(stats["total_rows"]) - 50) * 0.05
    figure_score = (
        case_type_count * 1000
        + min(case_row_count, 30) * 20
        + strict_fp * 10
        + low_tp * 5
        + bad_loc * 3
        + high_tp
        - complexity_penalty
    )
    return (
        case_type_count,
        case_row_count,
        figure_score,
        strict_fp,
        low_tp,
        bad_loc,
        high_tp,
        -int(stats["total_rows"]),
    )


def ranked_images(image_stats, image_rows):
    candidates = []
    for image_id, stats in image_stats.items():
        case_type_count = sum(1 for case_type in CASE_ORDER if stats[case_type] > 0)
        if case_type_count < 3:
            continue
        if not image_rows.get(image_id):
            continue
        candidates.append((image_id, stats))
    return sorted(candidates, key=lambda item: image_rank_tuple(item[0], item[1]), reverse=True)


def write_image_summary(path, selected):
    fieldnames = [
        "recommended_rank",
        "image_id",
        "local_image_path",
        "case_type_count",
        "total_case_rows",
        "total_detection_rows",
        "tp_total",
        "fp_total",
        *[f"{case_type}_count" for case_type in CASE_ORDER],
        "figure_priority_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (image_id, stats) in enumerate(selected, start=1):
            case_type_count = sum(1 for case_type in CASE_ORDER if stats[case_type] > 0)
            total_case_rows = sum(stats[case_type] for case_type in CASE_ORDER)
            row = {
                "recommended_rank": rank,
                "image_id": image_id,
                "local_image_path": str(local_image_path(image_id)),
                "case_type_count": case_type_count,
                "total_case_rows": total_case_rows,
                "total_detection_rows": int(stats["total_rows"]),
                "tp_total": int(stats["tp_total"]),
                "fp_total": int(stats["fp_total"]),
                "figure_priority_score": float(image_rank_tuple(image_id, stats)[2]),
            }
            for case_type in CASE_ORDER:
                row[f"{case_type}_count"] = int(stats[case_type])
            writer.writerow(row)


def write_detection_rows(path, selected, image_rows):
    fieldnames = [
        "recommended_rank",
        "case_type",
        "case_label",
        "local_image_path",
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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (image_id, _stats) in enumerate(selected, start=1):
            for case_type in CASE_ORDER:
                for row in image_rows[image_id].get(case_type, []):
                    out = {"recommended_rank": rank, **row}
                    writer.writerow(out)


def write_markdown(path, selected, image_rows):
    lines = [
        "# Introduction Score-Reliability Candidate Images",
        "",
        f"- Source CSV: `{TP_CSV}`",
        f"- Local image root: `{LOCAL_COCO_IMAGE_ROOT}`",
        "- Ranking prioritizes images containing multiple reliability cases in one scene.",
        "",
    ]
    for rank, (image_id, stats) in enumerate(selected[:20], start=1):
        lines.append(f"## {rank}. image_id={image_id}")
        lines.append(f"- Path: `{local_image_path(image_id)}`")
        lines.append(
            "- Counts: "
            + ", ".join(f"{CASE_DESCRIPTIONS[t]}={int(stats[t])}" for t in CASE_ORDER)
            + f", total detections={int(stats['total_rows'])}"
        )
        for case_type in CASE_ORDER:
            rows = image_rows[image_id].get(case_type, [])
            if not rows:
                continue
            lines.append(f"- {CASE_DESCRIPTIONS[case_type]} examples:")
            for row in rows:
                box = f"({float(row['xmin']):.1f}, {float(row['ymin']):.1f}, {float(row['xmax']):.1f}, {float(row['ymax']):.1f})"
                lines.append(
                    f"  - pred_idx={row['pred_idx']}, class={row['pred_class']}, "
                    f"score={float(row['score']):.3f}, IoU={float(row['gt_iou']):.3f}, "
                    f"type={row['error_type']}, box={box}"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not TP_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {TP_CSV}")
    image_stats, image_rows, missing_paths = collect_candidates()
    if missing_paths:
        sample = "\n".join(sorted(missing_paths)[:10])
        raise FileNotFoundError(f"Some mapped local image paths do not exist:\n{sample}")

    selected = ranked_images(image_stats, image_rows)[:MAX_IMAGES]
    if not selected:
        raise RuntimeError("No image-level candidates found.")
    if sum(1 for _image_id, stats in selected if all(stats[t] > 0 for t in CASE_ORDER)) == 0:
        raise RuntimeError("No selected image contains all four requested case types.")

    out_dir = OUTPUT_ROOT / f"{timestamp()}_intro_score_reliability_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_image_summary(out_dir / "candidate_images.csv", selected)
    write_detection_rows(out_dir / "candidate_detections.csv", selected, image_rows)
    write_markdown(out_dir / "candidate_examples.md", selected, image_rows)
    print(f"Saved: {out_dir}")
    print(f"Top image_id: {selected[0][0]}")


if __name__ == "__main__":
    main()
