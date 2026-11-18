import shutil
import csv
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_run_dir(uncertainty=None, unit=None, target_value=None, base_subdir=None):
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    uncertainty_name = str(uncertainty or "predict").lower()
    target_name = str(target_value or "").strip().lower()

    if base_subdir:
        base_dir = PROJECT_ROOT / "runs" / str(base_subdir)
    else:
        base_dir = PROJECT_ROOT / "runs"

    dir_name = (
        f"{run_name}_{uncertainty_name}"
        if not target_name
        else f"{run_name}_{uncertainty_name}_{target_name}"
    )
    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_used_config(config_path, run_dir):
    shutil.copy2(config_path, run_dir / "used_config.yaml")


def _count_ratio(csv_path: Path, col_name: str) -> tuple[int, int, float] | None:
    if not csv_path.is_file():
        return None

    total = 0
    positive = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if col_name not in (reader.fieldnames or []):
            return None
        for row in reader:
            total += 1
            value = str(row.get(col_name, "")).strip().lower()
            if value in {"1", "true"}:
                positive += 1

    ratio = (positive / total) if total else 0.0
    return total, positive, ratio


def _count_rows(csv_path: Path) -> int | None:
    if not csv_path.is_file():
        return None
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def save_run_summary(run_dir: Path, uncertainty: str) -> Path:
    run_dir = Path(run_dir)
    summary = {
        "uncertainty": str(uncertainty),
    }

    fn_stats = _count_ratio(run_dir / "fn.csv", "fn")
    if fn_stats is not None:
        total, fn_count, fn_ratio = fn_stats
        summary["num_images"] = total
        summary["fn_count"] = fn_count
        summary["fn_ratio"] = fn_ratio

    tp_stats = _count_ratio(run_dir / "tp.csv", "tp")
    if tp_stats is not None:
        total, tp_count, tp_ratio = tp_stats
        summary["num_bboxes"] = total
        summary["tp_count"] = tp_count
        summary["tp_ratio"] = tp_ratio

    score_rows = _count_rows(run_dir / "score.csv")
    if score_rows is not None:
        summary["score_rows"] = score_rows
    out_path = run_dir / "run_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_path
