import shutil
import hashlib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_RUN_DIR_NAME_LEN = 96


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip().lower()).strip("_")


def _shorten_run_dir_name(value: str, max_len: int = MAX_RUN_DIR_NAME_LEN) -> str:
    safe = _safe_path_part(value) or "run"
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_len - len(digest) - 1)
    return f"{safe[:keep].rstrip('_')}_{digest}"


def create_run_dir(task: str, model_group: str, cue: str, target_value: str = "", mode_subdir: str = "") -> Path:
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    cue_name = _safe_path_part(str(cue))
    target_name = _safe_path_part(str(target_value))
    dir_tail = cue_name if not target_name else f"{cue_name}_{target_name}"
    dir_name = f"{run_name}_{_shorten_run_dir_name(dir_tail)}"
    raw_parts = [
        part
        for part in str(model_group or "").replace("\\", "/").split("/")
        if _safe_path_part(part) and _safe_path_part(part) != "bbox_predictions"
    ]
    group_parts = [_safe_path_part(part) for part in raw_parts]
    safe_mode = _safe_path_part(str(mode_subdir or ""))

    safe_task = _safe_path_part(str(task))
    if safe_task not in {"meta_classifier", "meta_regressor"}:
        raise ValueError(f"Unsupported meta model task: {task}")

    base_dir = PROJECT_ROOT / "runs" / safe_task

    if len(group_parts) >= 2:
        base_dir = base_dir / group_parts[0]
        if safe_mode:
            base_dir = base_dir / safe_mode
        for part in group_parts[1:]:
            base_dir = base_dir / part
    elif group_parts and safe_mode:
        base_dir = base_dir / group_parts[0] / safe_mode
    elif group_parts:
        base_dir = base_dir / group_parts[0]
    else:
        if safe_mode:
            base_dir = base_dir / safe_mode

    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_used_config(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "used_config.yaml")
