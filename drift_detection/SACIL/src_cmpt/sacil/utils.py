from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def ensure_dir(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def dump_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def git_commit(root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def resolved_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and device.index is not None:
        visible_count = torch.cuda.device_count()
        if device.index >= visible_count:
            visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
            visibility_note = (
                "not set"
                if visibility is None or visibility == ""
                else repr(visibility)
            )
            raise ValueError(
                f"requested {device}, but this Python process sees only "
                f"{visible_count} CUDA device(s). CUDA_VISIBLE_DEVICES is "
                f"{visibility_note}. Clear CUDA_VISIBLE_DEVICES and use the "
                "physical index, or keep it set and use the remapped logical "
                "index (usually cuda:0)."
            )
    return device


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
