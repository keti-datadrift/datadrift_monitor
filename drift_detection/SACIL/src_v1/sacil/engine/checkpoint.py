from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sacil.utils import atomic_torch_save


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cuda_index(device: str | torch.device | int | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, int):
        return int(device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return (
        torch.cuda.current_device()
        if resolved.index is None
        else int(resolved.index)
    )


def restore_rng_state(
    state: dict[str, Any],
    *,
    source_cuda_device: str | torch.device | int | None = None,
    target_cuda_device: str | torch.device | int | None = None,
) -> None:
    """Restore RNG state across identical or remapped CUDA visibility.

    A shared base may be trained on ``cuda:1`` and continued on ``cuda:0`` or
    under a one-device ``CUDA_VISIBLE_DEVICES`` mask.  In that case the state
    consumed by the source device must be transplanted to the target logical
    device; calling ``set_rng_state_all`` is both fragile and semantically
    wrong for a cross-device continuation.
    """

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if not torch.cuda.is_available() or "cuda" not in state:
        return
    if (
        target_cuda_device is not None
        and not isinstance(target_cuda_device, int)
        and torch.device(target_cuda_device).type != "cuda"
    ):
        return
    cuda_states = list(state["cuda"])
    if not cuda_states:
        return
    source_index = _cuda_index(source_cuda_device)
    target_index = _cuda_index(target_cuda_device)
    if target_index is not None:
        if not 0 <= target_index < torch.cuda.device_count():
            raise ValueError(
                f"target CUDA device {target_index} is not visible"
            )
        if source_index is None:
            source_index = min(target_index, len(cuda_states) - 1)
        if not 0 <= source_index < len(cuda_states):
            raise ValueError(
                f"checkpoint has no CUDA RNG state for source device "
                f"{source_index}"
            )
        torch.cuda.set_rng_state(cuda_states[source_index], target_index)
        return
    # Backward-compatible generic resume: restore only the states for devices
    # visible to this process instead of indexing beyond the generator tuple.
    for index, cuda_state in enumerate(
        cuda_states[: torch.cuda.device_count()]
    ):
        torch.cuda.set_rng_state(cuda_state, index)


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    enriched = dict(payload)
    enriched["rng_state"] = capture_rng_state()
    atomic_torch_save(enriched, path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
