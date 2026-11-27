from __future__ import annotations

import random

import numpy as np
import torch

from sacil.engine.checkpoint import restore_rng_state


def _state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [
            torch.tensor([10], dtype=torch.uint8),
            torch.tensor([20], dtype=torch.uint8),
        ],
    }


def test_cuda_rng_is_transplanted_from_checkpoint_device(monkeypatch) -> None:
    restored: list[tuple[torch.Tensor, int]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: restored.append((value.clone(), device)),
    )

    state = _state()
    restore_rng_state(
        state,
        source_cuda_device="cuda:1",
        target_cuda_device="cuda:0",
    )

    assert len(restored) == 1
    assert restored[0][1] == 0
    assert torch.equal(restored[0][0], state["cuda"][1])


def test_generic_restore_ignores_hidden_checkpoint_devices(monkeypatch) -> None:
    restored: list[tuple[torch.Tensor, int]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: restored.append((value.clone(), device)),
    )

    state = _state()
    restore_rng_state(state)

    assert len(restored) == 1
    assert restored[0][1] == 0
    assert torch.equal(restored[0][0], state["cuda"][0])
