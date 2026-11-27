from __future__ import annotations

from pathlib import Path

import torch

from sacil.engine.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "test_artifacts"
        / "checkpoint_roundtrip.pt"
    )
    payload = {
        "session_id": 1,
        "tensor": torch.arange(5),
        "nested": {"value": "ok"},
    }
    save_checkpoint(payload, path)
    restored = load_checkpoint(path)
    assert restored["session_id"] == 1
    assert torch.equal(restored["tensor"], payload["tensor"])
    assert restored["nested"] == payload["nested"]
    assert "rng_state" in restored
