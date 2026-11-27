from __future__ import annotations

import pytest
import torch

from sacil.utils import resolved_device


def test_invalid_visible_cuda_ordinal_has_actionable_error(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    with pytest.raises(ValueError, match="remapped logical index"):
        resolved_device("cuda:1")


def test_valid_visible_cuda_ordinal_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolved_device("cuda:1") == torch.device("cuda:1")
