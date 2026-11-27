from __future__ import annotations

import torch

from sacil.memory import (
    ExemplarMemory,
    herding_select,
    icarl_herding_select,
)


def _stock_pycil_running_mean_herding(
    features: torch.Tensor, indices: list[int], count: int
) -> list[int]:
    """Literal tensor translation of PyCIL models/base.py lines 216-239."""

    vectors = features.detach().float()
    vectors = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)
    class_mean = vectors.mean(dim=0)
    remaining_indices = list(indices)
    selected_vectors: list[torch.Tensor] = []
    selected: list[int] = []
    for k in range(1, min(count, len(indices)) + 1):
        running_sum = (
            torch.stack(selected_vectors).sum(dim=0)
            if selected_vectors
            else torch.zeros_like(class_mean)
        )
        candidate_means = (vectors + running_sum) / k
        position = int(
            ((class_mean - candidate_means).square().sum(dim=1)).argmin()
        )
        selected.append(remaining_indices.pop(position))
        selected_vectors.append(vectors[position].clone())
        vectors = torch.cat((vectors[:position], vectors[position + 1 :]), dim=0)
    return selected


def test_herding_matches_stock_pycil_running_mean_selection() -> None:
    generator = torch.Generator().manual_seed(20260801)
    features = torch.randn(25, 64, generator=generator)
    indices = list(range(1000, 1025))
    expected = _stock_pycil_running_mean_herding(features, indices, 20)
    actual = herding_select(features, indices, 20)
    assert actual == expected


def test_herding_is_deterministic_unique_and_bounded() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    indices = [10, 11, 12, 13]
    first = herding_select(features, indices, 3)
    second = herding_select(features, indices, 3)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert set(first).issubset(indices)


def test_memory_roundtrip_preserves_class_indices() -> None:
    memory = ExemplarMemory(2)
    memory.set_class_indices(5, [10, 11])
    memory.set_class_indices(2, [3])
    restored = ExemplarMemory.from_state_dict(memory.state_dict())
    assert restored.exemplars_per_class == 2
    assert restored.class_ids == (2, 5)
    assert restored.indices_for_class(5) == (10, 11)
    assert restored.all_indices((5, 2)) == [10, 11, 3]


def test_icarl_herding_is_deterministic_unique_and_bounded() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    indices = [10, 11, 12, 13]
    first = icarl_herding_select(features, indices, 3)
    second = icarl_herding_select(features, indices, 3)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert set(first).issubset(indices)
