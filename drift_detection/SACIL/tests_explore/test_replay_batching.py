from __future__ import annotations

from collections import Counter

import pytest

from sacil.engine.table1_trainer import (
    ReplayBalancedBatchSampler,
    resolve_prototype_consolidation_options,
    resolve_replay_batching_options,
)


def test_dual_stream_sampler_has_exact_old_new_counts_and_indices() -> None:
    labels = [label for label in range(4) for _ in range(3)]
    sampler = ReplayBalancedBatchSampler(
        new_size=20,
        replay_labels=labels,
        batch_size=10,
        replay_fraction=0.4,
        batches=5,
        seed=7,
    )
    for batch in sampler:
        assert len(batch) == 10
        assert sum(position < 20 for position in batch) == 6
        assert sum(position >= 20 for position in batch) == 4
        assert all(0 <= position < 20 + len(labels) for position in batch)


def test_conflict_multiplier_changes_class_sampling_not_replay_mass() -> None:
    labels = [label for label in range(5) for _ in range(2)]
    sampler = ReplayBalancedBatchSampler(
        new_size=50,
        replay_labels=labels,
        batch_size=20,
        replay_fraction=0.5,
        batches=200,
        seed=3,
        conflict_labels={0},
        conflict_multiplier=4.0,
    )
    counts: Counter[int] = Counter()
    for batch in sampler:
        replay = [position for position in batch if position >= 50]
        assert len(replay) == 10
        for position in replay:
            counts[labels[position - 50]] += 1
    non_conflict_mean = sum(counts[label] for label in range(1, 5)) / 4
    assert counts[0] > 3.0 * non_conflict_mean


def test_hierarchy_conditioning_requires_bgs() -> None:
    with pytest.raises(ValueError, match="requires enabled BGS"):
        resolve_replay_batching_options(
            "icarl",
            {
                "replay_batching": {
                    "enabled": True,
                    "replay_fraction": 0.5,
                    "conflict_multiplier": 2.0,
                }
            },
        )


def test_uniform_dual_stream_options_are_explicit() -> None:
    options = resolve_replay_batching_options(
        "icarl",
        {
            "replay_batching": {
                "enabled": True,
                "replay_fraction": 0.5,
                "conflict_multiplier": 1.0,
            }
        },
    )
    assert options["enabled"] is True
    assert options["replay_fraction"] == 0.5
    assert options["conflict_multiplier"] == 1.0
    assert options["preserve_optimizer_steps"] is True


def test_hierarchy_prototype_consolidation_requires_bgs() -> None:
    with pytest.raises(ValueError, match="requires BGS"):
        resolve_prototype_consolidation_options(
            "icarl",
            {
                "prototype_consolidation": {
                    "enabled": True,
                    "lambda": 1.0,
                    "conflict_weight": 0.25,
                    "outside_weight": 1.0,
                }
            },
        )
