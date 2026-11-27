from __future__ import annotations

import pytest

from sacil.engine.table1_trainer import (
    base_recipe_signature,
    resolve_classification_reweighting_options,
)


def test_classification_reweighting_resolves_positive_weights() -> None:
    options = resolve_classification_reweighting_options(
        "icarl",
        {
            "classification_reweighting": {
                "enabled": True,
                "old_weight": 2,
                "new_weight": 1,
            }
        },
    )
    assert options["old_weight"] == 2.0
    assert options["new_weight"] == 1.0
    assert options["normalization"] == "sample_weight_sum"


def test_classification_reweighting_rejects_nonpositive_weight() -> None:
    with pytest.raises(ValueError, match="old_weight"):
        resolve_classification_reweighting_options(
            "icarl",
            {
                "classification_reweighting": {
                    "enabled": True,
                    "old_weight": 0,
                }
            },
        )


def test_classification_reweighting_does_not_change_base_signature() -> None:
    plain = {"method": {"name": "icarl"}, "training": {"base": {"epochs": 1}}}
    weighted = {
        "method": {
            "name": "icarl",
            "classification_reweighting": {
                "enabled": True,
                "old_weight": 2,
                "new_weight": 1,
            },
        },
        "training": {"base": {"epochs": 1}},
    }
    assert base_recipe_signature(plain) == base_recipe_signature(weighted)
