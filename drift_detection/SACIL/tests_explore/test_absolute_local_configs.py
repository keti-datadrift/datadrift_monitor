from __future__ import annotations

from pathlib import Path

import pytest

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import (
    base_recipe_signature,
    resolve_geometry_mode,
    resolve_geometry_options,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "lambda_geo"),
    [
        ("icarl_absolute_local_lambda4.yaml", 4.0),
        ("icarl_absolute_local_lambda16.yaml", 16.0),
    ],
)
def test_absolute_local_config_preserves_canonical_icarl_base(
    filename: str, lambda_geo: float
) -> None:
    control = load_config_tree(
        PROJECT_ROOT
        / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    candidate = load_config_tree(
        PROJECT_ROOT / "configs/explore/cifar100" / filename
    )

    for section in ("data", "model", "memory", "evaluation", "training"):
        assert candidate[section] == control[section]
    assert candidate["method"]["name"] == "icarl"
    assert candidate["method"]["kd_temperature"] == control["method"][
        "kd_temperature"
    ]
    assert resolve_geometry_mode("icarl", candidate["method"]) == "sacil"
    assert candidate["method"]["lambda_geo"] == lambda_geo

    geometry = resolve_geometry_options(candidate["method"])
    assert geometry["anchor_frame"] == "fixed"
    assert geometry["reliability"] == "uniform"
    assert geometry["objective"] == "mse"
    assert geometry["weight_normalization"] == "anchor_count"
    assert candidate["output"]["directory"] == (
        "outputs/explore/absolute_relaxation"
    )
    assert base_recipe_signature(candidate) == base_recipe_signature(control)
