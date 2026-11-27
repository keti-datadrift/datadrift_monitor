from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from sacil.config import load_config_tree
from sacil.data import ClassOrderProtocol
from sacil.engine.table1_trainer import (
    UnifiedTable1Trainer,
    base_recipe_signature,
    resolve_icarl_kd_weight,
)
from sacil.models import ExpandableLinearNet


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "explore" / "cifar100"


def test_icarl_kd_weight_defaults_to_one_and_validates() -> None:
    assert resolve_icarl_kd_weight("icarl", {"name": "icarl"}) == 1.0
    assert resolve_icarl_kd_weight(
        "icarl", {"name": "icarl", "kd_weight": 0.0}
    ) == 0.0
    assert resolve_icarl_kd_weight(
        "icarl", {"name": "icarl", "kd_weight": 0.25}
    ) == 0.25
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_icarl_kd_weight(
            "icarl", {"name": "icarl", "kd_weight": -1.0}
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_icarl_kd_weight(
            "icarl", {"name": "icarl", "kd_weight": float("nan")}
        )


def test_non_icarl_method_does_not_reinterpret_its_kd_weight() -> None:
    assert resolve_icarl_kd_weight(
        "create", {"name": "create", "kd_weight": 0.25}
    ) == 1.0


def test_kd_weight_is_incremental_only_in_base_signature() -> None:
    canonical = load_config_tree(
        ROOT / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    kd_off = copy.deepcopy(canonical)
    kd_off["method"]["kd_weight"] = 0.0
    assert base_recipe_signature(canonical) == base_recipe_signature(kd_off)


@pytest.mark.parametrize(
    "name, expected_topology",
    [
        ("icarl_kdoff_control.yaml", False),
        ("icarl_kdoff_edgecorr_r20_lambda15.yaml", True),
    ],
)
def test_kd_off_configs_are_explicit_resnet32_pairs(
    name: str, expected_topology: bool
) -> None:
    config = load_config_tree(CONFIG_ROOT / name)
    assert config["method"]["name"] == "icarl"
    assert config["method"]["kd_weight"] == 0.0
    assert config["model"]["backbone"] == "resnet32"
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["evaluation"]["classifier"] == "nme"
    assert bool(config["method"].get("edge_topology", {}).get("enabled", False)) is expected_topology


def test_kd_off_keeps_post_kd_geometry_path_and_has_zero_kd_gradient() -> None:
    trainer = object.__new__(UnifiedTable1Trainer)
    trainer.model = ExpandableLinearNet(num_classes=4, backbone="resnet32")
    teacher = copy.deepcopy(trainer.model)
    with torch.no_grad():
        next(teacher.parameters()).add_(0.01)
    trainer.protocol = ClassOrderProtocol(
        protocol_id="kd-off-unit",
        dataset="cifar100",
        num_classes=4,
        class_order=(0, 1, 2, 3),
        session_slices=(
            {"session_id": 0, "kind": "base", "start": 0, "stop": 2},
            {"session_id": 1, "kind": "incremental", "start": 2, "stop": 4},
        ),
    )
    trainer.method = "icarl"
    trainer.icarl_kd_weight = 0.0
    trainer.config = {"method": {"lambda_geo": 1.0}}
    trainer.selective_kd_options = {"enabled": False}
    trainer.branch_masked_kd_options = {"enabled": False}
    trainer.geodesic_distillation_options = {"enabled": False}
    trainer.bgs_options = {"enabled": False}
    trainer.casper_options = {"enabled": False}

    images = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    replay_mask = torch.tensor([True, True, False, False])

    class SquaredDrift(nn.Module):
        def forward(self, current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            return (current - reference).square().mean()

    components, _ = trainer._loss_components(
        1,
        images,
        targets,
        replay_mask,
        teacher,
        SquaredDrift(),
        None,
        None,
    )

    assert set(components) == {"classification", "kd", "geometry"}
    assert components["kd"].item() == 0.0
    kd_gradient = torch.autograd.grad(
        components["kd"], trainer.model.classifier.weight, retain_graph=True
    )[0]
    assert torch.count_nonzero(kd_gradient).item() == 0
    assert components["geometry"].item() > 0.0
