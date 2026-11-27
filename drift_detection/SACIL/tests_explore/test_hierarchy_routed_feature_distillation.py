from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import (
    UnifiedTable1Trainer,
    resolve_feature_cosine_distillation_options,
)
from sacil.methods import (
    cosine_feature_distillation_loss,
    hierarchy_routed_feature_sample_weights,
)
from sacil.models import ExpandableLinearNet


ROOT = Path(__file__).resolve().parents[1]


def test_routing_maps_fixed_bgs_regions_to_three_sample_groups() -> None:
    routed = hierarchy_routed_feature_sample_weights(
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([True, True, True, False, False]),
        known_classes=3,
        sample_region_ids=("node_7", None, "node_9"),
        old_conflict_weight=0.1,
        old_outside_weight=1.0,
        new_weight=0.1,
    )
    torch.testing.assert_close(
        routed.sample_weights,
        torch.tensor([0.1, 1.0, 0.1, 0.1, 0.1]),
    )
    metrics = routed.detached_metrics()
    assert metrics["old_conflict_count"] == 2
    assert metrics["old_outside_count"] == 1
    assert metrics["new_count"] == 2
    assert metrics["old_conflict_mean_weight"] == pytest.approx(0.1)
    assert metrics["old_outside_mean_weight"] == pytest.approx(1.0)
    assert metrics["new_mean_weight"] == pytest.approx(0.1)
    assert metrics["old_outside_effective_weight"] == pytest.approx(1.0 / 1.4)


def test_routing_rejects_stale_or_incompatible_bgs_mapping() -> None:
    with pytest.raises(ValueError, match="one entry per known class"):
        hierarchy_routed_feature_sample_weights(
            torch.tensor([0, 2]),
            torch.tensor([True, False]),
            known_classes=2,
            sample_region_ids=("node",),
            old_conflict_weight=0.1,
            old_outside_weight=1.0,
            new_weight=0.1,
        )
    with pytest.raises(ValueError, match="replay targets"):
        hierarchy_routed_feature_sample_weights(
            torch.tensor([2]),
            torch.tensor([True]),
            known_classes=2,
            sample_region_ids=("node", None),
            old_conflict_weight=0.1,
            old_outside_weight=1.0,
            new_weight=0.1,
        )


def test_all_one_routing_is_bitwise_identical_to_legacy_loss_and_gradient() -> None:
    torch.manual_seed(19)
    reference = torch.randn(9, 13)
    legacy_features = torch.randn(9, 13, requires_grad=True)
    routed_features = legacy_features.detach().clone().requires_grad_(True)
    legacy_loss = cosine_feature_distillation_loss(
        legacy_features, reference
    )
    routed_loss = cosine_feature_distillation_loss(
        routed_features,
        reference,
        sample_weights=torch.ones(9),
    )
    assert torch.equal(legacy_loss, routed_loss)
    legacy_loss.backward()
    routed_loss.backward()
    assert torch.equal(legacy_features.grad, routed_features.grad)


def test_routing_options_require_enabled_bgs() -> None:
    with pytest.raises(ValueError, match="requires enabled BGS"):
        resolve_feature_cosine_distillation_options(
            "icarl",
            {
                "feature_cosine_distillation": {
                    "enabled": True,
                    "hierarchy_routing": {"enabled": True},
                },
                "boundary_graph_surgery": {"enabled": False},
            },
        )


def test_routing_loss_fails_explicitly_when_bgs_reference_is_missing() -> None:
    trainer = UnifiedTable1Trainer.__new__(UnifiedTable1Trainer)
    trainer.model = ExpandableLinearNet(3, backbone="resnet32")
    teacher = copy.deepcopy(trainer.model).eval()
    trainer.method = "icarl"
    trainer.protocol = SimpleNamespace(
        session=lambda _: SimpleNamespace(start=2, stop=3, size=1)
    )
    trainer.icarl_kd_weight = 0.0
    trainer.config = {"method": {}}
    trainer.selective_kd_options = {"enabled": False}
    trainer.branch_masked_kd_options = {"enabled": False}
    trainer.geodesic_distillation_options = {"enabled": False}
    trainer.feature_cosine_distillation_options = {
        "enabled": True,
        "lambda": 1.0,
        "adaptive_mode": "none",
        "epsilon": 1e-12,
        "training_classifier": "linear",
        "hierarchy_routing": {
            "enabled": True,
            "old_conflict_weight": 0.1,
            "old_outside_weight": 1.0,
            "new_weight": 0.1,
        },
    }
    trainer.bgs_options = {"enabled": True}
    trainer.bgs_reference = None
    with pytest.raises(RuntimeError, match="requires a BGS reference"):
        trainer._loss_components(
            1,
            torch.randn(2, 3, 32, 32),
            torch.tensor([0, 2]),
            torch.tensor([True, False]),
            teacher,
            None,
            None,
            None,
        )


def test_routed_config_keeps_structured_bgs_and_disables_insertion() -> None:
    config = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_routed_bgs.yaml"
    )
    options = resolve_feature_cosine_distillation_options(
        "icarl", config["method"]
    )
    routing = options["hierarchy_routing"]
    assert routing == {
        "enabled": True,
        "old_conflict_weight": 0.1,
        "old_outside_weight": 1.0,
        "new_weight": 0.1,
        "partition_source": "bgs_reference.sample_region_ids",
    }
    assert config["model"]["backbone"] == "resnet32"
    assert config["evaluation"]["classifier"] == "nme"
    geometry = config["method"]["boundary_graph_surgery"]["geometry"]
    insertion = config["method"]["boundary_graph_surgery"]["insertion"]
    assert geometry["mask_mode"] == "structured"
    assert not insertion["enabled"]


def test_legacy_feature_config_does_not_enable_or_materialize_routing() -> None:
    config = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_cosinehead.yaml"
    )
    options = resolve_feature_cosine_distillation_options(
        "icarl", config["method"]
    )
    assert "hierarchy_routing" not in options
