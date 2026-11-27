from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLORE_SOURCE = PROJECT_ROOT / "src_explore"

import sacil  # noqa: E402
from sacil.config import load_config_tree  # noqa: E402
from sacil.engine.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from sacil.engine.table1_trainer import (  # noqa: E402
    UnifiedTable1Trainer,
    base_recipe_signature,
    resolve_edge_topology_options,
)
from sacil.methods import (  # noqa: E402
    HierarchicalEdgeCorrelationLoss,
    HierarchicalEdgeReference,
    conflict_subtree_inside_edge_weights,
    global_edge_weights,
    incident_edge_weights,
    pairwise_cosine_edge_vector,
)


def test_edge_vector_and_correlation_loss_have_finite_gradient() -> None:
    torch.manual_seed(11)
    reference_features = torch.randn(8, 6)
    current_features = (
        reference_features + 0.2 * torch.randn_like(reference_features)
    ).requires_grad_()
    reference_edges = pairwise_cosine_edge_vector(reference_features)
    module = HierarchicalEdgeCorrelationLoss(
        reference_edges, global_edge_weights(8)
    )
    value = module(current_features)
    assert value > 0
    value.backward()
    assert current_features.grad is not None
    assert torch.isfinite(current_features.grad).all()
    assert float(current_features.grad.abs().sum()) > 0


def test_explore_import_is_not_the_preserved_source_tree() -> None:
    package_path = Path(sacil.__file__).resolve()
    assert package_path.is_relative_to(EXPLORE_SOURCE.resolve())


def test_edge_objective_is_rotation_invariant_but_detects_topology_change() -> None:
    torch.manual_seed(19)
    reference_features = torch.randn(7, 5)
    rotation = torch.linalg.qr(torch.randn(5, 5)).Q
    rotated = (reference_features @ rotation).requires_grad_()
    reference_edges = pairwise_cosine_edge_vector(reference_features)
    module = HierarchicalEdgeCorrelationLoss(
        reference_edges, global_edge_weights(7)
    )
    invariant = module(rotated)
    assert float(invariant.detach()) == pytest.approx(0.0, abs=2e-6)

    changed = rotated.detach().clone()
    changed[0] = changed[0] + 1.5 * torch.randn(5)
    assert float(module(changed)) > 1e-4


def test_incident_rule_relaxes_every_edge_touching_a_conflict() -> None:
    representative_weights = torch.tensor([1.0, 0.1, 1.0])
    # Upper-triangle order: (0,1), (0,2), (1,2).
    torch.testing.assert_close(
        incident_edge_weights(representative_weights),
        torch.tensor([0.1, 1.0, 0.1]),
    )


def test_inside_rule_preserves_cut_boundary_in_edge_vector_order() -> None:
    # Upper-triangle order for [out, in, in, out]:
    # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    membership = torch.tensor([False, True, True, False])
    torch.testing.assert_close(
        conflict_subtree_inside_edge_weights(
            membership, min_edge_weight=0.1
        ),
        torch.tensor([1.0, 1.0, 1.0, 0.1, 1.0, 1.0]),
    )


def test_edge_reference_round_trip_is_loss_ready() -> None:
    reference = HierarchicalEdgeReference(
        session_id=1,
        representatives_per_class=2,
        representative_indices=(10, 11, 20, 21),
        representative_class_ids=(0, 0, 1, 1),
        reference_edges=torch.linspace(-0.4, 0.6, 6),
        edge_weights=torch.ones(6),
    )
    restored = HierarchicalEdgeReference.from_state_dict(
        reference.state_dict()
    )
    assert restored.representative_count == 4
    assert restored.edge_count == 6
    assert isinstance(restored.loss_module(), HierarchicalEdgeCorrelationLoss)


def test_reference_survives_checkpoint_round_trip(tmp_path: Path) -> None:
    reference = HierarchicalEdgeReference(
        session_id=2,
        representatives_per_class=2,
        representative_indices=(1, 2, 3, 4),
        representative_class_ids=(0, 0, 1, 1),
        reference_edges=torch.linspace(-0.8, 0.8, 6),
        edge_weights=torch.ones(6),
    )
    checkpoint = tmp_path / "edge_reference.pt"
    save_checkpoint(
        {"edge_topology_reference": reference.state_dict()}, checkpoint
    )
    restored = HierarchicalEdgeReference.from_state_dict(
        load_checkpoint(checkpoint)["edge_topology_reference"]
    )
    torch.testing.assert_close(restored.reference_edges, reference.reference_edges)
    torch.testing.assert_close(restored.edge_weights, reference.edge_weights)


class _TinyFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


def test_first_pre_update_component_is_numerically_zero() -> None:
    torch.manual_seed(3)
    trainer = object.__new__(UnifiedTable1Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = _TinyFeatureModel()
    images = torch.randn(6, 3)
    trainer.model.eval()
    with torch.no_grad():
        teacher_features = trainer.model.extract_features(images)
    reference = HierarchicalEdgeReference(
        session_id=1,
        representatives_per_class=2,
        representative_indices=tuple(range(6)),
        representative_class_ids=(0, 0, 1, 1, 2, 2),
        reference_edges=pairwise_cosine_edge_vector(teacher_features),
        edge_weights=global_edge_weights(6),
    )
    trainer.edge_topology_options = {
        "lambda_edge": 5.0,
        "update_interval_steps": 1,
    }
    trainer.edge_topology_reference = reference
    trainer._edge_topology_images = images
    trainer._edge_topology_loss = reference.loss_module()
    trainer.model.train()
    value = trainer._edge_topology_component(0)
    assert value is not None
    assert abs(float(value.detach())) < 1e-6
    value.backward()
    gradient = trainer.model.projection.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) < 1e-5


def test_trainer_component_has_key_ready_gradient_and_schedule() -> None:
    torch.manual_seed(5)
    trainer = object.__new__(UnifiedTable1Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = _TinyFeatureModel()
    images = torch.randn(6, 3)
    with torch.no_grad():
        teacher = trainer.model.extract_features(images) + 0.1 * torch.randn(6, 4)
    reference = HierarchicalEdgeReference(
        session_id=1,
        representatives_per_class=2,
        representative_indices=tuple(range(6)),
        representative_class_ids=(0, 0, 1, 1, 2, 2),
        reference_edges=pairwise_cosine_edge_vector(teacher),
        edge_weights=global_edge_weights(6),
    )
    trainer.edge_topology_options = {
        "lambda_edge": 5.0,
        "update_interval_steps": 2,
    }
    trainer.edge_topology_reference = reference
    trainer._edge_topology_images = images
    trainer._edge_topology_loss = reference.loss_module()

    components: dict[str, torch.Tensor] = {}
    edge = trainer._edge_topology_component(0)
    assert edge is not None
    components["hierarchical_edge_correlation"] = edge
    components["hierarchical_edge_correlation"].backward()
    gradient = trainer.model.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0
    assert trainer._edge_topology_component(1) is None


def test_session_zero_never_prepares_edge_topology() -> None:
    trainer = object.__new__(UnifiedTable1Trainer)
    trainer.edge_topology_options = {"enabled": True}
    trainer.edge_topology_reference = object()
    trainer._edge_topology_images = torch.empty(1)
    trainer._edge_topology_loss = object()
    trainer._prepare_edge_topology(0, None)
    assert trainer.edge_topology_reference is None
    assert trainer._edge_topology_images is None
    assert trainer._edge_topology_loss is None


@pytest.mark.parametrize(
    ("filename", "representatives", "weight"),
    [
        ("icarl_edgecorr_r2_lambda01.yaml", 2, 0.1),
        ("icarl_edgecorr_r2_lambda05.yaml", 2, 0.5),
        ("icarl_edgecorr_r2_lambda5.yaml", 2, 5.0),
        ("icarl_edgecorr_r2_lambda15.yaml", 2, 15.0),
        ("icarl_edgecorr_r4_lambda5.yaml", 4, 5.0),
        ("icarl_edgecorr_r4_lambda15.yaml", 4, 15.0),
        ("icarl_edgecorr_r20_lambda15.yaml", 20, 15.0),
        ("icarl_edgecorr_r20_lambda05.yaml", 20, 0.5),
    ],
)
def test_explore_configs_preserve_icarl_base_recipe(
    filename: str, representatives: int, weight: float
) -> None:
    control = load_config_tree(
        PROJECT_ROOT
        / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    candidate = load_config_tree(
        PROJECT_ROOT / "configs/explore/cifar100" / filename
    )
    options = resolve_edge_topology_options("icarl", candidate["method"])
    assert options["enabled"]
    assert options["representatives_per_class"] == representatives
    assert options["lambda_edge"] == weight
    if representatives == 20:
        assert options["update_interval_steps"] == 10
    assert candidate["output"]["directory"].startswith("outputs/explore/")
    assert base_recipe_signature(candidate) == base_recipe_signature(control)


@pytest.mark.parametrize(
    ("filename", "representatives", "weight"),
    [
        ("replay_edgecorr_r2_lambda01.yaml", 2, 0.1),
        ("replay_edgecorr_r2_lambda05.yaml", 2, 0.5),
        ("replay_edgecorr_r2_lambda5.yaml", 2, 5.0),
        ("replay_edgecorr_r2_lambda15.yaml", 2, 15.0),
        ("replay_edgecorr_r4_lambda5.yaml", 4, 5.0),
        ("replay_edgecorr_r4_lambda15.yaml", 4, 15.0),
        ("replay_edgecorr_r20_lambda15.yaml", 20, 15.0),
        ("replay_edgecorr_r20_lambda05.yaml", 20, 0.5),
    ],
)
def test_explore_configs_preserve_replay_base_recipe(
    filename: str, representatives: int, weight: float
) -> None:
    control = load_config_tree(
        PROJECT_ROOT
        / "configs/table1/cifar100/replay_nme_b50_inc5_resnet32.yaml"
    )
    candidate = load_config_tree(
        PROJECT_ROOT / "configs/explore/cifar100" / filename
    )
    options = resolve_edge_topology_options("replay", candidate["method"])
    assert options["enabled"]
    assert options["representatives_per_class"] == representatives
    assert options["lambda_edge"] == weight
    if representatives == 20:
        assert options["update_interval_steps"] == 10
    assert candidate["output"]["directory"].startswith("outputs/explore/")
    assert base_recipe_signature(candidate) == base_recipe_signature(control)


@pytest.mark.parametrize("substrate", ["replay", "icarl"])
def test_boundary_preserving_candidate_resolves_for_both_substrates(
    substrate: str,
) -> None:
    candidate = load_config_tree(
        PROJECT_ROOT
        / "configs/explore/cifar100"
        / f"{substrate}_edgecorr_inside_r2_lambda5.yaml"
    )
    options = resolve_edge_topology_options(substrate, candidate["method"])
    assert options["edge_weighting"] == "conflict_subtree_inside"


def test_edge_option_validation_and_base_signature() -> None:
    with pytest.raises(ValueError, match="Replay-CE or iCaRL"):
        resolve_edge_topology_options(
            "afc", {"edge_topology": {"enabled": True}}
        )
    with pytest.raises(ValueError, match="2, 4, or 20"):
        resolve_edge_topology_options(
            "icarl",
            {"edge_topology": {"enabled": True, "representatives_per_class": 3}},
        )
    control = load_config_tree(
        PROJECT_ROOT
        / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    candidate = copy.deepcopy(control)
    candidate["method"]["edge_topology"] = {
        "enabled": True,
        "representatives_per_class": 2,
    }
    assert base_recipe_signature(candidate) == base_recipe_signature(control)
