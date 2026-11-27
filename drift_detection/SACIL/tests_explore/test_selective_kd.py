from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import (
    base_recipe_signature,
    resolve_edge_topology_options,
    resolve_selective_kd_options,
)
from sacil.methods import (
    analytic_embedding_gradient_alignment,
    pycil_icarl_kd_loss,
    selective_pycil_icarl_kd_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_analytic_embedding_gradients_match_autograd() -> None:
    torch.manual_seed(21)
    features = torch.randn(1, 4, requires_grad=True)
    weights = torch.randn(5, 4)
    logits = features @ weights.T
    teacher = torch.randn(1, 3)
    target = torch.tensor([4])
    temperature = 2.0

    ce = F.cross_entropy(logits, target)
    ce_gradient = torch.autograd.grad(ce, features, retain_graph=True)[0]
    kd = pycil_icarl_kd_loss(
        logits[:, :3], teacher, temperature=temperature
    )
    kd_gradient = torch.autograd.grad(kd, features)[0]

    ce_delta = F.softmax(logits.detach(), dim=1)
    ce_delta[0, target.item()] -= 1.0
    expected_ce = ce_delta @ weights
    expected_kd = (
        (
            F.softmax(logits.detach()[:, :3] / temperature, dim=1)
            - F.softmax(teacher / temperature, dim=1)
        )
        / temperature
    ) @ weights[:3]
    torch.testing.assert_close(ce_gradient, expected_ce)
    torch.testing.assert_close(kd_gradient, expected_kd)

    alignment, keep = analytic_embedding_gradient_alignment(
        logits,
        teacher,
        target,
        weights,
        temperature=temperature,
        threshold=0.0,
    )
    expected_alignment = F.cosine_similarity(
        expected_ce, expected_kd, dim=1
    )
    torch.testing.assert_close(alignment, expected_alignment)
    assert not alignment.requires_grad and not keep.requires_grad


def test_all_new_rows_kept_is_exact_pycil_loss_and_gradient() -> None:
    torch.manual_seed(22)
    teacher = torch.randn(6, 4)
    weights = torch.randn(6, 5, requires_grad=True)
    replay = torch.tensor([True, False, True, False, True, False])
    targets = torch.tensor([0, 4, 1, 5, 2, 4])
    standard_logits = torch.randn(6, 6, requires_grad=True)
    candidate_logits = standard_logits.detach().clone().requires_grad_()

    standard = pycil_icarl_kd_loss(
        standard_logits[:, :4], teacher, temperature=2.0
    )
    result = selective_pycil_icarl_kd_loss(
        candidate_logits,
        teacher,
        targets,
        replay,
        weights,
        temperature=2.0,
        alignment_threshold=-2.0,
    )
    standard.backward()
    result.loss.backward()
    assert torch.equal(result.loss, standard)
    assert torch.equal(candidate_logits.grad, standard_logits.grad)
    assert float(result.new_keep_ratio) == 1.0
    # Routing is detached, so an independently supplied classifier matrix is
    # not part of the KD backward graph (no second-order path).
    assert weights.grad is None


def test_all_replay_batch_is_exact_pycil() -> None:
    torch.manual_seed(23)
    logits = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 3)
    result = selective_pycil_icarl_kd_loss(
        logits,
        teacher,
        torch.tensor([0, 1, 2, 0]),
        torch.ones(4, dtype=torch.bool),
        torch.randn(5, 4),
        alignment_threshold=0.0,
    )
    exact = pycil_icarl_kd_loss(logits[:, :3], teacher, temperature=2.0)
    assert torch.equal(result.loss, exact)
    assert result.old_count == 4 and result.new_count == 0


def test_all_new_batch_routes_without_replay_rows() -> None:
    torch.manual_seed(25)
    logits = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 3)
    result = selective_pycil_icarl_kd_loss(
        logits,
        teacher,
        torch.tensor([3, 4, 3, 4]),
        torch.zeros(4, dtype=torch.bool),
        torch.randn(5, 4),
        alignment_threshold=2.0,
    )
    assert result.old_count == 0 and result.new_count == 4
    assert float(result.old_kd) == 0.0
    assert float(result.new_keep_ratio) == 0.0
    assert float(result.loss) == 0.0
    gradient = torch.autograd.grad(result.loss, logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_dropped_new_rows_reduce_total_kd_force() -> None:
    torch.manual_seed(24)
    logits = torch.randn(5, 5, requires_grad=True)
    teacher = torch.randn(5, 3)
    replay = torch.tensor([True, False, True, False, False])
    targets = torch.tensor([0, 3, 1, 4, 3])
    result = selective_pycil_icarl_kd_loss(
        logits,
        teacher,
        targets,
        replay,
        torch.randn(5, 4),
        temperature=2.0,
        alignment_threshold=2.0,
    )
    old_exact = pycil_icarl_kd_loss(
        logits[replay, :3], teacher[replay], temperature=2.0
    )
    torch.testing.assert_close(result.loss, old_exact * (2 / 5))
    assert float(result.new_keep_ratio) == 0.0
    assert float(result.new_kd) == 0.0
    gradient = torch.autograd.grad(result.loss, logits)[0]
    torch.testing.assert_close(gradient[~replay], torch.zeros_like(gradient[~replay]))


def test_selective_config_preserves_icarl_base_signature() -> None:
    control = load_config_tree(
        PROJECT_ROOT
        / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    candidate = load_config_tree(
        PROJECT_ROOT
        / "configs/explore/cifar100/icarl_htpl_selective_kd_t0.yaml"
    )
    selective = resolve_selective_kd_options("icarl", candidate["method"])
    edge = resolve_edge_topology_options("icarl", candidate["method"])
    assert selective == {"enabled": True, "alignment_threshold": 0.0}
    assert edge["enabled"] and edge["representatives_per_class"] == 2
    assert edge["lambda_edge"] == 0.5
    assert edge["edge_weighting"] == "global"
    assert candidate["output"]["directory"] == "outputs/explore/selective_kd"
    assert base_recipe_signature(candidate) == base_recipe_signature(control)


def test_selective_option_validation() -> None:
    assert resolve_selective_kd_options("icarl", {}) == {
        "enabled": False,
        "alignment_threshold": 0.0,
    }
    with pytest.raises(ValueError, match="requires the iCaRL substrate"):
        resolve_selective_kd_options(
            "replay", {"selective_kd": {"enabled": True}}
        )
    with pytest.raises(ValueError, match="must be finite"):
        resolve_selective_kd_options(
            "icarl",
            {
                "selective_kd": {
                    "enabled": True,
                    "alignment_threshold": float("nan"),
                }
            },
        )
