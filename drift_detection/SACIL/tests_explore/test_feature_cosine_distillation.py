from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import (
    base_recipe_signature,
    resolve_feature_cosine_distillation_options,
)
from sacil.methods import (
    cosine_feature_distillation_loss,
    cosine_imprinted_weights,
    lucir_interclass_margin_ranking_loss,
    normalized_cosine_classifier_logits,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cosine_feature_loss_matches_cosine_embedding() -> None:
    torch.manual_seed(3)
    current = torch.randn(7, 11, requires_grad=True)
    reference = torch.randn(7, 11)
    actual = cosine_feature_distillation_loss(current, reference)
    expected = F.cosine_embedding_loss(
        current, reference, torch.ones(7), reduction="mean"
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert float(current.grad.abs().sum()) > 0


def test_cosine_feature_loss_weighting_and_detached_teacher() -> None:
    current = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    reference = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    weights = torch.tensor([1.0, 3.0])
    loss = cosine_feature_distillation_loss(
        current, reference, sample_weights=weights
    )
    torch.testing.assert_close(loss, torch.tensor(0.75))
    loss.backward()
    assert current.grad is not None
    assert reference.grad is None


def test_feature_cosine_options_and_base_signature() -> None:
    canonical = load_config_tree(
        ROOT / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    config = load_config_tree(
        ROOT / "configs/explore/cifar100/icarl_feature_cosine_b6.yaml"
    )
    options = resolve_feature_cosine_distillation_options(
        "icarl", config["method"]
    )
    assert options["lambda"] == 6.0
    assert config["model"]["backbone"] == "resnet32"
    assert config["evaluation"]["classifier"] == "nme"
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["method"]["kd_weight"] == 0.0
    assert base_recipe_signature(config) == base_recipe_signature(canonical)


def test_feature_cosine_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        resolve_feature_cosine_distillation_options(
            "icarl", {"feature_cosine_distillation": {"enabled": True, "lambda": -1}}
        )


def test_normalized_cosine_logits_ignore_feature_and_weight_norms() -> None:
    features = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    weights = torch.tensor([[2.0, 0.0], [0.0, 7.0]])
    logits = normalized_cosine_classifier_logits(
        features, weights, scale=10.0
    )
    torch.testing.assert_close(logits, 10.0 * torch.eye(2))


def test_cosine_imprinting_uses_normalized_class_means_and_old_norm() -> None:
    old = torch.tensor([[3.0, 0.0], [0.0, 5.0]])
    class_features = [
        torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
        torch.tensor([[0.0, 3.0], [0.0, 9.0]]),
    ]
    values = cosine_imprinted_weights(class_features, old)
    torch.testing.assert_close(values, torch.tensor([[4.0, 0.0], [0.0, 4.0]]))


def test_lucir_margin_ranking_matches_official_topk_formula() -> None:
    scores = torch.tensor(
        [
            [0.6, 0.1, 0.4, 0.2],
            [0.2, 0.7, 0.9, 0.8],
            [0.1, 0.2, 0.7, 0.6],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2])
    loss = lucir_interclass_margin_ranking_loss(
        scores, targets, known_classes=2, margin=0.5, top_k=2
    )
    # Only the two old-class rows participate.  Their four hinge values are
    # [0.3, 0.1, 0.7, 0.6].
    torch.testing.assert_close(loss, torch.tensor(0.425))
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_lucir_margin_returns_connected_zero_without_old_rows() -> None:
    scores = torch.randn(3, 5, requires_grad=True)
    loss = lucir_interclass_margin_ranking_loss(
        scores, torch.tensor([2, 3, 4]), known_classes=2
    )
    assert float(loss) == 0.0
    loss.backward()
    assert scores.grad is not None


def test_lucir_margin_config_matches_official_defaults() -> None:
    config = load_config_tree(
        ROOT / "configs/explore/cifar100/icarl_lucir_mr_cosinehead.yaml"
    )
    options = resolve_feature_cosine_distillation_options(
        "icarl", config["method"]
    )
    margin = options["margin_ranking"]
    assert margin == {
        "enabled": True,
        "margin": 0.5,
        "top_k": 2,
        "weight": 1.0,
        "implementation": "official_lucir_mr_lf_v1",
    }
    assert config["model"]["backbone"] == "resnet32"
    assert config["evaluation"]["classifier"] == "nme"


def test_cosinehead_config_is_explicit() -> None:
    config = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_b6_cosinehead.yaml"
    )
    options = resolve_feature_cosine_distillation_options(
        "icarl", config["method"]
    )
    assert options["training_classifier"] == "normalized_cosine"
    assert options["imprint_new_weights"]
    assert options["freeze_old_weights"]


def test_lucir_control_uses_old_over_new_adaptation() -> None:
    config = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_cosinehead.yaml"
    )
    options = resolve_feature_cosine_distillation_options(
        "icarl", config["method"]
    )
    assert options["lambda"] == 5.0
    assert options["adaptive_mode"] == "old_over_new"
    assert not options["adaptive_new_over_old"]


def test_throttled_l10_config_changes_runtime_only() -> None:
    control = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_l10_cosinehead.yaml"
    )
    throttled = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_l10_cosinehead_throttled.yaml"
    )
    for key in ("data", "protocol", "model", "method", "training", "memory", "evaluation"):
        if key in control:
            assert throttled[key] == control[key]
    assert throttled["runtime"] == {"gpu_throttle_ms": 50}


def test_feature_cosine_rejects_unknown_adaptive_mode() -> None:
    with pytest.raises(ValueError, match="adaptive_mode"):
        resolve_feature_cosine_distillation_options(
            "icarl",
            {
                "feature_cosine_distillation": {
                    "enabled": True,
                    "adaptive_mode": "sideways",
                }
            },
        )


def test_feature_cosine_resolves_replay_only_scope() -> None:
    options = resolve_feature_cosine_distillation_options(
        "icarl",
        {
            "feature_cosine_distillation": {
                "enabled": True,
                "sample_scope": "replay-only",
            }
        },
    )
    assert options["sample_scope"] == "replay_only"


def test_feature_cosine_resolves_explicit_old_logit_combination() -> None:
    options = resolve_feature_cosine_distillation_options(
        "icarl",
        {
            "feature_cosine_distillation": {
                "enabled": True,
                "combine_old_logit_kd": True,
            }
        },
    )
    assert options["combine_old_logit_kd"]


def test_feature_cosine_rejects_unknown_sample_scope() -> None:
    with pytest.raises(ValueError, match="sample_scope"):
        resolve_feature_cosine_distillation_options(
            "icarl",
            {
                "feature_cosine_distillation": {
                    "enabled": True,
                    "sample_scope": "confused-only",
                }
            },
        )


def test_full_cosinehead_control_uses_resnet32_and_matched_session_recipe() -> None:
    config = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_full_cosinehead.yaml"
    )
    assert config["model"]["backbone"] == "resnet32"
    assert config["training"]["base"]["epochs"] == 160
    assert config["training"]["incremental"]["epochs"] == 160
    assert config["training"]["base"]["milestones"] == [80, 120]
    assert config["training"]["incremental"]["milestones"] == [80, 120]


def test_cosinehead_bgs_candidate_changes_only_opt_in_geometry() -> None:
    control = load_config_tree(
        ROOT / "configs/explore/cifar100/icarl_feature_cosine_b6_cosinehead.yaml"
    )
    candidate = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_b6_cosinehead_bgs.yaml"
    )
    assert candidate["model"] == control["model"]
    assert candidate["training"] == control["training"]
    assert candidate["memory"] == control["memory"]
    assert candidate["evaluation"] == control["evaluation"]
    assert candidate["method"]["feature_cosine_distillation"] == control["method"][
        "feature_cosine_distillation"
    ]
    assert candidate["method"]["boundary_graph_surgery"]["enabled"]
    assert not candidate["method"]["boundary_graph_surgery"]["insertion"]["enabled"]


def test_lucir_cosinehead_bgs_candidate_is_a_matched_pair() -> None:
    control = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_cosinehead.yaml"
    )
    candidate = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_cosinehead_bgs.yaml"
    )
    for key in ("model", "training", "memory", "evaluation"):
        assert candidate[key] == control[key]
    assert candidate["method"]["feature_cosine_distillation"] == control["method"][
        "feature_cosine_distillation"
    ]
    assert candidate["method"]["boundary_graph_surgery"]["enabled"]
    assert candidate["method"]["boundary_graph_surgery"]["geometry"][
        "mask_mode"
    ] == "structured"
