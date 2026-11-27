from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import resolve_bgs_options
from sacil.methods import project_insertion_gradient


ROOT = Path(__file__).resolve().parents[1]


def test_conflicting_insertion_is_orthogonalized() -> None:
    stability = (torch.tensor([1.0, 0.0]),)
    insertion = (torch.tensor([-1.0, 1.0]),)
    result = project_insertion_gradient(stability, insertion)

    assert result.conflict
    assert result.cosine == pytest.approx(-(2.0 ** -0.5))
    # Projected insertion is [0, 1], so the optimizer gradient is [1, 1].
    assert torch.equal(result.gradients[0], torch.tensor([1.0, 1.0]))
    projected_insertion = result.gradients[0] - stability[0]
    assert torch.dot(projected_insertion, stability[0]).item() == pytest.approx(0.0)


def test_aligned_insertion_is_unchanged_and_none_gradients_are_supported() -> None:
    stability = (torch.tensor([1.0, 0.0]), None)
    insertion = (torch.tensor([2.0, 1.0]), torch.tensor([3.0]))
    result = project_insertion_gradient(stability, insertion)

    assert not result.conflict
    assert torch.equal(result.gradients[0], torch.tensor([3.0, 1.0]))
    assert torch.equal(result.gradients[1], torch.tensor([3.0]))
    assert result.insertion_retained_ratio == pytest.approx(1.0)


def test_projection_validates_shapes_and_epsilon() -> None:
    with pytest.raises(ValueError, match="identical lengths"):
        project_insertion_gradient((torch.ones(1),), ())
    with pytest.raises(ValueError, match="finite and positive"):
        project_insertion_gradient((torch.ones(1),), (torch.ones(1),), epsilon=0.0)


def test_pcli_config_is_explicit_resnet32_icarl() -> None:
    config = load_config_tree(
        ROOT / "configs/explore/cifar100/icarl_bgs_pcli.yaml"
    )
    options = resolve_bgs_options("icarl", config["method"])
    assert config["model"]["backbone"] == "resnet32"
    assert config["evaluation"]["classifier"] == "nme"
    assert config["memory"]["exemplars_per_class"] == 20
    assert options["insertion"]["enabled"]
    assert options["insertion"]["gradient_projection"]
    assert options["insertion"]["projection_epsilon"] == pytest.approx(1e-12)
