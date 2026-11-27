from __future__ import annotations

from pathlib import Path

import torch

from sacil.config import load_config_tree
from sacil.data import ClassOrderProtocol
from sacil.methods import (
    validate_imagenet100_b50_inc5_config,
    validate_imagenet100_b50_inc5_protocol,
)
from sacil.models import (
    AFCIncrementalNet,
    ExpandableLinearNet,
    FGPIncrementalNet,
    PyCILPODNet,
)


ROOT = Path(__file__).resolve().parents[1]


def test_imagenet100_protocol_is_b50_inc5() -> None:
    protocol = ClassOrderProtocol.from_json(
        ROOT
        / "experiment_configs/class_orders/imagenet100_b50_inc5_afc_order1.json"
    )
    validate_imagenet100_b50_inc5_protocol(protocol)
    assert protocol.num_sessions == 11
    assert [protocol.session(i).size for i in range(11)] == [50] + [5] * 10


def test_lucir_imagenet_config_contract() -> None:
    config = load_config_tree(
        ROOT / "configs/cmpt/imagenet100_b50_inc5/train_lucir.yaml"
    )
    validate_imagenet100_b50_inc5_config(config)
    assert config["seed"] == 1
    assert config["model"]["backbone"] == (
        "resnet18_imagenet_no_last_relu"
    )
    assert config["training"]["base"]["epochs"] == 90
    assert config["training"]["incremental"]["milestones"] == [30, 60]


def test_imagenet_resnet18_variants_have_standard_stem() -> None:
    lucir = ExpandableLinearNet(
        50, backbone="resnet18_imagenet_no_last_relu"
    )
    fgp = FGPIncrementalNet(
        50, backbone="fgp_resnet18_imagenet_no_last_relu"
    )
    assert lucir.backbone.conv1.kernel_size == (7, 7)
    assert lucir.backbone.conv1.stride == (2, 2)
    assert fgp.backbone.conv1.kernel_size == (7, 7)
    with torch.no_grad():
        lucir.eval()
        fgp.eval()
        images = torch.randn(1, 3, 64, 64)
        assert lucir(images).shape == (1, 50)
        assert fgp(images).shape == (1, 50)


def test_four_baseline_imagenet_configs_follow_shared_contract() -> None:
    expected_backbones = {
        "icarl": "resnet18_imagenet",
        "replay": "resnet18_imagenet",
        "podnet": "resnet18_imagenet",
        "afc": "afc_resnet18_imagenet_importance",
    }
    for method, backbone in expected_backbones.items():
        config = load_config_tree(
            ROOT
            / f"configs/cmpt/imagenet100_b50_inc5/train_{method}.yaml"
        )
        validate_imagenet100_b50_inc5_config(config)
        assert config["method"]["name"] == method
        assert config["model"]["backbone"] == backbone


def test_podnet_and_afc_have_imagenet_resnet18_adapters() -> None:
    podnet = PyCILPODNet(
        50, proxies_per_class=10, backbone="resnet18_imagenet"
    ).eval()
    afc = AFCIncrementalNet(
        50,
        initial_size=50,
        increment_size=5,
        proxies_per_class=10,
        backbone="afc_resnet18_imagenet_importance",
    ).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        podnet_output = podnet.forward_detailed(images)
        afc_output = afc.forward_detailed(images)
    assert podnet_output.logits.shape == (1, 50)
    assert podnet_output.features.shape == (1, 512)
    assert len(podnet_output.attentions) == 4
    assert afc_output.logits.shape == (1, 50)
    assert afc_output.features.shape == (1, 512)
    assert [value.shape[1] for value in afc_output.attentions] == [
        64,
        128,
        256,
        512,
    ]
