from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from sacil.config import load_config_tree
from sacil.methods import validate_imagenet100_b50_inc5_config
from sacil.models import (
    CSCCTImageNetResNet18,
    CSCCTIncrementalNet,
    ScaleShiftConv2d,
)


ROOT = Path(__file__).resolve().parents[1]


def _official_resnet_module():
    root = ROOT / "ref_codes/01_geometry_topology_preservation/CSCCT"
    path = root / "models/modified_resnet.py"
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "models" or name.startswith("models.")
    }
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(
            "official_cscct_imagenet_resnet", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load CSCCT reference: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(root))
        for name in tuple(sys.modules):
            if (name == "models" or name.startswith("models.")) and (
                name not in previous
            ):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_cscct_imagenet_resnet18_and_cosine_head_match_author_code() -> None:
    official = _official_resnet_module()
    torch.manual_seed(59)
    expected = official.resnet18(num_classes=50).eval()
    torch.manual_seed(59)
    actual = CSCCTIncrementalNet(
        50, backbone="cscct_modified_resnet18_imagenet"
    ).eval()

    assert isinstance(actual.first, CSCCTImageNetResNet18)
    expected_backbone = {
        key.replace(".downsample.", ".shortcut."): value
        for key, value in expected.state_dict().items()
        if not key.startswith("fc.")
    }
    actual_backbone = actual.first.state_dict()
    assert actual_backbone.keys() == expected_backbone.keys()
    for name in actual_backbone:
        assert torch.equal(actual_backbone[name], expected_backbone[name]), name
    assert torch.equal(actual.classifier.weight, expected.fc.weight)
    assert torch.equal(actual.classifier.sigma, expected.fc.sigma)
    assert len(actual.fusion) == 4

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(actual(images), expected(images))


def test_cscct_imagenet_expansion_uses_four_fusion_levels() -> None:
    model = CSCCTIncrementalNet(
        4, backbone="cscct_modified_resnet18_imagenet"
    ).eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = model.extract_features(images)
    imprint = F.normalize(torch.randn(2, 512), dim=1)
    model.expand_classes(imprint)
    model.eval()
    assert model.second is not None
    assert len(model.fusion) == 4
    assert any(
        isinstance(module, ScaleShiftConv2d)
        for module in model.first.modules()
    )
    with torch.no_grad():
        assert torch.allclose(
            features, model.extract_features(images), atol=1e-5
        )


def test_cscct_imagenet_config_uses_author_architecture_and_schedule() -> None:
    config = load_config_tree(
        ROOT / "configs/cmpt/imagenet100_b50_inc5/train_cscct.yaml"
    )
    validate_imagenet100_b50_inc5_config(config)
    assert config["method"]["name"] == "cscct"
    assert config["method"]["csc_weight"] == 3.0
    assert config["method"]["ct_weight"] == 1.5
    assert config["method"]["fusion_lr"] == 1e-8
    assert config["model"]["backbone"] == (
        "cscct_modified_resnet18_imagenet"
    )
    for phase in ("base", "incremental"):
        recipe = config["training"][phase]
        assert recipe["epochs"] == 160
        assert recipe["scheduler"] == "cosine"
        assert recipe["fusion_scheduler"] == "multistep"
        assert recipe["fusion_milestones"] == [53, 106]
