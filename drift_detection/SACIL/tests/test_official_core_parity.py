from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from sacil.models.pycil_podnet import (
    PyCILCosineLinear,
    PyCILSplitCosineLinear,
)
from sacil.models.pycil_linear import PyCILSimpleLinear
from sacil.methods.create import ClasswiseAutoencoderClassifier
from sacil.methods.fgp import RectifiedCosineLinear
from sacil.models.afc_resnet32 import AFCResNet32
from sacil.models.resnet32 import resnet32
from sacil.models.table1_models import CSCCTIncrementalNet, FGPResNet32


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_reference(relative_path: str, module_name: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference_with_import_root(
    relative_path: str,
    module_name: str,
    import_root: str,
):
    root = str(PROJECT_ROOT / import_root)
    generic_prefixes = ("models", "utils")
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in generic_prefixes
        )
    }
    sys.path.insert(0, root)
    try:
        return _load_reference(relative_path, module_name)
    finally:
        sys.path.remove(root)
        for name in tuple(sys.modules):
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in generic_prefixes
            ) and name not in previous_modules:
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def _load_afc_backbone_reference():
    names = ("inclearn", "inclearn.lib", "inclearn.lib.pooling")
    previous = {name: sys.modules.get(name) for name in names}
    inclearn = types.ModuleType("inclearn")
    library = types.ModuleType("inclearn.lib")
    pooling = types.ModuleType("inclearn.lib.pooling")
    pooling.WeldonPool2d = nn.AdaptiveAvgPool2d
    library.pooling = pooling
    inclearn.lib = library
    sys.modules.update(
        {
            "inclearn": inclearn,
            "inclearn.lib": library,
            "inclearn.lib.pooling": pooling,
        }
    )
    try:
        return _load_reference(
            "ref_codes/02_stability_plasticity_selective_preservation/AFC/"
            "inclearn/convnet/my_resnet_importance.py",
            "official_afc_rebuffi_importance",
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _assert_state_equal(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for name in left_state:
        assert torch.equal(left_state[name], right_state[name]), name


def test_stock_pycil_resnet32_state_and_forward_parity() -> None:
    official = _load_reference(
        "ref_codes/00_frameworks/PyCIL/convs/cifar_resnet.py",
        "official_pycil_cifar_resnet",
    )
    torch.manual_seed(19)
    expected = official.resnet32().eval()
    torch.manual_seed(19)
    actual = resnet32().eval()
    _assert_state_equal(actual, expected)

    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        expected_output = expected(images)
        actual_output = actual.forward_detailed(images)
    assert torch.equal(actual_output["features"], expected_output["features"])
    for actual_map, expected_map in zip(
        actual_output["fmaps"], expected_output["fmaps"]
    ):
        assert torch.equal(actual_map, expected_map)


def test_stock_pycil_simple_linear_initialization_parity() -> None:
    official = _load_reference(
        "ref_codes/00_frameworks/PyCIL/convs/linears.py",
        "official_pycil_linears_simple",
    )
    torch.manual_seed(23)
    expected = official.SimpleLinear(64, 50)
    torch.manual_seed(23)
    actual = PyCILSimpleLinear(64, 50)
    assert torch.equal(actual.weight, expected.weight)
    assert torch.equal(actual.bias, expected.bias)


def test_stock_pycil_cosine_linear_forward_and_gradient_parity() -> None:
    official = _load_reference(
        "ref_codes/00_frameworks/PyCIL/convs/linears.py",
        "official_pycil_linears_cosine",
    )
    torch.manual_seed(29)
    expected = official.CosineLinear(16, 7, 3, True)
    torch.manual_seed(29)
    actual = PyCILCosineLinear(16, 7, 3, True)
    _assert_state_equal(actual, expected)

    expected_input = torch.randn(5, 16, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected_loss = expected(expected_input)["logits"].square().mean()
    actual_loss = actual(actual_input)["logits"].square().mean()
    assert torch.equal(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    assert torch.equal(actual_input.grad, expected_input.grad)
    assert torch.equal(actual.weight.grad, expected.weight.grad)


def test_stock_pycil_split_cosine_forward_and_gradient_parity() -> None:
    official = _load_reference(
        "ref_codes/00_frameworks/PyCIL/convs/linears.py",
        "official_pycil_linears_split",
    )
    torch.manual_seed(31)
    expected = official.SplitCosineLinear(12, 6, 2, 4)
    torch.manual_seed(31)
    actual = PyCILSplitCosineLinear(12, 6, 2, 4)
    _assert_state_equal(actual, expected)

    expected_input = torch.randn(3, 12, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected_loss = expected(expected_input)["logits"].sum()
    actual_loss = actual(actual_input)["logits"].sum()
    assert torch.equal(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    assert torch.equal(actual_input.grad, expected_input.grad)
    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
        actual.named_parameters(), expected.named_parameters()
    ):
        assert actual_name == expected_name
        assert torch.equal(actual_parameter.grad, expected_parameter.grad)


def test_create_classwise_autoencoder_forward_parity() -> None:
    official = _load_reference(
        "ref_codes/01_geometry_topology_preservation/CREATE/convs/linears.py",
        "official_create_linears",
    )
    args = {"hidden_layers": [], "ae_latent": 32}
    torch.manual_seed(37)
    expected = official.CSSRClassifier(args, 64, 5).eval()
    torch.manual_seed(37)
    actual = ClasswiseAutoencoderClassifier(
        64,
        5,
        hidden_layers=(),
        latent_features=32,
        reconstruction_scale=0.1,
    ).eval()
    features = torch.randn(4, 64)
    with torch.no_grad():
        expected_output = expected(features)
        actual_output = actual(features)
    assert torch.equal(actual_output["logits"], expected_output["logits"])
    assert torch.equal(
        actual_output["error_logits"], expected_output["error"]
    )
    assert torch.equal(actual_output["latents"], expected_output["fm"])
    assert torch.equal(
        actual_output["reconstructions"], expected_output["recon"]
    )


def test_fgp_backbone_and_classifier_forward_parity() -> None:
    official_backbone = _load_reference(
        "ref_codes/01_geometry_topology_preservation/FGP-ICL/convnet/cifar_resnet.py",
        "official_fgp_cifar_resnet",
    )
    official_classifier = _load_reference(
        "ref_codes/01_geometry_topology_preservation/FGP-ICL/lib/normalized_fc.py",
        "official_fgp_normalized_fc",
    )
    torch.manual_seed(41)
    expected_backbone = official_backbone.resnet32(
        nf=64, zero_init_residual=True
    ).eval()
    torch.manual_seed(41)
    actual_backbone = FGPResNet32().eval()
    images = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(actual_backbone(images), expected_backbone(images))

    torch.manual_seed(43)
    expected_head = official_classifier.CosineLinear(
        64, 7, torch.device("cpu"), bias=True, eta=True
    )
    torch.manual_seed(43)
    actual_head = RectifiedCosineLinear(64, 7, bias=True)
    features = torch.randn(5, 64)
    assert torch.equal(actual_head(features), expected_head(features))


def test_cscct_base_resnet_and_cosine_head_parity() -> None:
    official = _load_reference_with_import_root(
        "ref_codes/01_geometry_topology_preservation/CSCCT/"
        "models/modified_resnet_cifar.py",
        "official_cscct_modified_resnet_cifar",
        "ref_codes/01_geometry_topology_preservation/CSCCT",
    )
    torch.manual_seed(45)
    expected = official.resnet32(num_classes=50).eval()
    torch.manual_seed(45)
    actual = CSCCTIncrementalNet(50).eval()

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
    images = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(actual(images), expected(images))


def test_cscct_first_incremental_dual_branch_feature_parity() -> None:
    official_base = _load_reference_with_import_root(
        "ref_codes/01_geometry_topology_preservation/CSCCT/"
        "models/modified_resnet_cifar.py",
        "official_cscct_incremental_base",
        "ref_codes/01_geometry_topology_preservation/CSCCT",
    )
    official_mtl = _load_reference_with_import_root(
        "ref_codes/01_geometry_topology_preservation/CSCCT/"
        "models/modified_resnetmtl_cifar.py",
        "official_cscct_incremental_mtl",
        "ref_codes/01_geometry_topology_preservation/CSCCT",
    )
    official_process = _load_reference_with_import_root(
        "ref_codes/01_geometry_topology_preservation/CSCCT/"
        "utils/process_fp.py",
        "official_cscct_process_fp",
        "ref_codes/01_geometry_topology_preservation/CSCCT",
    )

    torch.manual_seed(53)
    reference_base = official_base.resnet32(num_classes=50).eval()
    reference_state = reference_base.state_dict()
    first = official_mtl.resnetmtl32(num_classes=50).eval()
    first_state = first.state_dict()
    first_state.update(reference_state)
    first.load_state_dict(first_state)
    second = official_base.resnet32(num_classes=50).eval()
    second_state = second.state_dict()
    second_state.update(reference_state)
    second.load_state_dict(second_state)

    torch.manual_seed(53)
    actual = CSCCTIncrementalNet(50).eval()
    imprint = F.normalize(torch.randn(5, 64), dim=1)
    actual.expand_classes(imprint)
    actual.eval()

    images = torch.randn(4, 3, 32, 32)
    fusion = nn.ParameterList(
        nn.Parameter(torch.tensor([0.5])) for _ in range(3)
    )
    with torch.no_grad():
        expected_features = official_process.process_inputs_fp(
            SimpleNamespace(dataset="cifar100"),
            fusion,
            first,
            second,
            images,
            feature_mode=True,
        )
        actual_features = actual.extract_features(images)
    assert torch.equal(actual_features, expected_features)


def test_afc_rebuffi_importance_backbone_forward_parity() -> None:
    official = _load_afc_backbone_reference()
    torch.manual_seed(47)
    expected = official.CifarResNet(
        n=5,
        classifier_no_act=True,
        pooling_config={"type": "avg"},
        zero_residual=True,
    ).eval()
    torch.manual_seed(47)
    actual = AFCResNet32(blocks_per_stage=5).eval()
    images = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        expected_output = expected(images)
        actual_output = actual(images)
    assert torch.equal(actual_output.raw_features, expected_output["raw_features"])
    assert torch.equal(actual_output.features, expected_output["features"])
    for actual_map, expected_map in zip(
        actual_output.attentions, expected_output["attention"]
    ):
        assert torch.equal(actual_map, expected_map)
