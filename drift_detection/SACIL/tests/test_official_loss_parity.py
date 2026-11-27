from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD

from sacil.methods import (
    afc_nca_loss,
    afc_pod_loss,
    controlled_transfer_loss,
    create_classification_loss,
    create_kd_loss,
    cross_space_clustering_loss,
    fgp_graph_preservation_loss,
    old_logit_kl_loss,
    pod_flat_loss,
    pod_spatial_loss,
    podnet_nca_loss,
    scheduled_afc_factor,
)
from sacil.methods.casper import (
    _author_affinity,
    _normalize_affinity,
    casper_spectral_loss,
    pairwise_feature_distance,
)
from sacil.methods.create import ClasswiseAutoencoderClassifier
from sacil.models import ExpandableLinearNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_reference(relative_path: str, module_name: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference_functions(
    relative_path: str,
    names: set[str],
    namespace: dict[str, object],
) -> dict[str, object]:
    """Compile the selected function bodies directly from an official file."""

    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    if {node.name for node in selected} != names:
        missing = names - {node.name for node in selected}
        raise RuntimeError(f"reference functions not found: {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    values = dict(namespace)
    exec(compile(module, relative_path, "exec"), values)
    return {name: values[name] for name in names}


def _assert_parameter_values_equal(
    actual: nn.Module, expected: nn.Module
) -> None:
    actual_values = tuple(actual.parameters())
    expected_values = tuple(expected.parameters())
    assert len(actual_values) == len(expected_values)
    for actual_value, expected_value in zip(actual_values, expected_values):
        assert torch.equal(actual_value, expected_value)


def test_pycil_linear_network_ce_optimizer_step_parity() -> None:
    official_backbone = _load_reference(
        "ref_codes/00_frameworks/PyCIL/convs/cifar_resnet.py",
        "official_pycil_step_cifar_resnet",
    )
    official_linear = _load_reference(
        "ref_codes/00_frameworks/PyCIL/convs/linears.py",
        "official_pycil_step_linears",
    )

    class OfficialNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = official_backbone.resnet32()
            self.classifier = official_linear.SimpleLinear(64, 50)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.backbone(images)["features"]
            return self.classifier(features)["logits"]

    torch.manual_seed(101)
    expected = OfficialNetwork().train()
    torch.manual_seed(101)
    actual = ExpandableLinearNet(50).train()
    _assert_parameter_values_equal(actual, expected)

    images = torch.randn(8, 3, 32, 32)
    targets = torch.arange(8) % 50
    expected_optimizer = SGD(
        expected.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    actual_optimizer = SGD(
        actual.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    expected_logits = expected(images)
    actual_logits = actual(images)
    assert torch.equal(actual_logits, expected_logits)
    expected_loss = F.cross_entropy(expected_logits, targets)
    actual_loss = F.cross_entropy(actual_logits, targets)
    assert torch.equal(actual_loss, expected_loss)
    expected_optimizer.zero_grad()
    actual_optimizer.zero_grad()
    expected_loss.backward()
    actual_loss.backward()
    expected_optimizer.step()
    actual_optimizer.step()
    _assert_parameter_values_equal(actual, expected)


def test_podnet_nca_spatial_and_flat_losses_match_pycil_source() -> None:
    official = _load_reference_functions(
        "ref_codes/00_frameworks/PyCIL/models/podnet.py",
        {"nca", "pod_spatial_loss"},
        {"torch": torch, "F": F},
    )
    targets = torch.tensor([0, 2, 1, 3])
    expected_logits = torch.randn(4, 5, requires_grad=True)
    actual_logits = expected_logits.detach().clone().requires_grad_(True)
    expected_nca = official["nca"](expected_logits, targets)
    actual_nca = podnet_nca_loss(actual_logits, targets)
    assert torch.equal(actual_nca, expected_nca)
    expected_nca.backward()
    actual_nca.backward()
    assert torch.equal(actual_logits.grad, expected_logits.grad)

    expected_current = [torch.randn(3, 4, 5, 5, requires_grad=True)]
    actual_current = [expected_current[0].detach().clone().requires_grad_(True)]
    reference = [torch.randn(3, 4, 5, 5)]
    expected_spatial = official["pod_spatial_loss"](
        reference, expected_current
    )
    actual_spatial = pod_spatial_loss(actual_current, reference)
    assert torch.equal(actual_spatial, expected_spatial)
    expected_spatial.backward()
    actual_spatial.backward()
    assert torch.equal(actual_current[0].grad, expected_current[0].grad)

    current_features = torch.randn(4, 16)
    reference_features = torch.randn(4, 16)
    expected_flat = F.cosine_embedding_loss(
        current_features,
        reference_features,
        torch.ones(4),
    )
    assert torch.equal(
        pod_flat_loss(current_features, reference_features), expected_flat
    )


def test_afc_nca_pod_and_schedule_match_author_source() -> None:
    official_nca = _load_reference_functions(
        "ref_codes/02_stability_plasticity_selective_preservation/AFC/"
        "inclearn/lib/losses/base.py",
        {"nca"},
        {"torch": torch, "F": F},
    )["nca"]
    official_pod = _load_reference_functions(
        "ref_codes/02_stability_plasticity_selective_preservation/AFC/"
        "inclearn/lib/losses/distillation.py",
        {"pod"},
        {"torch": torch, "F": F},
    )["pod"]
    targets = torch.tensor([0, 2, 1, 3])
    expected_logits = torch.randn(4, 5, requires_grad=True)
    actual_logits = expected_logits.detach().clone().requires_grad_(True)
    expected_nca = official_nca(
        expected_logits,
        targets,
        scale=1.0,
        margin=0.6,
        exclude_pos_denominator=True,
    )
    actual_nca = afc_nca_loss(actual_logits, targets, 1.0)
    assert torch.equal(actual_nca, expected_nca)
    expected_nca.backward()
    actual_nca.backward()
    assert torch.equal(actual_logits.grad, expected_logits.grad)

    old_maps = [torch.randn(2, 4, 5, 5)]
    expected_maps = [torch.randn(2, 4, 5, 5, requires_grad=True)]
    actual_maps = [expected_maps[0].detach().clone().requires_grad_(True)]
    importance = [torch.rand(4)]
    expected_pod = official_pod(
        old_maps,
        expected_maps,
        collapse_channels="pixel",
        feature_distil_factor=importance,
    )
    actual_pod = afc_pod_loss(old_maps, actual_maps, importance)
    assert torch.equal(actual_pod, expected_pod)
    expected_pod.backward()
    actual_pod.backward()
    assert torch.equal(actual_maps[0].grad, expected_maps[0].grad)
    assert scheduled_afc_factor(55, 5, 4.0) == 4.0 * (55 / 5) ** 0.5


def test_create_classifier_and_kd_optimizer_step_match_author_source() -> None:
    official_linear = _load_reference(
        "ref_codes/01_geometry_topology_preservation/CREATE/convs/linears.py",
        "official_create_step_linears",
    )
    official_kd = _load_reference_functions(
        "ref_codes/01_geometry_topology_preservation/CREATE/models/create.py",
        {"_KD_loss"},
        {"torch": torch},
    )["_KD_loss"]
    args = {"hidden_layers": [], "ae_latent": 32}
    torch.manual_seed(107)
    expected = official_linear.CSSRClassifier(args, 64, 5).train()
    torch.manual_seed(107)
    actual = ClasswiseAutoencoderClassifier(
        64,
        5,
        hidden_layers=(),
        latent_features=32,
        reconstruction_scale=0.1,
    ).train()
    _assert_parameter_values_equal(actual, expected)
    features = torch.randn(8, 64)
    targets = torch.arange(8) % 5
    expected_output = expected(features)
    actual_output = actual(features)
    expected_loss = -(
        F.one_hot(targets, 5).to(expected_output["logits"])
        * torch.log(expected_output["logits"])
    ).sum(1).mean()
    actual_loss = create_classification_loss(actual_output["logits"], targets)
    assert torch.equal(actual_loss, expected_loss)
    expected_optimizer = SGD(expected.parameters(), lr=0.005, momentum=0.9)
    actual_optimizer = SGD(actual.parameters(), lr=0.005, momentum=0.9)
    expected_optimizer.zero_grad()
    actual_optimizer.zero_grad()
    expected_loss.backward()
    actual_loss.backward()
    expected_optimizer.step()
    actual_optimizer.step()
    _assert_parameter_values_equal(actual, expected)

    expected_old = torch.randn(6, 4, requires_grad=True)
    actual_old = expected_old.detach().clone().requires_grad_(True)
    teacher = torch.randn(6, 4)
    expected_kd = official_kd(expected_old, teacher, 2.0)
    actual_kd = create_kd_loss(
        actual_old, teacher, temperature=2.0
    )
    assert torch.equal(actual_kd, expected_kd)
    expected_kd.backward()
    actual_kd.backward()
    assert torch.equal(actual_old.grad, expected_old.grad)


def test_fgp_graph_loss_matches_author_equations_and_gradient() -> None:
    expected_features = torch.randn(6, 12, requires_grad=True)
    actual_features = expected_features.detach().clone().requires_grad_(True)
    old_features = torch.randn(6, 12)
    expected_weights = torch.randn(5, 12, requires_grad=True)
    actual_weights = expected_weights.detach().clone().requires_grad_(True)
    old_weights = torch.randn(5, 12)

    def euclidean(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distances = left.pow(2).sum(1, keepdim=True).expand(
            len(left), len(right)
        ) + right.pow(2).sum(1, keepdim=True).expand(
            len(right), len(left)
        ).T
        return distances.addmm(left, right.T, beta=1.0, alpha=-2.0)

    new_distance = euclidean(
        F.normalize(expected_features, dim=1),
        F.normalize(expected_weights, dim=1),
    )
    old_distance = euclidean(
        F.normalize(old_features, dim=1),
        F.normalize(old_weights, dim=1),
    )
    expected = (
        torch.exp(-0.5 * old_distance)
        * (new_distance - old_distance).square()
    ).sum() / len(new_distance)
    actual = fgp_graph_preservation_loss(
        actual_features,
        old_features,
        actual_weights,
        old_weights,
    )
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.equal(actual_features.grad, expected_features.grad)
    assert torch.equal(actual_weights.grad, expected_weights.grad)


def test_cscct_incremental_losses_match_author_equations() -> None:
    current_logits = torch.randn(10, 55)
    old_logits = torch.randn(10, 50)
    temperature = 2.0
    expected_kd = (
        nn.KLDivLoss()(
            F.log_softmax(current_logits[:, :50] / temperature, dim=1),
            F.softmax(old_logits / temperature, dim=1),
        )
        * temperature**2
        * 0.25
        * 50
    )
    actual_kd = 0.25 * old_logit_kl_loss(
        current_logits[:, :50], old_logits, temperature=temperature
    )
    torch.testing.assert_close(actual_kd, expected_kd, rtol=1e-6, atol=1e-7)

    current_features = torch.randn(10, 64)
    old_features = torch.randn(10, 64)
    targets = torch.tensor([0, 1, 2, 3, 4, 50, 51, 52, 53, 54])

    def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_norm = left.norm(dim=1)[:, None]
        right_norm = right.norm(dim=1)[:, None]
        left = left / torch.maximum(left_norm, torch.full_like(left_norm, 1e-8))
        right = right / torch.maximum(
            right_norm, torch.full_like(right_norm, 1e-8)
        )
        return left @ right.T

    signed = torch.where(
        targets[:, None] == targets[None, :], 1, -1
    ).flatten()
    expected_csc = ((1 - similarity(current_features, old_features).flatten()) * signed).mean()
    actual_csc = cross_space_clustering_loss(
        current_features, old_features, targets
    )
    assert torch.equal(actual_csc, expected_csc)

    expected_ct = (
        nn.KLDivLoss()(
            F.log_softmax(
                similarity(current_features[5:], current_features[:5])
                / temperature,
                dim=1,
            ),
            F.softmax(
                similarity(old_features[5:], old_features[:5]) / temperature,
                dim=1,
            ),
        )
        * temperature**2
    )
    actual_ct = controlled_transfer_loss(
        current_features,
        old_features,
        targets,
        known_classes=50,
        temperature=temperature,
    )
    torch.testing.assert_close(actual_ct, expected_ct, rtol=1e-6, atol=1e-7)


def _load_casper_spectral_reference():
    names = ("utils", "utils.knn")
    previous = {name: sys.modules.get(name) for name in names}
    package = types.ModuleType("utils")
    package.__path__ = []
    sys.modules["utils"] = package
    try:
        knn = _load_reference(
            "ref_codes/01_geometry_topology_preservation/CaSpeR-IL/utils/knn.py",
            "utils.knn",
        )
        sys.modules["utils.knn"] = knn
        return _load_reference(
            "ref_codes/01_geometry_topology_preservation/CaSpeR-IL/"
            "utils/spectral_analysis.py",
            "official_casper_spectral_analysis",
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_casper_affinity_and_eigengap_match_author_source() -> None:
    official = _load_casper_spectral_reference()
    features = torch.randn(32, 10)
    distances = pairwise_feature_distance(features)
    expected_affinity, expected_degree, _ = official.calc_ADL_knn(
        distances, k=4, symmetric=True
    )
    actual_affinity, actual_degree = _author_affinity(distances, 4)
    assert torch.equal(actual_affinity, expected_affinity)
    assert torch.equal(actual_degree, expected_degree)
    laplacian = torch.eye(32) - official.normalize_A(
        expected_affinity, expected_degree
    )
    eigenvalues, _ = official.find_eigs(laplacian, n_pairs=10)
    expected = eigenvalues[:6].sum() - eigenvalues[6]
    actual = casper_spectral_loss(
        features, num_classes=5, k=4, solver="xitorch"
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(
        _normalize_affinity(actual_affinity, actual_degree),
        official.normalize_A(expected_affinity, expected_degree),
    )
