from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sacil.engine.table1_trainer import (
    BalancedClassBatchSampler,
    StandaloneTable1Trainer,
    base_recipe_signature,
    geometry_preservation_component,
    resolve_casper_options,
    resolve_geometry_mode,
    resolve_geometry_options,
)
from sacil.methods import casper_spectral_loss
from sacil.methods import (
    pycil_finetune_loss,
    pycil_icarl_kd_loss,
    unified_method_contract,
)
from sacil.models import (
    AFCMultiProxyClassifier,
    CREATEIncrementalNet,
    CSCCTIncrementalNet,
    FGPIncrementalNet,
    FGPResNet32,
    ScaleShiftConv2d,
)
from sacil.config import load_config_tree


def test_afc_classifier_matches_scaled_negative_cosine_distance() -> None:
    classifier = AFCMultiProxyClassifier(
        3, [2], proxies_per_class=1, distance_scale=3.0
    )
    features = torch.tensor([[1.0, 0.0, 0.0]])
    with torch.no_grad():
        classifier._weights[0].copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
    assert torch.allclose(
        classifier(features), torch.tensor([[0.0, -18.0]]), atol=1e-6
    )


def test_fgp_backbone_and_incremental_classifier_shapes() -> None:
    backbone = FGPResNet32()
    features = backbone(torch.randn(2, 3, 32, 32))
    assert features.shape == (2, 64)
    model = FGPIncrementalNet(4)
    old = model.classifier.weight.detach().clone()
    model.expand_classes(6)
    assert torch.equal(model.classifier.weight[:4], old)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 6)


def test_create_native_model_expands_without_changing_old_autoencoders() -> None:
    model = CREATEIncrementalNet(3, hidden_layers=(), latent_features=4)
    old = {
        key: value.detach().clone()
        for key, value in model.classifier.class_autoencoders[0].state_dict().items()
    }
    model.expand_classes(5)
    for key, value in old.items():
        assert torch.equal(
            model.classifier.class_autoencoders[0].state_dict()[key], value
        )
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 5)


def test_cscct_expansion_builds_scale_shift_free_dual_branch() -> None:
    model = CSCCTIncrementalNet(4)
    images = torch.randn(2, 3, 32, 32)
    features = model.extract_features(images)
    imprint = F.normalize(torch.randn(2, 64), dim=1)
    model.expand_classes(imprint)
    assert model.second is not None
    assert any(isinstance(module, ScaleShiftConv2d) for module in model.first.modules())
    assert torch.allclose(features, model.extract_features(images), atol=1e-5)
    loss = model(images).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.fusion)


def test_casper_balanced_sampler_uses_exact_class_count() -> None:
    labels = [label for label in range(8) for _ in range(3)]
    sampler = BalancedClassBatchSampler(
        labels,
        batch_size=20,
        classes_per_batch=5,
        batches=4,
        seed=7,
    )
    for batch in sampler:
        assert len(batch) == 20
        assert len({labels[position] for position in batch}) == 5


def test_casper_balanced_sampler_matches_author_without_replacement() -> None:
    labels = [label for label in range(8) for _ in range(20)]
    sampler = BalancedClassBatchSampler(
        labels,
        batch_size=128,
        classes_per_batch=8,
        batches=2,
        seed=11,
    )
    for batch in sampler:
        for label in set(labels[position] for position in batch):
            positions = [position for position in batch if labels[position] == label]
            assert len(positions) == 16
            assert len(set(positions)) == len(positions)


def test_casper_partial_eigensolver_has_finite_gradient() -> None:
    features = torch.randn(64, 8, requires_grad=True)
    loss = casper_spectral_loss(
        features, num_classes=5, k=10, solver="partial"
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_casper_plugin_options_preserve_the_icarl_substrate() -> None:
    disabled = resolve_casper_options("icarl", {})
    assert disabled["enabled"] is False
    options = resolve_casper_options(
        "icarl",
        {
            "casper": {
                "enabled": True,
                "weight": 0.001,
                "knn": 8,
                "classes_per_graph": 8,
                "replay_batch_size": 64,
                "solver": "xitorch",
                "wd_reg": 0.0,
            }
        },
    )
    assert options == {
        "enabled": True,
        "weight": 0.001,
        "knn": 8,
        "classes_per_graph": 8,
        "replay_batch_size": 64,
        "solver": "xitorch",
        "wd_reg": 0.0,
    }
    with pytest.raises(ValueError, match="wd_reg=0"):
        resolve_casper_options(
            "icarl", {"casper": {"enabled": True, "wd_reg": 1e-5}}
        )
    with pytest.raises(ValueError, match="unsupported for substrate"):
        resolve_casper_options(
            "replay", {"casper": {"enabled": True}}
        )


def test_geometry_mode_keeps_the_cil_substrate_independent() -> None:
    assert resolve_geometry_mode("icarl", {}) == "none"
    assert resolve_geometry_mode("icarl", {"geometry_mode": "sacil"}) == "sacil"
    assert resolve_geometry_mode("afc", {"geometry_mode": "flat"}) == "flat"
    assert resolve_geometry_mode(
        "sacil", {"local_relaxation": False}
    ) == "global"
    assert resolve_geometry_mode(
        "sacil", {"use_internal_anchors": False}
    ) == "flat"
    with pytest.raises(ValueError, match="unsupported for substrate"):
        resolve_geometry_mode("replay", {"geometry_mode": "sacil"})


def test_geometry_anchor_ablation_options_are_explicit_and_validated() -> None:
    assert resolve_geometry_options({})["anchor_frame"] == "fixed"
    assert resolve_geometry_options({})["reliability"] == "uniform"
    assert resolve_geometry_options({})["objective"] == "mse"
    assert (
        resolve_geometry_options({})["weight_normalization"]
        == "weight_sum"
    )
    options = resolve_geometry_options(
        {
            "geometry": {
                "anchor_frame": "hybrid",
                "fixed_mix": 0.25,
                "reliability": "inverse_angular_dispersion",
                "refresh_interval_epochs": 2,
                "objective": "correlation",
                "weight_normalization": "anchor_count",
            }
        }
    )
    assert options["anchor_frame"] == "hybrid"
    assert options["fixed_mix"] == 0.25
    assert options["reliability"] == "inverse_angular_dispersion"
    assert options["refresh_interval_epochs"] == 2
    assert options["objective"] == "correlation"
    assert options["weight_normalization"] == "anchor_count"
    with pytest.raises(ValueError, match="anchor_frame"):
        resolve_geometry_options({"geometry": {"anchor_frame": "invalid"}})
    with pytest.raises(ValueError, match="objective"):
        resolve_geometry_options({"geometry": {"objective": "invalid"}})
    with pytest.raises(ValueError, match="weight_normalization"):
        resolve_geometry_options(
            {"geometry": {"weight_normalization": "invalid"}}
        )


def test_geometry_component_is_weighted_and_replay_only() -> None:
    class SquaredDifference(torch.nn.Module):
        def forward(
            self, current: torch.Tensor, reference: torch.Tensor
        ) -> torch.Tensor:
            return (current - reference).square().mean()

    current = torch.tensor(
        [[1.0, 2.0], [4.0, 6.0], [3.0, 5.0]], requires_grad=True
    )
    reference = torch.zeros_like(current)
    replay = torch.tensor([False, True, True])
    value = geometry_preservation_component(
        SquaredDifference(), current, reference, replay, weight=4.0
    )
    expected = 4.0 * current[replay].square().mean()
    assert value is not None
    assert torch.allclose(value, expected)
    assert geometry_preservation_component(
        SquaredDifference(),
        current,
        reference,
        torch.zeros_like(replay),
        weight=4.0,
    ) is None


def test_shared_base_signature_ignores_only_post_base_geometry() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config_tree(
        root
        / "configs"
        / "ablations"
        / "cifar100"
        / "icarl_sacil.yaml"
    )
    variant = copy.deepcopy(base)
    variant["method"].update(
        geometry_mode="flat",
        lambda_geo=64.0,
        hierarchy={"temperature": 0.05},
        conflict={"max_neighbors": 2},
        geometry={"anchor_frame": "co_moving"},
    )
    assert base_recipe_signature(variant) == base_recipe_signature(base)

    incompatible = copy.deepcopy(base)
    incompatible["training"]["base"]["epochs"] += 1
    assert base_recipe_signature(incompatible) != base_recipe_signature(base)

    incompatible = copy.deepcopy(base)
    incompatible["seed"] += 1
    assert base_recipe_signature(incompatible) != base_recipe_signature(base)


def test_icarl_casper_config_matches_icarl_except_for_incremental_regularizer() -> None:
    root = Path(__file__).resolve().parents[1]
    control = load_config_tree(
        root
        / "configs"
        / "table1"
        / "cifar100"
        / "icarl_nme_b50_inc5_resnet32.yaml"
    )
    casper = load_config_tree(
        root
        / "configs"
        / "table1"
        / "cifar100"
        / "icarl_casper_nme_b50_inc5_resnet32.yaml"
    )
    for section in ("data", "model", "memory", "evaluation", "training"):
        assert casper[section] == control[section]
    assert casper["method"]["name"] == "icarl"
    assert resolve_geometry_mode("icarl", casper["method"]) == "none"
    assert resolve_casper_options("icarl", casper["method"])["enabled"]
    assert base_recipe_signature(casper) == base_recipe_signature(control)


@pytest.mark.parametrize(
    ("filename", "frame", "reliability"),
    [
        ("icarl_sacil_fixed_lambda_16.yaml", "fixed", "uniform"),
        (
            "icarl_sacil_variance_fixed_lambda_16.yaml",
            "fixed",
            "inverse_angular_dispersion",
        ),
        ("icarl_sacil_co_moving_lambda_16.yaml", "co_moving", "uniform"),
        ("icarl_sacil_hybrid_lambda_16.yaml", "hybrid", "uniform"),
    ],
)
def test_anchor_frame_ablation_configs(
    filename: str, frame: str, reliability: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config_tree(
        root / "configs" / "ablations" / "cifar100" / filename
    )
    options = resolve_geometry_options(config["method"])
    assert config["method"]["geometry_mode"] == "sacil"
    assert config["method"]["lambda_geo"] == 16.0
    assert options["anchor_frame"] == frame
    assert options["reliability"] == reliability

def test_standalone_runner_import_graph_does_not_reference_pycil() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "scripts" / "train_table1.py",
        root / "src" / "sacil" / "engine" / "table1_trainer.py",
    )
    imported: list[str] = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
    assert not any("pycil" in module.lower() for module in imported)


def test_unified_reference_paths_are_metadata_only() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "scripts" / "train_table1.py",
        root / "src" / "sacil" / "engine" / "table1_trainer.py",
    )
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        assert "subprocess" not in names
        assert "importlib" not in names
    contract = unified_method_contract("create")
    assert contract.reference_only is True
    assert contract.implementation_module == "sacil.methods.create"
    assert contract.evaluation_classifier == "native_reconstruction_error"


def test_pycil_finetune_uses_only_the_new_head_slice() -> None:
    logits = torch.tensor(
        [[100.0, 90.0, 1.0, 2.0], [80.0, 70.0, 3.0, 1.0]],
        requires_grad=True,
    )
    targets = torch.tensor([3, 2])
    expected = F.cross_entropy(logits[:, 2:], targets - 2)
    actual = pycil_finetune_loss(logits, targets, known_classes=2)
    assert torch.allclose(actual, expected)


def test_unified_icarl_matches_pycil_softmax_kd_without_t_squared() -> None:
    current = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    reference = torch.tensor([[1.0, -1.0], [1.5, 0.5]])
    temperature = 2.0
    expected = -(
        F.softmax(reference / temperature, dim=1)
        * F.log_softmax(current / temperature, dim=1)
    ).sum() / current.shape[0]
    actual = pycil_icarl_kd_loss(
        current, reference, temperature=temperature
    )
    assert torch.allclose(actual, expected)


def test_standalone_runner_refuses_to_append_to_existing_run(tmp_path) -> None:
    trainer = StandaloneTable1Trainer.__new__(StandaloneTable1Trainer)
    trainer.run_dir = tmp_path
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_dir.mkdir()
    (tmp_path / "sessions.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new --run-name"):
        trainer.run()


def test_recursive_config_extends_preserves_method_and_debug_overrides(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    method = tmp_path / "method.yaml"
    validation = tmp_path / "validation.yaml"
    base.write_text("training:\n  base:\n    epochs: 200\n", encoding="utf-8")
    method.write_text(
        "extends: base.yaml\nmethod:\n  name: replay\n",
        encoding="utf-8",
    )
    validation.write_text(
        "extends: method.yaml\n"
        "training:\n  base:\n    epochs: 2\n"
        "debug:\n  max_sessions: 2\n",
        encoding="utf-8",
    )
    config = load_config_tree(validation)
    assert config["method"]["name"] == "replay"
    assert config["training"]["base"]["epochs"] == 2
    assert config["debug"]["max_sessions"] == 2


@pytest.mark.parametrize(
    ("substrate", "variant", "expected_mode", "expected_lambda"),
    [
        ("icarl", "control", "none", 0.0),
        ("icarl", "global_hap", "global", 1.0),
        ("icarl", "flat_lrhap", "flat", 1.0),
        ("icarl", "sacil", "sacil", 1.0),
        ("afc", "control", "none", 0.0),
        ("afc", "global_hap", "global", 1.0),
        ("afc", "flat_lrhap", "flat", 1.0),
        ("afc", "sacil", "sacil", 1.0),
    ],
)
def test_geometry_ablation_configs_preserve_substrate_recipe(
    substrate: str,
    variant: str,
    expected_mode: str,
    expected_lambda: float,
) -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config_tree(
        root
        / "configs"
        / "table1"
        / "cifar100"
        / f"{substrate}_nme_b50_inc5_resnet32.yaml"
    )
    ablation = load_config_tree(
        root
        / "configs"
        / "ablations"
        / "cifar100"
        / f"{substrate}_{variant}.yaml"
    )
    for section in ("data", "model", "memory", "evaluation", "training"):
        assert ablation[section] == base[section]
    assert ablation["method"]["name"] == substrate
    assert resolve_geometry_mode(substrate, ablation["method"]) == expected_mode
    assert ablation["method"]["lambda_geo"] == expected_lambda
    native_key = "afc" if substrate == "afc" else "kd_temperature"
    assert ablation["method"][native_key] == base["method"][native_key]
