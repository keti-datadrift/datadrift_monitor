from __future__ import annotations

from pathlib import Path

import pytest

from sacil.config import deep_update, load_config, load_config_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "table1" / "cifar100"
VALIDATION_ROOT = (
    PROJECT_ROOT / "configs" / "validation" / "two_session_25spc"
)
CMPT_COMMON_ROOT = PROJECT_ROOT / "configs" / "cmpt" / "common_recipe"


def _resolved(name: str) -> dict:
    path = CONFIG_ROOT / name
    child = load_config(path)
    parent = child.pop("extends", None)
    if parent is not None:
        child = deep_update(load_config(path.parent / parent), child)
    return child


@pytest.mark.parametrize(
    ("name", "epochs", "milestones"),
    [
        ("finetune_nme_b50_inc5_resnet32.yaml", 80, [40, 70]),
        ("replay_nme_b50_inc5_resnet32.yaml", 70, [30, 50]),
    ],
)
def test_control_baselines_use_their_reference_incremental_recipe(
    name: str, epochs: int, milestones: list[int]
):
    config = _resolved(name)
    incremental = config["training"]["incremental"]
    assert incremental == {
        "epochs": epochs,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 2e-4,
        "nesterov": False,
        "scheduler": "multistep",
        "milestones": milestones,
        "lr_decay": 0.1,
    }


def test_icarl_uses_pycil_optimization_recipe():
    config = _resolved("icarl_nme_b50_inc5_resnet32.yaml")
    assert config["training"]["batch_size"] == 128
    base = config["training"]["base"]
    assert base["epochs"] == 200
    assert base["lr"] == 0.1
    assert base["weight_decay"] == 5e-4
    assert base["milestones"] == [60, 120, 170]
    incremental = config["training"]["incremental"]
    assert incremental["epochs"] == 170
    assert incremental["lr"] == 0.1
    assert incremental["weight_decay"] == 2e-4
    assert incremental["milestones"] == [80, 120]


def test_casper_uses_paper_topology_hyperparameters():
    config = _resolved("casper_nme_b50_inc5_resnet32.yaml")
    casper = config["method"]["casper"]
    assert casper["weight"] == 0.001
    assert casper["knn"] == 8
    assert casper["classes_per_graph"] == 8
    assert casper["replay_batch_size"] == 64
    assert casper["wd_reg"] == 1e-5


def test_cmpt_common_cscct_keeps_author_recipe_and_shared_protocol():
    config = load_config_tree(CMPT_COMMON_ROOT / "train_cscct.yaml")
    assert config["method"] == {
        "name": "cscct",
        "replay_batching": {"enabled": False},
        "kd_weight": 0.25,
        "kd_temperature": 2.0,
        "csc_weight": 3.0,
        "ct_weight": 1.5,
        "ct_temperature": 2.0,
        "fusion_lr": 1e-8,
    }
    assert config["model"]["backbone"] == "cscct_modified_resnet32"
    assert config["training"]["batch_size"] == 128
    for phase in ("base", "incremental"):
        recipe = config["training"][phase]
        assert recipe["epochs"] == 160
        assert recipe["lr"] == 0.1
        assert recipe["momentum"] == 0.9
        assert recipe["weight_decay"] == 5e-4
        assert recipe["milestones"] == [80, 120]
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["memory"]["selection"] == "icarl_herding"
    assert config["data"]["protocol"].endswith(
        "cifar100_b50_t10_afc_order1.json"
    )


def test_create_uses_author_cosine_floor():
    config = _resolved("create_native_b50_inc5_resnet32.yaml")
    assert config["training"]["base"].get("eta_min", 0.0) == 0.0
    assert config["training"]["incremental"]["eta_min"] == 1e-8


@pytest.mark.parametrize(
    "name",
    [path.name for path in CONFIG_ROOT.glob("*.yaml") if not path.name.startswith("_")],
)
def test_table1_methods_share_protocol_memory_and_augmentation(name: str):
    config = _resolved(name)
    assert config["data"]["protocol"].endswith(
        "cifar100_b50_t10_afc_order1.json"
    )
    assert config["data"]["color_jitter"] is True
    assert config["memory"]["mode"] == "per_class"
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["memory"]["selection"] == "icarl_herding"
    assert "resnet18" not in config["model"]["backbone"]
    assert "32" in config["model"]["backbone"]
    expected_evaluator = (
        "native" if config["method"]["name"] == "create" else "nme"
    )
    assert config["evaluation"]["classifier"] == expected_evaluator
    assert config["comparison_contract"] == {
        "name": "annotation_1_cifar100_b50_inc5",
        "implementation": "in_repo_unified",
        "reference_code_policy": "reference_only",
    }


@pytest.mark.parametrize(
    "path",
    sorted(VALIDATION_ROOT.glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_two_session_validation_configs_only_shrink_data_and_epochs(
    path: Path,
) -> None:
    config = load_config_tree(path)
    assert config["debug"] == {
        "train_samples_per_class": 25,
        "max_sessions": 2,
    }
    assert config["training"]["base"]["epochs"] == 2
    assert config["training"]["incremental"]["epochs"] == 2
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["data"]["protocol"].endswith(
        "cifar100_b50_t10_afc_order1.json"
    )
