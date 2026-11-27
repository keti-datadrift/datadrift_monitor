from __future__ import annotations

from pathlib import Path

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import resolve_replay_batching_options
from sacil.methods import validate_annotation1_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "cmpt" / "common_recipe"


def _training_config(name: str) -> dict:
    return load_config_tree(CONFIG_ROOT / name)


def test_lucir_common_recipe_remains_fixed_ratio_control() -> None:
    config = _training_config("train_lucir.yaml")
    assert config["training"]["batch_size"] == 128
    assert config["training"]["incremental"]["lr"] == 0.1
    replay = resolve_replay_batching_options(
        config["method"]["name"], config["method"]
    )
    assert replay["enabled"] is True
    assert replay["replay_fraction"] == 0.5


def test_fgp_and_casper_share_topology_recipe_except_native_lr() -> None:
    fgp = _training_config("train_fgp_icl.yaml")
    casper = _training_config("train_casper_il.yaml")
    assert fgp["training"]["batch_size"] == 128
    assert casper["training"]["batch_size"] == 128
    for phase in ("base", "incremental"):
        for key in (
            "epochs",
            "momentum",
            "nesterov",
            "scheduler",
            "milestones",
            "lr_decay",
        ):
            assert fgp["training"][phase][key] == casper["training"][phase][key]
        assert fgp["training"][phase]["lr"] == 2.0
        assert casper["training"][phase]["lr"] == 0.3
    for config in (fgp, casper):
        replay = resolve_replay_batching_options(
            config["method"]["name"], config["method"]
        )
        assert replay["enabled"] is False


def test_casper_uses_author_icarl_loss_and_spectral_options() -> None:
    config = _training_config("train_casper_il.yaml")
    assert config["method"]["name"] == "casper"
    assert config["method"]["casper"] == {
        "weight": 0.001,
        "knn": 8,
        "classes_per_graph": 8,
        "replay_batch_size": 128,
        "solver": "xitorch",
        "wd_reg": 0.00001,
    }
    assert config["training"]["incremental"]["weight_decay"] == 0.0


def test_lucir_common_is_distinct_from_legacy_bridge() -> None:
    lucir = _training_config("train_lucir.yaml")
    legacy = _training_config("train_lucir_like_legacy.yaml")
    assert lucir["method"]["feature_cosine_distillation"]["margin_ranking"][
        "enabled"
    ] is True
    assert legacy["method"]["feature_cosine_distillation"]["margin_ranking"][
        "enabled"
    ] is False


def test_common_recipe_evaluators_target_expected_trajectories() -> None:
    expected = {
        "evaluate_lucir.yaml": ("icarl", False),
        "evaluate_fgp_icl.yaml": ("fgp", False),
        "evaluate_casper_il.yaml": ("casper", True),
    }
    for filename, (method, casper) in expected.items():
        config = load_config_tree(CONFIG_ROOT / filename)
        assert config["experiment"]["expected_checkpoint_method"] == method
        assert config["experiment"].get("expected_casper_enabled", False) is casper
        assert config["experiment"]["expected_sessions"] == 11
        assert config["cmpt"]["query_horizontal_flip"] is False


def test_affine_evaluators_reuse_the_same_frozen_trajectories() -> None:
    pairs = (
        ("evaluate_lucir.yaml", "evaluate_lucir_affine.yaml"),
        ("evaluate_fgp_icl.yaml", "evaluate_fgp_icl_affine.yaml"),
        ("evaluate_casper_il.yaml", "evaluate_casper_il_affine.yaml"),
    )
    for rigid_name, affine_name in pairs:
        rigid = load_config_tree(CONFIG_ROOT / rigid_name)
        affine = load_config_tree(CONFIG_ROOT / affine_name)
        assert rigid["experiment"]["checkpoints"] == affine["experiment"][
            "checkpoints"
        ]
        assert affine["cmpt"]["transport"] == "affine_ridge"
        assert affine["cmpt"]["affine_ridge"] == 0.01


def test_five_baselines_use_shared_cifar_contract_and_native_schedules() -> None:
    expected = {
        "icarl": (200, 170, "multistep"),
        "replay": (200, 70, "multistep"),
        "podnet": (160, 160, "cosine"),
        "afc": (160, 160, "cosine"),
        "cscct": (160, 160, "multistep"),
    }
    for method, (base_epochs, incremental_epochs, scheduler) in expected.items():
        config = _training_config(f"train_{method}.yaml")
        validate_annotation1_config(config)
        assert config["method"]["name"] == method
        assert config["training"]["base"]["epochs"] == base_epochs
        assert config["training"]["incremental"]["epochs"] == incremental_epochs
        assert config["training"]["incremental"]["scheduler"] == scheduler
        replay = resolve_replay_batching_options(method, config["method"])
        assert replay["enabled"] is False


def test_five_baseline_evaluators_pair_rigid_and_affine_checkpoints() -> None:
    for method in ("icarl", "replay", "podnet", "afc", "cscct"):
        rigid = load_config_tree(CONFIG_ROOT / f"evaluate_{method}.yaml")
        affine = load_config_tree(
            CONFIG_ROOT / f"evaluate_{method}_affine.yaml"
        )
        assert rigid["experiment"]["expected_checkpoint_method"] == method
        assert rigid["experiment"]["checkpoints"] == affine["experiment"][
            "checkpoints"
        ]
        assert affine["cmpt"]["transport"] == "affine_ridge"
