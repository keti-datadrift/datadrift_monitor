from __future__ import annotations

from pathlib import Path

from sacil.config import deep_update, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_casper_uses_paper_hyperparameters_on_the_common_protocol():
    child_path = (
        PROJECT_ROOT
        / "configs"
        / "table1"
        / "cifar100"
        / "casper_nme_b50_inc5_resnet32.yaml"
    )
    child = load_config(child_path)
    parent_path = child_path.parent / child.pop("extends")
    config = deep_update(load_config(parent_path), child)

    assert config["training"]["batch_size"] == 64
    for phase in ("base", "incremental"):
        settings = config["training"][phase]
        assert settings["epochs"] == 20
        assert settings["lr"] == 0.3
        assert settings["momentum"] == 0.0
        assert settings["weight_decay"] == 0.0
        assert settings["milestones"] == []
    casper = config["method"]["casper"]
    assert casper["wd_reg"] == 1e-5
    assert casper["weight"] == 0.001
    assert casper["classes_per_graph"] == 8
    assert casper["knn"] == 8
    assert config["model"]["backbone"] == "resnet32"
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["output"]["run_name"].startswith("casper_paper_hparams_")
