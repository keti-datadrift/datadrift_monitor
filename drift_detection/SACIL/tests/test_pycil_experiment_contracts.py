from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sacil.pycil.runtime import (
    _configure_seed_policy,
    _validate_implementation_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "configs" / "pycil" / "official" / "cifar100"
CONTROLLED_ROOT = (
    PROJECT_ROOT / "configs" / "pycil" / "controlled" / "cifar100"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "model_name", ["finetune", "replay", "icarl", "podnet"]
)
def test_official_configs_use_stock_pycil_names_only(model_name: str):
    config = _load(OFFICIAL_ROOT / f"{model_name}_b50_inc5_resnet32.json")
    assert config["implementation_source"] == "official_pycil"
    assert config["model_name"] == model_name
    assert config["init_cls"] == 50
    assert config["increment"] == 5
    assert config["convnet_type"] == "resnet32"
    assert config["memory_per_class"] == 20
    assert config["seed_policy"] == "config"
    assert config["class_order_path"].endswith(
        "cifar100_b50_t10_afc_order1.json"
    )
    # Algorithmic recipes come from the upstream learner module, not a local
    # config that could silently diverge from the reference implementation.
    assert "epochs" not in config
    assert "lr" not in config
    assert "batch_size" not in config


def test_controlled_configs_differ_only_in_geometry_contract():
    names = (
        "prototype_control_nme_b50_inc5_resnet32.json",
        "global_hap_nme_b50_inc5_resnet32.json",
        "flat_lrhap_nme_b50_inc5_resnet32.json",
        "sacil_nme_b50_inc5_resnet32.json",
    )
    configs = [_load(CONTROLLED_ROOT / name) for name in names]
    for config in configs:
        assert config["implementation_source"] == "sacil_pycil_extension"
        assert config["model_name"] == "sacil"
        assert config["classification_mode"] == "prototype_ce"
        assert config["lambda_kd"] == 0.0

    ignored = {"prefix", "artifact_dir", "geometry_mode", "lambda_geo"}
    reference = {
        key: value for key, value in configs[0].items() if key not in ignored
    }
    for config in configs[1:]:
        comparable = {
            key: value for key, value in config.items() if key not in ignored
        }
        assert comparable == reference

    assert [config["geometry_mode"] for config in configs] == [
        "none",
        "global",
        "flat",
        "sacil",
    ]
    assert [config["lambda_geo"] for config in configs] == [0.0, 1.0, 1.0, 1.0]


def test_implementation_source_rejects_crossed_routes():
    with pytest.raises(ValueError, match="stock PyCIL"):
        _validate_implementation_contract(
            {
                "implementation_source": "official_pycil",
                "model_name": "sacil",
            }
        )
    with pytest.raises(ValueError, match="model_name='sacil'"):
        _validate_implementation_contract(
            {
                "implementation_source": "sacil_pycil_extension",
                "model_name": "finetune",
            }
        )


def test_config_seed_policy_controls_python_numpy_and_torch():
    module = SimpleNamespace(_set_random=lambda: None)
    _configure_seed_policy({"seed_policy": "config", "seed": 17}, module)
    module._set_random()
    actual = (random.random(), np.random.rand(), torch.rand(2))

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
