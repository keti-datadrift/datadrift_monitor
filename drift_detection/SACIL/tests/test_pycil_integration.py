from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sacil.pycil.runtime import (
    activate_pycil,
    load_experiment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYCIL_ROOT = PROJECT_ROOT / "ref_codes" / "00_frameworks" / "PyCIL"


class _TensorTripletDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            index,
            torch.from_numpy(self.data[index]),
            int(self.targets[index]),
        )


class _TinyDataManager:
    def __init__(self) -> None:
        generator = np.random.default_rng(7)
        self.data = generator.normal(
            size=(16, 3, 32, 32)
        ).astype(np.float32)
        self.targets = np.repeat(np.arange(4), 4)
        self.nb_tasks = 2

    @staticmethod
    def get_task_size(task: int) -> int:
        if task not in (0, 1):
            raise IndexError(task)
        return 2

    def get_dataset(
        self,
        indices,
        source,
        mode,
        appendent=None,
        ret_data=False,
        **_,
    ):
        del source, mode
        pieces_data = []
        pieces_targets = []
        for class_id in indices:
            mask = self.targets == int(class_id)
            pieces_data.append(self.data[mask])
            pieces_targets.append(self.targets[mask])
        if appendent is not None and len(appendent) != 0:
            pieces_data.append(np.asarray(appendent[0]))
            pieces_targets.append(np.asarray(appendent[1]))
        if not pieces_data:
            raise ValueError("tiny dataset request is empty")
        data = np.concatenate(pieces_data)
        targets = np.concatenate(pieces_targets)
        dataset = _TensorTripletDataset(data, targets)
        if ret_data:
            return data, targets, dataset
        return dataset


@pytest.mark.skipif(
    not PYCIL_ROOT.is_dir(), reason="official PyCIL checkout is unavailable"
)
def test_pycil_runtime_registers_sacil_and_explicit_order(tmp_path):
    config = load_experiment_config(
        "configs/pycil/cifar100/sacil_b50_inc5.json",
        project_root=PROJECT_ROOT,
    )
    config["artifact_dir"] = str(tmp_path)
    resolved = activate_pycil(config, pycil_root=PYCIL_ROOT)

    from utils import factory

    args = dict(resolved)
    args.pop("resume_checkpoint", None)
    args["device"] = [torch.device("cpu")]
    learner = factory.get_model("sacil", args)
    assert type(learner).__name__ == "SACIL"
    assert learner.feature_dim == 64
    assert resolved["shuffle"] is False
    assert resolved["_explicit_class_order"][:5] == [87, 0, 52, 58, 44]

    from models import base

    dataset = _TensorTripletDataset(
        np.zeros((2, 3, 32, 32), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
    )
    upstream_loader = base.DataLoader(
        dataset, batch_size=2, num_workers=4
    )
    ordinary_loader = DataLoader(
        dataset, batch_size=2, num_workers=4
    )
    assert upstream_loader.num_workers == 0
    assert ordinary_loader.num_workers == 4


@pytest.mark.skipif(
    not PYCIL_ROOT.is_dir(), reason="official PyCIL checkout is unavailable"
)
def test_pycil_runtime_preserves_stock_baseline_classes():
    activate_pycil(
        {
            "dataset": "cifar100",
            "shuffle": False,
            "num_workers": 0,
            # A stock learner must ignore this project-side batch-size value.
            "batch_size": 7,
        },
        pycil_root=PYCIL_ROOT,
    )
    from utils import factory

    args = {
        "memory_size": 20,
        "memory_per_class": 1,
        "fixed_memory": True,
        "device": [torch.device("cpu")],
        "convnet_type": "resnet32",
        "num_workers": 0,
    }
    expected = {
        "finetune": ("models.finetune", "Finetune"),
        "replay": ("models.replay", "Replay"),
        "icarl": ("models.icarl", "iCaRL"),
        "podnet": ("models.podnet", "PODNet"),
    }
    for model_name, (module_name, class_name) in expected.items():
        learner = factory.get_model(model_name, dict(args))
        assert learner.__class__.__module__ == module_name
        assert learner.__class__.__name__ == class_name
        assert learner.feature_dim == 64

    import models.finetune as upstream_finetune

    assert upstream_finetune.num_workers == 0
    assert upstream_finetune.batch_size == 128
    with pytest.raises(AssertionError):
        factory.get_model("table1_finetune", dict(args))


@pytest.mark.skipif(
    not PYCIL_ROOT.is_dir(), reason="official PyCIL checkout is unavailable"
)
def test_pycil_sacil_two_task_smoke(tmp_path, monkeypatch):
    activate_pycil(
        {
            "dataset": "cifar100",
            "shuffle": False,
        },
        pycil_root=PYCIL_ROOT,
    )
    import sacil.pycil.learner as learner_module

    from sacil.anchors import compute_prototypes
    from utils import factory

    observed_reference_counts = []
    original_soft_confusion = learner_module.cosine_soft_confusion

    def prototype_reference_spy(
        features, targets, class_references, temperature
    ):
        expected = compute_prototypes(
            features,
            targets,
            range(class_references.shape[0]),
        )
        assert torch.allclose(class_references, expected)
        observed_reference_counts.append(class_references.shape[0])
        return original_soft_confusion(
            features,
            targets,
            class_references,
            temperature=temperature,
        )

    monkeypatch.setattr(
        learner_module,
        "cosine_soft_confusion",
        prototype_reference_spy,
    )

    args = {
        "memory_size": 4,
        "memory_per_class": 1,
        "fixed_memory": True,
        "device": [torch.device("cpu")],
        "convnet_type": "resnet32",
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "init_epochs": 1,
        "epochs": 1,
        "init_milestones": [],
        "milestones": [],
        "eval_interval": 0,
        "disable_tqdm": True,
        "save_checkpoints": False,
        "artifact_dir": str(tmp_path),
        "classification_mode": "prototype_ce",
        "prototype_temperature": 0.1,
        "lambda_kd": 0.0,
        "lambda_geo": 1.0,
    }
    learner = factory.get_model("sacil", args)
    learner.topk = 2
    manager = _TinyDataManager()

    learner.incremental_train(manager)
    learner.after_task()
    learner.incremental_train(manager)

    assert learner._total_classes == 4
    assert learner.exemplar_size == 4
    assert learner._geometry_loss is not None
    assert learner._conflict_weights is not None
    assert learner._artifacts.tree.num_leaves == 4
    assert learner._training_prototypes is not None
    assert observed_reference_counts == [2, 4]
    assert all(
        not parameter.requires_grad
        for parameter in learner._network.fc.parameters()
    )
    cnn, nme = learner.eval_task()
    assert nme is not None
    assert cnn["top1"] == nme["top1"]
    assert (tmp_path / "tree_task_00.json").is_file()
    assert (tmp_path / "tree_task_01.json").is_file()


@pytest.mark.skipif(
    not PYCIL_ROOT.is_dir(), reason="official PyCIL checkout is unavailable"
)
def test_pycil_posthoc_rng_guard_restores_all_cpu_states():
    activate_pycil(
        {"dataset": "cifar100", "shuffle": False},
        pycil_root=PYCIL_ROOT,
    )
    import random

    from sacil.pycil.learner import preserve_rng_state

    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    expected = (random.random(), np.random.rand(), torch.rand(1))

    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    with preserve_rng_state():
        random.random()
        np.random.rand()
        torch.rand(7)
    actual = (random.random(), np.random.rand(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(
    not PYCIL_ROOT.is_dir(), reason="official PyCIL checkout is unavailable"
)
def test_pycil_shared_base_resume_skips_training_and_restores_rng(tmp_path):
    activate_pycil(
        {"dataset": "cifar100", "shuffle": False},
        pycil_root=PYCIL_ROOT,
    )
    from utils import factory

    manager = _TinyDataManager()
    common = {
        "memory_size": 4,
        "memory_per_class": 1,
        "fixed_memory": True,
        "device": [torch.device("cpu")],
        "convnet_type": "resnet32",
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "init_epochs": 1,
        "epochs": 1,
        "init_milestones": [],
        "milestones": [],
        "eval_interval": 0,
        "disable_tqdm": True,
        "lambda_kd": 1.0,
        "lambda_geo": 1.0,
    }
    base_args = {
        **common,
        "save_checkpoints": True,
        "artifact_dir": str(tmp_path / "base"),
    }
    base = factory.get_model("sacil", base_args)
    base.topk = 2
    base.incremental_train(manager)
    base.eval_task()
    base.after_task()
    checkpoint = tmp_path / "base" / "task_00.pt"
    assert checkpoint.is_file()

    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(3)
    base_state = {
        name: value.detach().clone()
        for name, value in base._network.state_dict().items()
    }
    base_memory = base._data_memory.copy()

    resume_args = {
        **common,
        "save_checkpoints": False,
        "artifact_dir": str(tmp_path / "branch"),
        "resume_checkpoint": str(checkpoint),
    }
    resumed = factory.get_model("sacil", resume_args)
    resumed.topk = 2
    resumed.incremental_train(manager)

    assert resumed._cur_task == 0
    assert resumed._known_classes == 0
    assert resumed._total_classes == 2
    assert np.array_equal(resumed._data_memory, base_memory)
    for name, value in resumed._network.state_dict().items():
        assert torch.equal(value, base_state[name])

    resumed.eval_task()
    resumed.after_task()
    assert resumed._known_classes == 2
    assert resumed._old_network is not None
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)

    resumed.incremental_train(manager)
    assert resumed._cur_task == 1
    assert resumed._total_classes == 4
