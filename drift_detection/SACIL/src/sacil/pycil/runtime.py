from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PYCIL_ROOT = Path("ref_codes/00_frameworks/PyCIL")
_REQUIRED_PYCIL_FILES = (
    Path("trainer.py"),
    Path("models/base.py"),
    Path("utils/data_manager.py"),
    Path("utils/inc_net.py"),
)


def _resolved(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def validate_pycil_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    missing = [
        str(relative)
        for relative in _REQUIRED_PYCIL_FILES
        if not (root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{root} is not a PyCIL checkout; missing: {', '.join(missing)}"
        )
    return root


def load_experiment_config(
    path: str | Path, *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_path = _resolved(path, root)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("PyCIL experiment configuration must be a JSON object")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    for key in (
        "data_root",
        "class_order_path",
        "artifact_dir",
        "resume_checkpoint",
    ):
        if key in config:
            config[key] = str(_resolved(config[key], root))
    return config


def _load_class_order(path: str | Path) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("class_order") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("class-order file must contain a non-empty list")
    order = [int(value) for value in values]
    if len(set(order)) != len(order):
        raise ValueError("class order contains duplicate class IDs")
    if sorted(order) != list(range(len(order))):
        raise ValueError("class order must be a permutation of [0, N)")
    return order


def _pycil_dataset_class(data_module, dataset_name: str):
    name = dataset_name.lower()
    mapping = {
        "cifar10": data_module.iCIFAR10,
        "cifar10_aa": data_module.iCIFAR10_AA,
        "cifar100": data_module.iCIFAR100,
        "cifar100_aa": data_module.iCIFAR100_AA,
        "imagenet100": data_module.iImageNet100,
        "imagenet1000": data_module.iImageNet1000,
    }
    if name not in mapping:
        raise ValueError(f"explicit PyCIL dataset setup is unsupported: {name}")
    return mapping[name]


def _configure_class_order(config: dict[str, Any], data_module) -> None:
    order_path = config.get("class_order_path")
    if order_path is None:
        return
    order = _load_class_order(order_path)
    dataset_class = _pycil_dataset_class(data_module, config["dataset"])
    dataset_class.class_order = order
    # PyCIL uses ``idata.class_order`` only when shuffle is disabled.
    config["shuffle"] = False
    config["_explicit_class_order"] = order


def _configure_data_root(config: dict[str, Any], data_module) -> None:
    root_value = config.get("data_root")
    if root_value is None:
        return
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PyCIL data_root does not exist: {root}")

    dataset_name = str(config["dataset"]).lower()
    download = bool(config.get("download", False))
    if dataset_name in {"cifar10", "cifar10_aa"}:
        from torchvision import datasets

        def download_data(self):
            train = datasets.CIFAR10(
                str(root), train=True, download=download
            )
            test = datasets.CIFAR10(
                str(root), train=False, download=download
            )
            self.train_data = train.data
            self.train_targets = np.asarray(train.targets)
            self.test_data = test.data
            self.test_targets = np.asarray(test.targets)

    elif dataset_name in {"cifar100", "cifar100_aa"}:
        from torchvision import datasets

        def download_data(self):
            train = datasets.CIFAR100(
                str(root), train=True, download=download
            )
            test = datasets.CIFAR100(
                str(root), train=False, download=download
            )
            self.train_data = train.data
            self.train_targets = np.asarray(train.targets)
            self.test_data = test.data
            self.test_targets = np.asarray(test.targets)

    elif dataset_name in {"imagenet100", "imagenet1000"}:
        from torchvision import datasets
        from utils.toolkit import split_images_labels

        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet root must contain train/ and val/: {root}"
            )

        def download_data(self):
            train = datasets.ImageFolder(str(train_dir))
            test = datasets.ImageFolder(str(val_dir))
            self.train_data, self.train_targets = split_images_labels(
                train.imgs
            )
            self.test_data, self.test_targets = split_images_labels(test.imgs)

    else:
        raise ValueError(f"data_root patch is unsupported for {dataset_name}")

    dataset_class = _pycil_dataset_class(data_module, dataset_name)
    dataset_class.download_data = download_data


def _install_sacil_factory() -> None:
    factory = importlib.import_module("utils.factory")
    if getattr(factory, "_sacil_factory_installed", False):
        return
    original = factory.get_model

    def get_model(model_name, args):
        normalized = str(model_name).lower()
        if normalized in {"sacil", "pycil_sacil"}:
            learner_module = importlib.import_module(
                "sacil.pycil.learner"
            )
            return learner_module.SACIL(args)

        # Every non-SACIL name is resolved by the unmodified upstream
        # factory.  In particular, do not route stock names such as
        # ``finetune`` or aliases such as ``table1_finetune`` through a local
        # reimplementation.  Method-faithful baselines and controlled SACIL
        # extensions deliberately have separate code paths.
        model = original(model_name, args)

        # Stock PyCIL learners keep the worker count as a module global.
        # Overriding only this operational setting avoids Windows process
        # spawning overhead.  Batch size, optimizer, schedule, losses, and
        # parameter selection remain exactly as defined by upstream PyCIL.
        model_module = importlib.import_module(model.__class__.__module__)
        if "num_workers" in args and hasattr(model_module, "num_workers"):
            setattr(model_module, "num_workers", int(args["num_workers"]))
        return model

    factory.get_model = get_model
    factory._sacil_factory_installed = True
    factory._sacil_original_get_model = original


def _validate_implementation_contract(config: dict[str, Any]) -> None:
    """Reject configs that blur official and project-owned implementations."""

    source = str(config.get("implementation_source", "")).lower()
    model_name = str(config.get("model_name", "")).lower()
    custom = model_name in {"sacil", "pycil_sacil"}
    if source == "official_pycil" and custom:
        raise ValueError(
            "official_pycil configs must use a stock PyCIL model name"
        )
    if source == "sacil_pycil_extension" and not custom:
        raise ValueError(
            "sacil_pycil_extension configs must use model_name='sacil'"
        )
    if source and source not in {
        "official_pycil",
        "sacil_pycil_extension",
    }:
        raise ValueError(f"unknown implementation_source: {source}")


def _configure_base_memory_loader(config: dict[str, Any]) -> None:
    """Apply the experiment worker count to PyCIL's exemplar loaders.

    Upstream PyCIL hard-codes ``num_workers=4`` in BaseLearner's herding,
    class-mean, and exemplar-evaluation loaders. On Windows those short-lived
    loaders repeatedly spawn fresh Python processes. Keep the upstream
    algorithms unchanged while making their worker count an experiment
    setting.
    """

    base_module = importlib.import_module("models.base")
    original = getattr(
        base_module,
        "_sacil_original_data_loader",
        base_module.DataLoader,
    )
    num_workers = int(
        config.get("memory_num_workers", config.get("num_workers", 0))
    )
    if num_workers < 0:
        raise ValueError("memory_num_workers must be non-negative")

    def configured_data_loader(*args, **kwargs):
        kwargs["num_workers"] = num_workers
        if num_workers == 0:
            kwargs.pop("persistent_workers", None)
        return original(*args, **kwargs)

    base_module._sacil_original_data_loader = original
    base_module.DataLoader = configured_data_loader


def _configure_task_limit(config: dict[str, Any]) -> None:
    data_manager_module = importlib.import_module("utils.data_manager")
    original = getattr(
        data_manager_module,
        "_sacil_unlimited_data_manager",
        data_manager_module.DataManager,
    )
    value = config.get("max_tasks")
    if value is None:
        data_manager_module.DataManager = original
        return
    max_tasks = int(value)
    if max_tasks <= 0:
        raise ValueError("max_tasks must be positive")

    class LimitedDataManager(original):
        @property
        def nb_tasks(self):
            return min(max_tasks, super().nb_tasks)

    data_manager_module._sacil_unlimited_data_manager = original
    data_manager_module.DataManager = LimitedDataManager


def _configure_seed_policy(config: dict[str, Any], trainer_module) -> None:
    """Make an opt-in PyCIL run use the seed recorded in its config.

    Upstream PyCIL accepts a seed list but its trainer-level RNG helper always
    seeds torch with the literal value 1.  That makes repeated fixed-order
    runs identical.  The ``config`` policy fixes only experiment seeding; it
    does not alter any learner algorithm or optimization hyperparameter.
    """

    policy = str(config.get("seed_policy", "upstream_fixed_1")).lower()
    if policy == "upstream_fixed_1":
        return
    if policy != "config":
        raise ValueError(f"unknown seed_policy: {policy}")

    def set_random_from_config() -> None:
        seed_value = config.get("seed")
        if isinstance(seed_value, list):
            if len(seed_value) != 1:
                raise ValueError(
                    "seed_policy=config requires one active seed per run"
                )
            seed_value = seed_value[0]
        seed = int(seed_value)
        random.seed(seed)
        np.random.seed(seed)

        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainer_module._set_random = set_random_from_config


def activate_pycil(
    config: dict[str, Any], *, pycil_root: str | Path
) -> dict[str, Any]:
    """Activate an official PyCIL checkout and register the SACIL learner."""

    root = validate_pycil_root(pycil_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # PyCIL uses top-level module names (``utils``, ``models``). Refuse to
    # silently bind to an unrelated package with the same generic name.
    for module_name in ("utils", "models"):
        loaded = sys.modules.get(module_name)
        module_file = getattr(loaded, "__file__", None)
        if module_file is not None:
            origin = Path(module_file).resolve()
            if root not in origin.parents:
                raise RuntimeError(
                    f"top-level module {module_name!r} was already loaded "
                    f"from outside PyCIL: {origin}"
                )

    data_module = importlib.import_module("utils.data")
    resolved = copy.deepcopy(config)
    _validate_implementation_contract(resolved)
    _configure_data_root(resolved, data_module)
    _configure_class_order(resolved, data_module)
    _configure_task_limit(resolved)
    _configure_base_memory_loader(resolved)
    _install_sacil_factory()
    resolved["_pycil_root"] = str(root)
    return resolved


def run_pycil_experiment(
    config: dict[str, Any], *, pycil_root: str | Path
) -> None:
    resolved = activate_pycil(config, pycil_root=pycil_root)
    trainer = importlib.import_module("trainer")
    _configure_seed_policy(resolved, trainer)
    trainer.train(resolved)
