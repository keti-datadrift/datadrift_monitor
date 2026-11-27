from __future__ import annotations

import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank, compute_prototypes
from sacil.config import get_required
from sacil.data import ClassOrderProtocol, build_data_module
from sacil.engine.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from sacil.engine.evaluator import (
    compute_nme_class_means,
    evaluate,
    evaluate_nme,
)
from sacil.features import collect_features
from sacil.hierarchy import (
    GriffinPeronaGreedy,
    HierarchyTree,
    cosine_soft_confusion,
    symmetric_affinity,
)
from sacil.memory import ExemplarMemory, herding_select
from sacil.methods import (
    AnchorGeometryLoss,
    afc_nca_loss,
    afc_pod_loss,
    casper_spectral_loss,
    compute_conflict_weights,
    controlled_transfer_loss,
    create_classification_loss,
    create_contrastive_loss,
    create_kd_loss,
    cross_space_clustering_loss,
    fgp_graph_preservation_loss,
    global_preservation_weights,
    icarl_bce_loss,
    inverse_angular_dispersion_reliability,
    old_logit_kl_loss,
    parameter_l2_regularization,
    pod_flat_loss,
    pod_spatial_loss,
    podnet_nca_loss,
    prototype_cross_entropy,
    pycil_finetune_loss,
    pycil_icarl_kd_loss,
    replay_cross_entropy,
    reconstruction_confidence_weights,
    scheduled_afc_factor,
    scheduled_fgp_weight,
    SUPPORTED_UNIFIED_METHODS,
    unified_method_contract,
    validate_annotation1_config,
    validate_annotation1_protocol,
)
from sacil.metrics import CILMetricsTracker
from sacil.models import (
    AFCIncrementalNet,
    CREATEIncrementalNet,
    CSCCTIncrementalNet,
    ExpandableLinearNet,
    FGPIncrementalNet,
    PyCILPODNet,
    kmeans_imprinted_weights,
)
from sacil.utils import (
    dump_json,
    ensure_dir,
    git_commit,
    make_generator,
    resolved_device,
    seed_worker,
    set_seed,
)


SUPPORTED_TABLE1_METHODS = SUPPORTED_UNIFIED_METHODS
GEOMETRY_MODES = frozenset({"none", "global", "flat", "sacil"})
GEOMETRY_SUBSTRATES = frozenset({"icarl", "afc", "sacil"})
GEOMETRY_RELIABILITY_MODES = frozenset(
    {"uniform", "inverse_angular_dispersion"}
)
GEOMETRY_OBJECTIVES = frozenset({"mse", "correlation", "triplet_rank"})
CASPER_SUBSTRATES = frozenset({"icarl", "casper"})


def resolve_geometry_mode(
    method_name: str, method_config: dict[str, Any]
) -> str:
    """Resolve the geometry ablation without changing the CIL substrate.

    Older pure-SACIL configs exposed two boolean switches.  Explicit
    ``geometry_mode`` takes precedence, while the legacy switches remain a
    backward-compatible spelling for the same Global/Flat/SACIL variants.
    """

    method = str(method_name).lower()
    explicit = method_config.get("geometry_mode")
    if explicit is None:
        if method != "sacil":
            return "none"
        if not bool(method_config.get("local_relaxation", True)):
            return "global"
        if not bool(method_config.get("use_internal_anchors", True)):
            return "flat"
        return "sacil"
    mode = str(explicit).lower().replace("-", "_")
    aliases = {
        "off": "none",
        "global_hap": "global",
        "flat_lrhap": "flat",
        "full": "sacil",
    }
    mode = aliases.get(mode, mode)
    if mode not in GEOMETRY_MODES:
        raise ValueError(
            f"unknown geometry mode {explicit!r}; expected one of "
            f"{sorted(GEOMETRY_MODES)}"
        )
    if mode != "none" and method not in GEOMETRY_SUBSTRATES:
        raise ValueError(
            f"geometry mode {mode!r} is unsupported for substrate {method!r}"
        )
    return mode


def resolve_geometry_options(method_config: dict[str, Any]) -> dict[str, Any]:
    """Validate anchor-frame ablation options with legacy-safe defaults."""

    raw = method_config.get("geometry", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.geometry must be a mapping")
    frame = str(raw.get("anchor_frame", "fixed")).lower().replace("-", "_")
    frame = {"moving": "co_moving", "comoving": "co_moving"}.get(
        frame, frame
    )
    if frame not in {"fixed", "co_moving", "hybrid"}:
        raise ValueError(
            "method.geometry.anchor_frame must be fixed, co_moving, or hybrid"
        )
    reliability = str(raw.get("reliability", "uniform")).lower().replace(
        "-", "_"
    )
    reliability = {
        "none": "uniform",
        "inverse_dispersion": "inverse_angular_dispersion",
    }.get(reliability, reliability)
    if reliability not in GEOMETRY_RELIABILITY_MODES:
        raise ValueError(
            "method.geometry.reliability must be uniform or "
            "inverse_angular_dispersion"
        )
    requested_fixed_mix = float(raw.get("fixed_mix", 0.5))
    refresh = int(raw.get("refresh_interval_epochs", 1))
    power = float(raw.get("reliability_power", 1.0))
    epsilon = float(raw.get("reliability_epsilon", 1e-4))
    minimum = float(raw.get("reliability_min_weight", 0.25))
    maximum = float(raw.get("reliability_max_weight", 4.0))
    objective = str(raw.get("objective", "mse")).lower().replace("-", "_")
    objective = {
        "pearson": "correlation",
        "rank": "triplet_rank",
        "triplet": "triplet_rank",
    }.get(objective, objective)
    weight_normalization = str(
        raw.get("weight_normalization", "weight_sum")
    ).lower().replace("-", "_")
    weight_normalization = {
        "relative": "weight_sum",
        "absolute": "anchor_count",
    }.get(weight_normalization, weight_normalization)
    triplet_margin_scale = float(raw.get("triplet_margin_scale", 1.0))
    rank_tolerance = float(raw.get("rank_tolerance", 1e-4))
    if not 0.0 <= requested_fixed_mix <= 1.0:
        raise ValueError("method.geometry.fixed_mix must be in [0, 1]")
    if refresh <= 0:
        raise ValueError(
            "method.geometry.refresh_interval_epochs must be positive"
        )
    if power < 0 or epsilon <= 0:
        raise ValueError("invalid geometry reliability power/epsilon")
    if not 0 < minimum <= maximum:
        raise ValueError("invalid geometry reliability weight bounds")
    if objective not in GEOMETRY_OBJECTIVES:
        raise ValueError(
            "method.geometry.objective must be mse, correlation, or "
            "triplet_rank"
        )
    if weight_normalization not in {"weight_sum", "anchor_count"}:
        raise ValueError(
            "method.geometry.weight_normalization must be weight_sum or "
            "anchor_count"
        )
    if not 0.0 < triplet_margin_scale <= 1.0:
        raise ValueError(
            "method.geometry.triplet_margin_scale must be in (0, 1]"
        )
    if rank_tolerance < 0.0:
        raise ValueError(
            "method.geometry.rank_tolerance must be non-negative"
        )
    fixed_mix = (
        1.0
        if frame == "fixed"
        else 0.0
        if frame == "co_moving"
        else requested_fixed_mix
    )
    return {
        "anchor_frame": frame,
        "fixed_mix": fixed_mix,
        "refresh_interval_epochs": refresh,
        "reliability": reliability,
        "reliability_power": power,
        "reliability_epsilon": epsilon,
        "reliability_min_weight": minimum,
        "reliability_max_weight": maximum,
        "objective": objective,
        "weight_normalization": weight_normalization,
        "triplet_margin_scale": triplet_margin_scale,
        "rank_tolerance": rank_tolerance,
    }


def resolve_casper_options(
    method_name: str, method_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the CaSpeR regularizer without replacing its CIL substrate.

    ``method=casper`` retains the author-style standalone adapter.  The
    controlled comparison instead uses ``method=icarl`` with
    ``method.casper.enabled=true`` so the model, CE/KD objective, optimizer,
    epochs, memory, and evaluator remain exactly those of the iCaRL control.
    """

    method = str(method_name).lower()
    raw = method_config.get("casper", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.casper must be a mapping")
    enabled = method == "casper" or bool(raw.get("enabled", False))
    weight = float(raw.get("weight", 0.01))
    knn = int(raw.get("knn", 10))
    classes = int(raw.get("classes_per_graph", 5))
    batch_size = int(raw.get("replay_batch_size", 64))
    solver = str(raw.get("solver", "xitorch")).lower()
    wd_reg = float(raw.get("wd_reg", 1e-5 if method == "casper" else 0.0))

    if enabled and method not in CASPER_SUBSTRATES:
        raise ValueError(
            f"CaSpeR regularization is unsupported for substrate {method!r}"
        )
    if weight < 0 or wd_reg < 0:
        raise ValueError("CaSpeR weights must be non-negative")
    if knn <= 0 or classes <= 0 or batch_size <= 0:
        raise ValueError("CaSpeR k, class count, and batch size must be positive")
    if enabled and batch_size <= max(knn, classes + 1):
        raise ValueError(
            "CaSpeR replay_batch_size must exceed both knn and "
            "classes_per_graph + 1"
        )
    if solver not in {"xitorch", "partial"}:
        raise ValueError("CaSpeR solver must use the author xitorch path")
    if method == "icarl" and wd_reg != 0.0:
        raise ValueError(
            "the controlled iCaRL+CaSpeR plug-in requires wd_reg=0 so "
            "only the spectral term differs from the iCaRL substrate"
        )
    return {
        "enabled": enabled,
        "weight": weight,
        "knn": knn,
        "classes_per_graph": classes,
        "replay_batch_size": batch_size,
        "solver": solver,
        "wd_reg": wd_reg,
    }


def geometry_preservation_component(
    geometry: AnchorGeometryLoss | None,
    current_features: Tensor,
    reference_features: Tensor | None,
    replay_mask: Tensor,
    *,
    weight: float,
) -> Tensor | None:
    """Return the weighted replay-only geometry term, when applicable."""

    if weight < 0:
        raise ValueError("geometry weight must be non-negative")
    if (
        geometry is None
        or reference_features is None
        or not bool(replay_mask.any())
    ):
        return None
    return float(weight) * geometry(
        current_features[replay_mask], reference_features[replay_mask]
    )


def base_recipe_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Return every configuration field that can affect session 0."""

    method = copy.deepcopy(config.get("method", {}))
    for key in (
        "geometry_mode",
        "lambda_geo",
        "hierarchy",
        "conflict",
        "geometry",
        "local_relaxation",
        "use_internal_anchors",
    ):
        method.pop(key, None)
    # In the controlled iCaRL plug-in, CaSpeR is inactive in S0 and therefore
    # must not prevent reuse of the exact iCaRL base checkpoint.  The separate
    # method=casper adapter keeps this field because its explicit L2 term can
    # affect S0.
    if str(method.get("name", "")).lower() == "icarl":
        method.pop("casper", None)
    training = copy.deepcopy(config.get("training", {}))
    training.pop("incremental", None)
    debug = copy.deepcopy(config.get("debug", {}))
    debug.pop("max_sessions", None)
    return {
        "seed": int(config.get("seed", 1)),
        "deterministic": bool(config.get("deterministic", True)),
        "data": copy.deepcopy(config.get("data", {})),
        "model": copy.deepcopy(config.get("model", {})),
        "memory": copy.deepcopy(config.get("memory", {})),
        "evaluation": copy.deepcopy(config.get("evaluation", {})),
        "training": training,
        "method": method,
        "debug": debug,
    }


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Draw a fixed number of classes before drawing replay samples.

    This is the sampling contract used by CaSpeR.  It intentionally differs
    from ordinary class-balanced weighting: every graph contains exactly
    ``classes_per_batch`` old classes.
    """

    def __init__(
        self,
        labels: list[int],
        *,
        batch_size: int,
        classes_per_batch: int,
        batches: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.classes_per_batch = int(classes_per_batch)
        self.batches = int(batches)
        self.seed = int(seed)
        positions: dict[int, list[int]] = defaultdict(list)
        for position, label in enumerate(labels):
            positions[int(label)].append(position)
        self.positions = dict(positions)
        if self.classes_per_batch > len(self.positions):
            raise ValueError("CaSpeR requests more classes than memory contains")
        if self.classes_per_batch <= 0 or self.batch_size <= self.classes_per_batch:
            raise ValueError("invalid CaSpeR balanced batch dimensions")
        self._iteration = 0

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        classes = np.asarray(sorted(self.positions), dtype=np.int64)
        base, remainder = divmod(self.batch_size, self.classes_per_batch)
        for _ in range(self.batches):
            selected = rng.choice(
                classes, size=self.classes_per_batch, replace=False
            )
            batch: list[int] = []
            for offset, class_id in enumerate(selected.tolist()):
                count = base + int(offset < remainder)
                candidates = self.positions[int(class_id)]
                choices = rng.choice(
                    candidates,
                    size=count,
                    # Match CaSpeR Buffer.get_balanced_data: sample each
                    # selected class without replacement whenever possible.
                    replace=count > len(candidates),
                )
                batch.extend(int(value) for value in choices.tolist())
            rng.shuffle(batch)
            yield batch


class UnifiedTable1Trainer:
    """One in-repo Table-1 engine with method-specific adapters.

    Upstream repositories are algorithm references only.  This class has no
    runtime import or subprocess path into ``ref_codes``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        project_root: str | Path,
        *,
        max_sessions: int | None = None,
        base_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.project_root = Path(project_root).resolve()
        self.method = str(get_required(config, "method.name")).lower()
        if self.method not in SUPPORTED_TABLE1_METHODS:
            raise ValueError(f"unsupported unified method: {self.method}")
        validate_annotation1_config(self.config)
        self.geometry_mode = resolve_geometry_mode(
            self.method, self.config["method"]
        )
        self.geometry_options = resolve_geometry_options(
            self.config["method"]
        )
        self.casper_options = resolve_casper_options(
            self.method, self.config["method"]
        )
        self.base_checkpoint_path = (
            None
            if base_checkpoint is None
            else self._project_path(base_checkpoint)
        )
        self.start_session = 0
        self.method_contract = unified_method_contract(self.method)
        self.seed = int(config.get("seed", 1))
        set_seed(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.device = resolved_device(str(config.get("device", "cuda:0")))
        self.protocol = ClassOrderProtocol.from_json(
            self._project_path(get_required(config, "data.protocol"))
        )
        validate_annotation1_protocol(self.protocol)
        self.data = build_data_module(
            str(config["data"].get("name", "cifar100")),
            self._project_path(get_required(config, "data.root")),
            self.protocol,
            download=bool(config["data"].get("download", False)),
            color_jitter=bool(config["data"].get("color_jitter", True)),
        )
        initial_limit = self._memory_limit(self.protocol.session(0).stop)
        self.memory = ExemplarMemory(initial_limit)
        self.metrics = CILMetricsTracker()
        self.model: nn.Module | None = None
        self.class_means: Tensor | None = None
        self.sacil_tree: HierarchyTree | None = None
        self.sacil_prototypes: PrototypeBank | None = None
        self.sacil_anchors: HierarchicalAnchorBank | None = None
        configured_max_sessions = self.config.get("debug", {}).get(
            "max_sessions"
        )
        requested_max_sessions = (
            configured_max_sessions if max_sessions is None else max_sessions
        )
        if requested_max_sessions is None:
            self.max_sessions = self.protocol.num_sessions
        else:
            requested_max_sessions = int(requested_max_sessions)
            if requested_max_sessions <= 0:
                raise ValueError("max_sessions must be positive")
            self.max_sessions = min(
                requested_max_sessions, self.protocol.num_sessions
            )
        output = config["output"]
        self.run_dir = ensure_dir(
            self._project_path(output["directory"])
            / str(output["run_name"])
            / f"seed_{self.seed}"
        )
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        dump_json(
            {
                "config": self.config,
                "framework": "sacil-unified",
                "pycil_used": False,
                "reference_code_executed": False,
                "method_contract": self.method_contract.as_dict(),
                "geometry_mode": self.geometry_mode,
                "geometry_options": self.geometry_options,
                "casper_options": self.casper_options,
                "base_checkpoint": (
                    None
                    if self.base_checkpoint_path is None
                    else str(self.base_checkpoint_path)
                ),
                "protocol_id": self.protocol.protocol_id,
                "device": str(self.device),
                "git_commit": git_commit(self.project_root),
            },
            self.run_dir / "resolved_config.json",
        )

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def _memory_limit(self, seen_classes: int) -> int:
        memory = self.config["memory"]
        mode = str(memory.get("mode", "per_class"))
        if mode == "per_class":
            return int(memory.get("exemplars_per_class", 20))
        if mode == "fixed_total":
            return max(1, int(memory.get("capacity", 2000)) // int(seen_classes))
        raise ValueError(f"unknown memory mode: {mode}")

    @property
    def debug_train_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("train_samples_per_class")
        return None if value is None else int(value)

    @property
    def debug_test_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("test_samples_per_class")
        return None if value is None else int(value)

    def _loader(
        self,
        dataset,
        *,
        shuffle: bool,
        session_id: int,
        batch_size: int | None = None,
        batch_sampler=None,
    ) -> DataLoader:
        training = self.config["training"]
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "num_workers": int(training.get("num_workers", 0)),
            "pin_memory": bool(training.get("pin_memory", True)),
            "worker_init_fn": seed_worker,
            "generator": make_generator(self.seed * 1000 + session_id),
        }
        workers = kwargs["num_workers"]
        if workers > 0:
            kwargs["persistent_workers"] = bool(
                training.get("persistent_workers", False)
            )
        if batch_sampler is not None:
            kwargs["batch_sampler"] = batch_sampler
        else:
            kwargs.update(
                batch_size=int(batch_size or training.get("batch_size", 128)),
                shuffle=bool(shuffle),
            )
        return DataLoader(**kwargs)

    def _new_model(self, num_classes: int) -> nn.Module:
        model = self.config.get("model", {})
        if self.method == "podnet":
            return PyCILPODNet(
                num_classes,
                proxies_per_class=int(model.get("proxies_per_class", 10)),
            )
        if self.method == "afc":
            return AFCIncrementalNet(
                num_classes,
                initial_size=self.protocol.session(0).size,
                increment_size=self.protocol.session(1).size,
                proxies_per_class=int(model.get("proxies_per_class", 10)),
                classifier_scale=float(model.get("nca_scale", 1.0)),
                distance_scale=float(model.get("distance_scale", 3.0)),
            )
        if self.method == "create":
            return CREATEIncrementalNet(
                num_classes,
                hidden_layers=tuple(model.get("hidden_layers", [])),
                latent_features=int(model.get("latent_features", 32)),
                reconstruction_scale=float(model.get("reconstruction_scale", 0.1)),
                backbone=str(model.get("backbone", "resnet32")),
            )
        if self.method == "fgp":
            return FGPIncrementalNet(num_classes)
        if self.method == "cscct":
            return CSCCTIncrementalNet(num_classes)
        return ExpandableLinearNet(
            num_classes, backbone=str(model.get("backbone", "resnet32"))
        )

    def _phase(self, session_id: int) -> dict[str, Any]:
        name = "base" if session_id == 0 else "incremental"
        return dict(self.config["training"][name])

    @staticmethod
    def _scheduler(optimizer: SGD, phase: dict[str, Any], epochs: int):
        kind = str(phase.get("scheduler", "multistep"))
        if kind == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=float(phase.get("eta_min", 0.0)),
            )
        if kind == "multistep":
            return MultiStepLR(
                optimizer,
                milestones=[int(v) for v in phase.get("milestones", [])],
                gamma=float(phase.get("lr_decay", 0.1)),
            )
        raise ValueError(f"unknown scheduler: {kind}")

    def _training_dataset(self, session_id: int, old_indices: list[int]):
        if self.method == "joint":
            return self.data.train_dataset_for_classes(
                self.protocol.seen_classes(session_id),
                augment=True,
                samples_per_class=self.debug_train_samples_per_class,
            )
        if self.method == "finetune":
            return self.data.new_train_dataset(
                session_id,
                augment=True,
                samples_per_class=self.debug_train_samples_per_class,
            )
        return self.data.training_dataset(
            session_id,
            old_indices,
            samples_per_class=self.debug_train_samples_per_class,
        )

    def _prototype_dataset(self, session_id: int, old_indices: list[int]):
        new = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        if not old_indices:
            return new
        return ConcatDataset(
            [new, self.data.replay_dataset(old_indices, augment=False)]
        )

    @torch.inference_mode()
    def _incoming_collection(self, session_id: int):
        if self.model is None:
            raise RuntimeError("model is missing")
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        return collect_features(
            self.model,
            self._loader(dataset, shuffle=False, session_id=session_id + 4000),
            self.device,
        )

    def _expand_model(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("cannot expand a missing model")
        total = self.protocol.session(session_id).stop
        if isinstance(self.model, PyCILPODNet):
            # Stock PyCIL initializes the new proxy chunk through
            # SplitCosineLinear.reset_parameters; PODNet does not use AFC's
            # K-means weight imprinting.
            self.model.expand_classes(total)
        elif isinstance(self.model, AFCIncrementalNet):
            collection = self._incoming_collection(session_id)
            class_features = [
                collection.features[
                    collection.original_targets == int(class_id)
                ]
                for class_id in self.protocol.classes_for_session(session_id)
            ]
            weights = kmeans_imprinted_weights(
                class_features,
                self.model.classifier.weights,
                proxies_per_class=self.model.classifier.proxies_per_class,
                random_state=self.seed * 1000 + session_id * 100,
            )
            self.model.expand_classes(total, weights.to(self.device))
        elif isinstance(self.model, CSCCTIncrementalNet):
            collection = self._incoming_collection(session_id)
            old_norm = self.model.classifier.weight.detach().norm(dim=1).mean()
            values = []
            for class_id in self.protocol.classes_for_session(session_id):
                feature = collection.features[
                    collection.original_targets == int(class_id)
                ].mean(dim=0, keepdim=True)
                values.append(F.normalize(feature, dim=1) * old_norm.cpu())
            self.model.expand_classes(torch.cat(values).to(self.device))
        elif hasattr(self.model, "expand_classes"):
            self.model.expand_classes(total)
        else:
            raise TypeError("model has no expansion contract")

    def _source_base_record(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        if self.base_checkpoint_path is None:
            raise RuntimeError("base checkpoint path is missing")
        session_log = self.base_checkpoint_path.parent.parent / "sessions.jsonl"
        if session_log.exists():
            for line in session_log.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if int(record.get("session_id", -1)) == 0:
                    return record
        embedded = checkpoint.get("session_record")
        if embedded is not None:
            return copy.deepcopy(embedded)
        records = checkpoint.get("metrics", {}).get("records", [])
        if len(records) != 1:
            raise ValueError(
                "base checkpoint has no recoverable session-0 record"
            )
        return {
            "session_id": 0,
            "method": self.method,
            "geometry_mode": self.geometry_mode,
            "seen_class_count": self.protocol.session(0).stop,
            "memory_size": len(self.memory),
            "training": {
                "source": "shared_base_checkpoint",
                "epoch_logs": [],
            },
            "post_training": None,
            "evaluation": copy.deepcopy(records[0]),
        }

    def _validate_base_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("framework") != "sacil-unified":
            raise ValueError("base checkpoint is not from the unified runner")
        if int(checkpoint.get("session_id", -1)) != 0:
            raise ValueError("base checkpoint must be session_00.pt")
        if checkpoint.get("protocol_id") != self.protocol.protocol_id:
            raise ValueError("base checkpoint protocol does not match config")
        if str(checkpoint.get("method", "")).lower() != self.method:
            raise ValueError(
                "base checkpoint substrate does not match target config"
            )
        source_config = checkpoint.get("config")
        if not isinstance(source_config, dict):
            raise ValueError("base checkpoint has no resolved config")
        if base_recipe_signature(source_config) != base_recipe_signature(
            self.config
        ):
            raise ValueError(
                "base checkpoint session-0 recipe differs from target config"
            )
        records = checkpoint.get("metrics", {}).get("records", [])
        if len(records) != 1 or int(records[0].get("session_id", -1)) != 0:
            raise ValueError("base checkpoint must contain exactly one S0 metric")
        class_means = checkpoint.get("class_means")
        expected_classes = self.protocol.session(0).stop
        if (
            not isinstance(class_means, Tensor)
            or class_means.shape[0] != expected_classes
        ):
            raise ValueError("base checkpoint has invalid S0 class means")

    def validate_base_checkpoint(self) -> dict[str, Any] | None:
        """Validate a configured shared base without mutating trainer state."""

        if self.base_checkpoint_path is None:
            return None
        checkpoint = load_checkpoint(self.base_checkpoint_path, map_location="cpu")
        self._validate_base_checkpoint(checkpoint)
        return {
            "path": str(self.base_checkpoint_path),
            "method": checkpoint["method"],
            "session_id": int(checkpoint["session_id"]),
            "seed": int(checkpoint["config"].get("seed", 1)),
        }

    def _bootstrap_from_base_checkpoint(self) -> dict[str, Any]:
        if self.base_checkpoint_path is None:
            raise RuntimeError("base checkpoint path is missing")
        checkpoint = load_checkpoint(self.base_checkpoint_path, map_location="cpu")
        self._validate_base_checkpoint(checkpoint)
        base_classes = self.protocol.session(0).stop
        model = self._new_model(base_classes)
        if type(model).__name__ != str(checkpoint.get("model_type", "")):
            raise ValueError("base checkpoint model type does not match config")
        model.load_state_dict(checkpoint["model"], strict=True)
        self.model = model.to(self.device)
        self.memory = ExemplarMemory.from_state_dict(checkpoint["memory"])
        self.metrics = CILMetricsTracker.from_state_dict(checkpoint["metrics"])
        self.class_means = checkpoint["class_means"].detach().cpu().clone()

        source_hierarchy = (
            checkpoint.get("config", {})
            .get("method", {})
            .get("hierarchy", {})
        )
        target_hierarchy = self.config["method"].get("hierarchy", {})
        has_artifacts = all(
            key in checkpoint for key in ("tree", "prototypes", "anchors")
        )
        if self.geometry_mode == "none":
            self.sacil_tree = None
            self.sacil_prototypes = None
            self.sacil_anchors = None
        elif has_artifacts and source_hierarchy == target_hierarchy:
            self.sacil_tree = HierarchyTree.from_state_dict(checkpoint["tree"])
            self.sacil_prototypes = PrototypeBank.from_state_dict(
                checkpoint["prototypes"]
            )
            self.sacil_anchors = HierarchicalAnchorBank.from_state_dict(
                checkpoint["anchors"]
            )
            dump_json(
                self.sacil_tree.state_dict(),
                self.run_dir / "tree_session_00.json",
            )
        else:
            self._build_sacil_artifacts(0)

        if "rng_state" not in checkpoint:
            raise ValueError("base checkpoint has no RNG state")
        restore_rng_state(checkpoint["rng_state"])
        self.start_session = 1

        record = copy.deepcopy(self._source_base_record(checkpoint))
        record.update(
            method=self.method,
            geometry_mode=self.geometry_mode,
            geometry_options=copy.deepcopy(self.geometry_options),
            casper_options=copy.deepcopy(self.casper_options),
            checkpoint=str(self.checkpoint_dir / "session_00.pt"),
            base_reused=True,
            shared_from_checkpoint=str(self.base_checkpoint_path),
        )
        self._save_checkpoint(0, session_record=record)
        return record

    def run(self) -> dict[str, Any]:
        session_log = self.run_dir / "sessions.jsonl"
        metrics_path = self.run_dir / "metrics.json"
        existing_checkpoints = tuple(self.checkpoint_dir.glob("session_*.pt"))
        if session_log.exists() or metrics_path.exists() or existing_checkpoints:
            raise FileExistsError(
                "the unified run directory already contains training "
                f"artifacts: {self.run_dir}; choose a new --run-name"
            )
        if self.base_checkpoint_path is not None:
            base_record = self._bootstrap_from_base_checkpoint()
            with session_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(base_record, ensure_ascii=False) + "\n")
        for session_id in range(self.start_session, self.max_sessions):
            started = time.time()
            record = self._run_session(session_id)
            record["elapsed_seconds"] = time.time() - started
            with session_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = self.metrics.summary()
        dump_json(
            {"summary": summary, "sessions": self.metrics.records},
            metrics_path,
        )
        return summary

    def _run_session(self, session_id: int) -> dict[str, Any]:
        session = self.protocol.session(session_id)
        old_indices = self.memory.all_indices(self.protocol.class_order)
        if session_id == 0 or self.method == "joint":
            self.model = self._new_model(session.stop).to(self.device)
            teacher = None
        else:
            if self.model is None:
                raise RuntimeError("incremental session has no model")
            teacher = copy.deepcopy(self.model).to(self.device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self._expand_model(session_id)
            self.model.to(self.device)

        geometry = self._prepare_sacil_geometry(session_id, teacher)
        train_dataset = self._training_dataset(session_id, old_indices)
        train_loader = self._loader(
            train_dataset, shuffle=True, session_id=session_id
        )
        prototype_loader = (
            self._loader(
                self._prototype_dataset(session_id, old_indices),
                shuffle=False,
                session_id=session_id + 5000,
            )
            if self.method == "sacil"
            else None
        )
        training = self._train_session(
            session_id,
            train_loader,
            teacher,
            geometry,
            prototype_loader,
        )
        # PODNet, AFC, and CREATE construct a temporary balanced memory for
        # their classifier fine-tuning stage.  Their released lifecycles then
        # rebuild the incoming-class exemplars with the final, fine-tuned
        # representation before evaluation/after_task.
        self._update_memory(session_id)
        post = self._post_training(session_id, train_loader, teacher)
        if session_id > 0 and self.method in {"podnet", "afc", "create"}:
            self._update_memory(session_id)
        self._build_evaluation_state(session_id)
        evaluation = self._evaluate_session(session_id)
        metric = self.metrics.update(session_id, evaluation)
        record = {
            "session_id": session_id,
            "method": self.method,
            "geometry_mode": self.geometry_mode,
            "geometry_options": copy.deepcopy(self.geometry_options),
            "casper_options": copy.deepcopy(self.casper_options),
            "seen_class_count": session.stop,
            "memory_size": len(self.memory),
            "training": training,
            "post_training": post,
            "evaluation": metric,
        }
        checkpoint = self._save_checkpoint(
            session_id, session_record=record
        )
        record["checkpoint"] = str(checkpoint)
        return record

    def _prepare_sacil_geometry(
        self, session_id: int, teacher: nn.Module | None
    ) -> AnchorGeometryLoss | None:
        if self.geometry_mode == "none" or session_id == 0:
            return None
        if (
            teacher is None
            or self.sacil_anchors is None
            or self.sacil_tree is None
        ):
            raise RuntimeError("SACIL incremental session lacks old anchors")
        conflict = self.config["method"].get("conflict", {})
        if self.geometry_mode == "global":
            weights = global_preservation_weights(self.sacil_anchors)
        else:
            collection = self._incoming_collection(session_id)
            incoming = compute_prototypes(
                collection.features,
                collection.original_targets,
                self.protocol.classes_for_session(session_id),
            )
            weights = compute_conflict_weights(
                incoming,
                self.sacil_anchors,
                self.sacil_tree,
                max_neighbors=int(conflict.get("max_neighbors", 5)),
                old_class_ratio=float(conflict.get("old_class_ratio", 0.1)),
                temperature=float(conflict.get("temperature", 0.05)),
                min_preservation_weight=float(
                    conflict.get("min_preservation_weight", 0.1)
                ),
                ancestor_decay=float(conflict.get("ancestor_decay", 0.5)),
            )
        if self.geometry_options["reliability"] == "inverse_angular_dispersion":
            memory_loader = self._memory_loader(session_id, augment=False)
            old_collection = collect_features(teacher, memory_loader, self.device)
            leaf_reliability, internal_reliability = (
                inverse_angular_dispersion_reliability(
                    old_collection.features,
                    old_collection.original_targets,
                    self.sacil_anchors,
                    self.sacil_tree,
                    power=self.geometry_options["reliability_power"],
                    epsilon=self.geometry_options["reliability_epsilon"],
                    min_weight=self.geometry_options[
                        "reliability_min_weight"
                    ],
                    max_weight=self.geometry_options[
                        "reliability_max_weight"
                    ],
                )
            )
            weights.leaf_weights = weights.leaf_weights * leaf_reliability
            weights.internal_weights = (
                weights.internal_weights * internal_reliability
            )
        return AnchorGeometryLoss(
            self.sacil_anchors,
            weights.leaf_weights,
            weights.internal_weights,
            use_internal_anchors=self.geometry_mode != "flat",
            anchor_frame=self.geometry_options["anchor_frame"],
            fixed_mix=self.geometry_options["fixed_mix"],
            objective=self.geometry_options["objective"],
            weight_normalization=self.geometry_options[
                "weight_normalization"
            ],
            triplet_margin_scale=self.geometry_options[
                "triplet_margin_scale"
            ],
            rank_tolerance=self.geometry_options["rank_tolerance"],
        ).to(self.device)

    def _refresh_geometry_current_anchors(
        self,
        session_id: int,
        geometry: AnchorGeometryLoss,
    ) -> None:
        """Rebuild current anchors on old memory without changing the tree."""

        if not geometry.requires_current_anchor_refresh:
            return
        if (
            self.model is None
            or self.sacil_tree is None
            or self.sacil_anchors is None
        ):
            raise RuntimeError("current-anchor refresh lacks SACIL state")
        collection = collect_features(
            self.model,
            self._memory_loader(session_id, augment=False),
            self.device,
        )
        class_ids = self.sacil_anchors.leaf_class_ids
        prototypes = PrototypeBank(
            class_ids,
            compute_prototypes(
                collection.features,
                collection.original_targets,
                class_ids,
            ),
        )
        current_anchors = HierarchicalAnchorBank.from_tree(
            prototypes, self.sacil_tree
        )
        geometry.update_current_anchors(current_anchors)

    def _train_session(
        self,
        session_id: int,
        loader: DataLoader,
        teacher: nn.Module | None,
        geometry: AnchorGeometryLoss | None,
        prototype_loader: DataLoader | None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("training has no model")
        phase = self._phase(session_id)
        parameters = self._main_parameters()
        if self.method == "afc":
            # AFC author code clips every trainable parameter gradient through
            # per-parameter hooks registered at the beginning of each task.
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda gradient: torch.clamp(gradient, -5.0, 5.0)
                    )
        optimizer = SGD(
            parameters,
            lr=float(phase["lr"]),
            momentum=float(phase.get("momentum", 0.9)),
            weight_decay=float(phase.get("weight_decay", 5e-4)),
            nesterov=bool(phase.get("nesterov", False)),
        )
        epochs = int(phase["epochs"])
        scheduler = self._scheduler(optimizer, phase, epochs)
        fusion_loader = self._cscct_balanced_loader(session_id)
        fusion_optimizer = (
            SGD(
                list(self.model.fusion.parameters()),
                lr=float(self.config["method"].get("fusion_lr", 1e-8)),
                momentum=float(phase.get("momentum", 0.9)),
                weight_decay=float(phase.get("weight_decay", 5e-4)),
            )
            if isinstance(self.model, CSCCTIncrementalNet)
            and fusion_loader is not None
            else None
        )
        fusion_scheduler = (
            self._scheduler(fusion_optimizer, phase, epochs)
            if fusion_optimizer is not None
            else None
        )
        casper_loader = self._casper_loader(session_id, max(1, len(loader)))
        max_batches_value = self.config.get("debug", {}).get(
            "max_batches_per_epoch"
        )
        max_batches = (
            None if max_batches_value is None else int(max_batches_value)
        )
        logs = []
        geometry_anchor_refreshes = 0
        progress = tqdm(
            range(epochs), disable=bool(self.config.get("disable_tqdm", False))
        )
        for epoch in progress:
            if (
                geometry is not None
                and geometry.requires_current_anchor_refresh
                and epoch
                % int(self.geometry_options["refresh_interval_epochs"])
                == 0
            ):
                self._refresh_geometry_current_anchors(session_id, geometry)
                geometry_anchor_refreshes += 1
            if self.method == "cscct":
                # The CSCCT release advances both schedulers at the start of
                # every epoch, including the zeroth phase.  Preserve that
                # ordering instead of forcing the shared end-of-epoch order.
                scheduler.step()
                if fusion_scheduler is not None:
                    fusion_scheduler.step()
            prototypes = (
                self._refresh_training_prototypes(prototype_loader)
                if prototype_loader is not None
                else None
            )
            self.model.train()
            totals: dict[str, float] = defaultdict(float)
            count = batches = correct = 0
            casper_iterator = None if casper_loader is None else iter(casper_loader)
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True).long()
                replay_mask = batch["is_replay"].to(self.device).bool()
                replay_images = None
                if casper_iterator is not None:
                    try:
                        replay_batch = next(casper_iterator)
                    except StopIteration:
                        casper_iterator = iter(casper_loader)
                        replay_batch = next(casper_iterator)
                    replay_images = replay_batch["image"].to(
                        self.device, non_blocking=True
                    )
                components, prediction = self._loss_components(
                    session_id,
                    images,
                    targets,
                    replay_mask,
                    teacher,
                    geometry,
                    prototypes,
                    replay_images,
                )
                loss = sum(components.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_count = targets.numel()
                count += batch_count
                batches += 1
                correct += int(prediction.argmax(1).eq(targets).sum().item())
                totals["loss"] += float(loss.detach()) * batch_count
                for name, value in components.items():
                    totals[name] += float(value.detach()) * batch_count
            if self.method != "cscct":
                scheduler.step()
            if fusion_optimizer is not None and fusion_loader is not None:
                self._update_cscct_fusion(fusion_loader, fusion_optimizer)
                if fusion_scheduler is not None and self.method != "cscct":
                    fusion_scheduler.step()
            if count == 0:
                raise RuntimeError("unified training processed no samples")
            record = {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "batches": batches,
                "train_accuracy": correct / count,
                **{key: value / count for key, value in totals.items()},
            }
            logs.append(record)
            progress.set_description(
                f"{self.method} s{session_id} e{epoch + 1}/{epochs} "
                f"loss={record['loss']:.4f} acc={record['train_accuracy']:.3f}"
            )
        return {
            "epochs": epochs,
            "samples": len(loader.dataset),
            "geometry_anchor_refreshes": geometry_anchor_refreshes,
            "epoch_logs": logs,
        }

    def _main_parameters(self) -> list[nn.Parameter]:
        if self.model is None:
            raise RuntimeError("model is missing")
        if isinstance(self.model, PyCILPODNet):
            return self.model.main_trainable_parameters()
        if isinstance(self.model, AFCIncrementalNet):
            for parameter in self.model.classifier.old_weights:
                parameter.requires_grad_(False)
            return [
                *[p for p in self.model.backbone.parameters() if p.requires_grad],
                self.model.classifier.new_weights,
            ]
        if isinstance(self.model, CSCCTIncrementalNet):
            for parameter in self.model.classifier.old_weights:
                parameter.requires_grad_(False)
            return self.model.main_parameters()
        if self.method == "sacil" and isinstance(
            self.model, ExpandableLinearNet
        ):
            for parameter in self.model.classifier.parameters():
                parameter.requires_grad_(False)
            return list(self.model.backbone.parameters())
        return [p for p in self.model.parameters() if p.requires_grad]

    def _add_geometry_component(
        self,
        components: dict[str, Tensor],
        current_features: Tensor,
        reference_features: Tensor | None,
        replay_mask: Tensor,
        geometry: AnchorGeometryLoss | None,
    ) -> None:
        value = geometry_preservation_component(
            geometry,
            current_features,
            reference_features,
            replay_mask,
            weight=float(self.config["method"].get("lambda_geo", 1.0)),
        )
        if value is not None:
            components["geometry"] = value

    def _add_casper_component(
        self,
        components: dict[str, Tensor],
        replay_images: Tensor | None,
    ) -> None:
        """Append the author CaSpeR spectral term on a balanced replay graph."""

        if not self.casper_options["enabled"] or replay_images is None:
            return
        if self.model is None:
            raise RuntimeError("CaSpeR regularization has no model")
        replay_features = self.model.extract_features(replay_images)
        components["spectral"] = float(
            self.casper_options["weight"]
        ) * casper_spectral_loss(
            replay_features,
            num_classes=int(self.casper_options["classes_per_graph"]),
            k=int(self.casper_options["knn"]),
            solver=str(self.casper_options["solver"]),
        )

    def _loss_components(
        self,
        session_id: int,
        images: Tensor,
        targets: Tensor,
        replay_mask: Tensor,
        teacher: nn.Module | None,
        geometry: AnchorGeometryLoss | None,
        prototypes: Tensor | None,
        replay_images: Tensor | None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        if self.model is None:
            raise RuntimeError("loss has no model")
        known = self.protocol.session(session_id).start
        total = self.protocol.session(session_id).stop
        new_count = self.protocol.session(session_id).size

        if isinstance(self.model, ExpandableLinearNet):
            output = self.model.forward_detailed(images)
            if self.method == "joint":
                return {"classification": F.cross_entropy(output.logits, targets)}, output.logits
            if self.method == "replay":
                return {
                    "classification": replay_cross_entropy(
                        output.logits, targets
                    )
                }, output.logits
            if self.method == "finetune":
                return {
                    "classification": pycil_finetune_loss(
                        output.logits,
                        targets,
                        known_classes=known,
                    )
                }, output.logits
            reference = (
                None
                if teacher is None
                else teacher.forward_detailed(images)
            )
            if self.method == "icarl":
                components = {
                    "classification": F.cross_entropy(output.logits, targets)
                }
                if reference is not None:
                    components["kd"] = pycil_icarl_kd_loss(
                        output.logits[:, :known],
                        reference.logits,
                        temperature=float(
                            self.config["method"].get("kd_temperature", 2.0)
                        ),
                    )
                self._add_geometry_component(
                    components,
                    output.features,
                    None if reference is None else reference.features,
                    replay_mask,
                    geometry,
                )
                self._add_casper_component(components, replay_images)
                return components, output.logits
            if self.method == "casper":
                loss = icarl_bce_loss(
                    output.logits,
                    targets,
                    old_logits=(None if reference is None else reference.logits),
                    known_classes=known,
                )
                components = {"classification": loss}
                components["regularization"] = parameter_l2_regularization(
                    self.model,
                    float(self.casper_options["wd_reg"]),
                )
                self._add_casper_component(components, replay_images)
                return components, output.logits
            if self.method == "sacil":
                if prototypes is None:
                    raise RuntimeError("SACIL training prototypes are missing")
                classification, prediction = prototype_cross_entropy(
                    output.features,
                    targets,
                    prototypes,
                    temperature=float(
                        self.config["method"].get("prototype_temperature", 0.1)
                    ),
                )
                components = {"classification": classification}
                self._add_geometry_component(
                    components,
                    output.features,
                    None if reference is None else reference.features,
                    replay_mask,
                    geometry,
                )
                return components, prediction
            raise RuntimeError(f"unhandled linear method: {self.method}")

        if isinstance(self.model, PyCILPODNet):
            output = self.model.forward_detailed(images)
            components = {
                "classification": podnet_nca_loss(
                    output.logits,
                    targets,
                    scale=1.0,
                )
            }
            if teacher is not None:
                if not isinstance(teacher, PyCILPODNet):
                    raise TypeError("PODNet teacher has the wrong model type")
                reference = teacher.forward_detailed(images)
                factor = math.sqrt(total / new_count)
                pod = self.config["method"].get("podnet", {})
                components["pod_flat"] = float(
                    pod.get("flat_weight", 1.0)
                ) * factor * pod_flat_loss(
                    output.features, reference.features
                )
                components["pod_spatial"] = float(
                    pod.get("spatial_weight", 5.0)
                ) * factor * pod_spatial_loss(
                    output.attentions, reference.attentions
                )
            return components, output.logits

        if isinstance(self.model, AFCIncrementalNet):
            output = self.model.forward_detailed(images)
            components = {
                "classification": afc_nca_loss(
                    output.logits, targets, 1.0
                )
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                afc = self.config["method"].get("afc", {})
                factor = scheduled_afc_factor(
                    total, new_count, float(afc.get("pod_base_factor", 4.0))
                )
                components["afc"] = factor * afc_pod_loss(
                    reference.attentions,
                    output.attentions,
                    reference.importance[:-1],
                )
                self._add_geometry_component(
                    components,
                    output.features,
                    reference.features,
                    replay_mask,
                    geometry,
                )
            return components, output.logits

        if isinstance(self.model, CREATEIncrementalNet):
            output = self.model.forward_detailed(images)
            create = self.config["method"].get("create", {})
            components = {
                "classification": create_classification_loss(
                    output["logits"], targets
                )
            }
            weights = reconstruction_confidence_weights(
                output["error_logits"],
                alpha=float(create.get("confidence_alpha", 2.0)),
            )
            components["contrastive"] = float(
                create.get("contrastive_weight", 1.0)
            ) * create_contrastive_loss(
                output["latents"], targets, sample_weights=weights
            )
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                components["kd"] = float(create.get("kd_weight", 1.0)) * create_kd_loss(
                    output["error_logits"][:, :known],
                    reference["error_logits"],
                    temperature=float(create.get("kd_temperature", 2.0)),
                )
            return components, output["logits"]

        if isinstance(self.model, FGPIncrementalNet):
            output = self.model.forward_detailed(images)
            one_hot = F.one_hot(targets, num_classes=total).to(output.logits)
            components = {
                "classification": F.binary_cross_entropy_with_logits(
                    output.logits, one_hot
                )
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                weight = scheduled_fgp_weight(
                    known,
                    total,
                    base_weight=float(
                        self.config["method"].get("fgp_weight", 0.1)
                    ),
                )
                components["graph"] = weight * fgp_graph_preservation_loss(
                    output.features,
                    reference.features,
                    self.model.classifier.weight[:known],
                    teacher.classifier.weight,
                )
            return components, output.logits

        if isinstance(self.model, CSCCTIncrementalNet):
            output = self.model.forward_detailed(images)
            components = {
                "classification": F.cross_entropy(output.logits, targets)
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                method = self.config["method"]
                components["kd"] = float(method.get("kd_weight", 0.25)) * old_logit_kl_loss(
                    output.logits[:, :known],
                    reference.logits,
                    temperature=float(method.get("kd_temperature", 2.0)),
                )
                components["csc"] = float(method.get("csc_weight", 3.0)) * cross_space_clustering_loss(
                    output.features, reference.features, targets
                )
                components["ct"] = float(method.get("ct_weight", 1.5)) * controlled_transfer_loss(
                    output.features,
                    reference.features,
                    targets,
                    known_classes=known,
                    temperature=float(method.get("ct_temperature", 2.0)),
                )
            return components, output.logits
        raise TypeError(f"unsupported model type: {type(self.model).__name__}")

    def _refresh_training_prototypes(self, loader: DataLoader | None) -> Tensor:
        if loader is None or self.model is None:
            raise RuntimeError("prototype refresh lacks model or loader")
        collection = collect_features(self.model, loader, self.device)
        prototypes = compute_prototypes(
            collection.features,
            collection.targets,
            range(self.model.num_classes),
        )
        return prototypes.to(self.device)

    def _casper_loader(
        self, session_id: int, batches: int
    ) -> DataLoader | None:
        if not self.casper_options["enabled"] or session_id == 0:
            return None
        indices = self.memory.all_indices(self.protocol.class_order)
        dataset = self.data.replay_dataset(indices, augment=True)
        labels = [
            self.protocol.incremental_label(self.data.train_aug.targets[index])
            for index in indices
        ]
        sampler = BalancedClassBatchSampler(
            labels,
            batch_size=int(self.casper_options["replay_batch_size"]),
            classes_per_batch=int(self.casper_options["classes_per_graph"]),
            batches=batches,
            seed=self.seed * 10000 + session_id * 100,
        )
        return self._loader(
            dataset,
            shuffle=False,
            session_id=session_id + 6000,
            batch_sampler=sampler,
        )

    def _cscct_balanced_loader(self, session_id: int) -> DataLoader | None:
        if self.method != "cscct" or session_id == 0:
            return None
        old_indices = self.memory.all_indices(self.protocol.class_order)
        # CSCCT's official fusion update draws ``increment * exemplars`` new
        # images with replacement, then concatenates them with the complete
        # old memory (base_trainer.py::gen_balanced_loader).
        new_candidates = self.data.new_train_dataset(
            session_id,
            augment=True,
            samples_per_class=self.debug_train_samples_per_class,
        )
        rng = np.random.default_rng(self.seed * 10000 + session_id * 100 + 71)
        new_sample_count = (
            self.protocol.session(session_id).size
            * self._memory_limit(self.protocol.session(session_id).stop)
        )
        new_positions = rng.choice(
            len(new_candidates), size=new_sample_count, replace=True
        )
        new_indices = [
            new_candidates.indices[int(position)]
            for position in new_positions.tolist()
        ]
        dataset = ConcatDataset(
            [
                self.data.replay_dataset(old_indices, augment=True),
                self.data.train_dataset_from_indices(
                    new_indices, augment=True, is_replay=False
                ),
            ]
        )
        return self._loader(
            dataset, shuffle=True, session_id=session_id + 7000
        )

    def _update_cscct_fusion(
        self, loader: DataLoader, optimizer: SGD
    ) -> None:
        if not isinstance(self.model, CSCCTIncrementalNet):
            return
        self.model.eval()
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device).long()
            loss = F.cross_entropy(self.model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def _update_memory(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("memory update has no model")
        limit = self._memory_limit(self.protocol.session(session_id).stop)
        self.memory.resize_limit(limit)
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        collection = collect_features(
            self.model,
            self._loader(dataset, shuffle=False, session_id=session_id + 8000),
            self.device,
        )
        selection = str(self.config["memory"].get("selection", "icarl_herding"))
        for class_id in self.protocol.classes_for_session(session_id):
            mask = collection.original_targets == int(class_id)
            # The controlled Table-1 contract uses one selector for every
            # replay-based method.  ``icarl_herding`` means the running-mean
            # greedy procedure implemented by stock PyCIL BaseLearner, not
            # AFC's direction-update variant.
            function = herding_select
            if selection not in {"icarl_herding", "running_mean"}:
                raise ValueError(f"unsupported controlled herding: {selection}")
            selected = function(
                collection.features[mask],
                collection.indices[mask].tolist(),
                limit,
            )
            self.memory.set_class_indices(class_id, selected)

    def _post_training(
        self,
        session_id: int,
        train_loader: DataLoader,
        teacher: nn.Module | None,
    ) -> dict[str, Any] | None:
        if self.method == "podnet":
            return {"finetuning": self._podnet_finetune(session_id, teacher)}
        if self.method == "afc":
            return {
                "finetuning": self._afc_finetune(session_id),
                "importance": self._afc_importance(train_loader),
            }
        if self.method == "create":
            return {
                "finetuning": self._create_finetune(session_id, teacher)
            }
        return None

    def _podnet_finetune(
        self, session_id: int, teacher: nn.Module | None
    ) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, PyCILPODNet):
            return None
        config = self.config["method"].get("podnet", {}).get(
            "finetuning", {}
        )
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        parameters = self._main_parameters()
        optimizer = SGD(
            parameters,
            lr=float(config.get("lr", 0.005)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        loader = self._memory_loader(session_id, augment=True)
        losses: list[float] = []
        for _ in range(epochs):
            self.model.train()
            total_loss = 0.0
            count = 0
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device).long()
                components, _ = self._loss_components(
                    session_id,
                    images,
                    targets,
                    batch["is_replay"].to(self.device).bool(),
                    teacher,
                    None,
                    None,
                    None,
                )
                loss = sum(components.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * targets.numel()
                count += targets.numel()
            scheduler.step()
            losses.append(total_loss / count)
        return {"epochs": epochs, "losses": losses}

    def _memory_loader(self, session_id: int, *, augment: bool) -> DataLoader:
        indices = self.memory.all_indices(self.protocol.class_order)
        return self._loader(
            self.data.replay_dataset(indices, augment=augment),
            shuffle=augment,
            session_id=session_id + 9000 + int(augment),
        )

    def _afc_finetune(self, session_id: int) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, AFCIncrementalNet):
            return None
        config = self.config["method"].get("afc", {}).get("finetuning", {})
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(True)
        optimizer = SGD(
            self.model.classifier.parameters(),
            lr=float(config.get("lr", 0.05)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        loader = self._memory_loader(session_id, augment=True)
        losses = []
        for _ in range(epochs):
            total = count = 0
            self.model.train()
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).long()
                loss = afc_nca_loss(self.model(images), targets, 1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * targets.numel()
                count += targets.numel()
            losses.append(total / count)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        return {"epochs": epochs, "losses": losses}

    def _afc_importance(self, loader: DataLoader) -> dict[str, Any]:
        if not isinstance(self.model, AFCIncrementalNet):
            raise TypeError("AFC importance needs AFCIncrementalNet")
        self.model.backbone.reset_importance()
        self.model.backbone.start_importance_collection()
        self.model.train()
        batches = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device).long()
            self.model.zero_grad(set_to_none=True)
            loss = afc_nca_loss(self.model(images), targets, 1.0)
            loss.backward()
            batches += 1
        self.model.backbone.stop_importance_collection()
        self.model.backbone.normalize_importance()
        return {
            "batches": batches,
            "layer_means": [
                float(layer.importance.mean())
                for layer in self.model.backbone.importance_layers
            ],
        }

    def _create_finetune(
        self,
        session_id: int,
        teacher: nn.Module | None,
    ) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, CREATEIncrementalNet):
            return None
        config = self.config["method"].get("create", {}).get("finetuning", {})
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        optimizer = SGD(
            self.model.classifier.parameters(),
            lr=float(config.get("lr", 0.005)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        loader = self._memory_loader(session_id, augment=True)
        losses = []
        for _ in range(epochs):
            total = count = 0
            self.model.train()
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).long()
                components, _ = self._loss_components(
                    session_id,
                    images,
                    targets,
                    batch["is_replay"].to(self.device).bool(),
                    teacher,
                    None,
                    None,
                    None,
                )
                loss = sum(components.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * targets.numel()
                count += targets.numel()
            scheduler.step()
            losses.append(total / count)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        return {"epochs": epochs, "losses": losses}

    def _build_evaluation_state(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("evaluation state has no model")
        if self.method == "create":
            self.class_means = None
        else:
            if self.method == "joint":
                dataset = self.data.train_eval_dataset_for_classes(
                    self.protocol.seen_classes(session_id),
                    samples_per_class=self.debug_train_samples_per_class,
                )
                loader = self._loader(
                    dataset, shuffle=False, session_id=session_id + 10000
                )
            else:
                loader = self._memory_loader(session_id, augment=False)
            self.class_means = compute_nme_class_means(
                self.model,
                loader,
                self.device,
                self.protocol.session(session_id).stop,
                horizontal_flip=bool(
                    self.config.get("evaluation", {}).get(
                        "horizontal_flip", True
                    )
                ),
            ).cpu()
        if self.geometry_mode != "none":
            self._build_sacil_artifacts(session_id)

    def _build_sacil_artifacts(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("SACIL artifacts have no model")
        loader = self._memory_loader(session_id, augment=False)
        collection = collect_features(self.model, loader, self.device)
        class_ids = self.protocol.seen_classes(session_id)
        values = compute_prototypes(
            collection.features, collection.original_targets, class_ids
        )
        prototypes = PrototypeBank(class_ids, values)
        hierarchy = self.config["method"].get("hierarchy", {})
        confusion = cosine_soft_confusion(
            collection.features,
            collection.targets,
            values,
            temperature=float(hierarchy.get("taxonomy_temperature", 0.2)),
        )
        tree = GriffinPeronaGreedy().build(
            class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(prototypes, tree)
        self.sacil_tree = tree
        self.sacil_prototypes = prototypes
        self.sacil_anchors = anchors
        dump_json(
            tree.state_dict(), self.run_dir / f"tree_session_{session_id:02d}.json"
        )

    def _evaluate_session(self, session_id: int):
        if self.model is None:
            raise RuntimeError("evaluation has no model")
        dataset = self.data.cumulative_test_dataset(
            session_id, samples_per_class=self.debug_test_samples_per_class
        )
        loader = self._loader(
            dataset, shuffle=False, session_id=session_id + 11000
        )
        old = self.protocol.session(session_id).start
        if self.method == "create":
            return evaluate(self.model, loader, self.device, old)
        if self.class_means is None:
            raise RuntimeError("NME evaluation has no class means")
        return evaluate_nme(
            self.model, loader, self.device, old, self.class_means
        )

    def _save_checkpoint(
        self,
        session_id: int,
        *,
        session_record: dict[str, Any] | None = None,
    ) -> Path:
        if self.model is None:
            raise RuntimeError("checkpoint has no model")
        path = self.checkpoint_dir / f"session_{session_id:02d}.pt"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "framework": "sacil-unified",
            "pycil_used": False,
            "reference_code_executed": False,
            "method_contract": self.method_contract.as_dict(),
            "method": self.method,
            "geometry_mode": self.geometry_mode,
            "casper_options": copy.deepcopy(self.casper_options),
            "session_id": session_id,
            "protocol_id": self.protocol.protocol_id,
            "config": self.config,
            "model": self.model.state_dict(),
            "model_type": type(self.model).__name__,
            "memory": self.memory.state_dict(),
            "metrics": self.metrics.state_dict(),
            "class_means": self.class_means,
            "session_record": (
                None
                if session_record is None
                else copy.deepcopy(session_record)
            ),
        }
        if self.sacil_tree is not None:
            payload.update(
                tree=self.sacil_tree.state_dict(),
                prototypes=self.sacil_prototypes.state_dict(),
                anchors=self.sacil_anchors.state_dict(),
            )
        save_checkpoint(payload, path)
        return path


# Backward-compatible import name for old analysis utilities.  New experiment
# code and documentation use UnifiedTable1Trainer.
StandaloneTable1Trainer = UnifiedTable1Trainer
