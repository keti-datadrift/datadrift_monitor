from __future__ import annotations

import copy
import logging
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# These are intentionally the official PyCIL top-level imports. The runtime
# adapter places the pinned PyCIL checkout at the front of sys.path before
# importing this module.
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

from sacil.anchors import (
    HierarchicalAnchorBank,
    PrototypeBank,
    compute_prototypes,
)
from sacil.hierarchy import (
    GriffinPeronaGreedy,
    HierarchyTree,
    cosine_soft_confusion,
    symmetric_affinity,
)
from sacil.methods import (
    AnchorGeometryLoss,
    ConflictWeights,
    compute_conflict_weights,
    global_preservation_weights,
    icarl_bce_loss,
    prototype_cross_entropy,
    prototype_logits,
)
from sacil.utils import dump_json, ensure_dir


@dataclass
class _FeatureCollection:
    features: Tensor
    targets: Tensor


@dataclass
class _SessionArtifacts:
    tree: HierarchyTree
    prototypes: PrototypeBank
    anchors: HierarchicalAnchorBank


@contextmanager
def preserve_rng_state():
    """Keep SACIL's deterministic post-hoc work RNG-neutral to PyCIL."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def pycil_kd_loss(pred: Tensor, soft: Tensor, temperature: float) -> Tensor:
    """The old-logit distillation used by PyCIL's official iCaRL learner."""

    if pred.ndim != 2 or soft.ndim != 2 or pred.shape != soft.shape:
        raise ValueError("student and teacher old-logit shapes must match")
    if pred.shape[0] == 0:
        return pred.sum() * 0.0
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    log_prob = F.log_softmax(pred / float(temperature), dim=1)
    probability = F.softmax(soft / float(temperature), dim=1)
    return -(probability * log_prob).sum() / pred.shape[0]


@torch.inference_mode()
def collect_pycil_features(
    network: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> _FeatureCollection:
    was_training = network.training
    network.eval()
    features: list[Tensor] = []
    targets: list[Tensor] = []
    for _, images, labels in loader:
        output = network(images.to(device, non_blocking=True))
        features.append(output["features"].detach().cpu())
        targets.append(labels.detach().cpu().long())
    network.train(was_training)
    if not features:
        raise ValueError("cannot collect features from an empty PyCIL loader")
    return _FeatureCollection(
        features=torch.cat(features, dim=0),
        targets=torch.cat(targets, dim=0),
    )


class SACIL(BaseLearner):
    """SACIL on the official PyCIL iCaRL/replay lifecycle.

    PyCIL owns the dataset/task split, classifier expansion, exemplar memory,
    herding, and evaluation. SACIL contributes only the post-hoc hierarchical
    anchors, conflict-aware relaxation, and geometry-preservation loss.
    """

    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._artifacts: _SessionArtifacts | None = None
        self._geometry_loss: AnchorGeometryLoss | None = None
        self._conflict_weights: ConflictWeights | None = None
        self._prototype_loader: DataLoader | None = None
        self._training_prototypes: Tensor | None = None

        self.batch_size = int(args.get("batch_size", 128))
        self.num_workers = int(args.get("num_workers", 0))
        self.pin_memory = bool(args.get("pin_memory", True))

        self.init_epochs = int(args.get("init_epochs", 200))
        self.init_lr = float(args.get("init_lr", 0.1))
        self.init_milestones = [
            int(value)
            for value in args.get("init_milestones", [60, 120, 170])
        ]
        self.init_weight_decay = float(
            args.get("init_weight_decay", 5e-4)
        )

        self.epochs = int(args.get("epochs", 170))
        self.lr = float(args.get("lr", 0.1))
        self.milestones = [
            int(value) for value in args.get("milestones", [80, 120])
        ]
        self.weight_decay = float(args.get("weight_decay", 2e-4))
        self.momentum = float(args.get("momentum", 0.9))
        self.lr_decay = float(args.get("lr_decay", 0.1))
        self.nesterov = bool(args.get("nesterov", False))

        self.lambda_kd = float(args.get("lambda_kd", 1.0))
        self.kd_temperature = float(args.get("kd_temperature", 2.0))
        self.classification_mode = str(
            args.get("classification_mode", "ce_kd")
        ).lower()
        if self.classification_mode not in {
            "ce_kd",
            "icarl_bce",
            "prototype_ce",
        }:
            raise ValueError(
                "classification_mode must be ce_kd, icarl_bce, "
                "or prototype_ce"
            )
        self.prototype_temperature = float(
            args.get("prototype_temperature", 0.1)
        )
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if self.classification_mode == "prototype_ce" and self.lambda_kd != 0:
            raise ValueError(
                "prototype_ce uses no parametric old-logit KD; "
                "set lambda_kd to 0"
            )
        self.kd_scope = str(args.get("kd_scope", "all")).lower()
        if self.kd_scope not in {"all", "replay"}:
            raise ValueError("kd_scope must be 'all' or 'replay'")

        self.lambda_geo = float(args.get("lambda_geo", 1.0))
        self.geometry_mode = str(
            args.get("geometry_mode", "sacil")
        ).lower()
        if self.geometry_mode not in {"none", "global", "flat", "sacil"}:
            raise ValueError(
                "geometry_mode must be none, global, flat, or sacil"
            )
        self.taxonomy_temperature = float(
            args.get("taxonomy_temperature", 0.2)
        )
        self.conflict_max_neighbors = int(
            args.get("conflict_max_neighbors", 5)
        )
        self.conflict_old_class_ratio = float(
            args.get("conflict_old_class_ratio", 0.1)
        )
        self.conflict_temperature = float(
            args.get("conflict_temperature", 0.05)
        )
        self.min_preservation_weight = float(
            args.get("min_preservation_weight", 0.1)
        )
        self.ancestor_decay = float(args.get("ancestor_decay", 0.5))

        self.eval_interval = int(args.get("eval_interval", 5))
        self.max_batches_per_epoch = args.get("max_batches_per_epoch")
        if self.max_batches_per_epoch is not None:
            self.max_batches_per_epoch = int(self.max_batches_per_epoch)
        self.disable_tqdm = bool(args.get("disable_tqdm", False))
        self.save_checkpoints = bool(args.get("save_checkpoints", True))
        self.artifact_dir = ensure_dir(
            Path(args.get("artifact_dir", "outputs/pycil/sacil"))
        )
        self._resume_checkpoint_rng: dict[str, Any] | None = None
        self._resume_base_pending = False
        resume_checkpoint = args.get("resume_checkpoint")
        if resume_checkpoint is not None:
            self._load_base_checkpoint(Path(resume_checkpoint))

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if self._resume_checkpoint_rng is not None:
            # The trainer evaluates the restored base once before after_task.
            # Restore the exact post-base RNG state here so that the first
            # incremental classifier and shuffled loader match a direct
            # continuation from the shared base.
            self._restore_rng_state(self._resume_checkpoint_rng)
            self._resume_checkpoint_rng = None
        logging.info("Exemplar size: %d", self.exemplar_size)
        if self.save_checkpoints:
            self._save_checkpoint()

    def incremental_train(self, data_manager):
        if self._resume_base_pending:
            self._resume_base_pending = False
            test_dataset = data_manager.get_dataset(
                np.arange(0, self._total_classes),
                source="test",
                mode="test",
            )
            self.test_loader = self._loader(
                test_dataset, shuffle=False
            )
            self._network.to(self._device)
            logging.info(
                "Restored shared base task 0 (0-%d); skipping base training",
                self._total_classes,
            )
            return

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on %d-%d", self._known_classes, self._total_classes
        )

        self._geometry_loss = None
        self._conflict_weights = None
        if self._cur_task > 0:
            if self._old_network is None or self._artifacts is None:
                raise RuntimeError(
                    "incremental SACIL task requires teacher and anchors"
                )
            # Keep this extra feature pass from perturbing PyCIL's classifier
            # initialization and DataLoader RNG stream. It is executed for
            # every geometry mode so the controlled variants have the same
            # non-training lifecycle.
            with preserve_rng_state():
                incoming = self._incoming_prototypes(data_manager)
            if self.geometry_mode == "global":
                self._conflict_weights = global_preservation_weights(
                    self._artifacts.anchors
                )
            elif self.geometry_mode in {"flat", "sacil"}:
                self._conflict_weights = compute_conflict_weights(
                    incoming,
                    self._artifacts.anchors,
                    self._artifacts.tree,
                    max_neighbors=self.conflict_max_neighbors,
                    old_class_ratio=self.conflict_old_class_ratio,
                    temperature=self.conflict_temperature,
                    min_preservation_weight=self.min_preservation_weight,
                    ancestor_decay=self.ancestor_decay,
                )
            if (
                self.geometry_mode != "none"
                and self.lambda_geo != 0
                and self._conflict_weights is not None
            ):
                self._geometry_loss = AnchorGeometryLoss(
                    self._artifacts.anchors,
                    self._conflict_weights.leaf_weights,
                    self._conflict_weights.internal_weights,
                    use_internal_anchors=self.geometry_mode != "flat",
                ).to(self._device)
                logging.info(
                    "%s weights: leaf min=%.4f mean=%.4f",
                    self.geometry_mode,
                    self._conflict_weights.leaf_weights.min().item(),
                    self._conflict_weights.leaf_weights.mean().item(),
                )

        self._network.update_fc(self._total_classes)
        if self.classification_mode == "prototype_ce":
            for parameter in self._network.fc.parameters():
                parameter.requires_grad = False
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = self._loader(train_dataset, shuffle=True)
        if self.classification_mode == "prototype_ce":
            prototype_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="test",
                appendent=self._get_memory(),
            )
            self._prototype_loader = self._loader(
                prototype_dataset, shuffle=False
            )
        else:
            self._prototype_loader = None
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = self._loader(test_dataset, shuffle=False)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(
                self._network, self._multiple_gpus
            )
        self._train(
            self.train_loader,
            self.test_loader,
            prototype_loader=self._prototype_loader,
        )

        # These calls deliberately use the official PyCIL BaseLearner
        # implementation and its iCaRL herding policy.
        self.build_rehearsal_memory(
            data_manager, self.samples_per_class
        )
        # Artifact construction is analysis-only and must not shift PyCIL's
        # RNG stream before the following task.
        with preserve_rng_state():
            self._artifacts = self._build_posthoc_artifacts(data_manager)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _network_without_parallel(self) -> nn.Module:
        if isinstance(self._network, nn.DataParallel):
            return self._network.module
        return self._network

    def _classifier_weights(self) -> Tensor:
        network = self._network_without_parallel()
        classifier = network.fc
        if not hasattr(classifier, "weight"):
            raise TypeError(
                "PyCIL SACIL currently requires a single SimpleLinear head"
            )
        return classifier.weight.detach().cpu()

    def _taxonomy_class_references(self, prototypes: Tensor) -> Tensor:
        if self.classification_mode == "prototype_ce":
            # Prototype CE never optimizes the frozen FC head.  Its taxonomy
            # must therefore use the same post-hoc class representatives as
            # training/inference instead of arbitrary FC directions.
            return prototypes.detach().cpu()
        return self._classifier_weights()

    def _incoming_prototypes(self, data_manager) -> Tensor:
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        loader = self._loader(dataset, shuffle=False)
        collection = collect_pycil_features(
            self._old_network, loader, self._device
        )
        return compute_prototypes(
            collection.features,
            collection.targets,
            range(self._known_classes, self._total_classes),
        )

    def _refresh_training_prototypes(self, loader: DataLoader) -> Tensor:
        collection = collect_pycil_features(
            self._network, loader, self._device
        )
        prototypes = compute_prototypes(
            collection.features,
            collection.targets,
            range(self._total_classes),
        )
        return prototypes.to(self._device)

    @torch.inference_mode()
    def _compute_prototype_accuracy(
        self,
        loader: DataLoader,
        prototypes: Tensor,
    ) -> float:
        self._network.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            output = self._network(
                inputs.to(self._device, non_blocking=True)
            )
            logits = prototype_logits(
                output["features"],
                prototypes,
                temperature=self.prototype_temperature,
            )
            correct += int(
                logits.argmax(dim=1).cpu().eq(targets).sum()
            )
            total += targets.numel()
        if total == 0:
            raise RuntimeError("prototype evaluation loader is empty")
        return float(np.around(100.0 * correct / total, decimals=2))

    def _eval_cnn(self, loader):
        if self.classification_mode != "prototype_ce":
            return super()._eval_cnn(loader)
        if not hasattr(self, "_class_means"):
            raise RuntimeError(
                "prototype inference requires post-hoc class means"
            )
        # PyCIL names this first return path "CNN".  For Proto-SACIL it is
        # deliberately the same NME rule as the primary inference classifier,
        # so no unused parametric FC head is evaluated.
        return self._eval_nme(loader, self._class_means)

    def _phase(self) -> tuple[int, float, list[int], float]:
        if self._cur_task == 0:
            return (
                self.init_epochs,
                self.init_lr,
                self.init_milestones,
                self.init_weight_decay,
            )
        return self.epochs, self.lr, self.milestones, self.weight_decay

    def _train(
        self,
        train_loader,
        test_loader,
        *,
        prototype_loader: DataLoader | None = None,
    ):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        epochs, learning_rate, milestones, weight_decay = self._phase()
        parameters = [
            parameter
            for parameter in self._network.parameters()
            if parameter.requires_grad
        ]
        optimizer = optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=self.momentum,
            weight_decay=weight_decay,
            nesterov=self.nesterov,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=self.lr_decay
        )

        iterator = tqdm(range(epochs), disable=self.disable_tqdm)
        for epoch in iterator:
            training_prototypes = None
            if self.classification_mode == "prototype_ce":
                if prototype_loader is None:
                    raise RuntimeError(
                        "prototype_ce requires a prototype loader"
                    )
                training_prototypes = self._refresh_training_prototypes(
                    prototype_loader
                )
                self._training_prototypes = training_prototypes.detach().cpu()
            self._network.train()
            total_loss = 0.0
            total_clf = 0.0
            total_kd = 0.0
            total_geo = 0.0
            correct, total, batches = 0, 0, 0
            for batch_index, (_, inputs, targets) in enumerate(train_loader):
                if (
                    self.max_batches_per_epoch is not None
                    and batch_index >= self.max_batches_per_epoch
                ):
                    break
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True).long()
                output = self._network(inputs)
                logits = output["logits"]
                features = output["features"]
                classification = F.cross_entropy(logits, targets)
                prediction_logits = logits
                if self.classification_mode == "prototype_ce":
                    if training_prototypes is None:
                        raise RuntimeError("training prototypes are missing")
                    classification, prediction_logits = (
                        prototype_cross_entropy(
                            features,
                            targets,
                            training_prototypes,
                            temperature=self.prototype_temperature,
                        )
                    )
                distillation = features.sum() * 0.0
                geometry = features.sum() * 0.0

                if self._old_network is not None:
                    with torch.no_grad():
                        reference = self._old_network(inputs)
                    if self.classification_mode == "icarl_bce":
                        classification = icarl_bce_loss(
                            logits,
                            targets,
                            old_logits=reference["logits"],
                            known_classes=self._known_classes,
                        )
                    replay_mask = targets < self._known_classes
                    if (
                        self.classification_mode == "ce_kd"
                        and self.lambda_kd != 0
                    ):
                        if self.kd_scope == "all":
                            kd_current = logits[:, : self._known_classes]
                            kd_reference = reference["logits"]
                        else:
                            kd_current = logits[
                                replay_mask, : self._known_classes
                            ]
                            kd_reference = reference["logits"][replay_mask]
                        distillation = pycil_kd_loss(
                            kd_current,
                            kd_reference,
                            self.kd_temperature,
                        )
                    if (
                        self._geometry_loss is not None
                        and bool(replay_mask.any())
                    ):
                        geometry = self._geometry_loss(
                            features[replay_mask],
                            reference["features"][replay_mask],
                        )
                elif self.classification_mode == "icarl_bce":
                    classification = icarl_bce_loss(logits, targets)

                loss = (
                    classification
                    + self.lambda_kd * distillation
                    + self.lambda_geo * geometry
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                count = targets.numel()
                total += count
                batches += 1
                total_loss += float(loss.detach().item()) * count
                total_clf += float(classification.detach().item()) * count
                total_kd += float(distillation.detach().item()) * count
                total_geo += float(geometry.detach().item()) * count
                correct += (
                    prediction_logits.argmax(dim=1)
                    .eq(targets)
                    .detach()
                    .cpu()
                    .sum()
                )

            scheduler.step()
            if total == 0:
                raise RuntimeError("PyCIL SACIL processed no training samples")
            train_accuracy = np.around(
                tensor2numpy(correct) * 100 / total, decimals=2
            )
            info = (
                "Task {}, Epoch {}/{} => Loss {:.4f}, CE {:.4f}, "
                "KD {:.4f}, Geo {:.6f}, Train_accy {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                epochs,
                total_loss / total,
                total_clf / total,
                total_kd / total,
                total_geo / total,
                train_accuracy,
            )
            if (
                self.eval_interval > 0
                and epoch % self.eval_interval == 0
            ):
                if self.classification_mode == "prototype_ce":
                    test_accuracy = self._compute_prototype_accuracy(
                        test_loader, training_prototypes
                    )
                else:
                    test_accuracy = self._compute_accuracy(
                        self._network, test_loader
                    )
                info += ", Test_accy {:.2f}".format(test_accuracy)
            iterator.set_description(info)
            logging.info(info)

    def _build_posthoc_artifacts(self, data_manager) -> _SessionArtifacts:
        memory = self._get_memory()
        if memory is None:
            raise RuntimeError("SACIL anchors require exemplar memory")
        dataset = data_manager.get_dataset(
            [], source="train", mode="test", appendent=memory
        )
        collection = collect_pycil_features(
            self._network,
            self._loader(dataset, shuffle=False),
            self._device,
        )
        class_ids: Sequence[int] = tuple(range(self._total_classes))
        prototype_tensor = compute_prototypes(
            collection.features, collection.targets, class_ids
        )
        prototypes = PrototypeBank(class_ids, prototype_tensor)
        confusion = cosine_soft_confusion(
            collection.features,
            collection.targets,
            self._taxonomy_class_references(prototype_tensor),
            temperature=self.taxonomy_temperature,
        )
        tree = GriffinPeronaGreedy().build(
            class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(prototypes, tree)
        dump_json(
            tree.state_dict(),
            self.artifact_dir
            / f"tree_task_{self._cur_task:02d}.json",
        )
        return _SessionArtifacts(tree, prototypes, anchors)

    def _save_checkpoint(self) -> None:
        if self._artifacts is None:
            raise RuntimeError("cannot save SACIL without anchor artifacts")
        path = self.artifact_dir / f"task_{self._cur_task:02d}.pt"
        torch.save(
            {
                "schema_version": 2,
                "framework": "PyCIL",
                "task": self._cur_task,
                "known_classes": self._known_classes,
                "model_state_dict": self._network.state_dict(),
                "memory_data": copy.deepcopy(self._data_memory),
                "memory_targets": copy.deepcopy(self._targets_memory),
                "class_means": copy.deepcopy(self._class_means),
                "classification_mode": self.classification_mode,
                "prototype_temperature": self.prototype_temperature,
                "tree": self._artifacts.tree.state_dict(),
                "prototypes": self._artifacts.prototypes.state_dict(),
                "anchors": self._artifacts.anchors.state_dict(),
                "rng_state": self._capture_rng_state(),
                "conflict": (
                    None
                    if self._conflict_weights is None
                    else self._conflict_weights.state_dict()
                ),
            },
            path,
        )

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def _load_base_checkpoint(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"shared base checkpoint does not exist: {resolved}"
            )
        try:
            checkpoint = torch.load(
                resolved, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(resolved, map_location="cpu")

        if checkpoint.get("framework") != "PyCIL":
            raise ValueError("resume checkpoint is not a PyCIL checkpoint")
        if int(checkpoint.get("schema_version", 0)) < 2:
            raise ValueError(
                "resume checkpoint must use schema version 2 or newer"
            )
        checkpoint_mode = checkpoint.get("classification_mode")
        if checkpoint_mode is None:
            if self.classification_mode == "prototype_ce":
                raise ValueError(
                    "prototype_ce cannot resume a checkpoint without "
                    "its classification contract"
                )
        elif str(checkpoint_mode) != self.classification_mode:
            raise ValueError(
                "resume checkpoint classification mode does not match"
            )
        checkpoint_temperature = checkpoint.get("prototype_temperature")
        if (
            self.classification_mode == "prototype_ce"
            and checkpoint_temperature is not None
            and not math.isclose(
                float(checkpoint_temperature),
                self.prototype_temperature,
            )
        ):
            raise ValueError(
                "resume checkpoint prototype temperature does not match"
            )
        task = int(checkpoint["task"])
        if task != 0:
            raise ValueError(
                "shared resume currently requires a task-0 checkpoint"
            )
        known_classes = int(checkpoint["known_classes"])
        if known_classes <= 0:
            raise ValueError(
                "shared base checkpoint has no learned classes"
            )

        self._network.update_fc(known_classes)
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._data_memory = copy.deepcopy(checkpoint["memory_data"])
        self._targets_memory = copy.deepcopy(
            checkpoint["memory_targets"]
        )
        self._class_means = copy.deepcopy(checkpoint["class_means"])
        tree = HierarchyTree.from_state_dict(checkpoint["tree"])
        prototypes = PrototypeBank.from_state_dict(
            checkpoint["prototypes"]
        )
        anchors = HierarchicalAnchorBank.from_state_dict(
            checkpoint["anchors"]
        )
        self._artifacts = _SessionArtifacts(tree, prototypes, anchors)
        conflict = checkpoint.get("conflict")
        self._conflict_weights = (
            None
            if conflict is None
            else ConflictWeights.from_state_dict(conflict)
        )

        # The first trainer iteration evaluates this shared base without
        # retraining it. after_task then creates the teacher and restores the
        # saved RNG state before task 1 begins.
        self._cur_task = task
        self._known_classes = 0
        self._total_classes = known_classes
        self._resume_checkpoint_rng = checkpoint["rng_state"]
        self._resume_base_pending = True
        logging.info("Loaded shared base checkpoint: %s", resolved)
