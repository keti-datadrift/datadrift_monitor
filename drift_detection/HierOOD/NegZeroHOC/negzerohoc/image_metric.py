from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """Deterministic P-class/K-example batches for metric learning."""

    def __init__(
        self,
        targets: Sequence[int],
        *,
        classes_per_batch: int,
        examples_per_class: int,
        seed: int = 0,
    ):
        groups: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            groups[int(target)].append(index)
        if classes_per_batch < 2:
            raise ValueError("classes_per_batch must be at least 2")
        if examples_per_class < 2:
            raise ValueError("examples_per_class must be at least 2")
        if classes_per_batch > len(groups):
            raise ValueError("classes_per_batch exceeds the number of classes")
        self.groups = dict(groups)
        self.classes = sorted(groups)
        self.classes_per_batch = int(classes_per_batch)
        self.examples_per_class = int(examples_per_class)
        self.batch_size = self.classes_per_batch * self.examples_per_class
        self.num_batches = math.ceil(len(targets) / self.batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        for _ in range(self.num_batches):
            selected_classes = rng.sample(self.classes, self.classes_per_batch)
            batch = []
            for target in selected_classes:
                indices = self.groups[target]
                if len(indices) >= self.examples_per_class:
                    batch.extend(rng.sample(indices, self.examples_per_class))
                else:
                    batch.extend(rng.choices(indices, k=self.examples_per_class))
            rng.shuffle(batch)
            yield batch


class HierarchyPKBatchSampler(PKBatchSampler):
    """P/K batches containing both nearby and distant hierarchy classes.

    The first class is sampled uniformly.  The next two slots, when present,
    are filled by a nearest and a farthest class in tree distance.  Remaining
    slots are sampled uniformly without replacement.  This gives hierarchical
    metric losses usable positive/negative relations at coarse and fine levels
    instead of relying on accidental class co-occurrence in a random batch.
    """

    def __init__(
        self,
        targets: Sequence[int],
        *,
        class_paths: dict[int, Sequence[object]],
        classes_per_batch: int,
        examples_per_class: int,
        seed: int = 0,
    ):
        super().__init__(
            targets,
            classes_per_batch=classes_per_batch,
            examples_per_class=examples_per_class,
            seed=seed,
        )
        missing = sorted(set(self.classes) - set(class_paths))
        if missing:
            raise ValueError(f"class_paths is missing target classes: {missing}")
        self.class_paths = {
            int(target): tuple(class_paths[int(target)]) for target in self.classes
        }
        self.class_distances = {
            (first, second): self._tree_distance(
                self.class_paths[first], self.class_paths[second]
            )
            for first in self.classes
            for second in self.classes
        }

    @staticmethod
    def _tree_distance(first: Sequence[object], second: Sequence[object]) -> int:
        common = 0
        for first_node, second_node in zip(first, second):
            if first_node != second_node:
                break
            common += 1
        return len(first) + len(second) - 2 * common

    def _select_classes(self, rng: random.Random) -> list[int]:
        anchor = rng.choice(self.classes)
        selected = [anchor]
        remaining = [target for target in self.classes if target != anchor]

        if len(selected) < self.classes_per_batch:
            nearest_distance = min(self.class_distances[(anchor, target)] for target in remaining)
            nearest = [
                target for target in remaining
                if self.class_distances[(anchor, target)] == nearest_distance
            ]
            chosen = rng.choice(nearest)
            selected.append(chosen)
            remaining.remove(chosen)

        if len(selected) < self.classes_per_batch:
            farthest_distance = max(self.class_distances[(anchor, target)] for target in remaining)
            farthest = [
                target for target in remaining
                if self.class_distances[(anchor, target)] == farthest_distance
            ]
            chosen = rng.choice(farthest)
            selected.append(chosen)
            remaining.remove(chosen)

        if len(selected) < self.classes_per_batch:
            selected.extend(rng.sample(remaining, self.classes_per_batch - len(selected)))
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        for _ in range(self.num_batches):
            selected_classes = self._select_classes(rng)
            batch = []
            for target in selected_classes:
                indices = self.groups[target]
                if len(indices) >= self.examples_per_class:
                    batch.extend(rng.sample(indices, self.examples_per_class))
                else:
                    batch.extend(rng.choices(indices, k=self.examples_per_class))
            rng.shuffle(batch)
            yield batch


def _exposure_stats(
    sample_counts: Sequence[int],
    class_counts: dict[int, int],
) -> dict[str, float | int]:
    def summarize(values: Sequence[int], prefix: str) -> dict[str, float | int]:
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        return {
            f"{prefix}_min": min(values, default=0),
            f"{prefix}_max": max(values, default=0),
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_unseen": sum(value == 0 for value in values),
        }

    return {
        **summarize(sample_counts, "sample"),
        **summarize(list(class_counts.values()), "class_slots"),
    }


class MSLossPKBatchSampler(PKBatchSampler):
    """The queue sampler released with the Multi-Similarity loss.

    This follows ``random_identity_sampler.py`` from the official MS Loss
    repository: shuffle each class queue, pad only classes shorter than K by
    sampling with replacement, split into K-sized chunks (dropping a remainder
    shorter than K), sample P available
    classes per batch, and rebuild every queue once fewer than P classes remain.
    Queue and RNG state persist across DataLoader epochs so an epoch boundary
    does not silently discard the partially consumed official queue.
    """

    def __init__(
        self,
        targets: Sequence[int],
        *,
        classes_per_batch: int,
        examples_per_class: int,
        seed: int = 0,
    ):
        super().__init__(
            targets,
            classes_per_batch=classes_per_batch,
            examples_per_class=examples_per_class,
            seed=seed,
        )
        self._rng = random.Random(self.seed)
        self._batch_chunks: dict[int, list[list[int]]] = {}
        self._available_classes: list[int] = []
        self.sample_counts = [0] * len(targets)
        self.class_counts = {target: 0 for target in self.classes}
        self._scheduled_epoch: int | None = None
        self._epoch_batches: list[list[int]] = []

    def _prepare_cycle(self) -> None:
        self._batch_chunks = {}
        for target in self.classes:
            indices = list(self.groups[target])
            if len(indices) < self.examples_per_class:
                indices = self._rng.choices(
                    indices,
                    k=self.examples_per_class,
                )
            self._rng.shuffle(indices)
            self._batch_chunks[target] = [
                indices[start:start + self.examples_per_class]
                for start in range(
                    0,
                    len(indices) - self.examples_per_class + 1,
                    self.examples_per_class,
                )
            ]
        self._available_classes = list(self.classes)

    def _next_batch(self) -> list[int]:
        if len(self._available_classes) < self.classes_per_batch:
            self._prepare_cycle()
        selected_classes = self._rng.sample(
            self._available_classes,
            self.classes_per_batch,
        )
        batch = []
        for target in selected_classes:
            chunk = self._batch_chunks[target].pop(0)
            batch.extend(chunk)
            self.class_counts[target] += 1
            for index in chunk:
                self.sample_counts[index] += 1
            if not self._batch_chunks[target]:
                self._available_classes.remove(target)
        return batch

    def _schedule_epoch(self) -> None:
        if self._scheduled_epoch == self.epoch:
            return
        self._epoch_batches = [
            self._next_batch() for _ in range(self.num_batches)
        ]
        self._scheduled_epoch = self.epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._schedule_epoch()

    def __iter__(self) -> Iterator[list[int]]:
        self._schedule_epoch()
        yield from (list(batch) for batch in self._epoch_batches)

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "rng_state": self._rng.getstate(),
            "batch_chunks": {
                target: [list(chunk) for chunk in chunks]
                for target, chunks in self._batch_chunks.items()
            },
            "available_classes": list(self._available_classes),
            "sample_counts": list(self.sample_counts),
            "class_counts": dict(self.class_counts),
            "epoch": self.epoch,
            "scheduled_epoch": self._scheduled_epoch,
            "epoch_batches": [list(batch) for batch in self._epoch_batches],
        }

    def load_state_dict(self, state: dict) -> None:
        self._rng.setstate(state["rng_state"])
        self._batch_chunks = {
            int(target): [list(chunk) for chunk in chunks]
            for target, chunks in state["batch_chunks"].items()
        }
        self._available_classes = [
            int(target) for target in state["available_classes"]
        ]
        self.sample_counts = [int(count) for count in state["sample_counts"]]
        self.class_counts = {
            int(target): int(count)
            for target, count in state["class_counts"].items()
        }
        self.epoch = int(state.get("epoch", 0))
        self._scheduled_epoch = state.get("scheduled_epoch")
        self._epoch_batches = [
            list(batch) for batch in state.get("epoch_batches", [])
        ]

    def exposure_stats(self) -> dict[str, float | int]:
        return _exposure_stats(self.sample_counts, self.class_counts)


class GloballyBalancedPKBatchSampler(PKBatchSampler):
    """Greedy P/K sampler balancing cumulative exposure across the whole run.

    P distinct classes with the lowest mean per-sample exposure are chosen for
    every batch. Within each selected class, the K least-used samples are used.
    Random tie-breaking prevents a fixed index bias. Counts and RNG state are
    checkpointable, so the balance guarantee continues after resume.
    """

    def __init__(
        self,
        targets: Sequence[int],
        *,
        classes_per_batch: int,
        examples_per_class: int,
        seed: int = 0,
    ):
        super().__init__(
            targets,
            classes_per_batch=classes_per_batch,
            examples_per_class=examples_per_class,
            seed=seed,
        )
        self._rng = random.Random(self.seed)
        self.sample_counts = [0] * len(targets)
        self.class_counts = {target: 0 for target in self.classes}
        self._scheduled_epoch: int | None = None
        self._epoch_batches: list[list[int]] = []

    def _select_classes(self) -> list[int]:
        tie_breaks = {target: self._rng.random() for target in self.classes}
        ranked = sorted(
            self.classes,
            key=lambda target: (
                sum(self.sample_counts[index] for index in self.groups[target])
                / len(self.groups[target]),
                tie_breaks[target],
            ),
        )
        return ranked[:self.classes_per_batch]

    def _select_examples(self, target: int) -> list[int]:
        selected = []
        while len(selected) < self.examples_per_class:
            candidates = [
                index for index in self.groups[target] if index not in selected
            ]
            if not candidates:
                candidates = list(self.groups[target])
            minimum = min(self.sample_counts[index] for index in candidates)
            least_used = [
                index for index in candidates
                if self.sample_counts[index] == minimum
            ]
            index = self._rng.choice(least_used)
            selected.append(index)
            self.sample_counts[index] += 1
        return selected

    def _schedule_epoch(self) -> None:
        if self._scheduled_epoch == self.epoch:
            return
        self._epoch_batches = []
        for _ in range(self.num_batches):
            selected_classes = self._select_classes()
            batch = []
            for target in selected_classes:
                batch.extend(self._select_examples(target))
                self.class_counts[target] += 1
            self._rng.shuffle(batch)
            self._epoch_batches.append(batch)
        self._scheduled_epoch = self.epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._schedule_epoch()

    def __iter__(self) -> Iterator[list[int]]:
        self._schedule_epoch()
        yield from (list(batch) for batch in self._epoch_batches)

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "rng_state": self._rng.getstate(),
            "sample_counts": list(self.sample_counts),
            "class_counts": dict(self.class_counts),
            "epoch": self.epoch,
            "scheduled_epoch": self._scheduled_epoch,
            "epoch_batches": [list(batch) for batch in self._epoch_batches],
        }

    def load_state_dict(self, state: dict) -> None:
        self._rng.setstate(state["rng_state"])
        self.sample_counts = [int(count) for count in state["sample_counts"]]
        self.class_counts = {
            int(target): int(count)
            for target, count in state["class_counts"].items()
        }
        self.epoch = int(state.get("epoch", 0))
        self._scheduled_epoch = state.get("scheduled_epoch")
        self._epoch_batches = [
            list(batch) for batch in state.get("epoch_batches", [])
        ]

    def exposure_stats(self) -> dict[str, float | int]:
        return _exposure_stats(self.sample_counts, self.class_counts)


def supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    features = F.normalize(features.float(), dim=-1)
    targets = targets.long().to(features.device)
    batch_size = int(features.shape[0])
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = targets[:, None].eq(targets[None, :]) & ~self_mask
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every metric-learning sample needs an in-batch positive")

    logits = features @ features.t() / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    mean_positive_log_prob = (
        (log_prob * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)
    )
    loss = -mean_positive_log_prob.mean()
    similarities = features @ features.t()
    negative_mask = ~targets[:, None].eq(targets[None, :])
    return loss, {
        "supcon_loss": float(loss.detach().cpu()),
        "positive_cosine": float(similarities[positive_mask].mean().detach().cpu()),
        "negative_cosine": float(similarities[negative_mask].mean().detach().cpu()),
    }


def batch_hard_hierarchical_triplet_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    class_distances: torch.Tensor,
    *,
    base_margin: float,
    hierarchy_margin: float,
) -> tuple[torch.Tensor, dict]:
    features = F.normalize(features.float(), dim=-1)
    targets = targets.long().to(features.device)
    class_distances = class_distances.float().to(features.device)
    pairwise_distance = 1.0 - features @ features.t()
    same = targets[:, None].eq(targets[None, :])
    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = same & ~self_mask
    negative_mask = ~same
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every triplet anchor needs an in-batch positive")

    hardest_positive = pairwise_distance.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    selected_class_distances = class_distances.index_select(0, targets).index_select(1, targets)
    max_class_distance = class_distances.max().clamp_min(1.0)
    margins = float(base_margin) + float(hierarchy_margin) * (
        selected_class_distances / max_class_distance
    )
    violations = hardest_positive[:, None] - pairwise_distance + margins
    hardest_violation = violations.masked_fill(~negative_mask, float("-inf")).max(dim=1).values
    loss = F.relu(hardest_violation).mean()
    return loss, {
        "triplet_loss": float(loss.detach().cpu()),
        "hardest_positive_distance": float(hardest_positive.mean().detach().cpu()),
        "active_triplet_rate": float((hardest_violation > 0).float().mean().detach().cpu()),
    }


def cosine_proxy_loss(
    features: torch.Tensor,
    proxies: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float,
    margin: float,
) -> tuple[torch.Tensor, dict]:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    features = F.normalize(features.float(), dim=-1)
    proxies = F.normalize(proxies.float(), dim=-1)
    targets = targets.long().to(features.device)
    logits = features @ proxies.t()
    target_mask = F.one_hot(targets, num_classes=proxies.shape[0]).bool()
    margin_logits = logits - target_mask.to(logits.dtype) * float(margin)
    loss = F.cross_entropy(margin_logits / float(temperature), targets)
    predictions = logits.argmax(dim=1)
    return loss, {
        "proxy_loss": float(loss.detach().cpu()),
        "proxy_acc": float((predictions == targets).float().mean().detach().cpu()),
        "target_proxy_cosine": float(
            logits.gather(1, targets.unsqueeze(1)).mean().detach().cpu()
        ),
    }


def class_tree_distance_matrix(hierarchy, class_nodes: Sequence[str]) -> torch.Tensor:
    paths = []
    for node in class_nodes:
        ancestor_indices = hierarchy.node_ancestors.get(node, [])
        paths.append([
            hierarchy.id_node_list[int(index)] for index in ancestor_indices
        ] + [node])
    distances = torch.zeros(len(paths), len(paths), dtype=torch.float32)
    for first_index, first_path in enumerate(paths):
        for second_index, second_path in enumerate(paths):
            common = 0
            for first_node, second_node in zip(first_path, second_path):
                if first_node != second_node:
                    break
                common += 1
            distances[first_index, second_index] = (
                len(first_path) + len(second_path) - 2 * common
            )
    return distances
