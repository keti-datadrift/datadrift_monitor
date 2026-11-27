from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from .sessions import ClassOrderProtocol


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def cifar100_train_transform(
    *, color_jitter: bool = False
) -> transforms.Compose:
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter:
        augmentations.append(transforms.ColorJitter(brightness=63 / 255))
    return transforms.Compose(
        [
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def cifar100_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


class CILIndexedSubset(Dataset):
    """Index-preserving subset with original and incremental labels."""

    def __init__(
        self,
        dataset: CIFAR100,
        indices: Sequence[int],
        protocol: ClassOrderProtocol,
        is_replay: bool,
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.protocol = protocol
        self.is_replay = bool(is_replay)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = self.indices[position]
        image, original_target = self.dataset[index]
        return {
            "image": image,
            "target": self.protocol.incremental_label(original_target),
            "original_target": int(original_target),
            "index": int(index),
            "is_replay": self.is_replay,
        }


class CIFAR100DataModule:
    """CIFAR-100 views required by exemplar-based CIL."""

    def __init__(
        self,
        root: str | Path,
        protocol: ClassOrderProtocol,
        download: bool = False,
        color_jitter: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.protocol = protocol
        self.train_aug = CIFAR100(
            root=str(self.root),
            train=True,
            transform=cifar100_train_transform(color_jitter=color_jitter),
            download=download,
        )
        self.train_eval = CIFAR100(
            root=str(self.root),
            train=True,
            transform=cifar100_eval_transform(),
            download=download,
        )
        self.test_eval = CIFAR100(
            root=str(self.root),
            train=False,
            transform=cifar100_eval_transform(),
            download=download,
        )
        self.train_indices_by_class = self._index_targets(self.train_aug.targets)
        self.test_indices_by_class = self._index_targets(self.test_eval.targets)

    @staticmethod
    def _index_targets(targets: Sequence[int]) -> dict[int, tuple[int, ...]]:
        result: defaultdict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            result[int(target)].append(index)
        return {key: tuple(value) for key, value in result.items()}

    @staticmethod
    def _select_indices(
        mapping: dict[int, tuple[int, ...]],
        class_ids: Iterable[int],
        samples_per_class: int | None = None,
    ) -> list[int]:
        selected: list[int] = []
        for class_id in class_ids:
            indices = mapping[int(class_id)]
            if samples_per_class is not None:
                indices = indices[: int(samples_per_class)]
            selected.extend(indices)
        return selected

    def new_train_dataset(
        self,
        session_id: int,
        *,
        augment: bool = True,
        samples_per_class: int | None = None,
    ) -> CILIndexedSubset:
        indices = self._select_indices(
            self.train_indices_by_class,
            self.protocol.classes_for_session(session_id),
            samples_per_class,
        )
        dataset = self.train_aug if augment else self.train_eval
        return CILIndexedSubset(dataset, indices, self.protocol, is_replay=False)

    def replay_dataset(
        self, indices: Sequence[int], *, augment: bool = True
    ) -> CILIndexedSubset:
        dataset = self.train_aug if augment else self.train_eval
        return CILIndexedSubset(dataset, indices, self.protocol, is_replay=True)

    def train_eval_dataset_for_classes(
        self,
        class_ids: Iterable[int],
        samples_per_class: int | None = None,
    ) -> CILIndexedSubset:
        indices = self._select_indices(
            self.train_indices_by_class, class_ids, samples_per_class
        )
        return CILIndexedSubset(
            self.train_eval, indices, self.protocol, is_replay=False
        )

    def train_dataset_for_classes(
        self,
        class_ids: Iterable[int],
        *,
        augment: bool = True,
        samples_per_class: int | None = None,
    ) -> CILIndexedSubset:
        """Return a training split for arbitrary seen classes.

        Unlike ``new_train_dataset`` this is also suitable for joint-training
        upper bounds, which revisit the complete training data of every seen
        class at each session.
        """

        indices = self._select_indices(
            self.train_indices_by_class, class_ids, samples_per_class
        )
        dataset = self.train_aug if augment else self.train_eval
        return CILIndexedSubset(
            dataset, indices, self.protocol, is_replay=False
        )

    def train_eval_dataset_from_indices(
        self, indices: Sequence[int], *, is_replay: bool = True
    ) -> CILIndexedSubset:
        return CILIndexedSubset(
            self.train_eval, indices, self.protocol, is_replay=is_replay
        )

    def train_dataset_from_indices(
        self,
        indices: Sequence[int],
        *,
        augment: bool = True,
        is_replay: bool = False,
    ) -> CILIndexedSubset:
        """Return an augmented or deterministic training view of exact indices."""

        dataset = self.train_aug if augment else self.train_eval
        return CILIndexedSubset(
            dataset, indices, self.protocol, is_replay=is_replay
        )

    def cumulative_test_dataset(
        self,
        session_id: int,
        samples_per_class: int | None = None,
    ) -> CILIndexedSubset:
        indices = self._select_indices(
            self.test_indices_by_class,
            self.protocol.seen_classes(session_id),
            samples_per_class,
        )
        return CILIndexedSubset(
            self.test_eval, indices, self.protocol, is_replay=False
        )

    def training_dataset(
        self,
        session_id: int,
        memory_indices: Sequence[int],
        *,
        samples_per_class: int | None = None,
    ) -> Dataset:
        new_dataset = self.new_train_dataset(
            session_id,
            augment=True,
            samples_per_class=samples_per_class,
        )
        if not memory_indices:
            return new_dataset
        replay_dataset = self.replay_dataset(memory_indices, augment=True)
        return ConcatDataset([new_dataset, replay_dataset])
