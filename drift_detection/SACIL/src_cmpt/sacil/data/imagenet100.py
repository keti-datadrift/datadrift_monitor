from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .sessions import ClassOrderProtocol


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_train_transform() -> transforms.Compose:
    """The 224px training pipeline used by the LUCIR ImageNet release."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def imagenet_eval_transform() -> transforms.Compose:
    """Deterministic ImageNet validation/prototype view."""

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CILImageFolderSubset(Dataset):
    """Index-preserving ImageFolder subset with incremental labels."""

    def __init__(
        self,
        dataset: ImageFolder,
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


class ImageNet100DataModule:
    """ImageFolder views required by exemplar-based ImageNet-100 CIL."""

    def __init__(
        self,
        root: str | Path,
        protocol: ClassOrderProtocol,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.protocol = protocol
        train_root = self.root / "train"
        val_root = self.root / "val"
        if not train_root.is_dir() or not val_root.is_dir():
            raise FileNotFoundError(
                "ImageNet-100 root must contain train/ and val/: "
                f"{self.root}"
            )

        self.train_aug = ImageFolder(
            str(train_root), transform=imagenet_train_transform()
        )
        self.train_eval = ImageFolder(
            str(train_root), transform=imagenet_eval_transform()
        )
        self.test_eval = ImageFolder(
            str(val_root), transform=imagenet_eval_transform()
        )
        self._validate_label_mapping()
        self.train_indices_by_class = self._index_targets(
            self.train_aug.targets
        )
        self.test_indices_by_class = self._index_targets(
            self.test_eval.targets
        )

    def _validate_label_mapping(self) -> None:
        if self.protocol.num_classes != 100:
            raise ValueError("ImageNet100DataModule requires 100 classes")
        if self.train_aug.class_to_idx != self.train_eval.class_to_idx:
            raise ValueError("ImageNet train views have inconsistent labels")
        if self.train_aug.class_to_idx != self.test_eval.class_to_idx:
            raise ValueError("ImageNet train/val WNID mappings differ")

        class_index = self.root / "metadata" / "class_index.csv"
        if not class_index.is_file():
            raise FileNotFoundError(
                f"ImageNet-100 class mapping is missing: {class_index}"
            )
        with class_index.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = {
            str(row["wnid"]): int(row["label"])
            for row in rows
        }
        if len(expected) != 100 or expected != self.train_aug.class_to_idx:
            raise ValueError(
                "ImageFolder WNID ordering does not match metadata/class_index.csv"
            )

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
    ) -> CILImageFolderSubset:
        indices = self._select_indices(
            self.train_indices_by_class,
            self.protocol.classes_for_session(session_id),
            samples_per_class,
        )
        dataset = self.train_aug if augment else self.train_eval
        return CILImageFolderSubset(
            dataset, indices, self.protocol, is_replay=False
        )

    def replay_dataset(
        self, indices: Sequence[int], *, augment: bool = True
    ) -> CILImageFolderSubset:
        dataset = self.train_aug if augment else self.train_eval
        return CILImageFolderSubset(
            dataset, indices, self.protocol, is_replay=True
        )

    def train_eval_dataset_for_classes(
        self,
        class_ids: Iterable[int],
        samples_per_class: int | None = None,
    ) -> CILImageFolderSubset:
        indices = self._select_indices(
            self.train_indices_by_class, class_ids, samples_per_class
        )
        return CILImageFolderSubset(
            self.train_eval, indices, self.protocol, is_replay=False
        )

    def train_dataset_for_classes(
        self,
        class_ids: Iterable[int],
        *,
        augment: bool = True,
        samples_per_class: int | None = None,
    ) -> CILImageFolderSubset:
        indices = self._select_indices(
            self.train_indices_by_class, class_ids, samples_per_class
        )
        dataset = self.train_aug if augment else self.train_eval
        return CILImageFolderSubset(
            dataset, indices, self.protocol, is_replay=False
        )

    def train_eval_dataset_from_indices(
        self, indices: Sequence[int], *, is_replay: bool = True
    ) -> CILImageFolderSubset:
        return CILImageFolderSubset(
            self.train_eval, indices, self.protocol, is_replay=is_replay
        )

    def train_dataset_from_indices(
        self,
        indices: Sequence[int],
        *,
        augment: bool = True,
        is_replay: bool = False,
    ) -> CILImageFolderSubset:
        dataset = self.train_aug if augment else self.train_eval
        return CILImageFolderSubset(
            dataset, indices, self.protocol, is_replay=is_replay
        )

    def cumulative_test_dataset(
        self,
        session_id: int,
        samples_per_class: int | None = None,
    ) -> CILImageFolderSubset:
        indices = self._select_indices(
            self.test_indices_by_class,
            self.protocol.seen_classes(session_id),
            samples_per_class,
        )
        return CILImageFolderSubset(
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
        return ConcatDataset(
            [new_dataset, self.replay_dataset(memory_indices, augment=True)]
        )
