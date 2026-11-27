from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    accuracy: float
    old_accuracy: float | None
    new_accuracy: float
    harmonic_mean: float | None
    per_class_accuracy: dict[int, float]
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "old_accuracy": self.old_accuracy,
            "new_accuracy": self.new_accuracy,
            "harmonic_mean": self.harmonic_mean,
            "per_class_accuracy": {
                str(key): value
                for key, value in sorted(self.per_class_accuracy.items())
            },
            "sample_count": self.sample_count,
        }


def _safe_accuracy(correct: int, count: int) -> float | None:
    if count == 0:
        return None
    return correct / count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    old_class_count: int,
) -> EvaluationResult:
    was_training = model.training
    model.eval()
    total_correct = 0
    total_count = 0
    old_correct = 0
    old_count = 0
    new_correct = 0
    new_count = 0
    per_class_correct: dict[int, int] = {}
    per_class_count: dict[int, int] = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        predictions = model(images).argmax(dim=1)
        correct = predictions.eq(targets)
        total_correct += int(correct.sum().item())
        total_count += targets.numel()

        old_mask = targets < int(old_class_count)
        new_mask = ~old_mask
        old_correct += int(correct[old_mask].sum().item())
        old_count += int(old_mask.sum().item())
        new_correct += int(correct[new_mask].sum().item())
        new_count += int(new_mask.sum().item())

        original_targets = batch["original_target"].long()
        for class_id in original_targets.unique().tolist():
            class_mask = original_targets == int(class_id)
            class_correct = correct.detach().cpu()[class_mask]
            per_class_correct[int(class_id)] = (
                per_class_correct.get(int(class_id), 0)
                + int(class_correct.sum().item())
            )
            per_class_count[int(class_id)] = (
                per_class_count.get(int(class_id), 0)
                + int(class_mask.sum().item())
            )

    if was_training:
        model.train()
    if total_count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    old_accuracy = _safe_accuracy(old_correct, old_count)
    new_accuracy_value = _safe_accuracy(new_correct, new_count)
    if new_accuracy_value is None:
        raise ValueError("evaluation set contains no current-session classes")
    harmonic_mean = None
    if old_accuracy is not None:
        denominator = old_accuracy + new_accuracy_value
        harmonic_mean = (
            0.0
            if denominator == 0
            else 2.0 * old_accuracy * new_accuracy_value / denominator
        )
    return EvaluationResult(
        accuracy=total_correct / total_count,
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy_value,
        harmonic_mean=harmonic_mean,
        per_class_accuracy={
            class_id: per_class_correct[class_id] / per_class_count[class_id]
            for class_id in per_class_count
        },
        sample_count=total_count,
    )


@torch.inference_mode()
def compute_nme_class_means(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    horizontal_flip: bool = True,
) -> torch.Tensor:
    """Compute iCaRL-style normalized exemplar means.

    Official iCaRL-derived implementations average normalized features from
    each exemplar and its horizontal mirror.  Performing the flip after tensor
    normalization is equivalent because it only reverses the width axis.
    """

    was_training = model.training
    model.eval()
    sums: torch.Tensor | None = None
    counts = torch.zeros(int(num_classes), dtype=torch.long, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        features = F.normalize(model.extract_features(images).float(), dim=1)
        if horizontal_flip:
            flipped = F.normalize(
                model.extract_features(torch.flip(images, dims=(3,))).float(),
                dim=1,
            )
            features = 0.5 * (features + flipped)
        if sums is None:
            sums = torch.zeros(
                int(num_classes),
                features.shape[1],
                device=device,
                dtype=features.dtype,
            )
        sums.index_add_(0, targets, features)
        counts.index_add_(0, targets, torch.ones_like(targets))
    if was_training:
        model.train()
    if sums is None or bool((counts == 0).any()):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"cannot compute NME means; empty classes: {missing}")
    return F.normalize(sums / counts[:, None], dim=1).detach()


@torch.inference_mode()
def evaluate_nme(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    old_class_count: int,
    class_means: torch.Tensor,
) -> EvaluationResult:
    was_training = model.training
    model.eval()
    means = F.normalize(class_means.to(device).float(), dim=1)
    total_correct = total_count = 0
    old_correct = old_count = 0
    new_correct = new_count = 0
    per_class_correct: dict[int, int] = {}
    per_class_count: dict[int, int] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        features = F.normalize(model.extract_features(images).float(), dim=1)
        predictions = (features @ means.T).argmax(dim=1)
        correct = predictions.eq(targets)
        total_correct += int(correct.sum().item())
        total_count += int(targets.numel())
        old_mask = targets < int(old_class_count)
        new_mask = ~old_mask
        old_correct += int(correct[old_mask].sum().item())
        old_count += int(old_mask.sum().item())
        new_correct += int(correct[new_mask].sum().item())
        new_count += int(new_mask.sum().item())
        original_targets = batch["original_target"].long()
        cpu_correct = correct.cpu()
        for class_id in original_targets.unique().tolist():
            mask = original_targets == int(class_id)
            per_class_correct[int(class_id)] = (
                per_class_correct.get(int(class_id), 0)
                + int(cpu_correct[mask].sum().item())
            )
            per_class_count[int(class_id)] = (
                per_class_count.get(int(class_id), 0) + int(mask.sum().item())
            )
    if was_training:
        model.train()
    if total_count == 0 or new_count == 0:
        raise ValueError("cannot evaluate an empty NME split")
    old_accuracy = _safe_accuracy(old_correct, old_count)
    new_accuracy = new_correct / new_count
    harmonic = None
    if old_accuracy is not None:
        denominator = old_accuracy + new_accuracy
        harmonic = (
            0.0
            if denominator == 0
            else 2.0 * old_accuracy * new_accuracy / denominator
        )
    return EvaluationResult(
        accuracy=total_correct / total_count,
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy,
        harmonic_mean=harmonic,
        per_class_accuracy={
            class_id: per_class_correct[class_id] / count
            for class_id, count in per_class_count.items()
        },
        sample_count=total_count,
    )
