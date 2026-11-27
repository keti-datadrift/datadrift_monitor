from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from sacil.hierarchy import HierarchyTree


@dataclass(frozen=True)
class PrototypeTransportResult:
    prototypes: Tensor
    drifts: Tensor
    drift_norms: Tensor
    support_counts: Tensor


def orthogonal_procrustes_transport(
    prototypes: Tensor,
    old_features: Tensor,
    current_features: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, float]:
    """Rotate stored prototypes into the current feature frame.

    The row-vector rotation minimizes the paired exemplar residual and is
    orthogonal, so every cosine relation among transported prototypes is
    preserved up to numerical precision.
    """

    if prototypes.ndim != 2 or old_features.ndim != 2:
        raise ValueError("prototypes and Procrustes features must be matrices")
    if old_features.shape != current_features.shape:
        raise ValueError("Procrustes feature pairs must have one shape")
    if prototypes.shape[1] != old_features.shape[1]:
        raise ValueError("Procrustes feature dimensions do not match")
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    cross_covariance = old.T @ current
    left, _, right_h = torch.linalg.svd(cross_covariance, full_matrices=False)
    rotation = left @ right_h
    transported = F.normalize(
        F.normalize(prototypes.detach().float(), dim=1) @ rotation,
        dim=1,
        eps=float(epsilon),
    )
    residual = float(
        (old @ rotation - current).square().sum(dim=1).mean().item()
    )
    return transported, rotation, residual


def rigid_procrustes_transport(
    prototypes: Tensor,
    old_features: Tensor,
    current_features: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Fit a centered orthogonal map plus translation from exemplar pairs."""

    if prototypes.ndim != 2 or old_features.ndim != 2:
        raise ValueError("prototypes and rigid features must be matrices")
    if old_features.shape != current_features.shape:
        raise ValueError("rigid feature pairs must have one shape")
    if prototypes.shape[1] != old_features.shape[1]:
        raise ValueError("rigid feature dimensions do not match")
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    old_center = old.mean(dim=0, keepdim=True)
    current_center = current.mean(dim=0, keepdim=True)
    old_centered = old - old_center
    current_centered = current - current_center
    left, _, right_h = torch.linalg.svd(
        old_centered.T @ current_centered, full_matrices=False
    )
    rotation = left @ right_h
    translation = (current_center - old_center @ rotation)[0]
    transported = F.normalize(
        F.normalize(prototypes.detach().float(), dim=1) @ rotation
        + translation,
        dim=1,
        eps=float(epsilon),
    )
    residual = float(
        (old @ rotation + translation - current)
        .square()
        .sum(dim=1)
        .mean()
        .item()
    )
    return transported, rotation, translation, residual


def weighted_rigid_procrustes_transport(
    prototypes: Tensor,
    old_features: Tensor,
    current_features: Tensor,
    targets: Tensor,
    *,
    sigma: float = 0.2,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, Tensor, float, Tensor]:
    """Fit a rigid frame while downweighting non-representative exemplars."""

    if old_features.shape[0] != targets.numel():
        raise ValueError("weighted rigid feature and target counts differ")
    if prototypes.ndim != 2 or old_features.ndim != 2:
        raise ValueError("prototypes and weighted rigid features must be matrices")
    if old_features.shape != current_features.shape:
        raise ValueError("weighted rigid feature pairs must have one shape")
    if prototypes.shape[1] != old_features.shape[1]:
        raise ValueError("weighted rigid feature dimensions do not match")
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    labels = targets.detach().long().to(old.device)
    num_classes = prototypes.shape[0]
    if set(int(v) for v in labels.unique().tolist()) != set(range(num_classes)):
        raise ValueError("weighted rigid targets must cover contiguous old labels")
    weights = _representative_weights(
        old, labels, num_classes=num_classes, sigma=float(sigma)
    )
    weights = weights / weights.sum().clamp_min(float(epsilon))
    old_center = (weights[:, None] * old).sum(dim=0, keepdim=True)
    current_center = (weights[:, None] * current).sum(dim=0, keepdim=True)
    old_centered = old - old_center
    current_centered = current - current_center
    left, _, right_h = torch.linalg.svd(
        old_centered.T @ (weights[:, None] * current_centered),
        full_matrices=False,
    )
    rotation = left @ right_h
    translation = (current_center - old_center @ rotation)[0]
    transported = F.normalize(
        F.normalize(prototypes.detach().float(), dim=1) @ rotation
        + translation,
        dim=1,
        eps=float(epsilon),
    )
    residual = float(
        (
            weights
            * (old @ rotation + translation - current).square().sum(dim=1)
        ).sum().item()
    )
    return transported, rotation, translation, residual, weights


def similarity_procrustes_transport(
    prototypes: Tensor,
    old_features: Tensor,
    current_features: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, Tensor, float, float]:
    """Fit a rotation, isotropic scale, and translation to paired features."""

    if prototypes.ndim != 2 or old_features.ndim != 2:
        raise ValueError("prototypes and similarity features must be matrices")
    if old_features.shape != current_features.shape:
        raise ValueError("similarity feature pairs must have one shape")
    if prototypes.shape[1] != old_features.shape[1]:
        raise ValueError("similarity feature dimensions do not match")
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    old_center = old.mean(dim=0, keepdim=True)
    current_center = current.mean(dim=0, keepdim=True)
    old_centered = old - old_center
    current_centered = current - current_center
    left, singular_values, right_h = torch.linalg.svd(
        old_centered.T @ current_centered,
        full_matrices=False,
    )
    rotation = left @ right_h
    scale = singular_values.sum() / old_centered.square().sum().clamp_min(
        float(epsilon)
    )
    translation = (current_center - scale * old_center @ rotation)[0]
    transported = F.normalize(
        scale * F.normalize(prototypes.detach().float(), dim=1) @ rotation
        + translation,
        dim=1,
        eps=float(epsilon),
    )
    residual = float(
        (
            scale * old @ rotation + translation - current
        ).square().sum(dim=1).mean().item()
    )
    return transported, rotation, translation, float(scale.item()), residual


def empirical_bayes_residual_transport(
    prototypes: Tensor,
    old_features: Tensor,
    current_features: Tensor,
    targets: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, Tensor, float, float, Tensor]:
    """Relax a global rigid map by a data-estimated class residual.

    The rigid map is the shared co-moving frame.  Paired replay exemplars then
    estimate each class's remaining drift.  A single empirical-Bayes factor
    shrinks those noisy 20-shot residual means toward zero; no validation or
    test labels are used to choose the relaxation strength.
    """

    if old_features.shape[0] != targets.numel():
        raise ValueError("residual transport feature and target counts differ")
    transported, rotation, translation, residual = rigid_procrustes_transport(
        prototypes,
        old_features,
        current_features,
        epsilon=epsilon,
    )
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    fitted = F.normalize(old @ rotation + translation, dim=1)
    sample_residual = current - fitted
    labels = targets.detach().long().to(sample_residual.device)
    num_classes = prototypes.shape[0]
    if set(int(v) for v in labels.unique().tolist()) != set(range(num_classes)):
        raise ValueError("residual targets must cover contiguous old labels")

    class_means: list[Tensor] = []
    mean_noise_energies: list[Tensor] = []
    for class_index in range(num_classes):
        values = sample_residual[labels == class_index]
        mean = values.mean(dim=0)
        class_means.append(mean)
        if values.shape[0] > 1:
            # Trace of the covariance of the sample mean.
            centered_energy = (values - mean).square().sum()
            mean_noise = centered_energy / (
                float(values.shape[0]) * float(values.shape[0] - 1)
            )
        else:
            mean_noise = torch.zeros((), device=values.device)
        mean_noise_energies.append(mean_noise)

    residual_means = torch.stack(class_means, dim=0)
    noise_energy = torch.stack(mean_noise_energies).mean()
    centered_means = residual_means - residual_means.mean(dim=0, keepdim=True)
    observed_between = centered_means.square().sum(dim=1).mean()
    signal_energy = (observed_between - noise_energy).clamp_min(0.0)
    shrinkage = signal_energy / (signal_energy + noise_energy).clamp_min(
        float(epsilon)
    )
    relaxed = F.normalize(
        transported + shrinkage * residual_means.to(transported.device),
        dim=1,
        eps=float(epsilon),
    )
    return (
        relaxed,
        rotation,
        translation,
        residual,
        float(shrinkage.item()),
        residual_means,
    )


def affine_ridge_transport(
    prototypes: Tensor,
    old_features: Tensor,
    current_features: Tensor,
    *,
    ridge: float = 1.0e-2,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, float]:
    """Fit a regularized affine feature-frame map from exemplar pairs."""

    if ridge <= 0:
        raise ValueError("affine ridge must be positive")
    if prototypes.ndim != 2 or old_features.ndim != 2:
        raise ValueError("prototypes and affine features must be matrices")
    if old_features.shape != current_features.shape:
        raise ValueError("affine feature pairs must have one shape")
    if prototypes.shape[1] != old_features.shape[1]:
        raise ValueError("affine feature dimensions do not match")
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    design = torch.cat(
        [old, torch.ones(old.shape[0], 1, device=old.device)], dim=1
    )
    regularizer = torch.eye(
        design.shape[1], device=design.device, dtype=design.dtype
    ) * float(ridge)
    regularizer[-1, -1] = 0.0
    mapping = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ current,
    )
    prototype_design = torch.cat(
        [
            F.normalize(prototypes.detach().float(), dim=1),
            torch.ones(
                prototypes.shape[0],
                1,
                device=prototypes.device,
                dtype=prototypes.dtype,
            ),
        ],
        dim=1,
    )
    transported = F.normalize(
        prototype_design @ mapping, dim=1, eps=float(epsilon)
    )
    residual = float(
        (design @ mapping - current).square().sum(dim=1).mean().item()
    )
    return transported, mapping, residual


def _representative_weights(
    old_features: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    sigma: float,
) -> Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    weights = torch.empty(
        old_features.shape[0],
        device=old_features.device,
        dtype=old_features.dtype,
    )
    denominator = 2.0 * float(sigma) ** 2
    for class_index in range(num_classes):
        mask = targets == class_index
        if not bool(mask.any()):
            raise ValueError(f"missing transport support for class {class_index}")
        centered = old_features[mask] - old_features[mask].mean(
            dim=0, keepdim=True
        )
        squared_distance = centered.square().sum(dim=1)
        weights[mask] = torch.exp(-squared_distance / denominator)
    return weights


def transport_class_prototypes(
    prototypes: Tensor,
    old_features: Tensor,
    current_features: Tensor,
    targets: Tensor,
    class_ids: Sequence[int],
    *,
    mode: str = "global",
    tree: HierarchyTree | None = None,
    sigma: float = 0.2,
    epsilon: float = 1.0e-12,
) -> PrototypeTransportResult:
    """Move stored class prototypes with exemplar-observed feature drift.

    ``global`` reproduces the SDC principle with one robust drift vector.
    ``class`` estimates one drift from each class's retained exemplars.
    ``hierarchy_parent`` pools drift only within each leaf's immediate learned
    parent, yielding a co-moving local transport without pinning an absolute
    coordinate.
    """

    if prototypes.ndim != 2 or old_features.ndim != 2:
        raise ValueError("prototypes and features must be matrices")
    if old_features.shape != current_features.shape:
        raise ValueError("old and current features must have one shape")
    if old_features.shape[0] != targets.numel():
        raise ValueError("transport feature and target counts do not match")
    if old_features.shape[1] != prototypes.shape[1]:
        raise ValueError("transport feature dimensions do not match")
    if prototypes.shape[0] != len(class_ids):
        raise ValueError("class ID and prototype counts do not match")
    mode = str(mode).lower()
    if mode not in {"global", "class", "hierarchy_parent"}:
        raise ValueError(
            "transport mode must be global, class, or hierarchy_parent"
        )
    if mode == "hierarchy_parent" and tree is None:
        raise ValueError("hierarchy_parent transport requires a tree")

    values = F.normalize(prototypes.detach().float(), dim=1)
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    targets = targets.detach().long().to(old.device)
    num_classes = values.shape[0]
    if set(int(v) for v in targets.unique().tolist()) != set(range(num_classes)):
        raise ValueError("transport targets must cover contiguous old labels")
    weights = _representative_weights(
        old, targets, num_classes=num_classes, sigma=float(sigma)
    )
    sample_drift = current - old
    class_ids = tuple(int(value) for value in class_ids)

    drifts: list[Tensor] = []
    supports: list[int] = []
    global_drift = (weights[:, None] * sample_drift).sum(dim=0) / weights.sum().clamp_min(
        float(epsilon)
    )
    class_position = {
        class_id: position for position, class_id in enumerate(class_ids)
    }
    for position, class_id in enumerate(class_ids):
        if mode == "global":
            mask = torch.ones_like(targets, dtype=torch.bool)
            drift = global_drift
        elif mode == "class":
            mask = targets == position
            local_weights = weights[mask]
            drift = (
                local_weights[:, None] * sample_drift[mask]
            ).sum(dim=0) / local_weights.sum().clamp_min(float(epsilon))
        else:
            assert tree is not None
            parent_id = tree.parent(tree.leaf_node_id(class_id))
            members = (
                (class_id,)
                if parent_id is None
                else tree.descendants(parent_id)
            )
            positions = torch.tensor(
                [class_position[int(member)] for member in members],
                device=targets.device,
                dtype=targets.dtype,
            )
            mask = (targets[:, None] == positions[None, :]).any(dim=1)
            local_weights = weights[mask]
            drift = (
                local_weights[:, None] * sample_drift[mask]
            ).sum(dim=0) / local_weights.sum().clamp_min(float(epsilon))
        drifts.append(drift)
        supports.append(int(mask.sum().item()))

    drift_matrix = torch.stack(drifts, dim=0).to(values.device)
    transported = F.normalize(values + drift_matrix, dim=1)
    return PrototypeTransportResult(
        prototypes=transported,
        drifts=drift_matrix,
        drift_norms=drift_matrix.norm(dim=1),
        support_counts=torch.tensor(supports, dtype=torch.long),
    )
