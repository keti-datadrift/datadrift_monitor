from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class MomentTransportMap:
    affine_mapping: Tensor
    source_center: Tensor
    projection: Tensor
    quadratic_mean: Tensor
    quadratic_mapping: Tensor
    residual_covariance: Tensor
    affine_fit_residual: float
    quadratic_fit_residual: float
    rank: int


@dataclass(frozen=True)
class PolynomialKernelMomentTransportMap:
    affine_mapping: Tensor
    source_center: Tensor
    support_centered: Tensor
    support_kernel_mean: Tensor
    kernel_grand_mean: Tensor
    dual_mapping: Tensor
    residual_covariance: Tensor
    affine_fit_residual: float
    quadratic_fit_residual: float
    support_count: int


def class_first_second_moments(
    features: Tensor,
    targets: Tensor,
    class_ids: Tensor | list[int] | tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """Return E[z] and E[z^T z] for every requested class."""

    values = features.detach().cpu().float()
    labels = targets.detach().cpu().long().flatten()
    if values.ndim != 2 or values.shape[0] != labels.numel():
        raise ValueError("moment features and labels do not align")
    means: list[Tensor] = []
    seconds: list[Tensor] = []
    for class_id in [int(value) for value in class_ids]:
        class_values = values[labels == class_id]
        if class_values.shape[0] < 2:
            raise ValueError(f"class {class_id} has insufficient moment data")
        means.append(class_values.mean(dim=0))
        seconds.append(
            torch.einsum("nd,ne->de", class_values, class_values)
            / float(class_values.shape[0])
        )
    return torch.stack(means), torch.stack(seconds)


def _quadratic_features(values: Tensor) -> Tensor:
    if values.ndim != 2:
        raise ValueError("quadratic coordinates must be a matrix")
    row, column = torch.triu_indices(
        values.shape[1], values.shape[1], device=values.device
    )
    return values[:, row] * values[:, column]


def _quadratic_moment_features(projected_seconds: Tensor) -> Tensor:
    if projected_seconds.ndim != 3:
        raise ValueError("projected second moments must be [C,R,R]")
    row, column = torch.triu_indices(
        projected_seconds.shape[1],
        projected_seconds.shape[2],
        device=projected_seconds.device,
    )
    return projected_seconds[:, row, column]


def fit_low_rank_moment_transport(
    old_features: Tensor,
    current_features: Tensor,
    *,
    rank: int = 6,
    affine_ridge: float = 1.0e-2,
    quadratic_ridge: float = 1.0,
) -> MomentTransportMap:
    """Fit affine drift plus a low-rank quadratic conditional mean."""

    old = old_features.detach().float()
    current = current_features.detach().float().to(old.device)
    if old.ndim != 2 or old.shape != current.shape:
        raise ValueError("moment-transport feature pairs must have one shape")
    if affine_ridge <= 0.0 or quadratic_ridge <= 0.0:
        raise ValueError("moment-transport ridges must be positive")
    selected_rank = min(int(rank), old.shape[1], old.shape[0] - 1)
    if selected_rank <= 0:
        raise ValueError("moment-transport rank must be positive")

    design = torch.cat(
        [old, torch.ones(old.shape[0], 1, device=old.device)], dim=1
    )
    regularizer = torch.eye(
        design.shape[1], device=old.device, dtype=old.dtype
    ) * float(affine_ridge)
    regularizer[-1, -1] = 0.0
    affine = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ current,
    )
    affine_prediction = design @ affine
    residual = current - affine_prediction
    affine_fit = float(residual.square().sum(dim=1).mean().item())

    source_center = old.mean(dim=0)
    centered = old - source_center
    _, _, right_h = torch.linalg.svd(centered, full_matrices=False)
    projection = right_h[:selected_rank].T.contiguous()
    coordinates = centered @ projection
    quadratic = _quadratic_features(coordinates)
    quadratic_mean = quadratic.mean(dim=0)
    quadratic_centered = quadratic - quadratic_mean
    quadratic_regularizer = torch.eye(
        quadratic_centered.shape[1],
        device=old.device,
        dtype=old.dtype,
    ) * float(quadratic_ridge)
    quadratic_mapping = torch.linalg.solve(
        quadratic_centered.T @ quadratic_centered
        + quadratic_regularizer,
        quadratic_centered.T @ residual,
    )
    corrected_residual = residual - quadratic_centered @ quadratic_mapping
    quadratic_fit = float(
        corrected_residual.square().sum(dim=1).mean().item()
    )
    residual_covariance = (
        corrected_residual.T @ corrected_residual
        / float(corrected_residual.shape[0])
    )
    return MomentTransportMap(
        affine_mapping=affine,
        source_center=source_center,
        projection=projection,
        quadratic_mean=quadratic_mean,
        quadratic_mapping=quadratic_mapping,
        residual_covariance=residual_covariance,
        affine_fit_residual=affine_fit,
        quadratic_fit_residual=quadratic_fit,
        rank=selected_rank,
    )


def fit_low_rank_moment_transport_grid(
    old_features: Tensor,
    current_features: Tensor,
    *,
    ranks: tuple[int, ...] | list[int],
    quadratic_ridges: tuple[float, ...] | list[float],
    affine_ridge: float = 1.0e-2,
) -> dict[tuple[int, float], MomentTransportMap]:
    """Fit several low-rank maps while sharing affine fit and source SVD."""

    old = old_features.detach().float()
    current = current_features.detach().float().to(old.device)
    if old.ndim != 2 or old.shape != current.shape:
        raise ValueError("moment-transport feature pairs must have one shape")
    selected_ranks = tuple(sorted({int(value) for value in ranks}))
    selected_ridges = tuple(sorted({float(value) for value in quadratic_ridges}))
    if not selected_ranks or any(value <= 0 for value in selected_ranks):
        raise ValueError("moment-transport grid ranks must be positive")
    if not selected_ridges or any(value <= 0.0 for value in selected_ridges):
        raise ValueError("moment-transport grid ridges must be positive")
    if affine_ridge <= 0.0:
        raise ValueError("moment-transport affine ridge must be positive")

    design = torch.cat(
        [old, torch.ones(old.shape[0], 1, device=old.device)], dim=1
    )
    regularizer = torch.eye(
        design.shape[1], device=old.device, dtype=old.dtype
    ) * float(affine_ridge)
    regularizer[-1, -1] = 0.0
    affine = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ current,
    )
    residual = current - design @ affine
    affine_fit = float(residual.square().sum(dim=1).mean().item())
    source_center = old.mean(dim=0)
    centered = old - source_center
    _, _, right_h = torch.linalg.svd(centered, full_matrices=False)

    mappings: dict[tuple[int, float], MomentTransportMap] = {}
    for requested_rank in selected_ranks:
        rank = min(requested_rank, old.shape[1], old.shape[0] - 1)
        if rank <= 0:
            raise ValueError("moment-transport grid rank must be positive")
        projection = right_h[:rank].T.contiguous()
        coordinates = centered @ projection
        quadratic = _quadratic_features(coordinates)
        quadratic_mean = quadratic.mean(dim=0)
        quadratic_centered = quadratic - quadratic_mean
        gram = quadratic_centered.T @ quadratic_centered
        target = quadratic_centered.T @ residual
        identity = torch.eye(
            gram.shape[0], device=old.device, dtype=old.dtype
        )
        for ridge in selected_ridges:
            quadratic_mapping = torch.linalg.solve(
                gram + float(ridge) * identity,
                target,
            )
            corrected_residual = (
                residual - quadratic_centered @ quadratic_mapping
            )
            quadratic_fit = float(
                corrected_residual.square().sum(dim=1).mean().item()
            )
            residual_covariance = (
                corrected_residual.T @ corrected_residual
                / float(corrected_residual.shape[0])
            )
            mappings[(requested_rank, ridge)] = MomentTransportMap(
                affine_mapping=affine,
                source_center=source_center,
                projection=projection,
                quadratic_mean=quadratic_mean,
                quadratic_mapping=quadratic_mapping,
                residual_covariance=residual_covariance,
                affine_fit_residual=affine_fit,
                quadratic_fit_residual=quadratic_fit,
                rank=rank,
            )
    return mappings


def fit_polynomial_kernel_moment_transport(
    old_features: Tensor,
    current_features: Tensor,
    *,
    affine_ridge: float = 1.0e-2,
    quadratic_ridge: float = 1.0,
) -> PolynomialKernelMomentTransportMap:
    """Fit affine drift plus full-space homogeneous degree-2 kernel drift."""

    old = old_features.detach().float()
    current = current_features.detach().float().to(old.device)
    if old.ndim != 2 or old.shape != current.shape:
        raise ValueError("moment-transport feature pairs must have one shape")
    if old.shape[0] < 2:
        raise ValueError("kernel moment transport needs at least two pairs")
    if affine_ridge <= 0.0 or quadratic_ridge <= 0.0:
        raise ValueError("moment-transport ridges must be positive")

    design = torch.cat(
        [old, torch.ones(old.shape[0], 1, device=old.device)], dim=1
    )
    regularizer = torch.eye(
        design.shape[1], device=old.device, dtype=old.dtype
    ) * float(affine_ridge)
    regularizer[-1, -1] = 0.0
    affine = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ current,
    )
    affine_prediction = design @ affine
    residual = current - affine_prediction
    affine_fit = float(residual.square().sum(dim=1).mean().item())

    source_center = old.mean(dim=0)
    support_centered = old - source_center
    kernel = (support_centered @ support_centered.T).square()
    support_kernel_mean = kernel.mean(dim=0)
    kernel_grand_mean = kernel.mean()
    centered_kernel = (
        kernel
        - kernel.mean(dim=1, keepdim=True)
        - support_kernel_mean.unsqueeze(0)
        + kernel_grand_mean
    )
    kernel_regularizer = torch.eye(
        centered_kernel.shape[0],
        device=old.device,
        dtype=old.dtype,
    ) * float(quadratic_ridge)
    dual_mapping = torch.linalg.solve(
        centered_kernel + kernel_regularizer,
        residual,
    )
    corrected_residual = residual - centered_kernel @ dual_mapping
    quadratic_fit = float(
        corrected_residual.square().sum(dim=1).mean().item()
    )
    residual_covariance = (
        corrected_residual.T @ corrected_residual
        / float(corrected_residual.shape[0])
    )
    return PolynomialKernelMomentTransportMap(
        affine_mapping=affine,
        source_center=source_center,
        support_centered=support_centered,
        support_kernel_mean=support_kernel_mean,
        kernel_grand_mean=kernel_grand_mean,
        dual_mapping=dual_mapping,
        residual_covariance=residual_covariance,
        affine_fit_residual=affine_fit,
        quadratic_fit_residual=quadratic_fit,
        support_count=int(old.shape[0]),
    )


def apply_polynomial_kernel_moment_transport(
    means: Tensor,
    second_moments: Tensor,
    mapping: PolynomialKernelMomentTransportMap,
    *,
    residual_covariance_scale: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Propagate class moments with a full-space degree-2 kernel map."""

    device = mapping.support_centered.device
    old_means = means.detach().float().to(device)
    old_seconds = second_moments.detach().float().to(device)
    if old_means.ndim != 2 or old_seconds.shape != (
        old_means.shape[0],
        old_means.shape[1],
        old_means.shape[1],
    ):
        raise ValueError("class first/second moments have incompatible shapes")
    affine = mapping.affine_mapping.to(device)
    linear = affine[:-1]
    bias = affine[-1]
    center = mapping.source_center.to(device)
    support = mapping.support_centered.to(device)

    affine_means = old_means @ linear + bias
    centered_seconds = (
        old_seconds
        - torch.einsum("cd,e->cde", old_means, center)
        - torch.einsum("d,ce->cde", center, old_means)
        + torch.einsum("d,e->de", center, center).unsqueeze(0)
    )
    expected_kernel = torch.einsum(
        "nd,cde,ne->cn", support, centered_seconds, support
    )
    expected_centered_kernel = (
        expected_kernel
        - expected_kernel.mean(dim=1, keepdim=True)
        - mapping.support_kernel_mean.to(device).unsqueeze(0)
        + mapping.kernel_grand_mean.to(device)
    )
    correction = expected_centered_kernel @ mapping.dual_mapping.to(device)
    transported_means = affine_means + correction

    old_covariances = (
        old_seconds
        - torch.einsum("cd,ce->cde", old_means, old_means)
    )
    transported_covariances = torch.einsum(
        "dr,cde,es->crs", linear, old_covariances, linear
    )
    scale = float(residual_covariance_scale)
    if scale < 0.0:
        raise ValueError("residual covariance scale must be non-negative")
    if scale > 0.0:
        transported_covariances = transported_covariances + scale * (
            mapping.residual_covariance.to(device).unsqueeze(0)
        )
    transported_covariances = 0.5 * (
        transported_covariances
        + transported_covariances.transpose(1, 2)
    )
    transported_seconds = transported_covariances + torch.einsum(
        "cd,ce->cde", transported_means, transported_means
    )
    prototypes = F.normalize(transported_means, dim=1)
    return transported_means, transported_seconds, prototypes


def apply_low_rank_moment_transport(
    means: Tensor,
    second_moments: Tensor,
    mapping: MomentTransportMap,
    *,
    residual_covariance_scale: float = 0.0,
    correction_scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Propagate class moments and return normalized transported means."""

    old_means = means.detach().float()
    old_seconds = second_moments.detach().float().to(old_means.device)
    if old_means.ndim != 2 or old_seconds.shape != (
        old_means.shape[0],
        old_means.shape[1],
        old_means.shape[1],
    ):
        raise ValueError("class first/second moments have incompatible shapes")
    affine = mapping.affine_mapping.to(old_means.device)
    linear = affine[:-1]
    bias = affine[-1]
    center = mapping.source_center.to(old_means.device)
    projection = mapping.projection.to(old_means.device)

    affine_means = old_means @ linear + bias
    centered_seconds = (
        old_seconds
        - torch.einsum("cd,e->cde", old_means, center)
        - torch.einsum("d,ce->cde", center, old_means)
        + torch.einsum("d,e->de", center, center).unsqueeze(0)
    )
    projected_seconds = torch.einsum(
        "dr,cde,es->crs", projection, centered_seconds, projection
    )
    quadratic_expectation = _quadratic_moment_features(projected_seconds)
    quadratic_centered = (
        quadratic_expectation
        - mapping.quadratic_mean.to(old_means.device)
    )
    scale = float(correction_scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("quadratic correction scale must lie in [0, 1]")
    correction = scale * (
        quadratic_centered
        @ mapping.quadratic_mapping.to(old_means.device)
    )
    transported_means = affine_means + correction

    old_covariances = (
        old_seconds
        - torch.einsum("cd,ce->cde", old_means, old_means)
    )
    transported_covariances = torch.einsum(
        "dr,cde,es->crs", linear, old_covariances, linear
    )
    covariance_scale = float(residual_covariance_scale)
    if covariance_scale < 0.0:
        raise ValueError("residual covariance scale must be non-negative")
    if covariance_scale > 0.0:
        transported_covariances = transported_covariances + covariance_scale * (
            mapping.residual_covariance.to(old_means.device).unsqueeze(0)
        )
    transported_covariances = 0.5 * (
        transported_covariances
        + transported_covariances.transpose(1, 2)
    )
    transported_seconds = transported_covariances + torch.einsum(
        "cd,ce->cde", transported_means, transported_means
    )
    prototypes = F.normalize(transported_means, dim=1)
    return transported_means, transported_seconds, prototypes
