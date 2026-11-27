from __future__ import annotations

import math

import torch
from torch import Tensor

from .persistent_quadrature import project_probability_simplex


CALIBRATION_MODES = ("mean", "second", "combined")


def _relative_moment_errors(
    features: Tensor,
    weights: Tensor,
    target_mean: Tensor,
    target_second: Tensor,
    *,
    epsilon: float,
) -> tuple[float, float]:
    mean = weights @ features
    centered = features - target_mean.unsqueeze(0)
    covariance = target_second - torch.outer(target_mean, target_mean)
    covariance = 0.5 * (covariance + covariance.T)
    estimated_covariance = torch.einsum(
        "n,nd,ne->de", weights, centered, centered
    )
    mean_error = float(
        ((mean - target_mean).square().sum()
        / target_mean.square().sum().clamp_min(epsilon)).item()
    )
    second_error = float(
        ((estimated_covariance - covariance).square().sum()
        / covariance.square().sum().clamp_min(epsilon)).item()
    )
    return mean_error, second_error


def fit_moment_calibration_weights(
    exemplar_features: Tensor,
    targets: Tensor,
    target_means: Tensor,
    target_second_moments: Tensor,
    *,
    mode: str,
    uniform_ridge: float = 1.0e-3,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-10,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, dict[str, object]]:
    """Fit class-wise population-moment calibration weights.

    The optimization is performed on the probability simplex.  Mean matching
    uses a degree-1 Gram matrix.  Covariance matching uses the exact full-space
    degree-2 Gram matrix

        <x_i x_i^T, x_j x_j^T>_F = (x_i^T x_j)^2,

    so no PCA rank or explicit D-by-D vectorization is required.
    """

    selected_mode = str(mode).lower()
    if selected_mode not in CALIBRATION_MODES:
        raise ValueError(
            f"moment calibration mode must be one of {CALIBRATION_MODES}"
        )
    features = exemplar_features.detach().cpu().float()
    labels = targets.detach().cpu().long().flatten()
    means = target_means.detach().cpu().float()
    seconds = target_second_moments.detach().cpu().float()
    if features.ndim != 2 or labels.numel() != features.shape[0]:
        raise ValueError("calibration features and targets do not align")
    if means.ndim != 2 or means.shape[1] != features.shape[1]:
        raise ValueError("calibration target means have an invalid shape")
    if seconds.shape != (means.shape[0], means.shape[1], means.shape[1]):
        raise ValueError("calibration second moments have an invalid shape")
    if float(uniform_ridge) < 0.0:
        raise ValueError("moment calibration uniform ridge must be non-negative")
    if int(max_iterations) <= 0:
        raise ValueError("moment calibration max_iterations must be positive")
    if float(tolerance) < 0.0 or float(epsilon) <= 0.0:
        raise ValueError("moment calibration tolerances are invalid")

    class_ids = labels.unique(sorted=True)
    if class_ids.tolist() != list(range(means.shape[0])):
        raise ValueError(
            "moment calibration requires contiguous old-class targets"
        )
    all_weights = torch.empty(features.shape[0], dtype=torch.float32)
    class_diagnostics: list[dict[str, float | int]] = []
    for class_id in class_ids.tolist():
        indices = torch.nonzero(labels == int(class_id)).flatten()
        values = features[indices]
        count = values.shape[0]
        if count < 2:
            raise ValueError(
                f"class {class_id} has insufficient calibration exemplars"
            )
        uniform = torch.full((count,), 1.0 / float(count))
        target_mean = means[class_id]
        target_second = seconds[class_id]
        gram = torch.zeros(count, count)
        rhs = torch.zeros(count)

        if selected_mode in {"mean", "combined"}:
            mean_scale = float(
                target_mean.square().sum().clamp_min(epsilon).item()
            )
            gram = gram + (values @ values.T) / mean_scale
            rhs = rhs + (values @ target_mean) / mean_scale

        if selected_mode in {"second", "combined"}:
            covariance = target_second - torch.outer(
                target_mean, target_mean
            )
            covariance = 0.5 * (covariance + covariance.T)
            centered = values - target_mean.unsqueeze(0)
            covariance_scale = float(
                covariance.square().sum().clamp_min(epsilon).item()
            )
            gram = gram + (centered @ centered.T).square() / covariance_scale
            rhs = rhs + torch.einsum(
                "nd,de,ne->n", centered, covariance, centered
            ) / covariance_scale

        ridge = float(uniform_ridge)
        if ridge > 0.0:
            gram = gram + ridge * torch.eye(count)
            rhs = rhs + ridge * uniform
        largest = float(torch.linalg.eigvalsh(gram).max().item())
        step = 1.0 / max(2.0 * largest, epsilon)
        weights = uniform.clone()
        momentum = weights.clone()
        acceleration = 1.0
        completed = 0
        for iteration in range(int(max_iterations)):
            gradient = 2.0 * (gram @ momentum - rhs)
            updated = project_probability_simplex(
                momentum - step * gradient
            )
            completed = iteration + 1
            if float((updated - weights).abs().max().item()) <= tolerance:
                weights = updated
                break
            next_acceleration = 0.5 * (
                1.0 + math.sqrt(1.0 + 4.0 * acceleration**2)
            )
            momentum = updated + (
                (acceleration - 1.0) / next_acceleration
            ) * (updated - weights)
            weights = updated
            acceleration = next_acceleration

        uniform_mean_error, uniform_second_error = _relative_moment_errors(
            values,
            uniform,
            target_mean,
            target_second,
            epsilon=epsilon,
        )
        weighted_mean_error, weighted_second_error = _relative_moment_errors(
            values,
            weights,
            target_mean,
            target_second,
            epsilon=epsilon,
        )
        positive = weights[weights > 0.0]
        entropy = float(
            (
                -(positive * positive.log()).sum()
                / math.log(float(count))
            ).item()
        )
        all_weights[indices] = weights
        class_diagnostics.append(
            {
                "class_id": int(class_id),
                "iterations": int(completed),
                "effective_sample_size": float(
                    1.0 / weights.square().sum().clamp_min(epsilon).item()
                ),
                "normalized_entropy": entropy,
                "minimum_weight": float(weights.min().item()),
                "maximum_weight": float(weights.max().item()),
                "uniform_mean_relative_error": uniform_mean_error,
                "weighted_mean_relative_error": weighted_mean_error,
                "uniform_second_relative_error": uniform_second_error,
                "weighted_second_relative_error": weighted_second_error,
            }
        )

    def average(key: str) -> float:
        return sum(float(item[key]) for item in class_diagnostics) / len(
            class_diagnostics
        )

    diagnostics: dict[str, object] = {
        "mode": selected_mode,
        "class_count": len(class_diagnostics),
        "support_count": int(features.shape[0]),
        "uniform_ridge": float(uniform_ridge),
        "mean_effective_sample_size": average("effective_sample_size"),
        "mean_normalized_entropy": average("normalized_entropy"),
        "mean_uniform_mean_relative_error": average(
            "uniform_mean_relative_error"
        ),
        "mean_weighted_mean_relative_error": average(
            "weighted_mean_relative_error"
        ),
        "mean_uniform_second_relative_error": average(
            "uniform_second_relative_error"
        ),
        "mean_weighted_second_relative_error": average(
            "weighted_second_relative_error"
        ),
        "classes": class_diagnostics,
    }
    return all_weights, diagnostics


def expand_exemplar_weights(
    exemplar_weights: Tensor,
    fit_support_count: int,
) -> Tensor:
    """Repeat weights for a concatenated regular/flip affine support."""

    weights = exemplar_weights.detach().float().flatten()
    count = int(fit_support_count)
    if count == weights.numel():
        return weights
    if count == 2 * weights.numel():
        return torch.cat([weights, weights], dim=0)
    raise ValueError("calibration weights do not align with affine support")


def transport_first_second_moments_affine(
    means: Tensor,
    second_moments: Tensor,
    mapping: Tensor,
) -> tuple[Tensor, Tensor]:
    """Transport population first/second moments through y=xA+b."""

    source_means = means.detach().float()
    source_seconds = second_moments.detach().float().to(source_means.device)
    affine = mapping.detach().float().to(source_means.device)
    if source_means.ndim != 2:
        raise ValueError("affine moment means must be a matrix")
    if source_seconds.shape != (
        source_means.shape[0],
        source_means.shape[1],
        source_means.shape[1],
    ):
        raise ValueError("affine moment second moments have an invalid shape")
    if affine.shape != (source_means.shape[1] + 1, source_means.shape[1]):
        raise ValueError("affine moment mapping has an invalid shape")
    linear = affine[:-1]
    bias = affine[-1]
    covariances = source_seconds - torch.einsum(
        "cd,ce->cde", source_means, source_means
    )
    covariances = 0.5 * (covariances + covariances.transpose(1, 2))
    transported_means = source_means @ linear + bias
    transported_covariances = torch.einsum(
        "dr,cde,es->crs", linear, covariances, linear
    )
    transported_covariances = 0.5 * (
        transported_covariances
        + transported_covariances.transpose(1, 2)
    )
    transported_seconds = transported_covariances + torch.einsum(
        "cd,ce->cde", transported_means, transported_means
    )
    return transported_means, transported_seconds
