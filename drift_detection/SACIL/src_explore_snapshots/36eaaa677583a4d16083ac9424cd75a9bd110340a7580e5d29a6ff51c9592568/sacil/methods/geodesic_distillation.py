from __future__ import annotations

import math

import torch
from torch import Tensor


def principal_subspace(features: Tensor, rank: int) -> Tensor:
    """Return a centered PCA basis with shape ``[feature_dim, rank]``."""

    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim]")
    batch, dimension = features.shape
    rank = int(rank)
    maximum = min(int(batch) - 1, int(dimension) - 1)
    if rank <= 0 or rank > maximum:
        raise ValueError(f"subspace rank must be in [1, {maximum}]")
    centered = features - features.mean(dim=0, keepdim=True)
    basis, _, _ = torch.linalg.svd(centered.transpose(0, 1), full_matrices=False)
    return basis[:, :rank]


def geodesic_flow_kernel(
    source_basis: Tensor,
    target_basis: Tensor,
    *,
    epsilon: float = 1e-7,
) -> Tensor:
    """Closed-form Grassmann geodesic-flow kernel from GeoDL Eq. (7).

    The common factor one half is included.  It cancels in the normalized
    GeoDL cosine, but keeps the identical-subspace limit equal to an
    orthogonal projector instead of twice that projector.
    """

    if source_basis.ndim != 2 or target_basis.ndim != 2:
        raise ValueError("subspace bases must be matrices")
    if source_basis.shape != target_basis.shape:
        raise ValueError("source and target bases must have identical shape")
    dimension, rank = source_basis.shape
    if rank <= 0 or rank >= dimension:
        raise ValueError("basis rank must be between zero and feature dimension")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("kernel epsilon must be finite and positive")

    source_basis, _ = torch.linalg.qr(source_basis, mode="reduced")
    target_basis, _ = torch.linalg.qr(target_basis, mode="reduced")
    complete_source, _ = torch.linalg.qr(source_basis, mode="complete")
    complement = complete_source[:, rank:]

    alignment = source_basis.transpose(0, 1) @ target_basis
    u1, cosine, vh = torch.linalg.svd(alignment, full_matrices=False)
    cosine = cosine.clamp(0.0, 1.0)
    v = vh.transpose(0, 1)
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    complement_coordinates = complement.transpose(0, 1) @ target_basis @ v
    safe_sine = sine.clamp_min(epsilon)
    u2 = -complement_coordinates / safe_sine.unsqueeze(0)
    u2 = torch.where(
        (sine > epsilon).unsqueeze(0), u2, torch.zeros_like(u2)
    )

    ratio = torch.where(
        theta > epsilon,
        torch.sin(2.0 * theta) / (2.0 * theta),
        torch.ones_like(theta),
    )
    cross = torch.where(
        theta > epsilon,
        (torch.cos(2.0 * theta) - 1.0) / (2.0 * theta),
        torch.zeros_like(theta),
    )
    lambda_1 = 1.0 + ratio
    lambda_2 = cross
    lambda_3 = 1.0 - ratio

    delta = torch.cat((source_basis @ u1, complement @ u2), dim=1)
    diagonal_1 = torch.diag(lambda_1)
    diagonal_2 = torch.diag(lambda_2)
    diagonal_3 = torch.diag(lambda_3)
    block = torch.cat(
        (
            torch.cat((diagonal_1, diagonal_2), dim=1),
            torch.cat((diagonal_2, diagonal_3), dim=1),
        ),
        dim=0,
    )
    kernel = 0.5 * delta @ block @ delta.transpose(0, 1)
    return 0.5 * (kernel + kernel.transpose(0, 1))


def geodesic_distillation_loss(
    current_features: Tensor,
    reference_features: Tensor,
    *,
    subspace_rank: int,
    epsilon: float = 1e-7,
) -> Tensor:
    """GeoDL Eq. (9), with the minibatch kernel treated as a target metric."""

    if current_features.shape != reference_features.shape:
        raise ValueError("current and reference features must have identical shape")
    if current_features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim]")
    with torch.no_grad():
        source = principal_subspace(reference_features.detach(), subspace_rank)
        target = principal_subspace(current_features.detach(), subspace_rank)
        kernel = geodesic_flow_kernel(source, target, epsilon=epsilon)

    current_q = current_features @ kernel
    reference_q = reference_features.detach() @ kernel
    numerator = (current_q * reference_features.detach()).sum(dim=1)
    current_norm = (current_q * current_features).sum(dim=1).clamp_min(epsilon).sqrt()
    reference_norm = (
        (reference_q * reference_features.detach()).sum(dim=1)
        .clamp_min(epsilon)
        .sqrt()
    )
    similarity = numerator / (current_norm * reference_norm).clamp_min(epsilon)
    return (1.0 - similarity.clamp(-1.0, 1.0)).mean()
