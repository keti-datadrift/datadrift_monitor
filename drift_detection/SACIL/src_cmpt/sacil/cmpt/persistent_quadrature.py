from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def project_probability_simplex(values: Tensor) -> Tensor:
    """Euclidean projection of a vector onto the probability simplex."""

    vector = values.detach().float().flatten()
    if vector.numel() == 0:
        raise ValueError("cannot project an empty vector")
    ordered, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(ordered, dim=0) - 1.0
    positions = torch.arange(
        1,
        vector.numel() + 1,
        dtype=vector.dtype,
        device=vector.device,
    )
    active = ordered - cumulative / positions > 0.0
    if not bool(active.any()):
        raise RuntimeError("simplex projection has no active coordinates")
    last = int(torch.nonzero(active, as_tuple=False)[-1].item())
    threshold = cumulative[last] / float(last + 1)
    projected = torch.clamp(vector - threshold, min=0.0)
    return projected / projected.sum().clamp_min(1.0e-12)


def fit_persistent_quadrature_weights(
    exemplar_features: Tensor,
    full_population_features: Tensor,
    *,
    uniform_ridge: float = 1.0e-3,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-10,
) -> tuple[Tensor, dict[str, float]]:
    """Fit non-negative exemplar mass weights to an introduction-time mean.

    Both matrices contain one per-image NME contribution per row.  The target
    is the *unnormalized* population mean, because the final prototype is
    normalized only after its finite-sample expectation has been estimated.
    The simplex constraint makes the coefficients population-mass weights;
    the ridge term selects a stable solution close to uniform herding when
    several weight vectors reconstruct the same target.
    """

    weights, diagnostics = fit_multiview_persistent_quadrature_weights(
        exemplar_features.detach().cpu().float().unsqueeze(0),
        full_population_features.detach().cpu().float().unsqueeze(0),
        uniform_ridge=uniform_ridge,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return weights, diagnostics


def fit_multiview_persistent_quadrature_weights(
    exemplar_feature_views: Tensor,
    full_population_feature_views: Tensor,
    *,
    uniform_ridge: float = 1.0e-3,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-10,
) -> tuple[Tensor, dict[str, float]]:
    """Fit one simplex weight vector across several representation views.

    Views may be deterministic image transformations and/or embeddings from
    the previous and current backbones.  Sharing one set of population-mass
    weights across them discourages a coreset correction that is specific to
    only the introduction model's coordinate system.
    """

    exemplars = exemplar_feature_views.detach().cpu().float()
    population = full_population_feature_views.detach().cpu().float()
    if exemplars.ndim != 3 or population.ndim != 3:
        raise ValueError("multi-view feature inputs must have shape [V,N,D]")
    if exemplars.shape[0] != population.shape[0]:
        raise ValueError("exemplar and population view counts differ")
    if exemplars.shape[2] != population.shape[2]:
        raise ValueError("exemplar and population feature dimensions differ")
    if exemplars.shape[1] < 2 or population.shape[1] <= exemplars.shape[1]:
        raise ValueError(
            "quadrature fitting requires a strict full-population superset"
        )
    ridge = float(uniform_ridge)
    if ridge < 0.0:
        raise ValueError("uniform_ridge must be non-negative")
    iterations = int(max_iterations)
    if iterations <= 0:
        raise ValueError("max_iterations must be positive")

    view_count = exemplars.shape[0]
    count = exemplars.shape[1]
    uniform = torch.full((count,), 1.0 / float(count))
    targets = population.mean(dim=1)
    gram = torch.einsum("vkd,vld->vkl", exemplars, exemplars).mean(dim=0)
    rhs = torch.einsum("vkd,vd->vk", exemplars, targets).mean(dim=0)
    if ridge > 0.0:
        gram = gram + ridge * torch.eye(count)
        rhs = rhs + ridge * uniform

    # The gradient of w'Gw - 2r'w has Lipschitz constant 2*lambda_max(G).
    largest = float(torch.linalg.eigvalsh(gram).max().item())
    step = 1.0 / max(2.0 * largest, 1.0e-12)
    weights = uniform.clone()
    momentum = weights.clone()
    acceleration = 1.0
    completed = 0
    for iteration in range(iterations):
        gradient = 2.0 * (gram @ momentum - rhs)
        updated = project_probability_simplex(momentum - step * gradient)
        completed = iteration + 1
        if float((updated - weights).abs().max().item()) <= tolerance:
            weights = updated
            break
        next_acceleration = 0.5 * (
            1.0 + (1.0 + 4.0 * acceleration**2) ** 0.5
        )
        momentum = updated + (
            (acceleration - 1.0) / next_acceleration
        ) * (updated - weights)
        weights = updated
        acceleration = next_acceleration

    uniform_means = torch.einsum("k,vkd->vd", uniform, exemplars)
    weighted_means = torch.einsum("k,vkd->vd", weights, exemplars)
    uniform_directions = F.normalize(uniform_means, dim=1)
    weighted_directions = F.normalize(weighted_means, dim=1)
    target_directions = F.normalize(targets, dim=1)
    uniform_distances = 1.0 - F.cosine_similarity(
        uniform_directions, target_directions, dim=1
    )
    weighted_distances = 1.0 - F.cosine_similarity(
        weighted_directions, target_directions, dim=1
    )
    positive = weights[weights > 0.0]
    entropy = float(
        (-(positive * positive.log()).sum() / torch.log(torch.tensor(float(count))))
        .item()
    )
    diagnostics = {
        "uniform_ridge": ridge,
        "iterations": float(completed),
        "view_count": float(view_count),
        "uniform_target_cosine_distance": float(
            uniform_distances.mean().item()
        ),
        "weighted_target_cosine_distance": float(
            weighted_distances.mean().item()
        ),
        "maximum_weighted_view_cosine_distance": float(
            weighted_distances.max().item()
        ),
        "effective_sample_size": float(
            (1.0 / weights.square().sum().clamp_min(1.0e-12)).item()
        ),
        "normalized_entropy": entropy,
        "minimum_weight": float(weights.min().item()),
        "maximum_weight": float(weights.max().item()),
    }
    return weights, diagnostics


def weighted_class_prototypes(
    exemplar_features: Tensor,
    targets: Tensor,
    class_weights: Tensor,
) -> Tensor:
    """Apply persistent class-wise simplex weights in the current space."""

    features = exemplar_features.detach().cpu().float()
    labels = targets.detach().cpu().long().flatten()
    weights = class_weights.detach().cpu().float()
    if features.ndim != 2:
        raise ValueError("exemplar_features must have shape [N, D]")
    if labels.numel() != features.shape[0]:
        raise ValueError("feature and target counts do not match")
    if weights.ndim != 2:
        raise ValueError("class_weights must have shape [C, K]")
    prototypes: list[Tensor] = []
    for class_id in range(weights.shape[0]):
        class_features = features[labels == class_id]
        class_weight = weights[class_id]
        if class_features.shape[0] != class_weight.numel():
            raise ValueError(
                f"class {class_id} has {class_features.shape[0]} features but "
                f"{class_weight.numel()} persistent weights"
            )
        prototypes.append(F.normalize(class_weight @ class_features, dim=0))
    return torch.stack(prototypes)
