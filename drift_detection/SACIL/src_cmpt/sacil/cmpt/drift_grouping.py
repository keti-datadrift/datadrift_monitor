from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr
from torch import Tensor
from torch.nn import functional as F

from sacil.methods.prototype_transport import (
    affine_ridge_transport,
    apply_affine_mapping,
    paired_class_means,
)


def _upper_triangle(matrix: Tensor) -> Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity matrix must be square")
    indices = torch.triu_indices(
        int(matrix.shape[0]), int(matrix.shape[1]), offset=1
    )
    return matrix[indices[0], indices[1]]


def _finite_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
) -> float:
    if left.size < 3 or right.size != left.size:
        return float("nan")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    if method == "spearman":
        value = spearmanr(left, right).statistic
    elif method == "pearson":
        value = pearsonr(left, right).statistic
    else:
        raise ValueError("unknown correlation method")
    return float(value)


def direction_similarity_correlation(
    prototypes: Tensor,
    drifts: Tensor,
    *,
    permutations: int = 500,
    seed: int = 1,
    epsilon: float = 1.0e-8,
) -> dict[str, Any]:
    """Relate old-prototype similarity to class-drift direction similarity.

    The one-sided Mantel-style permutation test shuffles class identities in
    the drift-similarity matrix. Pairwise class entries are therefore never
    treated as independent observations.
    """

    if prototypes.ndim != 2 or drifts.ndim != 2:
        raise ValueError("prototypes and drifts must be matrices")
    if prototypes.shape != drifts.shape:
        raise ValueError("prototype and drift matrices must have one shape")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    norms = drifts.detach().float().norm(dim=1)
    valid = norms > float(epsilon)
    prototype = F.normalize(prototypes.detach().float()[valid], dim=1)
    drift = F.normalize(drifts.detach().float()[valid], dim=1)
    class_count = int(prototype.shape[0])
    if class_count < 3:
        return {
            "valid_class_count": class_count,
            "pair_count": 0,
            "spearman_rho": None,
            "pearson_r": None,
            "permutation_p_positive": None,
        }

    feature_similarity = prototype @ prototype.T
    drift_similarity = drift @ drift.T
    feature_pairs = _upper_triangle(feature_similarity).cpu().numpy()
    drift_pairs = _upper_triangle(drift_similarity).cpu().numpy()
    observed = _finite_correlation(
        feature_pairs, drift_pairs, method="spearman"
    )
    pearson = _finite_correlation(
        feature_pairs, drift_pairs, method="pearson"
    )
    if not math.isfinite(observed):
        p_value = None
    else:
        generator = np.random.default_rng(int(seed))
        exceedances = 0
        upper = np.triu_indices(class_count, k=1)
        feature_ranks = rankdata(feature_pairs, method="average")
        drift_ranks = rankdata(drift_pairs, method="average")
        feature_centered = feature_ranks - feature_ranks.mean()
        drift_centered = drift_ranks - drift_ranks.mean()
        denominator = float(
            np.linalg.norm(feature_centered)
            * np.linalg.norm(drift_centered)
        )
        drift_rank_matrix = np.zeros(
            (class_count, class_count), dtype=np.float64
        )
        drift_rank_matrix[upper] = drift_centered
        drift_rank_matrix[(upper[1], upper[0])] = drift_centered
        for _ in range(int(permutations)):
            order = generator.permutation(class_count)
            permuted_centered = drift_rank_matrix[
                np.ix_(order, order)
            ][upper]
            candidate = float(
                np.dot(feature_centered, permuted_centered) / denominator
            )
            if candidate >= observed:
                exceedances += 1
        p_value = (exceedances + 1.0) / (int(permutations) + 1.0)
    return {
        "valid_class_count": class_count,
        "pair_count": int(feature_pairs.size),
        "spearman_rho": observed if math.isfinite(observed) else None,
        "pearson_r": pearson if math.isfinite(pearson) else None,
        "permutation_p_positive": p_value,
    }


def split_half_drift_reliability(
    old_features: Tensor,
    current_features: Tensor,
    targets: Tensor,
    mapping: Tensor,
    *,
    num_classes: int,
    repeats: int = 50,
    seed: int = 1,
    epsilon: float = 1.0e-8,
) -> dict[str, float | int | None]:
    """Estimate whether class drift directions reproduce on held-out exemplars."""

    if old_features.shape != current_features.shape:
        raise ValueError("split-half paired features must have one shape")
    if old_features.ndim != 2 or old_features.shape[0] != targets.numel():
        raise ValueError("invalid split-half feature/target shapes")
    if repeats <= 0:
        raise ValueError("split-half repeats must be positive")
    old = F.normalize(old_features.detach().float(), dim=1)
    current = F.normalize(current_features.detach().float(), dim=1)
    labels = targets.detach().cpu().long()
    class_indices = [
        torch.where(labels == class_index)[0]
        for class_index in range(int(num_classes))
    ]
    if any(indices.numel() < 2 for indices in class_indices):
        raise ValueError("each class needs at least two paired exemplars")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw_scores: list[float] = []
    residual_scores: list[float] = []
    raw_valid = 0
    residual_valid = 0
    possible = 0
    for _ in range(int(repeats)):
        halves: list[tuple[Tensor, Tensor]] = []
        for indices in class_indices:
            order = indices[
                torch.randperm(indices.numel(), generator=generator)
            ]
            boundary = int(indices.numel() // 2)
            halves.append((order[:boundary], order[boundary:]))

        old_means: list[Tensor] = []
        current_means: list[Tensor] = []
        for side in (0, 1):
            old_means.append(
                torch.stack(
                    [
                        F.normalize(
                            old[pair[side]].mean(dim=0),
                            dim=0,
                        )
                        for pair in halves
                    ]
                )
            )
            current_means.append(
                torch.stack(
                    [
                        F.normalize(
                            current[pair[side]].mean(dim=0),
                            dim=0,
                        )
                        for pair in halves
                    ]
                )
            )

        raw = [
            current_means[side] - old_means[side] for side in (0, 1)
        ]
        residual = [
            current_means[side]
            - apply_affine_mapping(old_means[side], mapping)
            for side in (0, 1)
        ]
        for first, second, scores, kind in (
            (raw[0], raw[1], raw_scores, "raw"),
            (residual[0], residual[1], residual_scores, "residual"),
        ):
            valid = (first.norm(dim=1) > float(epsilon)) & (
                second.norm(dim=1) > float(epsilon)
            )
            possible += int(valid.numel()) if kind == "raw" else 0
            if kind == "raw":
                raw_valid += int(valid.sum().item())
            else:
                residual_valid += int(valid.sum().item())
            if bool(valid.any()):
                values = F.cosine_similarity(first[valid], second[valid])
                scores.extend(float(value) for value in values.tolist())

    return {
        "repeats": int(repeats),
        "raw_mean_cosine": (
            sum(raw_scores) / len(raw_scores) if raw_scores else None
        ),
        "residual_mean_cosine": (
            sum(residual_scores) / len(residual_scores)
            if residual_scores
            else None
        ),
        "raw_valid_fraction": raw_valid / possible if possible else None,
        "residual_valid_fraction": (
            residual_valid / possible if possible else None
        ),
    }


def cross_fitted_global_residuals(
    old_fit_features: Tensor,
    current_fit_features: Tensor,
    fit_targets: Tensor,
    old_means: Tensor,
    current_means: Tensor,
    *,
    folds: int,
    ridge: float,
    seed: int,
) -> Tensor:
    """Predict each class with a global map fitted without that class fold."""

    class_count = int(old_means.shape[0])
    fold_count = min(int(folds), class_count)
    if fold_count < 2:
        raise ValueError("class cross-fitting requires at least two folds")
    labels = fit_targets.detach().cpu().long()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    class_order = torch.randperm(class_count, generator=generator)
    class_folds = torch.tensor_split(class_order, fold_count)
    predicted = torch.empty_like(current_means)
    for held_out in class_folds:
        held_mask = torch.zeros(class_count, dtype=torch.bool)
        held_mask[held_out] = True
        fit_mask = ~held_mask[labels]
        held_prediction, _, _ = affine_ridge_transport(
            old_means[held_out],
            old_fit_features[fit_mask],
            current_fit_features[fit_mask],
            ridge=ridge,
        )
        predicted[held_out] = held_prediction
    return current_means - predicted


def analyze_drift_grouping_support(
    old_fit_features: Tensor,
    current_fit_features: Tensor,
    fit_targets: Tensor,
    old_exemplar_features: Tensor,
    current_exemplar_features: Tensor,
    exemplar_targets: Tensor,
    *,
    num_classes: int,
    ridge: float,
    permutations: int,
    split_repeats: int,
    class_crossfit_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze raw and global-residual class drift on one CIL transition."""

    old_means, current_means = paired_class_means(
        old_fit_features,
        current_fit_features,
        fit_targets,
        num_classes=num_classes,
        epsilon=1.0e-12,
    )
    _, mapping, fit_residual = affine_ridge_transport(
        old_means,
        old_fit_features,
        current_fit_features,
        ridge=ridge,
    )
    predicted_means = apply_affine_mapping(old_means, mapping)
    raw_drifts = current_means - old_means
    residual_drifts = current_means - predicted_means
    cross_fitted_residual_drifts = cross_fitted_global_residuals(
        old_fit_features,
        current_fit_features,
        fit_targets,
        old_means,
        current_means,
        folds=class_crossfit_folds,
        ridge=ridge,
        seed=seed + 2,
    )
    raw_energy = float(raw_drifts.square().sum().item())
    residual_energy = float(residual_drifts.square().sum().item())
    explained = (
        None
        if raw_energy <= 1.0e-12
        else 1.0 - residual_energy / raw_energy
    )
    return {
        "class_count": int(num_classes),
        "fit_support_count": int(old_fit_features.shape[0]),
        "affine_sample_fit_residual": float(fit_residual),
        "class_mean_raw_drift_norm": float(
            raw_drifts.norm(dim=1).mean().item()
        ),
        "class_mean_residual_drift_norm": float(
            residual_drifts.norm(dim=1).mean().item()
        ),
        "class_mean_drift_energy_explained_by_global": explained,
        "raw_drift": direction_similarity_correlation(
            old_means,
            raw_drifts,
            permutations=permutations,
            seed=seed,
        ),
        "global_residual_drift": direction_similarity_correlation(
            old_means,
            residual_drifts,
            permutations=permutations,
            seed=seed + 1,
        ),
        "cross_fitted_global_residual_drift": (
            direction_similarity_correlation(
                old_means,
                cross_fitted_residual_drifts,
                permutations=permutations,
                seed=seed + 2,
            )
        ),
        "split_half_reliability": split_half_drift_reliability(
            old_exemplar_features,
            current_exemplar_features,
            exemplar_targets,
            mapping,
            num_classes=num_classes,
            repeats=split_repeats,
            seed=seed + 3,
        ),
    }


def summarize_drift_grouping_sessions(
    sessions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not sessions:
        raise ValueError("drift-grouping summary requires sessions")

    def values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for session in sessions:
            current: Any = session
            for key in path:
                current = current[key]
            if current is not None and math.isfinite(float(current)):
                result.append(float(current))
        return result

    def describe(items: Sequence[float]) -> dict[str, Any]:
        if not items:
            return {"mean": None, "median": None, "positive_sessions": 0}
        array = np.asarray(items, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "positive_sessions": int((array > 0.0).sum()),
            "session_count": int(array.size),
        }

    raw_rho = values(("raw_drift", "spearman_rho"))
    residual_rho = values(("global_residual_drift", "spearman_rho"))
    cross_fitted_residual_rho = values(
        ("cross_fitted_global_residual_drift", "spearman_rho")
    )
    raw_p = values(("raw_drift", "permutation_p_positive"))
    residual_p = values(
        ("global_residual_drift", "permutation_p_positive")
    )
    cross_fitted_residual_p = values(
        (
            "cross_fitted_global_residual_drift",
            "permutation_p_positive",
        )
    )
    return {
        "session_count": len(sessions),
        "raw_spearman": describe(raw_rho),
        "global_residual_spearman": describe(residual_rho),
        "cross_fitted_global_residual_spearman": describe(
            cross_fitted_residual_rho
        ),
        "raw_positive_permutation_p_below_0_05": sum(
            value < 0.05 for value in raw_p
        ),
        "residual_positive_permutation_p_below_0_05": sum(
            value < 0.05 for value in residual_p
        ),
        "cross_fitted_residual_positive_permutation_p_below_0_05": sum(
            value < 0.05 for value in cross_fitted_residual_p
        ),
        "mean_global_explained_class_drift_energy": float(
            np.mean(
                values(("class_mean_drift_energy_explained_by_global",))
            )
        ),
        "mean_raw_split_half_cosine": float(
            np.mean(
                values(("split_half_reliability", "raw_mean_cosine"))
            )
        ),
        "mean_residual_split_half_cosine": float(
            np.mean(
                values(
                    (
                        "split_half_reliability",
                        "residual_mean_cosine",
                    )
                )
            )
        ),
    }
