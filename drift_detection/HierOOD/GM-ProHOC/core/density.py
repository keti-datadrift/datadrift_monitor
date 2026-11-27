from __future__ import annotations

import math
from typing import Any

import torch


def build_leaf_to_ancestor_mask(hierarchy, leaf_class_names: list[str]) -> torch.Tensor:
    n_leaves = len(leaf_class_names)
    n_nodes = len(hierarchy.id_node_list)
    mask = torch.zeros((n_leaves, n_nodes), dtype=torch.bool)
    for leaf_idx, leaf_name in enumerate(leaf_class_names):
        mask[leaf_idx, hierarchy.id_node_list.index(leaf_name)] = True
        for anc in hierarchy.node_ancestors[leaf_name]:
            mask[leaf_idx, anc] = True
    return mask


def fit_node_distributions(
    features: torch.Tensor,
    leaf_targets: torch.Tensor,
    hierarchy,
    leaf_class_names: list[str],
    covariance_type: str = "diag",
    eps: float = 1e-6,
):
    covariance_type = covariance_type.lower()
    mask = build_leaf_to_ancestor_mask(hierarchy, leaf_class_names)
    n_nodes = mask.shape[1]
    feat_dim = features.shape[1]
    means = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    mean_directions = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    variances = torch.zeros((n_nodes, feat_dim), dtype=features.dtype)
    covariance_matrices = None
    shared_covariance = None
    counts = torch.zeros((n_nodes,), dtype=torch.long)
    node_features_by_idx: list[torch.Tensor] = []

    if covariance_type in {"full", "shared_full"}:
        covariance_matrices = torch.zeros((n_nodes, feat_dim, feat_dim), dtype=features.dtype)

    for node_idx in range(n_nodes):
        member_leaf_mask = mask[:, node_idx]
        member_leaf_indices = torch.nonzero(member_leaf_mask, as_tuple=False).squeeze(1)
        sample_mask = torch.isin(leaf_targets, member_leaf_indices)
        node_features = features[sample_mask]
        node_features_by_idx.append(node_features)
        counts[node_idx] = node_features.shape[0]
        if node_features.shape[0] == 0:
            variances[node_idx] = torch.full((feat_dim,), eps, dtype=features.dtype)
            if covariance_matrices is not None:
                covariance_matrices[node_idx] = torch.eye(feat_dim, dtype=features.dtype) * eps
            continue
        means[node_idx] = node_features.mean(dim=0)
        mean_directions[node_idx] = torch.nn.functional.normalize(
            means[node_idx].unsqueeze(0),
            dim=-1,
            eps=eps,
        ).squeeze(0)
        variances[node_idx] = _fit_diag_variance(node_features, eps)
        if covariance_matrices is not None:
            covariance_matrices[node_idx] = _fit_full_covariance(node_features, means[node_idx], eps)

    if covariance_type == "shared_full":
        shared_covariance = _fit_depth_shared_full_covariance(
            node_features_by_idx,
            means,
            hierarchy,
            feat_dim,
            features.dtype,
            features.device,
            eps,
        )
        covariance_matrices = None

    return {
        "node_names": hierarchy.id_node_list,
        "node_to_index": {name: idx for idx, name in enumerate(hierarchy.id_node_list)},
        "feature_dim": feat_dim,
        "covariance_type": covariance_type,
        "means": means,
        "mean_directions": mean_directions,
        "variances": variances,
        "covariance_matrices": covariance_matrices,
        "shared_covariance": shared_covariance,
        "counts": counts,
        "leaf_to_ancestor_mask": mask,
    }


def fit_diagonal_gaussians(
    features: torch.Tensor,
    leaf_targets: torch.Tensor,
    hierarchy,
    leaf_class_names: list[str],
    eps: float = 1e-6,
):
    return fit_node_distributions(
        features,
        leaf_targets,
        hierarchy,
        leaf_class_names,
        covariance_type="diag",
        eps=eps,
    )


def score_nodes(
    features: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    shared_covariance: torch.Tensor | None = None,
    mean_directions: torch.Tensor | None = None,
    covariance_type: str = "diag",
    score_type: str = "gaussian_loglik",
    kappa: float = 20.0,
) -> torch.Tensor:
    covariance_type = covariance_type.lower()
    score_type = score_type.lower()
    diff = features[:, None, :] - means[None, :, :]

    if score_type == "vmf":
        if mean_directions is None:
            mean_directions = torch.nn.functional.normalize(means, dim=-1, eps=1e-12)
        normalized_features = torch.nn.functional.normalize(features, dim=-1, eps=1e-12)
        normalized_means = torch.nn.functional.normalize(mean_directions, dim=-1, eps=1e-12)
        scores = float(kappa) * torch.matmul(normalized_features, normalized_means.transpose(0, 1))
    else:
        maha, log_det = _gaussian_terms(
            diff,
            variances=variances,
            covariance_matrices=covariance_matrices,
            shared_covariance=shared_covariance,
            covariance_type=covariance_type,
        )
        if score_type == "gaussian_loglik":
            const = means.shape[-1] * math.log(2.0 * math.pi)
            scores = -0.5 * (maha + log_det[None, :] + const)
        elif score_type == "raw_gaussian":
            const = means.shape[-1] * math.log(2.0 * math.pi)
            scores = torch.exp(-0.5 * (maha + log_det[None, :] + const))
        elif score_type == "mahalanobis":
            scores = -1.0 * maha
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")

    return scores


def _fit_diag_variance(node_features: torch.Tensor, eps: float) -> torch.Tensor:
    feat_dim = node_features.shape[1]
    if node_features.shape[0] <= 1:
        return torch.full((feat_dim,), eps, dtype=node_features.dtype)
    return node_features.var(dim=0, unbiased=False) + eps


def _fit_full_covariance(node_features: torch.Tensor, mean: torch.Tensor, eps: float) -> torch.Tensor:
    feat_dim = node_features.shape[1]
    eye = torch.eye(feat_dim, dtype=node_features.dtype, device=node_features.device)
    if node_features.shape[0] <= 1:
        return eye * eps
    centered = node_features - mean.unsqueeze(0)
    cov = centered.transpose(0, 1).matmul(centered) / node_features.shape[0]
    return cov + eye * eps


def _fit_depth_shared_full_covariance(
    node_features_by_idx: list[torch.Tensor],
    means: torch.Tensor,
    hierarchy,
    feat_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    eye = torch.eye(feat_dim, dtype=dtype, device=device)
    n_nodes = len(node_features_by_idx)
    covariances = torch.empty((n_nodes, feat_dim, feat_dim), dtype=dtype, device=device)
    depth_to_nodes: dict[int, list[int]] = {}

    for node_idx, node_name in enumerate(hierarchy.id_node_list):
        depth = len(hierarchy.node_ancestors[node_name])
        depth_to_nodes.setdefault(depth, []).append(node_idx)

    for node_indices in depth_to_nodes.values():
        total_count = 0
        scatter = torch.zeros((feat_dim, feat_dim), dtype=dtype, device=device)
        for node_idx in node_indices:
            node_features = node_features_by_idx[node_idx]
            if node_features.shape[0] == 0:
                continue
            centered = node_features - means[node_idx].unsqueeze(0)
            scatter += centered.transpose(0, 1).matmul(centered)
            total_count += node_features.shape[0]

        if total_count == 0:
            shared_cov = eye * eps
        else:
            shared_cov = scatter / total_count + eye * eps

        for node_idx in node_indices:
            covariances[node_idx] = shared_cov

    return covariances


def _gaussian_terms(
    diff: torch.Tensor,
    variances: torch.Tensor | None,
    covariance_matrices: torch.Tensor | None,
    shared_covariance: torch.Tensor | None,
    covariance_type: str,
):
    if covariance_type == "diag":
        if variances is None:
            raise ValueError("variances are required for diagonal covariance scoring")
        log_det = torch.log(variances).sum(dim=-1)
        maha = (diff.pow(2) / variances[None, :, :]).sum(dim=-1)
        return maha, log_det

    if covariance_type == "full":
        if covariance_matrices is None:
            raise ValueError("covariance_matrices are required for full covariance scoring")
        inv_cov = torch.linalg.inv(covariance_matrices)
        sign, logabsdet = torch.linalg.slogdet(covariance_matrices)
        if not torch.all(sign > 0):
            raise ValueError("full covariance matrices must be positive definite")
        maha = torch.einsum("bnd,nde,bne->bn", diff, inv_cov, diff)
        return maha, logabsdet

    if covariance_type == "shared_full":
        if shared_covariance is None:
            raise ValueError("shared_covariance is required for shared_full scoring")
        if shared_covariance.dim() == 3:
            inv_cov = torch.linalg.inv(shared_covariance)
            sign, logabsdet = torch.linalg.slogdet(shared_covariance)
            if not torch.all(sign > 0):
                raise ValueError("depth-shared full covariance matrices must be positive definite")
            maha = torch.einsum("bnd,nde,bne->bn", diff, inv_cov, diff)
            return maha, logabsdet
        inv_cov = torch.linalg.inv(shared_covariance)
        sign, logabsdet = torch.linalg.slogdet(shared_covariance)
        if not bool(sign.item() > 0):
            raise ValueError("shared full covariance must be positive definite")
        projected = torch.matmul(diff, inv_cov)
        maha = (projected * diff).sum(dim=-1)
        log_det = torch.full(
            (diff.shape[1],),
            logabsdet,
            dtype=diff.dtype,
            device=diff.device,
        )
        return maha, log_det

    raise ValueError(f"Unsupported covariance_type: {covariance_type}")
