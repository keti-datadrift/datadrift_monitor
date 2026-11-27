from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import torch

from core.density import (
    gaussian_bump,
    gaussian_bump_integrals,
    gaussian_bump_integrals_from_covariance,
    gaussian_chi2_ellipsoid_membership,
    gaussian_chi2_survival_membership,
    gaussian_covariance_matrices,
    gaussian_logpdf,
    gaussian_logpdf_from_covariance,
    masked_gaussian_logpdf,
    reweighted_gaussian_logpdf,
    estimate_vmf_kappas,
    score_nodes,
    vmf_bump,
    vmf_bump_integrals,
    vmf_log_normalizer,
    vmf_logpdf,
)
from libs.utils.hierarchy_utils import expected_hdist, get_hdist_matrix


def build_depth_maps(hierarchy):
    nodes_by_depth = defaultdict(list)
    for idx, name in enumerate(hierarchy.id_node_list):
        depth = len(hierarchy.node_ancestors[name])
        nodes_by_depth[depth].append(idx)
    return {depth: sorted(indices) for depth, indices in nodes_by_depth.items()}


def _normalize_depthwise_temperature(temperature, depths: list[int]) -> dict[int, float]:
    if isinstance(temperature, (list, tuple)):
        if len(temperature) != len(depths):
            raise ValueError(
                f"temperature vector length {len(temperature)} does not match "
                f"number of non-root depths {len(depths)}"
            )
        return {depth: max(float(temp), 1e-8) for depth, temp in zip(depths, temperature)}
    return {depth: max(float(temperature), 1e-8) for depth in depths}


def compute_depth_probs(node_scores: torch.Tensor, nodes_by_depth: dict[int, list[int]], temperature: float | list[float] = 1.0):
    depth_probs = {}
    non_root_depths = sorted(depth for depth in nodes_by_depth.keys() if depth != 0)
    temp_by_depth = _normalize_depthwise_temperature(temperature, non_root_depths)
    for depth, indices in nodes_by_depth.items():
        scaled_scores = node_scores[:, indices]
        if depth != 0:
            scaled_scores = scaled_scores / temp_by_depth[depth]
        depth_probs[depth] = torch.softmax(scaled_scores, dim=-1)
    return depth_probs


def _local_probabilities_for_node(
    parent_name: str,
    hierarchy,
    depth_probs: dict[int, torch.Tensor],
    nodes_by_depth: dict[int, list[int]],
    ood_scale: float = 1.0,
):
    children = hierarchy.parent2children.get(parent_name, [])
    if not children:
        return None

    child_indices = [hierarchy.id_node_list.index(child) for child in children]
    child_depth = len(hierarchy.node_ancestors[children[0]])
    depth_indices = nodes_by_depth[child_depth]
    depth_index_map = {node_idx: pos for pos, node_idx in enumerate(depth_indices)}
    child_depth_probs = torch.stack(
        [depth_probs[child_depth][:, depth_index_map[idx]] for idx in child_indices],
        dim=1,
    )
    child_group_sum = child_depth_probs.sum(dim=1, keepdim=True)
    p_comp = 1.0 - child_group_sum.squeeze(1)
    p_comp = torch.clamp(p_comp, min=0.0)
    eps = 1e-12
    child_local_probs = child_depth_probs / child_group_sum.clamp_min(eps)
    entropy = -torch.sum(child_local_probs * torch.log(child_local_probs + eps), dim=1)
    ood_mass = float(ood_scale) * (p_comp + entropy)
    ood_mass = torch.clamp(ood_mass, min=0.0)

    # Stay in probability space: use depth-wise child probabilities and
    # the derived OOD mass, then renormalize locally within siblings.
    local_mass = torch.cat([child_depth_probs, ood_mass.unsqueeze(1)], dim=1)
    local_normalizer = local_mass.sum(dim=1, keepdim=True).clamp_min(eps)
    local_probs = local_mass / local_normalizer
    return {
        "child_indices": child_indices,
        "child_depth_probs": child_depth_probs,
        "child_local_probs": child_local_probs,
        "child_probs": local_probs[:, :-1],
        "ood_prob": local_probs[:, -1],
        "ood_score": ood_mass,
        "p_comp": p_comp,
        "entropy": entropy,
        "ood_scale": float(ood_scale),
    }


def _cgm_enabled(cgm_cfg: dict[str, Any] | None) -> bool:
    return bool(cgm_cfg and cgm_cfg.get("enabled", False))


def _is_unit_temperature(temperature) -> bool:
    if isinstance(temperature, (list, tuple)):
        return all(abs(float(temp) - 1.0) <= 1e-8 for temp in temperature)
    return abs(float(temperature) - 1.0) <= 1e-8


def _validate_cgm_cfg(cgm_cfg: dict[str, Any], score_type: str, temperature):
    strict_pdf = bool(cgm_cfg.get("strict_pdf", False))
    density_family = cgm_cfg.get("density_family", "gaussian").lower()
    if density_family not in {"gaussian", "vmf"}:
        raise ValueError(f"Unsupported CGM density_family: {density_family}")
    ood_density = cgm_cfg.get("ood_density", "parent_mask").lower()
    if ood_density not in {
        "parent_mask",
        "complement_mixture",
        "hybrid_mixture",
        "multiscale_parent_mask",
        "child_mixture_mask",
        "child_tail_mixture",
        "empirical_child_tail_mixture",
        "soft_child_tail_mixture",
        "posterior_entropy_mixture",
        "posterior_residual_mixture",
        "conformal_maxprob_stop",
        "conformal_logpdf_stop",
        "random_effects_parent",
        "random_effects_mixture",
        "random_complement_mixture",
        "poe_random_effects_parent",
        "positive_density_residual",
        "softset_complement",
        "vmf_parent_mask",
    }:
        raise ValueError(f"Unsupported CGM ood_density: {ood_density}")
    complement_reduce = cgm_cfg.get("complement_reduce", "sum").lower()
    if complement_reduce not in {"sum", "mean"}:
        raise ValueError(f"Unsupported CGM complement_reduce: {complement_reduce}")
    mask_type = cgm_cfg.get("mask_type", "sum").lower()
    if mask_type not in {"sum", "product", "max", "residual_sigmoid"}:
        raise ValueError(f"Unsupported CGM mask_type: {mask_type}")
    membership_type = cgm_cfg.get("membership_type", "peak_bump").lower()
    if membership_type not in {"peak_bump", "chi2_survival", "chi2_ellipsoid"}:
        raise ValueError(f"Unsupported CGM membership_type: {membership_type}")
    membership_probability = float(cgm_cfg.get("membership_probability", 0.95))
    if membership_probability <= 0.0 or membership_probability >= 1.0:
        raise ValueError("CGM membership_probability must satisfy 0 < value < 1")
    membership_correction = cgm_cfg.get("membership_correction", "none").lower()
    if membership_correction not in {"none", "sidak"}:
        raise ValueError(f"Unsupported CGM membership_correction: {membership_correction}")
    if ood_density in {
        "multiscale_parent_mask",
        "child_mixture_mask",
        "random_effects_parent",
        "random_effects_mixture",
        "random_complement_mixture",
        "poe_random_effects_parent",
    }:
        product_mask_allowed = (
            ood_density in {"random_effects_parent", "poe_random_effects_parent"}
            and mask_type == "product"
        )
        if mask_type != "sum" and not product_mask_allowed:
            raise ValueError(f"{ood_density} requires mask_type: sum")
        if not bool(cgm_cfg.get("normalize_ood_pdf", True)):
            raise ValueError(f"{ood_density} requires normalize_ood_pdf: true")
    if ood_density == "softset_complement":
        if mask_type != "product":
            raise ValueError("softset_complement requires mask_type: product")
        if not bool(cgm_cfg.get("normalize_ood_pdf", True)):
            raise ValueError("softset_complement requires normalize_ood_pdf: true")
    if ood_density == "child_tail_mixture":
        if mask_type != "product" or membership_type != "chi2_ellipsoid":
            raise ValueError(
                "child_tail_mixture requires mask_type: product and membership_type: chi2_ellipsoid"
            )
        if not bool(cgm_cfg.get("normalize_ood_pdf", True)):
            raise ValueError("child_tail_mixture requires normalize_ood_pdf: true")
    if ood_density == "empirical_child_tail_mixture":
        if mask_type != "product":
            raise ValueError("empirical_child_tail_mixture requires mask_type: product")
        if not bool(cgm_cfg.get("normalize_ood_pdf", True)):
            raise ValueError("empirical_child_tail_mixture requires normalize_ood_pdf: true")
    if ood_density == "soft_child_tail_mixture":
        if mask_type != "product" or membership_type != "chi2_survival":
            raise ValueError(
                "soft_child_tail_mixture requires mask_type: product and membership_type: chi2_survival"
            )
        if not bool(cgm_cfg.get("normalize_ood_pdf", True)):
            raise ValueError("soft_child_tail_mixture requires normalize_ood_pdf: true")
    local_mode = cgm_cfg.get("local_mode", "density_softmax").lower()
    if local_mode not in {
        "density_softmax",
        "depth_gate",
        "blend",
        "binary_model_selection",
        "density_joint",
    }:
        raise ValueError(f"Unsupported CGM local_mode: {local_mode}")
    lam = float(cgm_cfg.get("lambda", 0.9))
    if lam < 0.0 or lam >= 1.0:
        raise ValueError("CGM lambda must satisfy 0 <= lambda < 1")
    child_weight = cgm_cfg.get("child_weight", "uniform").lower()
    if child_weight not in {"uniform", "count"}:
        raise ValueError(f"Unsupported CGM child_weight: {child_weight}")
    candidate_prior = cgm_cfg.get("candidate_prior", "uniform").lower()
    if candidate_prior not in {
        "uniform",
        "hierarchy_leaf_count",
        "balanced_terminal",
        "mixed_balanced_terminal",
    }:
        raise ValueError(f"Unsupported CGM candidate_prior: {candidate_prior}")
    if ood_density == "positive_density_residual" and (
        candidate_prior != "uniform" or child_weight != "uniform"
    ):
        raise ValueError(
            "positive_density_residual requires uniform child weights and no external candidate prior"
        )
    if local_mode == "binary_model_selection" and (
        candidate_prior != "uniform" or child_weight != "uniform"
    ):
        raise ValueError(
            "binary_model_selection requires uniform child weights and no external candidate prior"
        )
    between_cov_estimator = cgm_cfg.get("between_cov_estimator", "parent_residual").lower()
    if between_cov_estimator not in {
        "parent_residual",
        "empirical_child_means",
        "shrunk_child_means",
    }:
        raise ValueError(f"Unsupported CGM between_cov_estimator: {between_cov_estimator}")
    between_cov_shrinkage_strength = float(
        cgm_cfg.get("between_cov_shrinkage_strength", 3.0)
    )
    if between_cov_shrinkage_strength < 0.0:
        raise ValueError("CGM between_cov_shrinkage_strength must be nonnegative")
    product_mask_samples = int(cgm_cfg.get("product_mask_samples", 1024))
    if product_mask_samples <= 0:
        raise ValueError("CGM product_mask_samples must be positive")
    if strict_pdf:
        if ood_density not in {
            "parent_mask",
            "multiscale_parent_mask",
            "child_mixture_mask",
            "child_tail_mixture",
            "empirical_child_tail_mixture",
            "soft_child_tail_mixture",
            "posterior_entropy_mixture",
            "posterior_residual_mixture",
            "conformal_maxprob_stop",
            "conformal_logpdf_stop",
            "random_effects_parent",
            "random_effects_mixture",
            "random_complement_mixture",
            "poe_random_effects_parent",
            "positive_density_residual",
            "softset_complement",
            "vmf_parent_mask",
        }:
            raise ValueError(
                "strict_pdf CGM requires a normalized parent-mask density"
            )
        if ood_density == "random_complement_mixture" and complement_reduce != "mean":
            raise ValueError("strict random_complement_mixture requires complement_reduce: mean")
        product_mask_allowed = (
            ood_density
            in {
                "random_effects_parent",
                "poe_random_effects_parent",
                "softset_complement",
                "child_tail_mixture",
                "empirical_child_tail_mixture",
                "soft_child_tail_mixture",
            }
            and mask_type == "product"
        )
        if mask_type != "sum" and not product_mask_allowed:
            raise ValueError("strict_pdf CGM requires a normalized sum or product mask")
        if not bool(cgm_cfg.get("normalize_ood_pdf", True)):
            raise ValueError("strict_pdf CGM requires normalize_ood_pdf: true")
        if local_mode not in {
            "density_softmax",
            "binary_model_selection",
            "density_joint",
        }:
            raise ValueError(
                "strict_pdf CGM requires density_softmax, binary_model_selection, or density_joint"
            )
    if density_family == "gaussian" and score_type.lower() != "gaussian_loglik":
        raise ValueError("CGM inference requires inference.score_type: gaussian_loglik")
    if density_family == "vmf" and ood_density != "vmf_parent_mask":
        raise ValueError("vMF CGM requires ood_density: vmf_parent_mask")
    if not _is_unit_temperature(temperature):
        raise ValueError("CGM base inference requires temperature 1.0")


def _cgm_child_weights(
    child_indices: list[int],
    density_payload: dict[str, Any],
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.full((len(child_indices),), 1.0 / len(child_indices), dtype=dtype, device=device)

    counts = density_payload.get("counts")
    if counts is None:
        return torch.full((len(child_indices),), 1.0 / len(child_indices), dtype=dtype, device=device)
    weights = counts[child_indices].to(device=device, dtype=dtype)
    total = weights.sum()
    if not bool(total.item() > 0):
        return torch.full((len(child_indices),), 1.0 / len(child_indices), dtype=dtype, device=device)
    return weights / total


def _depth_pooled_child_mean_covariance(
    child_depth: int,
    hierarchy,
    density_payload: dict[str, Any],
    child_weight_mode: str,
    eps: float,
) -> torch.Tensor:
    cache = density_payload.setdefault("_cgm_depth_between_covariance_cache", {})
    cache_key = (child_depth, child_weight_mode)
    if cache_key in cache:
        return cache[cache_key]

    means = density_payload["means"]
    numerator = torch.zeros(
        (means.shape[1], means.shape[1]),
        dtype=means.dtype,
        device=means.device,
    )
    total_correction = torch.zeros((), dtype=means.dtype, device=means.device)
    for children in hierarchy.parent2children.values():
        if not children:
            continue
        current_depth = len(hierarchy.node_ancestors[children[0]])
        if current_depth != child_depth:
            continue
        child_indices = [hierarchy.id_node_list.index(child) for child in children]
        if len(child_indices) <= 1:
            continue
        child_weights = _cgm_child_weights(
            child_indices,
            density_payload,
            child_weight_mode,
            dtype=means.dtype,
            device=means.device,
        )
        child_means = means[child_indices]
        local_mean = torch.sum(child_weights.unsqueeze(1) * child_means, dim=0)
        centered_means = child_means - local_mean.unsqueeze(0)
        numerator += torch.einsum(
            "n,nd,ne->de",
            child_weights,
            centered_means,
            centered_means,
        )
        total_correction += 1.0 - torch.sum(child_weights.square())

    if bool(total_correction.item() > eps):
        pooled_covariance = numerator / total_correction
    else:
        pooled_covariance = numerator
    pooled_covariance = 0.5 * (
        pooled_covariance + pooled_covariance.transpose(0, 1)
    )
    cache[cache_key] = pooled_covariance
    return pooled_covariance


def _product_mask_log_normalizer(
    parent_idx: int,
    base_mean: torch.Tensor,
    base_covariance: torch.Tensor,
    child_indices: list[int],
    density_payload: dict[str, Any],
    mask_cov_scale: float,
    lam: float,
    num_samples: int,
    eps: float,
    membership_type: str = "peak_bump",
    membership_probability: float = 0.95,
) -> torch.Tensor:
    cache = density_payload.setdefault("_cgm_product_mask_normalizer_cache", {})
    cache_key = (
        parent_idx,
        tuple(child_indices),
        float(mask_cov_scale),
        float(lam),
        int(num_samples),
        membership_type,
        float(membership_probability),
        tuple(base_covariance.shape),
        float(base_covariance.diagonal().sum().item()),
    )
    if cache_key in cache:
        return cache[cache_key]

    sobol = torch.quasirandom.SobolEngine(
        dimension=base_mean.numel(),
        scramble=True,
        seed=parent_idx,
    )
    uniforms = sobol.draw(num_samples).to(
        device=base_mean.device,
        dtype=base_mean.dtype,
    )
    uniforms = uniforms.clamp(min=eps, max=1.0 - eps)
    standard_normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniforms - 1.0)
    cholesky = torch.linalg.cholesky(base_covariance)
    samples = base_mean.unsqueeze(0) + standard_normal @ cholesky.transpose(0, 1)
    if membership_type == "chi2_survival":
        membership_fn = gaussian_chi2_survival_membership
        membership_kwargs = {}
    elif membership_type == "chi2_ellipsoid":
        membership_fn = gaussian_chi2_ellipsoid_membership
        membership_kwargs = {"probability_mass": membership_probability}
    else:
        membership_fn = gaussian_bump
        membership_kwargs = {}
    sample_bumps = membership_fn(
        samples,
        density_payload["means"],
        density_payload.get("variances"),
        covariance_matrices=density_payload.get("covariance_matrices"),
        shared_covariance=density_payload.get("shared_covariance"),
        covariance_type=density_payload.get(
            "covariance_type",
            density_payload.get("config", {}).get("covariance_type", "diag"),
        ),
        node_indices=child_indices,
        covariance_scale=mask_cov_scale,
        **membership_kwargs,
    )
    safe_max = 1.0 - max(eps, torch.finfo(sample_bumps.dtype).eps)
    sample_bumps = torch.nan_to_num(
        sample_bumps,
        nan=0.0,
        posinf=safe_max,
        neginf=0.0,
    )
    sample_bumps = sample_bumps.clamp(min=0.0, max=safe_max)
    sample_log_mask = torch.log1p(-lam * sample_bumps).sum(dim=1)
    log_normalizer = torch.logsumexp(sample_log_mask, dim=0) - math.log(num_samples)
    cache[cache_key] = log_normalizer
    return log_normalizer


def _resolve_cgm_ood_prior(raw_prior, child_depth: int, max_depth: int) -> float:
    if isinstance(raw_prior, (list, tuple)):
        if len(raw_prior) != max_depth:
            raise ValueError(
                f"CGM depth-wise ood_prior length {len(raw_prior)} does not match "
                f"number of non-root depths {max_depth}"
            )
        prior = float(raw_prior[child_depth - 1])
    else:
        prior = float(raw_prior)
    if prior <= 0.0:
        raise ValueError("CGM ood_prior must be positive")
    return prior


def _resolve_cgm_between_cov_scale(raw_scale, child_depth: int, max_depth: int) -> float:
    if isinstance(raw_scale, (list, tuple)):
        if len(raw_scale) != max_depth:
            raise ValueError(
                f"CGM depth-wise between_cov_scale length {len(raw_scale)} does not match "
                f"number of non-root depths {max_depth}"
            )
        scale = float(raw_scale[child_depth - 1])
    else:
        scale = float(raw_scale)
    if scale < 0.0:
        raise ValueError("CGM between_cov_scale must be nonnegative")
    return scale


def _hierarchy_leaf_candidate_priors(
    parent_name: str,
    children: list[str],
    hierarchy,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_name = "_cgm_ood_leaf_counts"
    ood_leaf_counts = getattr(hierarchy, cache_name, None)
    if ood_leaf_counts is None:
        mapped_indices = hierarchy.gen_ds2node_map(hierarchy.ood_train_classes)
        mapped_names = [hierarchy.id_node_list[idx] for idx in mapped_indices.tolist()]
        ood_leaf_counts = Counter(mapped_names)
        setattr(hierarchy, cache_name, ood_leaf_counts)

    child_counts = torch.tensor(
        [float(len(hierarchy.get_leaf_descendants(child))) for child in children],
        dtype=dtype,
        device=device,
    )
    ood_count = torch.tensor(
        float(ood_leaf_counts.get(parent_name, 0)),
        dtype=dtype,
        device=device,
    )
    total = child_counts.sum() + ood_count
    return child_counts / total, ood_count / total


def _balanced_terminal_candidate_priors(
    parent_name: str,
    children: list[str],
    hierarchy,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_name = "_cgm_balanced_terminal_names"
    terminal_names = getattr(hierarchy, cache_name, None)
    if terminal_names is None:
        id_leaf_names = {
            name
            for name in hierarchy.id_node_list
            if name not in hierarchy.parent2children
        }
        mapped_indices = hierarchy.gen_ds2node_map(hierarchy.ood_train_classes)
        ood_terminal_names = {
            hierarchy.id_node_list[index] for index in mapped_indices.tolist()
        }
        terminal_names = id_leaf_names | ood_terminal_names
        setattr(hierarchy, cache_name, terminal_names)

    child_counts = []
    for child in children:
        child_idx = hierarchy.id_node_list.index(child)
        count = sum(
            terminal == child
            or child_idx in hierarchy.node_ancestors[terminal]
            for terminal in terminal_names
        )
        child_counts.append(float(count))
    child_counts_tensor = torch.tensor(
        child_counts,
        dtype=dtype,
        device=device,
    )
    ood_count = torch.tensor(
        float(parent_name in terminal_names),
        dtype=dtype,
        device=device,
    )
    total = child_counts_tensor.sum() + ood_count
    return child_counts_tensor / total, ood_count / total


def _mixed_balanced_terminal_candidate_priors(
    parent_name: str,
    children: list[str],
    hierarchy,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_name = "_cgm_mixed_balanced_terminal_weights"
    terminal_weights = getattr(hierarchy, cache_name, None)
    if terminal_weights is None:
        id_leaf_names = {
            name
            for name in hierarchy.id_node_list
            if name not in hierarchy.parent2children
        }
        mapped_indices = hierarchy.gen_ds2node_map(hierarchy.ood_train_classes)
        ood_terminal_names = {
            hierarchy.id_node_list[index] for index in mapped_indices.tolist()
        }
        id_weight = 0.5 / len(id_leaf_names)
        ood_weight = 0.5 / len(ood_terminal_names)
        terminal_weights = {
            name: id_weight for name in id_leaf_names
        }
        for name in ood_terminal_names:
            terminal_weights[name] = terminal_weights.get(name, 0.0) + ood_weight
        setattr(hierarchy, cache_name, terminal_weights)

    child_masses = []
    for child in children:
        child_idx = hierarchy.id_node_list.index(child)
        mass = sum(
            weight
            for terminal, weight in terminal_weights.items()
            if terminal == child
            or child_idx in hierarchy.node_ancestors[terminal]
        )
        child_masses.append(mass)
    child_masses_tensor = torch.tensor(
        child_masses,
        dtype=dtype,
        device=device,
    )
    ood_mass = torch.tensor(
        terminal_weights.get(parent_name, 0.0),
        dtype=dtype,
        device=device,
    )
    total = child_masses_tensor.sum() + ood_mass
    return child_masses_tensor / total, ood_mass / total


def _vmf_kappas(
    density_payload: dict[str, Any],
    scale: float,
) -> torch.Tensor:
    cache = density_payload.setdefault("_cgm_vmf_kappa_cache", {})
    if scale not in cache:
        cache[scale] = estimate_vmf_kappas(density_payload["means"]) * scale
    return cache[scale]


def _cgm_vmf_local_probabilities_for_node(
    parent_name: str,
    features: torch.Tensor,
    hierarchy,
    node_logpdf: torch.Tensor,
    density_payload: dict[str, Any],
    cgm_cfg: dict[str, Any],
):
    children = hierarchy.parent2children.get(parent_name, [])
    if not children:
        return None

    eps = float(cgm_cfg.get("eps", 1e-12))
    lam = float(cgm_cfg.get("lambda", 0.9))
    child_weight_mode = cgm_cfg.get("child_weight", "uniform").lower()
    candidate_prior_mode = cgm_cfg.get("candidate_prior", "uniform").lower()
    kappa_scale = float(cgm_cfg.get("vmf_kappa_scale", 1.0))
    ood_kappa_scale = float(cgm_cfg.get("vmf_ood_kappa_scale", 1.0))
    mask_kappa = float(cgm_cfg.get("vmf_mask_kappa", 100.0))
    ood_base_mode = cgm_cfg.get("vmf_ood_base", "parent").lower()
    if ood_base_mode not in {"parent", "child_mixture"}:
        raise ValueError(f"Unsupported CGM vmf_ood_base: {ood_base_mode}")
    if kappa_scale <= 0.0 or ood_kappa_scale <= 0.0 or mask_kappa <= 0.0:
        raise ValueError("CGM vMF concentration scales must be positive")

    parent_idx = hierarchy.id_node_list.index(parent_name)
    child_indices = [hierarchy.id_node_list.index(child) for child in children]
    directions = density_payload.get("mean_directions")
    if directions is None:
        directions = torch.nn.functional.normalize(
            density_payload["means"],
            dim=1,
            eps=eps,
        )
    kappas = _vmf_kappas(density_payload, kappa_scale)
    parent_direction = directions[parent_idx]
    normalized_features = torch.nn.functional.normalize(features, dim=1, eps=eps)
    child_logpdf = node_logpdf[:, child_indices]
    parent_logpdf = node_logpdf[:, parent_idx]
    child_weights = _cgm_child_weights(
        child_indices,
        density_payload,
        child_weight_mode,
        dtype=features.dtype,
        device=features.device,
    )
    if ood_base_mode == "child_mixture":
        component_directions = directions[child_indices]
        component_kappas = kappas[child_indices] * ood_kappa_scale
        component_logpdf = vmf_logpdf(
            features,
            component_directions,
            component_kappas,
        )
        ood_base_logpdf = torch.logsumexp(
            component_logpdf
            + torch.log(child_weights.clamp_min(eps)).unsqueeze(0),
            dim=1,
        )
        parent_direction = torch.sum(
            child_weights.unsqueeze(1) * component_directions,
            dim=0,
        )
        base_kappa = torch.sum(child_weights * component_kappas)
    else:
        base_kappa = kappas[parent_idx] * ood_kappa_scale
        normalized_parent = torch.nn.functional.normalize(
            parent_direction,
            dim=0,
            eps=eps,
        )
        ood_base_logpdf = (
            base_kappa * (normalized_features @ normalized_parent)
            + vmf_log_normalizer(base_kappa.reshape(1), features.shape[1])[0]
        )
    child_bumps = vmf_bump(
        features,
        directions[child_indices],
        mask_kappa,
    )
    mask_driver = (child_bumps * child_weights.unsqueeze(0)).sum(dim=1)
    mask = 1.0 - lam * mask_driver
    log_mask = torch.log(mask.clamp_min(eps))
    integral_cache = density_payload.setdefault("_cgm_vmf_bump_integral_cache", {})
    integral_cache_key = (
        parent_idx,
        ood_base_mode,
        float(kappa_scale),
        float(ood_kappa_scale),
        float(mask_kappa),
        child_weight_mode,
    )
    bump_integrals = integral_cache.get(integral_cache_key)
    if bump_integrals is None:
        if ood_base_mode == "child_mixture":
            component_integrals = torch.stack(
                [
                    vmf_bump_integrals(
                        component_direction,
                        component_kappa,
                        directions[child_indices],
                        mask_kappa,
                    )
                    for component_direction, component_kappa in zip(
                        component_directions,
                        component_kappas,
                    )
                ],
                dim=0,
            )
            bump_integrals = torch.sum(
                child_weights.unsqueeze(1) * component_integrals,
                dim=0,
            )
        else:
            bump_integrals = vmf_bump_integrals(
                parent_direction,
                base_kappa,
                directions[child_indices],
                mask_kappa,
            )
        integral_cache[integral_cache_key] = bump_integrals
    normalizer = 1.0 - lam * torch.sum(child_weights * bump_integrals)
    log_normalizer = torch.log(normalizer.clamp_min(eps))
    ood_logpdf = ood_base_logpdf + log_mask - log_normalizer

    child_logits = child_logpdf
    density_ood_logit = ood_logpdf
    if candidate_prior_mode in {
        "hierarchy_leaf_count",
        "balanced_terminal",
        "mixed_balanced_terminal",
    }:
        prior_fn = {
            "hierarchy_leaf_count": _hierarchy_leaf_candidate_priors,
            "balanced_terminal": _balanced_terminal_candidate_priors,
            "mixed_balanced_terminal": _mixed_balanced_terminal_candidate_priors,
        }[candidate_prior_mode]
        child_candidate_priors, ood_candidate_prior = prior_fn(
            parent_name,
            children,
            hierarchy,
            dtype=features.dtype,
            device=features.device,
        )
        child_logits = child_logits + torch.log(
            child_candidate_priors.clamp_min(eps)
        ).unsqueeze(0)
        if bool(ood_candidate_prior.item() > 0):
            density_ood_logit = density_ood_logit + torch.log(ood_candidate_prior)
        else:
            density_ood_logit = torch.full_like(density_ood_logit, -torch.inf)
    else:
        prior = 1.0 / (len(children) + 1)
        child_candidate_priors = torch.full(
            (len(children),),
            prior,
            dtype=features.dtype,
            device=features.device,
        )
        ood_candidate_prior = torch.tensor(
            prior,
            dtype=features.dtype,
            device=features.device,
        )
        density_ood_logit = density_ood_logit + math.log(
            float(cgm_cfg.get("ood_prior", 1.0))
        )

    density_logits = torch.cat(
        [child_logits, density_ood_logit.unsqueeze(1)],
        dim=1,
    )
    density_probs = torch.softmax(density_logits, dim=1)
    log_child_weights = torch.log(child_weights.clamp_min(eps))
    child_mixture_logpdf = torch.logsumexp(
        child_logpdf + log_child_weights.unsqueeze(0),
        dim=1,
    )
    empty_vector = torch.empty((0,), dtype=features.dtype, device=features.device)
    empty_matrix = torch.empty((0, 0), dtype=features.dtype, device=features.device)
    return {
        "child_indices": child_indices,
        "child_probs": density_probs[:, :-1],
        "ood_prob": density_probs[:, -1],
        "child_logpdf": child_logpdf,
        "parent_logpdf": parent_logpdf,
        "ood_base_logpdf": ood_base_logpdf,
        "ood_base_cov_scale": ood_kappa_scale,
        "mask_cov_scale": mask_kappa,
        "between_cov_scale": 0.0,
        "between_cov_estimator": f"vmf_{ood_base_mode}",
        "between_cov_shrinkage_strength": 0.0,
        "product_mask_samples": 0,
        "random_effects_weight": 0.0,
        "ood_base_mean": parent_direction,
        "random_effects_covariance": empty_matrix,
        "child_mixture_logpdf": child_mixture_logpdf,
        "child_bumps": child_bumps,
        "child_weights": child_weights,
        "mask_driver": mask_driver,
        "mask": mask,
        "log_mask": log_mask,
        "bump_integrals": bump_integrals,
        "normalizer": normalizer,
        "ood_prior": 1.0,
        "candidate_prior_mode": candidate_prior_mode,
        "child_candidate_priors": child_candidate_priors,
        "ood_candidate_prior": ood_candidate_prior,
        "ood_density": "vmf_parent_mask",
        "complement_reduce": "sum",
        "complement_weight": 0.0,
        "complement_indices": [],
        "parent_covariance_scales": [],
        "parent_scale_weights": empty_vector,
        "parent_scale_normalizers": empty_vector,
        "local_mode": "density_softmax",
        "child_log_scale": 1.0,
        "ood_log_scale": 1.0,
        "gate_log_scale": 1.0,
        "gate_bias": 0.0,
        "blend_weight": 0.0,
        "child_logits": child_logits,
        "ood_logit": density_ood_logit,
        "ood_logpdf": ood_logpdf,
    }


def _cgm_local_probabilities_for_node(
    parent_name: str,
    features: torch.Tensor,
    hierarchy,
    node_logpdf: torch.Tensor,
    density_payload: dict[str, Any],
    cgm_cfg: dict[str, Any],
    depth_probs: dict[int, torch.Tensor] | None = None,
    nodes_by_depth: dict[int, list[int]] | None = None,
):
    if cgm_cfg.get("density_family", "gaussian").lower() == "vmf":
        return _cgm_vmf_local_probabilities_for_node(
            parent_name,
            features,
            hierarchy,
            node_logpdf,
            density_payload,
            cgm_cfg,
        )
    children = hierarchy.parent2children.get(parent_name, [])
    if not children:
        return None

    eps = float(cgm_cfg.get("eps", 1e-12))
    lam = float(cgm_cfg.get("lambda", 0.9))
    raw_ood_prior = cgm_cfg.get("ood_prior", 1.0)
    child_log_scale = float(cgm_cfg.get("child_log_scale", 1.0))
    ood_log_scale = float(cgm_cfg.get("ood_log_scale", 1.0))
    if child_log_scale <= 0.0 or ood_log_scale <= 0.0:
        raise ValueError("CGM child_log_scale and ood_log_scale must be positive")
    mask_type = cgm_cfg.get("mask_type", "sum").lower()
    ood_density = cgm_cfg.get("ood_density", "parent_mask").lower()
    membership_type = cgm_cfg.get(
        "membership_type",
        "chi2_survival" if ood_density == "softset_complement" else "peak_bump",
    ).lower()
    membership_probability = float(cgm_cfg.get("membership_probability", 0.95))
    membership_correction = cgm_cfg.get("membership_correction", "none").lower()
    complement_reduce = cgm_cfg.get("complement_reduce", "sum").lower()
    local_mode = cgm_cfg.get("local_mode", "density_softmax").lower()
    child_weight_mode = cgm_cfg.get("child_weight", "uniform").lower()
    candidate_prior_mode = cgm_cfg.get("candidate_prior", "uniform").lower()
    normalize_ood_pdf = bool(cgm_cfg.get("normalize_ood_pdf", True))
    ood_base_cov_scale = float(cgm_cfg.get("ood_base_cov_scale", 1.0))
    if ood_base_cov_scale <= 0.0:
        raise ValueError("CGM ood_base_cov_scale must be positive")
    mask_cov_scale = float(
        cgm_cfg.get("mask_cov_scale", cgm_cfg.get("mask_covariance_scale", 1.0))
    )
    if mask_cov_scale <= 0.0:
        raise ValueError("CGM mask_cov_scale must be positive")
    raw_between_cov_scale = cgm_cfg.get("between_cov_scale", 1.0)
    between_cov_estimator = cgm_cfg.get("between_cov_estimator", "parent_residual").lower()
    between_cov_shrinkage_strength = float(
        cgm_cfg.get("between_cov_shrinkage_strength", 3.0)
    )
    product_mask_samples = int(cgm_cfg.get("product_mask_samples", 1024))
    gate_log_scale = float(cgm_cfg.get("gate_log_scale", 1.0))
    if gate_log_scale <= 0.0:
        raise ValueError("CGM gate_log_scale must be positive")
    gate_bias = float(cgm_cfg.get("gate_bias", 0.0))
    blend_weight = float(cgm_cfg.get("blend_weight", 0.5))
    if blend_weight < 0.0 or blend_weight > 1.0:
        raise ValueError("CGM blend_weight must satisfy 0 <= blend_weight <= 1")
    complement_weight = float(cgm_cfg.get("complement_weight", 0.5))
    if complement_weight < 0.0 or complement_weight > 1.0:
        raise ValueError("CGM complement_weight must satisfy 0 <= complement_weight <= 1")
    random_effects_weight = float(cgm_cfg.get("random_effects_weight", 0.5))
    if random_effects_weight < 0.0 or random_effects_weight > 1.0:
        raise ValueError("CGM random_effects_weight must satisfy 0 <= value <= 1")
    parent_covariance_scales = [
        float(value) for value in cgm_cfg.get("parent_covariance_scales", [1.0, 2.0, 4.0])
    ]
    if not parent_covariance_scales or any(value <= 0.0 for value in parent_covariance_scales):
        raise ValueError("CGM parent_covariance_scales must contain positive values")
    raw_parent_scale_weights = cgm_cfg.get("parent_scale_weights")
    if raw_parent_scale_weights is None:
        parent_scale_weights = torch.full(
            (len(parent_covariance_scales),),
            1.0 / len(parent_covariance_scales),
            dtype=features.dtype,
            device=features.device,
        )
    else:
        if len(raw_parent_scale_weights) != len(parent_covariance_scales):
            raise ValueError("CGM parent_scale_weights must match parent_covariance_scales")
        parent_scale_weights = torch.tensor(
            [float(value) for value in raw_parent_scale_weights],
            dtype=features.dtype,
            device=features.device,
        )
        if bool(torch.any(parent_scale_weights < 0).item()) or not bool(
            parent_scale_weights.sum().item() > 0
        ):
            raise ValueError("CGM parent_scale_weights must be nonnegative with a positive sum")
        parent_scale_weights = parent_scale_weights / parent_scale_weights.sum()

    parent_idx = hierarchy.id_node_list.index(parent_name)
    child_indices = [hierarchy.id_node_list.index(child) for child in children]
    local_membership_probability = membership_probability
    if membership_correction == "sidak":
        local_membership_probability = membership_probability ** (
            1.0 / max(len(child_indices), 1)
        )
    child_depth = len(hierarchy.node_ancestors[children[0]])
    max_depth = max(len(hierarchy.node_ancestors[name]) for name in hierarchy.id_node_list)
    ood_prior = _resolve_cgm_ood_prior(raw_ood_prior, child_depth, max_depth)
    between_cov_scale = _resolve_cgm_between_cov_scale(
        raw_between_cov_scale,
        child_depth,
        max_depth,
    )
    covariance_type = density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type", "diag"))
    means = density_payload["means"]
    variances = density_payload.get("variances")
    covariance_matrices = density_payload.get("covariance_matrices")
    shared_covariance = density_payload.get("shared_covariance")

    child_logpdf = node_logpdf[:, child_indices]
    parent_logpdf = node_logpdf[:, parent_idx]
    random_effects_covariance = torch.empty(
        (0, 0),
        dtype=features.dtype,
        device=features.device,
    )
    ood_base_mean = means[parent_idx]
    if ood_density in {
        "random_effects_parent",
        "random_effects_mixture",
        "random_complement_mixture",
        "poe_random_effects_parent",
    }:
        ood_base_logpdf = parent_logpdf
    elif abs(ood_base_cov_scale - 1.0) <= 1e-12:
        ood_base_logpdf = parent_logpdf
    else:
        ood_base_logpdf = gaussian_logpdf(
            features,
            means,
            variances,
            covariance_matrices=covariance_matrices,
            shared_covariance=shared_covariance,
            covariance_type=covariance_type,
            node_indices=[parent_idx],
            covariance_scale=ood_base_cov_scale,
        ).squeeze(1)
    if membership_type == "chi2_survival":
        membership_fn = gaussian_chi2_survival_membership
        membership_kwargs = {}
    elif membership_type == "chi2_ellipsoid":
        membership_fn = gaussian_chi2_ellipsoid_membership
        membership_kwargs = {"probability_mass": local_membership_probability}
    else:
        membership_fn = gaussian_bump
        membership_kwargs = {}
    membership_features = features
    if density_payload.get("density_feature_mask_type") == "depth_fisher":
        depth_masks = density_payload.get("depth_feature_masks", {})
        mask_dims = depth_masks.get(int(child_depth))
        if mask_dims is None:
            raise ValueError(f"Missing depth feature mask for child depth {child_depth}")
        if not torch.is_tensor(mask_dims):
            mask_dims = torch.tensor(mask_dims, dtype=torch.long, device=features.device)
        else:
            mask_dims = mask_dims.to(device=features.device, dtype=torch.long)
        membership_features = features[:, mask_dims]
    elif density_payload.get("density_feature_mask_type") == "depth_fisher_reweight":
        depth_weights = density_payload.get("depth_feature_weights", {})
        weights = depth_weights.get(int(child_depth))
        if weights is None:
            raise ValueError(f"Missing depth feature weights for child depth {child_depth}")
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights, dtype=features.dtype, device=features.device)
        else:
            weights = weights.to(device=features.device, dtype=features.dtype)
        membership_features = features * weights.unsqueeze(0)
    child_bumps = membership_fn(
        membership_features,
        means,
        variances,
        covariance_matrices=covariance_matrices,
        shared_covariance=shared_covariance,
        covariance_type=covariance_type,
        node_indices=child_indices,
        covariance_scale=mask_cov_scale,
        **membership_kwargs,
    )
    mask_lam = 1.0 if ood_density == "softset_complement" else lam
    child_weights = _cgm_child_weights(
        child_indices,
        density_payload,
        child_weight_mode,
        dtype=features.dtype,
        device=features.device,
    )
    if ood_density in {"conformal_maxprob_stop", "conformal_logpdf_stop"}:
        calibration_key = (
            "conformal_posterior_stop"
            if ood_density == "conformal_maxprob_stop"
            else "conformal_logpdf_stop"
        )
        calibration_root = density_payload.get("cgm_calibration", {})
        calibration = calibration_root.get(calibration_key, {})
        if not calibration and ood_density == "conformal_maxprob_stop":
            calibration = calibration_root.get("conformal_maxprob_stop", {})
        thresholds = calibration.get("thresholds")
        if thresholds is None:
            raise ValueError(
                f"{ood_density} requires density_payload['cgm_calibration']['{calibration_key}']"
            )
        threshold = float(thresholds.get(parent_name, 1.0))
        log_child_weights = torch.log(child_weights.clamp_min(eps))
        child_logits = child_logpdf + log_child_weights.unsqueeze(0)
        child_posteriors = torch.softmax(child_logits, dim=1)
        confidence = child_posteriors.max(dim=1).values
        if ood_density == "conformal_maxprob_stop":
            nonconformity = 1.0 - confidence
        else:
            nonconformity = -child_logpdf.max(dim=1).values
        stop = (nonconformity > threshold).to(features.dtype)
        child_probs = child_posteriors * (1.0 - stop).unsqueeze(1)
        empty_vector = torch.empty((0,), dtype=features.dtype, device=features.device)
        empty_matrix = torch.empty((0, 0), dtype=features.dtype, device=features.device)
        return {
            "child_indices": child_indices,
            "child_probs": child_probs,
            "ood_prob": stop,
            "child_logpdf": child_logpdf,
            "parent_logpdf": parent_logpdf,
            "ood_base_logpdf": parent_logpdf,
            "ood_base_cov_scale": 1.0,
            "mask_cov_scale": mask_cov_scale,
            "between_cov_scale": 0.0,
            "between_cov_estimator": ood_density,
            "between_cov_shrinkage_strength": 0.0,
            "product_mask_samples": 0,
            "random_effects_weight": 0.0,
            "ood_base_mean": means[parent_idx],
            "random_effects_covariance": empty_matrix,
            "child_mixture_logpdf": torch.logsumexp(child_logits, dim=1),
            "child_bumps": child_posteriors,
            "child_weights": child_weights,
            "mask_driver": nonconformity,
            "mask": stop.unsqueeze(1).expand_as(child_logpdf),
            "log_mask": torch.log(stop.clamp_min(eps)).unsqueeze(1).expand_as(child_logpdf),
            "bump_integrals": empty_vector,
            "normalizer": torch.tensor(threshold, dtype=features.dtype, device=features.device),
            "ood_prior": "id_conformal_coverage",
            "candidate_prior_mode": ood_density,
            "child_candidate_priors": child_weights,
            "ood_candidate_prior": torch.tensor(1.0, dtype=features.dtype, device=features.device),
            "ood_density": ood_density,
            "complement_reduce": ood_density,
            "complement_weight": 0.0,
            "complement_indices": [],
            "parent_covariance_scales": [],
            "parent_scale_weights": empty_vector,
            "parent_scale_normalizers": empty_vector,
            "local_mode": local_mode,
            "child_log_scale": 1.0,
            "ood_log_scale": 1.0,
            "gate_log_scale": gate_log_scale,
            "gate_bias": gate_bias,
            "blend_weight": blend_weight,
            "child_logits": child_logits,
            "ood_logit": torch.log(stop.clamp_min(eps)),
            "ood_logpdf": torch.log(stop.clamp_min(eps)),
        }
    if ood_density in {"posterior_entropy_mixture", "posterior_residual_mixture"}:
        log_child_weights = torch.log(child_weights.clamp_min(eps))
        child_logits = child_logpdf + log_child_weights.unsqueeze(0)
        child_mixture_logpdf = torch.logsumexp(child_logits, dim=1)
        child_posteriors = torch.softmax(child_logits, dim=1)
        if ood_density == "posterior_entropy_mixture":
            entropy = -torch.sum(
                child_posteriors * torch.log(child_posteriors.clamp_min(eps)),
                dim=1,
            )
            if len(child_indices) > 1:
                ambiguity = entropy / math.log(float(len(child_indices)))
            else:
                ambiguity = torch.zeros_like(entropy)
            estimator_name = "posterior_entropy_mixture"
        else:
            ambiguity = 1.0 - child_posteriors.max(dim=1).values
            estimator_name = "posterior_residual_mixture"
        density_ood_logit = child_mixture_logpdf + torch.log(ambiguity.clamp_min(eps))
        density_logits = torch.cat([child_logits, density_ood_logit.unsqueeze(1)], dim=1)
        density_probs = torch.softmax(density_logits, dim=1)
        empty_vector = torch.empty((0,), dtype=features.dtype, device=features.device)
        empty_matrix = torch.empty((0, 0), dtype=features.dtype, device=features.device)
        return {
            "child_indices": child_indices,
            "child_probs": density_probs[:, :-1],
            "ood_prob": density_probs[:, -1],
            "child_logpdf": child_logpdf,
            "parent_logpdf": parent_logpdf,
            "ood_base_logpdf": child_mixture_logpdf,
            "ood_base_cov_scale": 1.0,
            "mask_cov_scale": mask_cov_scale,
            "between_cov_scale": 0.0,
            "between_cov_estimator": estimator_name,
            "between_cov_shrinkage_strength": 0.0,
            "product_mask_samples": 0,
            "random_effects_weight": 0.0,
            "ood_base_mean": means[parent_idx],
            "random_effects_covariance": empty_matrix,
            "child_mixture_logpdf": child_mixture_logpdf,
            "child_bumps": child_posteriors,
            "child_weights": child_weights,
            "mask_driver": ambiguity,
            "mask": ambiguity.unsqueeze(1).expand_as(child_logpdf),
            "log_mask": torch.log(ambiguity.clamp_min(eps)).unsqueeze(1).expand_as(child_logpdf),
            "bump_integrals": empty_vector,
            "normalizer": torch.tensor(1.0, dtype=features.dtype, device=features.device),
            "ood_prior": estimator_name,
            "candidate_prior_mode": ood_density,
            "child_candidate_priors": child_weights,
            "ood_candidate_prior": torch.tensor(1.0, dtype=features.dtype, device=features.device),
            "ood_density": ood_density,
            "complement_reduce": estimator_name,
            "complement_weight": 0.0,
            "complement_indices": [],
            "parent_covariance_scales": [],
            "parent_scale_weights": empty_vector,
            "parent_scale_normalizers": empty_vector,
            "local_mode": local_mode,
            "child_log_scale": 1.0,
            "ood_log_scale": 1.0,
            "gate_log_scale": gate_log_scale,
            "gate_bias": gate_bias,
            "blend_weight": blend_weight,
            "child_logits": child_logits,
            "ood_logit": density_ood_logit,
            "ood_logpdf": density_ood_logit,
        }
    if ood_density in {
        "child_tail_mixture",
        "empirical_child_tail_mixture",
        "soft_child_tail_mixture",
    }:
        log_child_weights = torch.log(child_weights.clamp_min(eps))
        if ood_density == "empirical_child_tail_mixture":
            calibration = (
                density_payload.get("cgm_calibration", {})
                .get("conformal_logpdf_stop", {})
            )
            node_thresholds = calibration.get("node_thresholds")
            if node_thresholds is None:
                raise ValueError(
                    "empirical_child_tail_mixture requires conformal_logpdf_stop node_thresholds"
                )
            threshold_values = torch.tensor(
                [
                    float(node_thresholds.get(hierarchy.id_node_list[idx], float("inf")))
                    for idx in child_indices
                ],
                dtype=features.dtype,
                device=features.device,
            )
            tail_masks = (-child_logpdf > threshold_values.unsqueeze(0)).to(features.dtype)
            child_bumps = 1.0 - tail_masks
        else:
            safe_max = 1.0 - max(eps, torch.finfo(child_bumps.dtype).eps)
            tail_masks = 1.0 - torch.nan_to_num(
                child_bumps,
                nan=0.0,
                posinf=safe_max,
                neginf=0.0,
            ).clamp(min=0.0, max=safe_max)
        component_tail_logits = (
            child_logpdf
            + log_child_weights.unsqueeze(0)
            + torch.log(tail_masks.clamp_min(eps))
        )
        tail_joint_logpdf = torch.logsumexp(component_tail_logits, dim=1)
        if ood_density == "soft_child_tail_mixture":
            normalizer = torch.tensor(0.5, dtype=features.dtype, device=features.device)
            correction_name = "chi2_cdf_tail"
        elif ood_density == "empirical_child_tail_mixture":
            normalizer = torch.tensor(
                1.0 - float(
                    density_payload.get("cgm_calibration", {})
                    .get("conformal_logpdf_stop", {})
                    .get("coverage", local_membership_probability)
                ),
                dtype=features.dtype,
                device=features.device,
            ).clamp_min(eps)
            correction_name = "empirical_logpdf_tail"
        else:
            normalizer = torch.tensor(
                1.0 - local_membership_probability,
                dtype=features.dtype,
                device=features.device,
            ).clamp_min(eps)
            correction_name = "hard_ellipsoid_tail"
        ood_logpdf = tail_joint_logpdf - torch.log(normalizer)
        child_logits = child_logpdf + log_child_weights.unsqueeze(0)
        if local_mode == "density_joint":
            density_ood_logit = tail_joint_logpdf
        else:
            density_ood_logit = ood_logpdf
        density_logits = torch.cat([child_logits, density_ood_logit.unsqueeze(1)], dim=1)
        density_probs = torch.softmax(density_logits, dim=1)
        empty_vector = torch.empty((0,), dtype=features.dtype, device=features.device)
        empty_matrix = torch.empty((0, 0), dtype=features.dtype, device=features.device)
        return {
            "child_indices": child_indices,
            "child_probs": density_probs[:, :-1],
            "ood_prob": density_probs[:, -1],
            "child_logpdf": child_logpdf,
            "parent_logpdf": parent_logpdf,
            "ood_base_logpdf": tail_joint_logpdf,
            "ood_base_cov_scale": 1.0,
            "mask_cov_scale": mask_cov_scale,
            "between_cov_scale": 0.0,
            "between_cov_estimator": correction_name,
            "between_cov_shrinkage_strength": 0.0,
            "product_mask_samples": 0,
            "random_effects_weight": 0.0,
            "ood_base_mean": means[parent_idx],
            "random_effects_covariance": empty_matrix,
            "child_mixture_logpdf": torch.logsumexp(child_logits, dim=1),
            "child_bumps": child_bumps,
            "child_weights": child_weights,
            "mask_driver": tail_masks.sum(dim=1),
            "mask": tail_masks,
            "log_mask": torch.log(tail_masks.clamp_min(eps)),
            "bump_integrals": empty_vector,
            "normalizer": normalizer,
            "ood_prior": "induced_child_tail_mass",
            "candidate_prior_mode": ood_density,
            "child_candidate_priors": child_weights,
            "ood_candidate_prior": normalizer,
            "ood_density": ood_density,
            "complement_reduce": "mixture_tail",
            "complement_weight": 0.0,
            "complement_indices": [],
            "parent_covariance_scales": [],
            "parent_scale_weights": empty_vector,
            "parent_scale_normalizers": empty_vector,
            "local_mode": local_mode,
            "child_log_scale": 1.0,
            "ood_log_scale": 1.0,
            "gate_log_scale": gate_log_scale,
            "gate_bias": gate_bias,
            "blend_weight": blend_weight,
            "child_logits": child_logits,
            "ood_logit": density_ood_logit,
            "ood_logpdf": ood_logpdf,
        }
    if ood_density == "positive_density_residual":
        log_child_weights = torch.log(child_weights.clamp_min(eps))
        child_logits = child_logpdf + log_child_weights.unsqueeze(0)
        child_mixture_logpdf = torch.logsumexp(child_logits, dim=1)

        residual_support = parent_logpdf > child_mixture_logpdf
        log_ratio = child_mixture_logpdf - parent_logpdf
        safe_log_ratio = log_ratio.clamp(max=-torch.finfo(features.dtype).eps)
        positive_log_residual = parent_logpdf + torch.log(-torch.expm1(safe_log_ratio))
        ood_logpdf = torch.where(
            residual_support,
            positive_log_residual,
            torch.full_like(positive_log_residual, -torch.inf),
        )
        density_logits = torch.cat([child_logits, ood_logpdf.unsqueeze(1)], dim=1)
        density_probs = torch.softmax(density_logits, dim=1)

        mixture_to_parent_ratio = torch.exp(log_ratio.clamp(max=0.0))
        mask = torch.where(
            residual_support,
            1.0 - mixture_to_parent_ratio,
            torch.zeros_like(mixture_to_parent_ratio),
        )
        log_mask = torch.where(
            residual_support,
            torch.log(mask.clamp_min(eps)),
            torch.full_like(mask, -torch.inf),
        )
        empty_vector = torch.empty((0,), dtype=features.dtype, device=features.device)
        empty_matrix = torch.empty((0, 0), dtype=features.dtype, device=features.device)
        return {
            "child_indices": child_indices,
            "child_probs": density_probs[:, :-1],
            "ood_prob": density_probs[:, -1],
            "child_logpdf": child_logpdf,
            "parent_logpdf": parent_logpdf,
            "ood_base_logpdf": parent_logpdf,
            "ood_base_cov_scale": 1.0,
            "mask_cov_scale": 1.0,
            "between_cov_scale": 0.0,
            "between_cov_estimator": "none",
            "between_cov_shrinkage_strength": 0.0,
            "product_mask_samples": 0,
            "random_effects_weight": 0.0,
            "ood_base_mean": means[parent_idx],
            "random_effects_covariance": empty_matrix,
            "child_mixture_logpdf": child_mixture_logpdf,
            "child_bumps": child_bumps,
            "child_weights": child_weights,
            "mask_driver": mixture_to_parent_ratio,
            "mask": mask,
            "log_mask": log_mask,
            "bump_integrals": empty_vector,
            "normalizer": torch.tensor(float("nan"), dtype=features.dtype, device=features.device),
            "ood_prior": "induced_residual_mass",
            "candidate_prior_mode": "induced_residual",
            "child_candidate_priors": child_weights,
            "ood_candidate_prior": torch.tensor(float("nan"), dtype=features.dtype, device=features.device),
            "ood_density": ood_density,
            "complement_reduce": "sum",
            "complement_weight": 0.0,
            "complement_indices": [],
            "parent_covariance_scales": [],
            "parent_scale_weights": empty_vector,
            "parent_scale_normalizers": empty_vector,
            "local_mode": "density_softmax",
            "child_log_scale": 1.0,
            "ood_log_scale": 1.0,
            "gate_log_scale": 1.0,
            "gate_bias": 0.0,
            "blend_weight": 0.0,
            "child_logits": child_logits,
            "ood_logit": ood_logpdf,
            "ood_logpdf": ood_logpdf,
        }
    if ood_density in {
        "random_effects_parent",
        "random_effects_mixture",
        "random_complement_mixture",
        "poe_random_effects_parent",
    }:
        parent_covariance = gaussian_covariance_matrices(
            variances,
            covariance_matrices,
            shared_covariance,
            covariance_type,
            [parent_idx],
        )[0]
        child_covariances = gaussian_covariance_matrices(
            variances,
            covariance_matrices,
            shared_covariance,
            covariance_type,
            child_indices,
        )
        within_covariance = torch.sum(
            child_weights[:, None, None] * child_covariances,
            dim=0,
        )
        if between_cov_estimator in {"empirical_child_means", "shrunk_child_means"}:
            child_means = means[child_indices]
            ood_base_mean = torch.sum(child_weights.unsqueeze(1) * child_means, dim=0)
            centered_means = child_means - ood_base_mean.unsqueeze(0)
            between_covariance = torch.einsum(
                "n,nd,ne->de",
                child_weights,
                centered_means,
                centered_means,
            )
            correction = 1.0 - torch.sum(child_weights.square())
            if bool(correction.item() > eps):
                between_covariance = between_covariance / correction
            if between_cov_estimator == "shrunk_child_means":
                pooled_covariance = _depth_pooled_child_mean_covariance(
                    child_depth,
                    hierarchy,
                    density_payload,
                    child_weight_mode,
                    eps,
                )
                local_degrees_of_freedom = max(len(child_indices) - 1, 0)
                denominator = (
                    local_degrees_of_freedom + between_cov_shrinkage_strength
                )
                local_weight = (
                    local_degrees_of_freedom / denominator
                    if denominator > 0.0
                    else 1.0
                )
                between_covariance = (
                    local_weight * between_covariance
                    + (1.0 - local_weight) * pooled_covariance
                )
        else:
            between_raw = 0.5 * (
                parent_covariance - within_covariance
                + (parent_covariance - within_covariance).transpose(0, 1)
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(between_raw)
            between_covariance = (
                eigenvectors * eigenvalues.clamp_min(0.0).unsqueeze(0)
            ) @ eigenvectors.transpose(0, 1)
        eye = torch.eye(
            parent_covariance.shape[0],
            dtype=features.dtype,
            device=features.device,
        )
        random_effects_covariance = (
            ood_base_cov_scale
            * (within_covariance + between_cov_scale * between_covariance)
            + eps * eye
        )
        ood_base_logpdf = gaussian_logpdf_from_covariance(
            features,
            ood_base_mean,
            random_effects_covariance,
        )
        if ood_density == "poe_random_effects_parent":
            parent_precision = torch.linalg.inv(parent_covariance + eps * eye)
            random_precision = torch.linalg.inv(random_effects_covariance)
            poe_precision = 0.5 * (parent_precision + random_precision)
            random_natural = random_precision @ ood_base_mean
            parent_natural = parent_precision @ means[parent_idx]
            poe_covariance = torch.linalg.inv(poe_precision + eps * eye)
            poe_mean = poe_covariance @ (0.5 * (parent_natural + random_natural))
            random_effects_covariance = 0.5 * (
                poe_covariance + poe_covariance.transpose(0, 1)
            ) + eps * eye
            ood_base_mean = poe_mean
            ood_base_logpdf = gaussian_logpdf_from_covariance(
                features,
                ood_base_mean,
                random_effects_covariance,
            )
    if mask_type == "sum":
        mask_driver = (child_bumps * child_weights.unsqueeze(0)).sum(dim=1)
        mask = 1.0 - mask_lam * mask_driver
        log_mask = torch.log(mask.clamp_min(eps))
    elif mask_type == "product":
        safe_max = 1.0 - max(eps, torch.finfo(child_bumps.dtype).eps)
        child_bumps = torch.nan_to_num(
            child_bumps,
            nan=0.0,
            posinf=safe_max,
            neginf=0.0,
        )
        safe_child_bumps = child_bumps.clamp(min=0.0, max=safe_max)
        mask_driver = safe_child_bumps.sum(dim=1)
        log_mask = torch.log1p(-mask_lam * safe_child_bumps).sum(dim=1)
        mask = torch.exp(log_mask)
    elif mask_type == "max":
        mask_driver = child_bumps.max(dim=1).values
        mask = 1.0 - mask_lam * mask_driver
        log_mask = torch.log(mask.clamp_min(eps))
    elif mask_type == "residual_sigmoid":
        log_child_weights = torch.log(child_weights.clamp_min(eps))
        child_mixture_logpdf = torch.logsumexp(child_logpdf + log_child_weights.unsqueeze(0), dim=1)
        mask_driver = torch.sigmoid(child_mixture_logpdf - parent_logpdf)
        mask = 1.0 - mask_lam * mask_driver
        log_mask = torch.log(mask.clamp_min(eps))
    else:
        raise ValueError(f"Unsupported CGM mask_type: {mask_type}")

    if normalize_ood_pdf and mask_type == "sum":
        if ood_density in {
            "random_effects_parent",
            "random_effects_mixture",
            "random_complement_mixture",
        }:
            bump_integrals = gaussian_bump_integrals_from_covariance(
                ood_base_mean,
                random_effects_covariance,
                means[child_indices],
                child_covariances * mask_cov_scale,
            )
        else:
            bump_integrals = gaussian_bump_integrals(
                parent_idx,
                child_indices,
                means,
                variances,
                covariance_matrices=covariance_matrices,
                shared_covariance=shared_covariance,
                covariance_type=covariance_type,
                parent_covariance_scale=ood_base_cov_scale,
                child_covariance_scale=mask_cov_scale,
            )
        normalizer = 1.0 - lam * torch.sum(child_weights * bump_integrals)
        log_normalizer = torch.log(normalizer.clamp_min(eps))
    elif normalize_ood_pdf and mask_type == "product":
        bump_integrals = torch.empty(
            (0,),
            dtype=features.dtype,
            device=features.device,
        )
        if random_effects_covariance.numel() == 0:
            normalizer_base_covariance = (
                gaussian_covariance_matrices(
                    variances,
                    covariance_matrices,
                    shared_covariance,
                    covariance_type,
                    [parent_idx],
                )[0]
                * ood_base_cov_scale
            )
        else:
            normalizer_base_covariance = random_effects_covariance
        log_normalizer = _product_mask_log_normalizer(
            parent_idx,
            ood_base_mean,
            normalizer_base_covariance,
            child_indices,
            density_payload,
            mask_cov_scale,
            mask_lam,
            product_mask_samples,
            eps,
            membership_type,
            local_membership_probability,
        )
        normalizer = torch.exp(log_normalizer)
    else:
        bump_integrals = torch.empty((len(child_indices),), dtype=features.dtype, device=features.device)
        normalizer = torch.ones((), dtype=features.dtype, device=features.device)
        log_normalizer = torch.zeros((), dtype=features.dtype, device=features.device)

    if local_mode == "binary_model_selection" or candidate_prior_mode in {
        "hierarchy_leaf_count",
        "balanced_terminal",
        "mixed_balanced_terminal",
    }:
        log_ood_prior = torch.zeros((), dtype=features.dtype, device=features.device)
    else:
        log_ood_prior = torch.log(
            torch.tensor(ood_prior, dtype=features.dtype, device=features.device)
        )
    parent_mask_logpdf = ood_base_logpdf + log_mask - log_normalizer
    complement_indices: list[int] = []
    parent_scale_normalizers = torch.empty((0,), dtype=features.dtype, device=features.device)
    if ood_density in {
        "parent_mask",
        "random_effects_parent",
        "poe_random_effects_parent",
        "softset_complement",
    }:
        ood_logpdf = parent_mask_logpdf + log_ood_prior
    elif ood_density == "random_effects_mixture":
        standard_bump_integrals = gaussian_bump_integrals(
            parent_idx,
            child_indices,
            means,
            variances,
            covariance_matrices=covariance_matrices,
            shared_covariance=shared_covariance,
            covariance_type=covariance_type,
            child_covariance_scale=mask_cov_scale,
        )
        standard_normalizer = 1.0 - lam * torch.sum(
            child_weights * standard_bump_integrals
        )
        standard_parent_logpdf = (
            parent_logpdf
            + log_mask
            - torch.log(standard_normalizer.clamp_min(eps))
        )
        if random_effects_weight <= 0.0:
            mixture_logpdf = standard_parent_logpdf
        elif random_effects_weight >= 1.0:
            mixture_logpdf = parent_mask_logpdf
        else:
            mixture_logpdf = torch.logaddexp(
                standard_parent_logpdf
                + torch.log(
                    torch.tensor(
                        1.0 - random_effects_weight,
                        dtype=features.dtype,
                        device=features.device,
                    )
                ),
                parent_mask_logpdf
                + torch.log(
                    torch.tensor(
                        random_effects_weight,
                        dtype=features.dtype,
                        device=features.device,
                    )
                ),
            )
        ood_logpdf = mixture_logpdf + log_ood_prior
    elif ood_density == "multiscale_parent_mask":
        component_logpdfs = []
        component_normalizers = []
        for covariance_scale in parent_covariance_scales:
            scaled_parent_logpdf = gaussian_logpdf(
                features,
                means,
                variances,
                covariance_matrices=covariance_matrices,
                shared_covariance=shared_covariance,
                covariance_type=covariance_type,
                node_indices=[parent_idx],
                covariance_scale=covariance_scale,
            ).squeeze(1)
            scaled_bump_integrals = gaussian_bump_integrals(
                parent_idx,
                child_indices,
                means,
                variances,
                covariance_matrices=covariance_matrices,
                shared_covariance=shared_covariance,
                covariance_type=covariance_type,
                parent_covariance_scale=covariance_scale,
                child_covariance_scale=mask_cov_scale,
            )
            scaled_normalizer = 1.0 - lam * torch.sum(child_weights * scaled_bump_integrals)
            component_normalizers.append(scaled_normalizer)
            component_logpdfs.append(
                scaled_parent_logpdf
                + log_mask
                - torch.log(scaled_normalizer.clamp_min(eps))
            )
        parent_scale_normalizers = torch.stack(component_normalizers)
        component_logpdf = torch.stack(component_logpdfs, dim=1)
        ood_logpdf = torch.logsumexp(
            component_logpdf + torch.log(parent_scale_weights.clamp_min(eps)).unsqueeze(0),
            dim=1,
        ) + log_ood_prior
    elif ood_density == "child_mixture_mask":
        base_component_logpdf = gaussian_logpdf(
            features,
            means,
            variances,
            covariance_matrices=covariance_matrices,
            shared_covariance=shared_covariance,
            covariance_type=covariance_type,
            node_indices=child_indices,
            covariance_scale=ood_base_cov_scale,
        )
        base_mixture_logpdf = torch.logsumexp(
            base_component_logpdf + torch.log(child_weights.clamp_min(eps)).unsqueeze(0),
            dim=1,
        )
        pairwise_integrals = torch.stack(
            [
                gaussian_bump_integrals(
                    base_child_idx,
                    child_indices,
                    means,
                    variances,
                    covariance_matrices=covariance_matrices,
                    shared_covariance=shared_covariance,
                    covariance_type=covariance_type,
                    parent_covariance_scale=ood_base_cov_scale,
                    child_covariance_scale=mask_cov_scale,
                )
                for base_child_idx in child_indices
            ],
            dim=0,
        )
        weighted_integral = torch.sum(
            child_weights.unsqueeze(1)
            * child_weights.unsqueeze(0)
            * pairwise_integrals
        )
        normalizer = 1.0 - lam * weighted_integral
        log_normalizer = torch.log(normalizer.clamp_min(eps))
        bump_integrals = pairwise_integrals
        ood_logpdf = base_mixture_logpdf + log_mask - log_normalizer + log_ood_prior
    elif ood_density in {
        "complement_mixture",
        "hybrid_mixture",
        "random_complement_mixture",
    }:
        if nodes_by_depth is None:
            raise ValueError("nodes_by_depth is required for complement-based CGM")
        depth_indices = nodes_by_depth[child_depth]
        child_index_set = set(child_indices)
        complement_indices = [idx for idx in depth_indices if idx not in child_index_set]
        complement_logpdf = torch.full_like(parent_logpdf, -torch.inf)
        if complement_indices:
            complement_logpdf = torch.logsumexp(node_logpdf[:, complement_indices], dim=1)
            if complement_reduce == "mean":
                complement_logpdf = complement_logpdf - torch.log(
                    torch.tensor(float(len(complement_indices)), dtype=features.dtype, device=features.device)
                )
        if ood_density == "complement_mixture":
            ood_logpdf = complement_logpdf + log_ood_prior
        elif complement_weight <= 0.0:
            ood_logpdf = parent_mask_logpdf + log_ood_prior
        elif not complement_indices and ood_density in {
            "hybrid_mixture",
            "random_complement_mixture",
        }:
            ood_logpdf = parent_mask_logpdf + log_ood_prior
        else:
            if complement_weight >= 1.0:
                ood_logpdf = complement_logpdf + log_ood_prior
            else:
                log_parent_weight = torch.log(
                    torch.tensor(1.0 - complement_weight, dtype=features.dtype, device=features.device)
                )
                log_complement_weight = torch.log(
                    torch.tensor(complement_weight, dtype=features.dtype, device=features.device)
                )
                ood_logpdf = torch.logaddexp(
                    parent_mask_logpdf + log_parent_weight,
                    complement_logpdf + log_complement_weight,
                ) + log_ood_prior
    else:
        raise ValueError(f"Unsupported CGM ood_density: {ood_density}")
    log_child_weights = torch.log(child_weights.clamp_min(eps))
    child_mixture_logpdf = torch.logsumexp(child_logpdf + log_child_weights.unsqueeze(0), dim=1)
    if local_mode == "binary_model_selection":
        hypothesis_logits = torch.stack([child_mixture_logpdf, ood_logpdf], dim=1)
        hypothesis_probs = torch.softmax(hypothesis_logits, dim=1)
        conditional_child_logits = child_logpdf + log_child_weights.unsqueeze(0)
        conditional_child_probs = torch.softmax(conditional_child_logits, dim=1)
        child_probs = hypothesis_probs[:, :1] * conditional_child_probs
        ood_prob = hypothesis_probs[:, 1]
        return {
            "child_indices": child_indices,
            "child_probs": child_probs,
            "ood_prob": ood_prob,
            "child_logpdf": child_logpdf,
            "parent_logpdf": parent_logpdf,
            "ood_base_logpdf": ood_base_logpdf,
            "ood_base_cov_scale": ood_base_cov_scale,
            "mask_cov_scale": mask_cov_scale,
            "between_cov_scale": between_cov_scale,
            "between_cov_estimator": between_cov_estimator,
            "between_cov_shrinkage_strength": between_cov_shrinkage_strength,
            "product_mask_samples": product_mask_samples,
            "random_effects_weight": random_effects_weight,
            "ood_base_mean": ood_base_mean,
            "random_effects_covariance": random_effects_covariance,
            "child_mixture_logpdf": child_mixture_logpdf,
            "child_bumps": child_bumps,
            "child_weights": child_weights,
            "mask_driver": mask_driver,
            "mask": mask,
            "log_mask": log_mask,
            "bump_integrals": bump_integrals,
            "normalizer": normalizer,
            "ood_prior": "binary_symmetric",
            "candidate_prior_mode": "binary_symmetric",
            "child_candidate_priors": child_weights,
            "ood_candidate_prior": torch.tensor(0.5, dtype=features.dtype, device=features.device),
            "ood_density": ood_density,
            "complement_reduce": complement_reduce,
            "complement_weight": complement_weight,
            "complement_indices": complement_indices,
            "parent_covariance_scales": parent_covariance_scales,
            "parent_scale_weights": parent_scale_weights,
            "parent_scale_normalizers": parent_scale_normalizers,
            "local_mode": local_mode,
            "child_log_scale": 1.0,
            "ood_log_scale": 1.0,
            "gate_log_scale": 1.0,
            "gate_bias": 0.0,
            "blend_weight": 0.0,
            "child_logits": conditional_child_logits,
            "ood_logit": ood_logpdf,
            "ood_logpdf": ood_logpdf,
        }
    if local_mode == "density_joint":
        child_logits = child_logpdf + log_child_weights.unsqueeze(0)
        density_ood_logit = parent_mask_logpdf + log_normalizer
        density_logits = torch.cat([child_logits, density_ood_logit.unsqueeze(1)], dim=1)
        density_probs = torch.softmax(density_logits, dim=1)
        return {
            "child_indices": child_indices,
            "child_probs": density_probs[:, :-1],
            "ood_prob": density_probs[:, -1],
            "child_logpdf": child_logpdf,
            "parent_logpdf": parent_logpdf,
            "ood_base_logpdf": ood_base_logpdf,
            "ood_base_cov_scale": ood_base_cov_scale,
            "mask_cov_scale": mask_cov_scale,
            "between_cov_scale": between_cov_scale,
            "between_cov_estimator": between_cov_estimator,
            "between_cov_shrinkage_strength": between_cov_shrinkage_strength,
            "product_mask_samples": product_mask_samples,
            "random_effects_weight": random_effects_weight,
            "ood_base_mean": ood_base_mean,
            "random_effects_covariance": random_effects_covariance,
            "child_mixture_logpdf": child_mixture_logpdf,
            "child_bumps": child_bumps,
            "child_weights": child_weights,
            "mask_driver": mask_driver,
            "mask": mask,
            "log_mask": log_mask,
            "bump_integrals": bump_integrals,
            "normalizer": normalizer,
            "ood_prior": "induced_joint_mass",
            "candidate_prior_mode": "density_joint",
            "child_candidate_priors": child_weights,
            "ood_candidate_prior": normalizer,
            "ood_density": ood_density,
            "complement_reduce": complement_reduce,
            "complement_weight": complement_weight,
            "complement_indices": complement_indices,
            "parent_covariance_scales": parent_covariance_scales,
            "parent_scale_weights": parent_scale_weights,
            "parent_scale_normalizers": parent_scale_normalizers,
            "local_mode": local_mode,
            "child_log_scale": 1.0,
            "ood_log_scale": 1.0,
            "gate_log_scale": gate_log_scale,
            "gate_bias": gate_bias,
            "blend_weight": blend_weight,
            "child_logits": child_logits,
            "ood_logit": density_ood_logit,
            "ood_logpdf": parent_mask_logpdf,
        }
    child_logits = child_log_scale * child_logpdf
    density_ood_logit = ood_log_scale * ood_logpdf
    if candidate_prior_mode in {
        "hierarchy_leaf_count",
        "balanced_terminal",
        "mixed_balanced_terminal",
    }:
        prior_fn = {
            "hierarchy_leaf_count": _hierarchy_leaf_candidate_priors,
            "balanced_terminal": _balanced_terminal_candidate_priors,
            "mixed_balanced_terminal": _mixed_balanced_terminal_candidate_priors,
        }[candidate_prior_mode]
        child_candidate_priors, ood_candidate_prior = prior_fn(
            parent_name,
            children,
            hierarchy,
            dtype=features.dtype,
            device=features.device,
        )
        child_logits = child_logits + torch.log(child_candidate_priors.clamp_min(eps)).unsqueeze(0)
        if bool(ood_candidate_prior.item() > 0):
            density_ood_logit = density_ood_logit + torch.log(ood_candidate_prior)
        else:
            density_ood_logit = torch.full_like(density_ood_logit, -torch.inf)
    else:
        child_candidate_priors = torch.full(
            (len(children),),
            1.0 / (len(children) + 1),
            dtype=features.dtype,
            device=features.device,
        )
        ood_candidate_prior = torch.tensor(
            1.0 / (len(children) + 1),
            dtype=features.dtype,
            device=features.device,
        )
    density_logits = torch.cat([child_logits, density_ood_logit.unsqueeze(1)], dim=1)
    density_probs = torch.softmax(density_logits, dim=1)

    if local_mode == "density_softmax":
        child_probs = density_probs[:, :-1]
        ood_prob = density_probs[:, -1]
        ood_logit = density_ood_logit
    elif local_mode in {"depth_gate", "blend"}:
        if depth_probs is None or nodes_by_depth is None:
            raise ValueError("depth_probs and nodes_by_depth are required for CGM depth-based modes")
        depth_indices = nodes_by_depth[child_depth]
        depth_index_map = {node_idx: pos for pos, node_idx in enumerate(depth_indices)}
        child_depth_probs = torch.stack(
            [depth_probs[child_depth][:, depth_index_map[idx]] for idx in child_indices],
            dim=1,
        )
        child_group_sum = child_depth_probs.sum(dim=1, keepdim=True)
        child_local_probs = child_depth_probs / child_group_sum.clamp_min(eps)
        gate_logit = gate_log_scale * ((ood_log_scale * ood_logpdf) - (child_log_scale * child_mixture_logpdf)) + gate_bias
        gate_ood_prob = torch.sigmoid(gate_logit)
        gate_child_probs = (1.0 - gate_ood_prob).unsqueeze(1) * child_local_probs
        if local_mode == "depth_gate":
            child_probs = gate_child_probs
            ood_prob = gate_ood_prob
            child_logits = torch.log(child_depth_probs.clamp_min(eps))
            ood_logit = gate_logit
        else:
            child_probs = (1.0 - blend_weight) * density_probs[:, :-1] + blend_weight * gate_child_probs
            ood_prob = (1.0 - blend_weight) * density_probs[:, -1] + blend_weight * gate_ood_prob
            ood_logit = density_ood_logit
    else:
        raise ValueError(f"Unsupported CGM local_mode: {local_mode}")

    return {
        "child_indices": child_indices,
        "child_probs": child_probs,
        "ood_prob": ood_prob,
        "child_logpdf": child_logpdf,
        "parent_logpdf": parent_logpdf,
        "ood_base_logpdf": ood_base_logpdf,
        "ood_base_cov_scale": ood_base_cov_scale,
        "mask_cov_scale": mask_cov_scale,
        "between_cov_scale": between_cov_scale,
        "between_cov_estimator": between_cov_estimator,
        "between_cov_shrinkage_strength": between_cov_shrinkage_strength,
        "product_mask_samples": product_mask_samples,
        "random_effects_weight": random_effects_weight,
        "ood_base_mean": ood_base_mean,
        "random_effects_covariance": random_effects_covariance,
        "child_mixture_logpdf": child_mixture_logpdf,
        "child_bumps": child_bumps,
        "child_weights": child_weights,
        "mask_driver": mask_driver,
        "mask": mask,
        "log_mask": log_mask,
        "bump_integrals": bump_integrals,
        "normalizer": normalizer,
        "ood_prior": ood_prior,
        "candidate_prior_mode": candidate_prior_mode,
        "child_candidate_priors": child_candidate_priors,
        "ood_candidate_prior": ood_candidate_prior,
        "ood_density": ood_density,
        "complement_reduce": complement_reduce,
        "complement_weight": complement_weight,
        "complement_indices": complement_indices,
        "parent_covariance_scales": parent_covariance_scales,
        "parent_scale_weights": parent_scale_weights,
        "parent_scale_normalizers": parent_scale_normalizers,
        "local_mode": local_mode,
        "child_log_scale": child_log_scale,
        "ood_log_scale": ood_log_scale,
        "gate_log_scale": gate_log_scale,
        "gate_bias": gate_bias,
        "blend_weight": blend_weight,
        "child_logits": child_logits,
        "ood_logit": ood_logit,
        "ood_logpdf": ood_logpdf,
    }


def hierarchical_node_probabilities(
    features: torch.Tensor | None,
    hierarchy,
    density_payload: dict[str, Any],
    score_type: str = "gaussian_loglik",
    temperature: float = 1.0,
    kappa: float = 20.0,
    include_debug: bool = True,
    node_scores: torch.Tensor | None = None,
    cgm_cfg: dict[str, Any] | None = None,
    ood_scale: float = 1.0,
):
    use_cgm = _cgm_enabled(cgm_cfg)
    if use_cgm:
        if features is None:
            raise ValueError("features are required for CGM inference")
        _validate_cgm_cfg(cgm_cfg or {}, score_type, temperature)
        density_family = (cgm_cfg or {}).get("density_family", "gaussian").lower()
        if density_family == "vmf":
            directions = density_payload.get("mean_directions")
            if directions is None:
                directions = torch.nn.functional.normalize(
                    density_payload["means"],
                    dim=1,
                    eps=1e-12,
                )
            kappa_scale = float((cgm_cfg or {}).get("vmf_kappa_scale", 1.0))
            node_scores = vmf_logpdf(
                features,
                directions,
                _vmf_kappas(density_payload, kappa_scale),
            )
        elif density_payload.get("density_feature_mask_type") == "depth_fisher":
            node_scores = masked_gaussian_logpdf(features, density_payload, hierarchy)
        elif density_payload.get("density_feature_mask_type") == "depth_fisher_reweight":
            node_scores = reweighted_gaussian_logpdf(features, density_payload, hierarchy)
        else:
            node_scores = gaussian_logpdf(
                features,
                density_payload["means"],
                density_payload.get("variances"),
                covariance_matrices=density_payload.get("covariance_matrices"),
                shared_covariance=density_payload.get("shared_covariance"),
                covariance_type=density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type", "diag")),
            )
    elif node_scores is None:
        if features is None:
            raise ValueError("features are required when node_scores are not provided")
        if density_payload.get("density_feature_mask_type") == "depth_fisher":
            node_scores = masked_gaussian_logpdf(features, density_payload, hierarchy)
        elif density_payload.get("density_feature_mask_type") == "depth_fisher_reweight":
            node_scores = reweighted_gaussian_logpdf(features, density_payload, hierarchy)
        else:
            node_scores = score_nodes(
                features,
                density_payload["means"],
                density_payload.get("variances"),
                covariance_matrices=density_payload.get("covariance_matrices"),
                shared_covariance=density_payload.get("shared_covariance"),
                mean_directions=density_payload.get("mean_directions"),
                covariance_type=density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type", "diag")),
                score_type=score_type,
                kappa=kappa,
            )
    nodes_by_depth = build_depth_maps(hierarchy)
    depth_probs = compute_depth_probs(node_scores, nodes_by_depth, temperature=temperature)

    n_samples = node_scores.shape[0]
    n_nodes = len(hierarchy.id_node_list)
    final_probs = torch.zeros((n_samples, n_nodes), dtype=node_scores.dtype, device=node_scores.device)
    local_info = {}

    for parent in hierarchy.parent2children:
        if use_cgm:
            local = _cgm_local_probabilities_for_node(
                parent,
                features,
                hierarchy,
                node_scores,
                density_payload,
                cgm_cfg or {},
                depth_probs=depth_probs,
                nodes_by_depth=nodes_by_depth,
            )
        else:
            local = _local_probabilities_for_node(
                parent,
                hierarchy,
                depth_probs,
                nodes_by_depth,
                ood_scale=ood_scale,
            )
        if local is not None:
            local_info[parent] = local

    def recurse(node_name: str, incoming_prob: torch.Tensor):
        node_idx = hierarchy.id_node_list.index(node_name)
        local = local_info.get(node_name)
        if local is None:
            final_probs[:, node_idx] += incoming_prob
            return
        final_probs[:, node_idx] += incoming_prob * local["ood_prob"]
        for child_idx, child_node_idx in enumerate(local["child_indices"]):
            child_name = hierarchy.id_node_list[child_node_idx]
            child_prob = incoming_prob * local["child_probs"][:, child_idx]
            recurse(child_name, child_prob)

    root_name = "root"
    recurse(root_name, torch.ones((n_samples,), dtype=node_scores.dtype, device=node_scores.device))
    final_probs = final_probs / final_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    debug = {
        "temperature": temperature,
        "cgm": cgm_cfg if use_cgm else {"enabled": False},
    }
    if include_debug:
        if use_cgm:
            debug.update(
                {
                    "node_logpdf": node_scores.cpu(),
                    "local_info": {
                        name: {
                            "child_indices": payload["child_indices"],
                            "child_probs": payload["child_probs"].cpu(),
                            "ood_prob": payload["ood_prob"].cpu(),
                            "child_logpdf": payload["child_logpdf"].cpu(),
                            "parent_logpdf": payload["parent_logpdf"].cpu(),
                            "ood_base_logpdf": payload["ood_base_logpdf"].cpu(),
                            "ood_base_cov_scale": payload["ood_base_cov_scale"],
                            "mask_cov_scale": payload["mask_cov_scale"],
                            "between_cov_scale": payload["between_cov_scale"],
                            "between_cov_estimator": payload["between_cov_estimator"],
                            "between_cov_shrinkage_strength": payload[
                                "between_cov_shrinkage_strength"
                            ],
                            "product_mask_samples": payload["product_mask_samples"],
                            "random_effects_weight": payload["random_effects_weight"],
                            "ood_base_mean": payload["ood_base_mean"].cpu(),
                            "random_effects_covariance": payload["random_effects_covariance"].cpu(),
                            "child_mixture_logpdf": payload["child_mixture_logpdf"].cpu(),
                            "child_bumps": payload["child_bumps"].cpu(),
                            "child_weights": payload["child_weights"].cpu(),
                            "mask_driver": payload["mask_driver"].cpu(),
                            "mask": payload["mask"].cpu(),
                            "log_mask": payload["log_mask"].cpu(),
                            "bump_integrals": payload["bump_integrals"].cpu(),
                            "normalizer": payload["normalizer"].detach().cpu(),
                            "ood_prior": payload["ood_prior"],
                            "candidate_prior_mode": payload["candidate_prior_mode"],
                            "child_candidate_priors": payload["child_candidate_priors"].cpu(),
                            "ood_candidate_prior": payload["ood_candidate_prior"].cpu(),
                            "ood_density": payload["ood_density"],
                            "complement_reduce": payload["complement_reduce"],
                            "complement_weight": payload["complement_weight"],
                            "complement_indices": payload["complement_indices"],
                            "parent_covariance_scales": payload["parent_covariance_scales"],
                            "parent_scale_weights": payload["parent_scale_weights"].cpu(),
                            "parent_scale_normalizers": payload["parent_scale_normalizers"].cpu(),
                            "local_mode": payload["local_mode"],
                            "child_log_scale": payload["child_log_scale"],
                            "ood_log_scale": payload["ood_log_scale"],
                            "gate_log_scale": payload["gate_log_scale"],
                            "gate_bias": payload["gate_bias"],
                            "blend_weight": payload["blend_weight"],
                            "child_logits": payload["child_logits"].cpu(),
                            "ood_logit": payload["ood_logit"].cpu(),
                            "ood_logpdf": payload["ood_logpdf"].cpu(),
                        }
                        for name, payload in local_info.items()
                    },
                }
            )
        else:
            debug.update(
                {
                    "node_scores": node_scores.cpu(),
                    "depth_probs": {depth: probs.cpu() for depth, probs in depth_probs.items()},
                    "local_info": {
                        name: {
                            "child_indices": payload["child_indices"],
                            "child_depth_probs": payload["child_depth_probs"].cpu(),
                            "child_local_probs": payload["child_local_probs"].cpu(),
                            "child_probs": payload["child_probs"].cpu(),
                            "ood_prob": payload["ood_prob"].cpu(),
                            "ood_score": payload["ood_score"].cpu(),
                            "p_comp": payload["p_comp"].cpu(),
                            "entropy": payload["entropy"].cpu(),
                            "ood_scale": payload["ood_scale"],
                        }
                        for name, payload in local_info.items()
                    },
                }
            )
    return final_probs, debug


def predict_from_probabilities(
    final_probs: torch.Tensor,
    hierarchy,
    mode: str = "argmax",
    hdist_mat: torch.Tensor | None = None,
):
    if mode == "argmax":
        return final_probs.argmax(dim=1)
    if mode == "min_hdist":
        if hdist_mat is None:
            hdist_mat = get_hdist_matrix(hierarchy, range(len(hierarchy.id_node_list)))
        neg_dists = -1.0 * expected_hdist(final_probs.cpu(), hdist_mat)
        return neg_dists.argmax(dim=1)
    raise ValueError(f"Unsupported prediction mode: {mode}")
