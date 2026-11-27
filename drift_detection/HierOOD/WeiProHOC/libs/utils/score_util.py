import math

import torch


DEFAULT_EPS = 1e-12


def compute_group_sums(p,
                       children_map,
                       n_samples,
                       n_parents,
                       device="cpu"):

    group_sums = torch.zeros(n_samples, n_parents, device=device, dtype=p.dtype)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p)
    return group_sums


def compute_local_child_probs(p,
                              children_map,
                              group_sums,
                              eps=DEFAULT_EPS):

    p_safe = p + eps
    denom = group_sums[:, children_map] + eps
    return p_safe / denom


def compute_entropy_from_local_probs(local_child_probs,
                                     children_map,
                                     n_samples,
                                     n_parents,
                                     device="cpu",
                                     eps=DEFAULT_EPS):

    x = torch.zeros(n_samples, n_parents, device=device, dtype=local_child_probs.dtype)
    p_safe = local_child_probs.clamp_min(eps)
    x.scatter_add_(1,
                   children_map.expand(n_samples, -1),
                   -1.0 * p_safe * torch.log(p_safe))
    return x


def compute_normalized_entropy(entropy,
                               group_sizes,
                               eps=DEFAULT_EPS):

    group_sizes = group_sizes.to(entropy.device)
    denom = torch.log(group_sizes.clamp_min(2.0))
    entropy_norm = torch.zeros_like(entropy)
    valid_mask = group_sizes > 1
    if valid_mask.any():
        entropy_norm[:, valid_mask] = entropy[:, valid_mask] / denom[valid_mask].clamp_min(eps)
    return entropy_norm


def compute_complementary_probability(group_sums):
    return (1.0 - group_sums).clamp(min=0.0, max=1.0)


def normalize_local_conditional(p,
                                group_sums,
                                score,
                                children_map,
                                eps=DEFAULT_EPS):

    total_sums = group_sums + score
    safe_total_sums = total_sums.clone()
    zero_mask = safe_total_sums <= eps
    safe_total_sums[zero_mask] = 1.0

    result = p / safe_total_sums[:, children_map]
    p_ood = score / safe_total_sums

    if zero_mask.any():
        p_ood = p_ood.clone()
        p_ood[zero_mask] = 1.0

    return result, p_ood


def _resolve_depth_weight(values, depth, default=1.0):
    if values is None:
        return default
    return float(values[depth])


def _resolve_node_weights(values, depth, n_parents, device, dtype):
    if values is None:
        return torch.ones(n_parents, device=device, dtype=dtype)

    depth_values = values[depth]
    if not isinstance(depth_values, torch.Tensor):
        depth_values = torch.tensor(depth_values, device=device, dtype=dtype)
    else:
        depth_values = depth_values.to(device=device, dtype=dtype)

    if depth_values.numel() != n_parents:
        raise ValueError(
            f"Node-wise weight length mismatch at depth {depth}: "
            f"expected {n_parents}, got {depth_values.numel()}"
        )
    return depth_values


def _get_fixed_beta(num_classes, beta_rule):
    if beta_rule == "ones":
        return 1.0
    if beta_rule == "inv":
        return 1.0 / max(num_classes, 1)
    if beta_rule == "inv_log":
        return 1.0 / max(math.log(max(num_classes, 2)), DEFAULT_EPS)
    if beta_rule == "inv_sqrt":
        return 1.0 / math.sqrt(max(num_classes, 1))
    raise ValueError(f"Unknown beta_rule: {beta_rule}")


def _resolve_scheduled_beta(depth,
                            beta_schedule="constant",
                            beta0=1.0,
                            beta_gamma=0.5,
                            beta_k=0.5,
                            beta_min=0.0):
    d = depth + 1
    if beta_schedule == "constant":
        return float(beta0)
    if beta_schedule == "inverse_depth":
        return 1.0 / d
    if beta_schedule == "exp_decay":
        return float(beta0) * (float(beta_gamma) ** (d - 1))
    if beta_schedule == "linear_decay":
        return max(float(beta0) - float(beta_k) * (d - 1), float(beta_min))
    raise ValueError(f"Unknown beta_schedule: {beta_schedule}")


def _compute_score_terms(p,
                         children_map,
                         group_sizes,
                         n_samples,
                         n_parents,
                         device="cpu",
                         eps=DEFAULT_EPS):

    validate_tensor(p)

    group_sums = compute_group_sums(p, children_map, n_samples, n_parents, device=device)
    group_sums_eps = compute_group_sums(p + eps,
                                        children_map,
                                        n_samples,
                                        n_parents,
                                        device=device)
    local_child_probs = compute_local_child_probs(p, children_map, group_sums_eps, eps=eps)
    validate_tensor(local_child_probs)

    entropy = compute_entropy_from_local_probs(local_child_probs,
                                               children_map,
                                               n_samples,
                                               n_parents,
                                               device=device,
                                               eps=eps)
    assert torch.isfinite(entropy).all()

    entropy_norm = compute_normalized_entropy(entropy, group_sizes, eps=eps)
    p_comp = compute_complementary_probability(group_sums)

    return group_sums, entropy, entropy_norm, p_comp


def _finalize_score(p,
                    children_map,
                    group_sizes,
                    n_samples,
                    n_parents,
                    score,
                    device="cpu",
                    eps=DEFAULT_EPS):

    group_sums = compute_group_sums(p, children_map, n_samples, n_parents, device=device)
    result, p_ood = normalize_local_conditional(p, group_sums, score, children_map, eps=eps)

    validate_tensor(result)
    validate_tensor(p_ood)

    result_sums = compute_group_sums(result, children_map, n_samples, n_parents, device=device)
    final_sums = result_sums + p_ood
    assert torch.allclose(final_sums, torch.ones_like(final_sums), atol=1e-4)

    return result, p_ood


def compprob(p,
             children_map,
             group_sizes,
             n_samples,
             n_parents,
             device="cpu",
             **kwargs):

    group_sums = compute_group_sums(p, children_map, n_samples, n_parents, device=device)
    p_comp = compute_complementary_probability(group_sums)
    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           p_comp,
                           device=device)


def entcompprob(p,
                children_map,
                group_sizes,
                n_samples,
                n_parents,
                device="cpu",
                **kwargs):

    _, entropy, _, p_comp = _compute_score_terms(p,
                                                 children_map,
                                                 group_sizes,
                                                 n_samples,
                                                 n_parents,
                                                 device=device)
    score = entropy + p_comp
    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           score,
                           device=device)


def legacy_entcompprob(p,
                       children_map,
                       group_sizes,
                       n_samples,
                       n_parents,
                       device="cpu",
                       **kwargs):

    eps = DEFAULT_EPS

    validate_tensor(p)

    group_sums = torch.zeros(n_samples, n_parents, device=device, dtype=p.dtype)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p + eps)
    p_norm = (p + eps) / (group_sums[:, children_map] + eps)

    validate_tensor(p_norm)

    entropy = torch.zeros(n_samples, n_parents, device=device, dtype=p.dtype)
    entropy.scatter_add_(1,
                         children_map.expand(n_samples, -1),
                         -1.0 * p_norm * torch.log(p_norm + eps))

    assert torch.isfinite(entropy).all()

    total_sums = group_sums + entropy
    result = p / total_sums[:, children_map]
    p_ood = entropy / total_sums

    validate_tensor(p_ood)
    validate_tensor(result)

    result_sums = torch.zeros(n_samples, n_parents, device=device, dtype=p.dtype)
    result_sums.scatter_add_(1, children_map.expand(n_samples, -1), result)
    final_sums = result_sums + p_ood
    assert torch.allclose(final_sums, torch.ones_like(final_sums), atol=1e-4)

    return result, p_ood


def normentropy_compprob(p,
                         children_map,
                         group_sizes,
                         n_samples,
                         n_parents,
                         device="cpu",
                         **kwargs):

    _, _, entropy_norm, _ = _compute_score_terms(p,
                                                 children_map,
                                                 group_sizes,
                                                 n_samples,
                                                 n_parents,
                                                 device=device)
    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           entropy_norm,
                           device=device)


def depth_weighted_raw(p,
                       children_map,
                       group_sizes,
                       n_samples,
                       n_parents,
                       device="cpu",
                       depth=None,
                       depth_alpha=None,
                       depth_beta=None,
                       **kwargs):

    if depth is None:
        raise ValueError("depth_weighted_raw requires the current depth")

    _, entropy, _, p_comp = _compute_score_terms(p,
                                                 children_map,
                                                 group_sizes,
                                                 n_samples,
                                                 n_parents,
                                                 device=device)
    alpha = _resolve_depth_weight(depth_alpha, depth)
    beta = _resolve_depth_weight(depth_beta, depth)
    score = alpha * entropy + beta * p_comp

    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           score,
                           device=device)


def depth_weighted_norm(p,
                        children_map,
                        group_sizes,
                        n_samples,
                        n_parents,
                        device="cpu",
                        depth=None,
                        depth_alpha=None,
                        depth_beta=None,
                        **kwargs):

    if depth is None:
        raise ValueError("depth_weighted_norm requires the current depth")

    _, _, entropy_norm, p_comp = _compute_score_terms(p,
                                                      children_map,
                                                      group_sizes,
                                                      n_samples,
                                                      n_parents,
                                                      device=device)
    alpha = _resolve_depth_weight(depth_alpha, depth)
    beta = _resolve_depth_weight(depth_beta, depth)
    score = alpha * entropy_norm + beta * p_comp

    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           score,
                           device=device)


def scheduled_raw(p,
                  children_map,
                  group_sizes,
                  n_samples,
                  n_parents,
                  device="cpu",
                  depth=None,
                  beta_schedule="constant",
                  beta0=1.0,
                  beta_gamma=0.5,
                  beta_k=0.5,
                  beta_min=0.0,
                  **kwargs):

    if depth is None:
        raise ValueError("scheduled_raw requires the current depth")

    _, entropy, _, p_comp = _compute_score_terms(p,
                                                 children_map,
                                                 group_sizes,
                                                 n_samples,
                                                 n_parents,
                                                 device=device)
    beta = _resolve_scheduled_beta(depth,
                                   beta_schedule=beta_schedule,
                                   beta0=beta0,
                                   beta_gamma=beta_gamma,
                                   beta_k=beta_k,
                                   beta_min=beta_min)
    score = entropy + beta * p_comp

    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           score,
                           device=device)


def scheduled_norm(p,
                   children_map,
                   group_sizes,
                   n_samples,
                   n_parents,
                   device="cpu",
                   depth=None,
                   beta_schedule="constant",
                   beta0=1.0,
                   beta_gamma=0.5,
                   beta_k=0.5,
                   beta_min=0.0,
                   **kwargs):

    if depth is None:
        raise ValueError("scheduled_norm requires the current depth")

    _, _, entropy_norm, p_comp = _compute_score_terms(p,
                                                      children_map,
                                                      group_sizes,
                                                      n_samples,
                                                      n_parents,
                                                      device=device)
    beta = _resolve_scheduled_beta(depth,
                                   beta_schedule=beta_schedule,
                                   beta0=beta0,
                                   beta_gamma=beta_gamma,
                                   beta_k=beta_k,
                                   beta_min=beta_min)
    score = entropy_norm + beta * p_comp

    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           score,
                           device=device)


def fixedbeta_norm(p,
                   children_map,
                   group_sizes,
                   n_samples,
                   n_parents,
                   device="cpu",
                   depth=None,
                   beta_rule="ones",
                   num_classes=None,
                   **kwargs):

    if num_classes is None:
        raise ValueError("fixedbeta_norm requires num_classes for the current depth")

    _, _, entropy_norm, p_comp = _compute_score_terms(p,
                                                      children_map,
                                                      group_sizes,
                                                      n_samples,
                                                      n_parents,
                                                      device=device)
    beta = _get_fixed_beta(num_classes, beta_rule)
    score = entropy_norm + beta * p_comp

    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           score,
                           device=device)


def node_weighted_norm(p,
                       children_map,
                       group_sizes,
                       n_samples,
                       n_parents,
                       device="cpu",
                       depth=None,
                       node_alpha_by_depth=None,
                       node_beta_by_depth=None,
                       **kwargs):

    if depth is None:
        raise ValueError("node_weighted_norm requires the current depth")

    _, _, entropy_norm, p_comp = _compute_score_terms(p,
                                                      children_map,
                                                      group_sizes,
                                                      n_samples,
                                                      n_parents,
                                                      device=device)

    alpha = _resolve_node_weights(node_alpha_by_depth,
                                  depth,
                                  n_parents,
                                  device,
                                  entropy_norm.dtype)
    beta = _resolve_node_weights(node_beta_by_depth,
                                 depth,
                                 n_parents,
                                 device,
                                 entropy_norm.dtype)

    score = entropy_norm * alpha.unsqueeze(0) + p_comp * beta.unsqueeze(0)

    return _finalize_score(p,
                           children_map,
                           group_sizes,
                           n_samples,
                           n_parents,
                           score,
                           device=device)


def validate_tensor(tensor):
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values!"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values!"
    assert (tensor >= 0.0).all(), "Tensor contains values less than 0.0!"
    assert (tensor <= 1.0).all(), "Tensor contains values greater than 1.0!"
