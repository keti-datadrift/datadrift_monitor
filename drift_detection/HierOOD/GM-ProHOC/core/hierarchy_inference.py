from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from core.density import score_nodes
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


def _normalize_depthwise_param(param, depths: list[int], min_value: float | None = None) -> dict[int, float]:
    if isinstance(param, (list, tuple)):
        if len(param) != len(depths):
            raise ValueError(
                f"depth-wise parameter length {len(param)} does not match "
                f"number of non-root depths {len(depths)}"
            )
        values = [float(x) for x in param]
    else:
        values = [float(param) for _ in depths]

    if min_value is not None:
        values = [max(v, min_value) for v in values]
    return {depth: value for depth, value in zip(depths, values)}


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
    alpha_by_depth: dict[int, float],
    beta_by_depth: dict[int, float],
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
    alpha = alpha_by_depth[child_depth]
    beta = beta_by_depth[child_depth]
    ood_mass = alpha * p_comp + beta * entropy
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
        "alpha": alpha,
        "beta": beta,
        "p_comp": p_comp,
        "entropy": entropy,
    }


def hierarchical_node_probabilities(
    features: torch.Tensor | None,
    hierarchy,
    density_payload: dict[str, Any],
    score_type: str = "gaussian_loglik",
    temperature: float = 1.0,
    kappa: float = 20.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    include_debug: bool = True,
    node_scores: torch.Tensor | None = None,
):
    if node_scores is None:
        if features is None:
            raise ValueError("features are required when node_scores are not provided")
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
    non_root_depths = sorted(depth for depth in nodes_by_depth.keys() if depth != 0)
    depth_probs = compute_depth_probs(node_scores, nodes_by_depth, temperature=temperature)
    alpha_by_depth = _normalize_depthwise_param(alpha, non_root_depths)
    beta_by_depth = _normalize_depthwise_param(beta, non_root_depths)

    n_samples = node_scores.shape[0]
    n_nodes = len(hierarchy.id_node_list)
    final_probs = torch.zeros((n_samples, n_nodes), dtype=node_scores.dtype, device=node_scores.device)
    local_info = {}

    for parent in hierarchy.parent2children:
        local = _local_probabilities_for_node(
            parent,
            hierarchy,
            depth_probs,
            nodes_by_depth,
            alpha_by_depth,
            beta_by_depth,
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
        "alpha": alpha,
        "beta": beta,
    }
    if include_debug:
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
                        "alpha": payload["alpha"],
                        "beta": payload["beta"],
                        "p_comp": payload["p_comp"].cpu(),
                        "entropy": payload["entropy"].cpu(),
                    }
                    for name, payload in local_info.items()
                },
            }
        )
    return final_probs, debug


def predict_from_probabilities(final_probs: torch.Tensor, hierarchy, mode: str = "argmax"):
    if mode == "argmax":
        return final_probs.argmax(dim=1)
    if mode == "min_hdist":
        hdist_mat = get_hdist_matrix(hierarchy, range(len(hierarchy.id_node_list)))
        neg_dists = -1.0 * expected_hdist(final_probs.cpu(), hdist_mat)
        return neg_dists.argmax(dim=1)
    raise ValueError(f"Unsupported prediction mode: {mode}")
