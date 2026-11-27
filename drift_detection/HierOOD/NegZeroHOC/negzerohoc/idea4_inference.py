from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from .unknown_scoring import grouped_unknown_logits


def terminal_global_path_scores(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    logit_scale: float = 1.0,
    allow_root_unknown: bool = False,
    unknown_aggregation: str = "logmeanexp",
    return_local_scores: bool = False,
) -> dict:
    """Return differentiable scores for every leaf and unknown terminal."""
    if float(logit_scale) <= 0.0:
        raise ValueError("Terminal global-path logit_scale must be positive")

    features = F.normalize(features.float(), dim=-1)
    device = features.device
    batch_size = int(features.shape[0])
    node_scores = {"root": torch.zeros(batch_size, dtype=features.dtype, device=device)}
    terminal_scores: dict[str, torch.Tensor] = {}
    edge_log_probs = {} if return_local_scores else None
    unknown_log_probs = {} if return_local_scores else None

    ordered_parents = sorted(
        hierarchy.parent2children,
        key=lambda parent: (len(hierarchy.node_ancestors.get(parent, [])), parent),
    )
    for parent in ordered_parents:
        if parent not in node_scores:
            continue

        local = semantic_index[parent]
        child_features = F.normalize(local.child_features.to(device).float(), dim=-1)
        has_unknown = local.unknown_feature is not None and (
            parent != "root" or allow_root_unknown
        )
        if has_unknown:
            logits = grouped_unknown_logits(
                features,
                child_features,
                local.unknown_feature.to(device),
                logit_scale=float(logit_scale),
                aggregation=unknown_aggregation,
            )
        else:
            logits = float(logit_scale) * (features @ child_features.t())
        log_probs = F.log_softmax(logits, dim=1)
        parent_score = node_scores[parent]

        for child_index, child in enumerate(local.children):
            node_scores[child] = parent_score + log_probs[:, child_index]
            if return_local_scores:
                edge_log_probs[(parent, child)] = log_probs[:, child_index]

        if has_unknown:
            local_unknown_log_prob = log_probs[:, len(local.children)]
            terminal_scores[parent] = parent_score + local_unknown_log_prob
            if return_local_scores:
                unknown_log_probs[parent] = local_unknown_log_prob

    leaves = [node for node in hierarchy.id_node_list if node not in hierarchy.parent2children]
    missing = [leaf for leaf in leaves if leaf not in node_scores]
    if missing:
        raise RuntimeError(f"Terminal global-path inference could not reach {len(missing)} leaves")
    if not leaves:
        raise RuntimeError("Terminal global-path inference found no ID leaves")

    candidate_nodes = list(leaves) + list(terminal_scores)
    candidate_kinds = ["leaf"] * len(leaves) + ["unknown"] * len(terminal_scores)
    candidate_scores = [node_scores[leaf] for leaf in leaves]
    candidate_scores.extend(terminal_scores[parent] for parent in terminal_scores)
    return {
        "score_matrix": torch.stack(candidate_scores, dim=1),
        "candidate_nodes": candidate_nodes,
        "candidate_kinds": candidate_kinds,
        "node_scores": node_scores,
        "terminal_scores": terminal_scores,
        "edge_log_probs": edge_log_probs,
        "unknown_log_probs": unknown_log_probs,
    }


@torch.no_grad()
def predict_features_terminal_global_path(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    logit_scale: float = 1.0,
    allow_root_unknown: bool = False,
    unknown_aggregation: str = "logmeanexp",
    return_trace: bool = False,
) -> dict:
    """Decode ID leaves and parent-local unknown terminals in one path space.

    Known edges and a parent's unknown terminal share one local softmax. The
    score of a complete candidate is the sum of its root-to-terminal local log
    probabilities. This is exact MAP decoding over every ID leaf and every
    enabled local unknown terminal, without a threshold or beam search.
    """
    scores = terminal_global_path_scores(
        features,
        hierarchy,
        semantic_index,
        logit_scale=logit_scale,
        allow_root_unknown=allow_root_unknown,
        unknown_aggregation=unknown_aggregation,
        return_local_scores=return_trace,
    )
    candidate_nodes = scores["candidate_nodes"]
    candidate_kinds = scores["candidate_kinds"]
    score_matrix = scores["score_matrix"]
    edge_log_probs = scores["edge_log_probs"]
    unknown_log_probs = scores["unknown_log_probs"]
    batch_size = int(score_matrix.shape[0])
    winner_indices = torch.argmax(score_matrix, dim=1).detach().cpu().tolist()
    winner_nodes = [candidate_nodes[index] for index in winner_indices]
    winner_kinds = [candidate_kinds[index] for index in winner_indices]

    node_to_index = {node: index for index, node in enumerate(hierarchy.id_node_list)}
    preds = torch.tensor([node_to_index[node] for node in winner_nodes], dtype=torch.long)
    depth_counts = Counter(len(hierarchy.node_ancestors.get(node, [])) for node in winner_nodes)
    node_counts = Counter(winner_nodes)
    kind_counts = Counter(winner_kinds)
    diagnostics = {
        "stop_depth_counts": dict(sorted(depth_counts.items())),
        "stop_node_counts": dict(node_counts.most_common()),
        "candidate_type_counts": dict(kind_counts),
        "unknown_selection_rate": kind_counts.get("unknown", 0) / max(1, batch_size),
    }

    traces = None
    if return_trace:
        traces = []
        for sample_index, (winner_node, winner_kind) in enumerate(zip(winner_nodes, winner_kinds)):
            ancestor_indices = hierarchy.node_ancestors.get(winner_node, [])
            path = [hierarchy.id_node_list[index] for index in ancestor_indices] + [winner_node]
            decisions = []
            cumulative = 0.0
            for parent, child in zip(path[:-1], path[1:]):
                edge_score = float(edge_log_probs[(parent, child)][sample_index])
                cumulative += edge_score
                decisions.append({
                    "parent": parent,
                    "candidate": child,
                    "candidate_type": "known",
                    "local_log_probability": edge_score,
                    "cumulative_log_probability": cumulative,
                })
            if winner_kind == "unknown":
                unknown_score = float(unknown_log_probs[winner_node][sample_index])
                cumulative += unknown_score
                decisions.append({
                    "parent": winner_node,
                    "candidate": f"__unknown__:{winner_node}",
                    "candidate_type": "unknown",
                    "local_log_probability": unknown_score,
                    "cumulative_log_probability": cumulative,
                })
            traces.append(decisions)

    return {
        "preds": preds,
        "traces": traces,
        "diagnostics": diagnostics,
    }
