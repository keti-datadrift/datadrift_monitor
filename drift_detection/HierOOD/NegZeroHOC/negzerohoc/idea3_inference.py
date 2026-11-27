from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .unknown_scoring import grouped_unknown_logits

from .semantic_index import LocalSemanticCandidates


POSITIVE_GLOBAL_PATH_MODES = {"positive_global_path", "positive_pathscore_diagnostic"}
SUPPORTED_IDEA3_MODES = {"positive_child_only", "parent_unknown"} | POSITIVE_GLOBAL_PATH_MODES


@dataclass
class Idea3SemanticIndex:
    locals: dict[str, LocalSemanticCandidates]


@torch.no_grad()
def build_idea3_semantic_index(
    hierarchy,
    positive_learner,
    unknown_learner=None,
    mode: str = "positive_child_only",
    allow_root_unknown: bool = False,
) -> dict[str, LocalSemanticCandidates]:
    if mode not in SUPPORTED_IDEA3_MODES:
        raise ValueError(f"Unsupported Idea 3 inference mode: {mode}")

    index = {}
    was_training = positive_learner.training
    positive_learner.eval()
    if unknown_learner is not None:
        unknown_was_training = unknown_learner.training
        unknown_learner.eval()
    else:
        unknown_was_training = False

    for parent, children in hierarchy.parent2children.items():
        child_features = positive_learner.encode_children(parent, list(children)).detach().cpu()
        unknown_feature = None
        candidate_names = list(children)
        prompts = {
            child: [positive_learner.edge_text(parent, child)]
            for child in children
        }

        if mode == "parent_unknown" and unknown_learner is not None and (parent != "root" or allow_root_unknown):
            unknown_feature = unknown_learner.encode_unknown(parent).detach().cpu()
            unknown_name = f"__unknown__:{parent}"
            candidate_names.append(unknown_name)
            prompts[unknown_name] = [unknown_learner.unknown_text(parent)]

        index[parent] = LocalSemanticCandidates(
            parent=parent,
            children=list(children),
            child_features=child_features,
            unknown_feature=unknown_feature,
            candidate_names=candidate_names,
            prompts=prompts,
        )

    if was_training:
        positive_learner.train()
    if unknown_learner is not None and unknown_was_training:
        unknown_learner.train()
    return index


def predict_one_idea3(
    image_feature: torch.Tensor,
    hierarchy,
    semantic_index,
    mode: str,
    tau: float = 1.0,
    unknown_aggregation: str = "logmeanexp",
    return_trace: bool = False,
):
    node = "root"
    trace = []

    while node in hierarchy.parent2children:
        local = semantic_index[node]
        feats = local.child_features.to(image_feature.device)
        names = list(local.children)

        if mode == "parent_unknown" and local.unknown_feature is not None:
            logits = grouped_unknown_logits(
                image_feature.unsqueeze(0),
                feats,
                local.unknown_feature.to(image_feature.device),
                logit_scale=float(tau),
                aggregation=unknown_aggregation,
            )[0]
            names.append(f"__unknown__:{node}")
        else:
            logits = float(tau) * torch.mv(feats, image_feature)
        winner_idx = int(torch.argmax(logits).item())
        winner = names[winner_idx]

        if return_trace:
            trace.append({
                "parent": node,
                "candidates": names,
                "winner": winner,
                "logits": logits.detach().cpu().tolist(),
            })

        if winner.startswith("__unknown__:"):
            return node, trace
        node = winner

    return node, trace


def predict_features_idea3(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    mode: str,
    tau: float = 1.0,
    unknown_aggregation: str = "logmeanexp",
    return_trace: bool = False,
) -> dict:
    if mode in POSITIVE_GLOBAL_PATH_MODES:
        return predict_features_global_path(
            features,
            hierarchy,
            semantic_index,
            logit_scale=tau,
            return_trace=return_trace,
        )

    features = F.normalize(features.float(), dim=-1)
    preds = []
    traces = [] if return_trace else None
    stop_nodes = []

    for feat in features:
        node, trace = predict_one_idea3(
            feat,
            hierarchy,
            semantic_index,
            mode=mode,
            tau=tau,
            unknown_aggregation=unknown_aggregation,
            return_trace=return_trace,
        )
        preds.append(hierarchy.id_node_list.index(node))
        stop_nodes.append(node)
        if return_trace:
            traces.append(trace)

    return {
        "preds": torch.tensor(preds, dtype=torch.long),
        "traces": traces,
        "diagnostics": summarize_stop_nodes(hierarchy, stop_nodes),
    }


def predict_features_global_path(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    logit_scale: float = 1.0,
    return_trace: bool = False,
) -> dict:
    """Exact MAP decoding under the hierarchy's local conditional models.

    Each local softmax defines ``P(child | image, parent)``. Dynamic
    programming accumulates log probabilities from root to every leaf and
    returns the globally most probable complete path. Complexity is linear in
    the number of hierarchy edges and no beam-width approximation is used.
    """
    if float(logit_scale) <= 0.0:
        raise ValueError("Global path logit_scale must be positive")

    features = F.normalize(features.float(), dim=-1)
    device = features.device
    batch_size = int(features.shape[0])
    node_scores = {"root": torch.zeros(batch_size, dtype=features.dtype, device=device)}
    edge_log_probs = {} if return_trace else None

    ordered_parents = sorted(
        hierarchy.parent2children,
        key=lambda parent: (len(hierarchy.node_ancestors.get(parent, [])), parent),
    )
    for parent in ordered_parents:
        if parent not in node_scores:
            continue
        local = semantic_index[parent]
        child_features = F.normalize(local.child_features.to(device).float(), dim=-1)
        logits = float(logit_scale) * (features @ child_features.t())
        log_probs = F.log_softmax(logits, dim=1)
        parent_score = node_scores[parent]
        for child_index, child in enumerate(local.children):
            node_scores[child] = parent_score + log_probs[:, child_index]
            if return_trace:
                edge_log_probs[(parent, child)] = log_probs[:, child_index].detach().cpu()

    leaves = [node for node in hierarchy.id_node_list if node not in hierarchy.parent2children]
    if not leaves:
        raise RuntimeError("Global path inference found no leaf nodes")
    missing = [leaf for leaf in leaves if leaf not in node_scores]
    if missing:
        raise RuntimeError(f"Global path inference could not reach {len(missing)} leaves")

    leaf_scores = torch.stack([node_scores[leaf] for leaf in leaves], dim=1)
    winner_indices = torch.argmax(leaf_scores, dim=1)
    winner_nodes = [leaves[int(index)] for index in winner_indices.detach().cpu().tolist()]
    node_to_index = {node: index for index, node in enumerate(hierarchy.id_node_list)}
    preds = torch.tensor([node_to_index[node] for node in winner_nodes], dtype=torch.long)

    traces = None
    if return_trace:
        traces = []
        for sample_index, leaf in enumerate(winner_nodes):
            path_indices = hierarchy.node_ancestors.get(leaf, [])
            path = [hierarchy.id_node_list[index] for index in path_indices] + [leaf]
            decisions = []
            cumulative = 0.0
            for parent, child in zip(path[:-1], path[1:]):
                edge_score = float(edge_log_probs[(parent, child)][sample_index])
                cumulative += edge_score
                decisions.append({
                    "parent": parent,
                    "child": child,
                    "edge_log_probability": edge_score,
                    "cumulative_log_probability": cumulative,
                })
            traces.append(decisions)

    return {
        "preds": preds,
        "traces": traces,
        "diagnostics": summarize_stop_nodes(hierarchy, winner_nodes),
    }


def summarize_stop_nodes(hierarchy, stop_nodes: list[str]) -> dict:
    depth_counts = Counter(len(hierarchy.node_ancestors.get(node, [])) for node in stop_nodes)
    node_counts = Counter(stop_nodes)
    return {
        "stop_depth_counts": dict(sorted(depth_counts.items())),
        "stop_node_counts": dict(node_counts.most_common()),
    }
