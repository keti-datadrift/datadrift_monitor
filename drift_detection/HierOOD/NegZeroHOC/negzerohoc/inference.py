from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F


def predict_one(
    image_feature: torch.Tensor,
    hierarchy,
    semantic_index,
    mode: str,
    tau: float = 1.0,
    return_trace: bool = False,
):
    node = "root"
    trace = []

    while node in hierarchy.parent2children:
        local = semantic_index[node]
        feats = local.child_features.to(image_feature.device)
        names = list(local.children)

        if mode == "manual_unknown" and local.unknown_feature is not None:
            feats = torch.cat([feats, local.unknown_feature.to(image_feature.device).unsqueeze(0)], dim=0)
            names.append(f"__unknown__:{node}")

        logits = tau * torch.mv(feats, image_feature)
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


def predict_features(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    mode: str,
    tau: float = 1.0,
    return_trace: bool = False,
):
    features = F.normalize(features.float(), dim=-1)
    preds = []
    traces = [] if return_trace else None
    stop_nodes = []

    for feat in features:
        node, trace = predict_one(
            feat,
            hierarchy,
            semantic_index,
            mode=mode,
            tau=tau,
            return_trace=return_trace,
        )
        preds.append(hierarchy.id_node_list.index(node))
        stop_nodes.append(node)
        if return_trace:
            traces.append(trace)

    return {
        "preds": torch.tensor(preds, dtype=torch.long),
        "traces": traces,
        "diagnostics": summarize_predictions(hierarchy, stop_nodes),
    }


def summarize_predictions(hierarchy, stop_nodes: list[str]) -> dict:
    depth_counts = Counter(len(hierarchy.node_ancestors.get(node, [])) for node in stop_nodes)
    node_counts = Counter(stop_nodes)
    return {
        "stop_depth_counts": dict(sorted(depth_counts.items())),
        "stop_node_counts": dict(node_counts.most_common()),
    }
