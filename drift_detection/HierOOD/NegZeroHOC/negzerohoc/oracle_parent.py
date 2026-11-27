from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from .unknown_scoring import grouped_unknown_logits


def _balanced_rate(values: torch.Tensor, groups: list[str]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values.detach().cpu().tolist(), groups):
        grouped[str(group)].append(float(value))
    if not grouped:
        return 0.0
    return sum(sum(items) / len(items) for items in grouped.values()) / len(grouped)


@torch.no_grad()
def oracle_parent_diagnostics(
    features: torch.Tensor,
    classes: list[str],
    targets: torch.Tensor,
    hierarchy,
    semantic_index,
    *,
    logit_scale: float,
    allow_root_unknown: bool = False,
    unknown_aggregation: str = "logmeanexp",
) -> dict:
    """Separate OOD parent routing from rejection at the true parent.

    The nearest retained ID ancestor of each OOD class is its rejection parent.
    ``oracle_unknown`` ignores routing and evaluates the unknown candidate only at
    that parent. ``positive_route`` uses frozen positive children on every edge
    before the rejection parent, while ``student_route`` also lets earlier
    unknown candidates compete.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [images, dim]")
    if int(features.shape[0]) != int(targets.numel()):
        raise ValueError("features and targets must contain the same number of samples")
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be positive")

    features = F.normalize(features.float(), dim=-1)
    node_indices = hierarchy.gen_ds2node_map(list(classes))[targets.long().cpu()]
    rejection_nodes = [hierarchy.id_node_list[int(index)] for index in node_indices]
    dataset_classes = [classes[int(index)] for index in targets.long().cpu().tolist()]

    supported_indices = [
        index
        for index, node in enumerate(rejection_nodes)
        if node in semantic_index
        and semantic_index[node].unknown_feature is not None
        and (node != "root" or allow_root_unknown)
    ]
    supported_set = set(supported_indices)
    unsupported_indices = [
        index for index in range(len(rejection_nodes)) if index not in supported_set
    ]
    if not supported_indices:
        return {
            "num_samples": int(features.shape[0]),
            "supported_samples": 0,
            "unsupported_samples": len(unsupported_indices),
            "unsupported_nodes": sorted({rejection_nodes[index] for index in unsupported_indices}),
            "oracle_unknown_recall": 0.0,
            "oracle_unknown_balanced_recall": 0.0,
            "positive_route_reach_rate": 0.0,
            "student_route_reach_rate": 0.0,
            "joint_student_exact_rate": 0.0,
            "by_rejection_parent": {},
        }

    oracle_unknown = torch.zeros(len(rejection_nodes), dtype=torch.bool)
    positive_route = torch.zeros(len(rejection_nodes), dtype=torch.bool)
    student_route = torch.zeros(len(rejection_nodes), dtype=torch.bool)
    local_margins = torch.zeros(len(rejection_nodes), dtype=torch.float32)
    per_parent: dict[str, dict] = {}

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index in supported_indices:
        grouped_indices[rejection_nodes[index]].append(index)

    for rejection_parent, indices in sorted(grouped_indices.items()):
        index_tensor = torch.tensor(indices, dtype=torch.long, device=features.device)
        parent_features = features.index_select(0, index_tensor)
        local = semantic_index[rejection_parent]
        local_logits = grouped_unknown_logits(
            parent_features,
            local.child_features.to(features.device),
            local.unknown_feature.to(features.device),
            logit_scale=float(logit_scale),
            aggregation=unknown_aggregation,
        )
        unknown_index = len(local.children)
        local_predictions = local_logits.argmax(dim=1)
        local_wins = local_predictions == unknown_index
        margins = local_logits[:, unknown_index] - local_logits[:, :unknown_index].max(dim=1).values

        path = [
            hierarchy.id_node_list[index]
            for index in hierarchy.node_ancestors.get(rejection_parent, [])
        ] + [rejection_parent]
        positive_reaches = torch.ones(len(indices), dtype=torch.bool, device=features.device)
        student_reaches = torch.ones(len(indices), dtype=torch.bool, device=features.device)
        for ancestor, route_child in zip(path[:-1], path[1:]):
            route_local = semantic_index[ancestor]
            child_features = F.normalize(
                route_local.child_features.to(features.device).float(), dim=-1
            )
            target_index = list(route_local.children).index(route_child)
            positive_logits = float(logit_scale) * (parent_features @ child_features.t())
            positive_reaches &= positive_logits.argmax(dim=1) == target_index

            has_unknown = route_local.unknown_feature is not None and (
                ancestor != "root" or allow_root_unknown
            )
            if has_unknown:
                student_logits = grouped_unknown_logits(
                    parent_features,
                    child_features,
                    route_local.unknown_feature.to(features.device),
                    logit_scale=float(logit_scale),
                    aggregation=unknown_aggregation,
                )
            else:
                student_logits = positive_logits
            student_reaches &= student_logits.argmax(dim=1) == target_index

        oracle_unknown[index_tensor.cpu()] = local_wins.cpu()
        positive_route[index_tensor.cpu()] = positive_reaches.cpu()
        student_route[index_tensor.cpu()] = student_reaches.cpu()
        local_margins[index_tensor.cpu()] = margins.cpu()
        per_parent[rejection_parent] = {
            "count": len(indices),
            "oracle_unknown_recall": float(local_wins.float().mean().cpu()),
            "positive_route_reach_rate": float(positive_reaches.float().mean().cpu()),
            "student_route_reach_rate": float(student_reaches.float().mean().cpu()),
            "joint_student_exact_rate": float(
                (student_reaches & local_wins).float().mean().cpu()
            ),
            "mean_unknown_margin": float(margins.mean().cpu()),
        }

    supported_mask = torch.tensor(
        [index in supported_set for index in range(len(rejection_nodes))],
        dtype=torch.bool,
    )
    supported_nodes = [rejection_nodes[index] for index in supported_indices]
    supported_classes = [dataset_classes[index] for index in supported_indices]
    oracle_values = oracle_unknown[supported_mask]
    positive_values = positive_route[supported_mask]
    student_values = student_route[supported_mask]
    joint_values = oracle_values & student_values

    return {
        "num_samples": int(features.shape[0]),
        "supported_samples": len(supported_indices),
        "unsupported_samples": len(unsupported_indices),
        "unsupported_nodes": sorted({rejection_nodes[index] for index in unsupported_indices}),
        "oracle_unknown_recall": float(oracle_values.float().mean()),
        "oracle_unknown_balanced_recall": _balanced_rate(oracle_values, supported_nodes),
        "oracle_unknown_class_balanced_recall": _balanced_rate(
            oracle_values, supported_classes
        ),
        "positive_route_reach_rate": float(positive_values.float().mean()),
        "positive_route_balanced_reach_rate": _balanced_rate(
            positive_values, supported_nodes
        ),
        "student_route_reach_rate": float(student_values.float().mean()),
        "student_route_balanced_reach_rate": _balanced_rate(
            student_values, supported_nodes
        ),
        "joint_student_exact_rate": float(joint_values.float().mean()),
        "joint_student_balanced_exact_rate": _balanced_rate(
            joint_values, supported_nodes
        ),
        "mean_unknown_margin": float(local_margins[supported_mask].mean()),
        "by_rejection_parent": per_parent,
    }
