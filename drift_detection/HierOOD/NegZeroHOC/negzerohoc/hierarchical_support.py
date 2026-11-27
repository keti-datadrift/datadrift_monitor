from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class HierarchicalSupportCalibration:
    """ID-train-only prototype and conformal calibration state."""

    prototype_nodes: tuple[str, ...]
    prototypes: torch.Tensor
    node_prototype_indices: dict[str, torch.Tensor]
    node_calibration_scores: dict[str, torch.Tensor]
    prototype_calibration_scores: tuple[torch.Tensor, ...]
    reference_indices: torch.Tensor
    calibration_indices: torch.Tensor


def stratified_reference_calibration_split(
    targets: torch.Tensor,
    *,
    reference_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < float(reference_fraction) < 1.0:
        raise ValueError("reference_fraction must be strictly between zero and one")
    targets = targets.detach().long().cpu()
    generator = torch.Generator().manual_seed(int(seed))
    reference = []
    calibration = []
    for target in sorted(set(targets.tolist())):
        indices = torch.nonzero(targets == int(target), as_tuple=False).flatten()
        if int(indices.numel()) < 2:
            raise ValueError(
                f"Class {target} needs at least two samples for an internal split"
            )
        indices = indices[torch.randperm(int(indices.numel()), generator=generator)]
        reference_count = int(round(float(reference_fraction) * int(indices.numel())))
        reference_count = min(max(1, reference_count), int(indices.numel()) - 1)
        reference.extend(indices[:reference_count].tolist())
        calibration.extend(indices[reference_count:].tolist())
    return (
        torch.tensor(sorted(reference), dtype=torch.long),
        torch.tensor(sorted(calibration), dtype=torch.long),
    )


def _node_path(hierarchy, node: str) -> tuple[str, ...]:
    ancestors = tuple(
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.node_ancestors.get(node, [])
    )
    return ancestors + (node,)


def build_hierarchical_support_calibration(
    hierarchy,
    features: torch.Tensor,
    classes: list[str],
    targets: torch.Tensor,
    *,
    reference_fraction: float = 0.8,
    seed: int = 0,
    reference_indices: torch.Tensor | None = None,
    calibration_indices: torch.Tensor | None = None,
) -> HierarchicalSupportCalibration:
    features = F.normalize(features.detach().float().cpu(), dim=-1)
    targets = targets.detach().long().cpu()
    if int(features.shape[0]) != int(targets.shape[0]):
        raise ValueError("features and targets must contain the same number of samples")
    if (reference_indices is None) != (calibration_indices is None):
        raise ValueError(
            "reference_indices and calibration_indices must be supplied together"
        )
    if reference_indices is None:
        reference_indices, calibration_indices = (
            stratified_reference_calibration_split(
                targets,
                reference_fraction=reference_fraction,
                seed=seed,
            )
        )
    else:
        reference_indices = reference_indices.detach().long().cpu()
        calibration_indices = calibration_indices.detach().long().cpu()
        if int(reference_indices.numel()) == 0 or int(calibration_indices.numel()) == 0:
            raise ValueError("Explicit reference and calibration splits cannot be empty")
        overlap = set(reference_indices.tolist()) & set(calibration_indices.tolist())
        if overlap:
            raise ValueError("Explicit reference and calibration splits must be disjoint")
        largest = max(reference_indices.max().item(), calibration_indices.max().item())
        if largest >= int(targets.numel()):
            raise IndexError("Explicit support split index exceeds the payload length")
    class_to_node = hierarchy.gen_ds2node_map(classes).long().cpu()
    sample_node_indices = class_to_node[targets]
    sample_nodes = [
        hierarchy.id_node_list[int(index)] for index in sample_node_indices.tolist()
    ]

    prototype_nodes = []
    prototypes = []
    for node in sorted(set(sample_nodes)):
        node_index = hierarchy.id_node_list.index(node)
        mask = sample_node_indices.index_select(0, reference_indices) == node_index
        selected = reference_indices[mask]
        if int(selected.numel()) == 0:
            raise RuntimeError(f"No reference samples were retained for {node!r}")
        prototype_nodes.append(node)
        prototypes.append(F.normalize(features.index_select(0, selected).mean(dim=0), dim=0))
    prototype_tensor = torch.stack(prototypes)

    node_prototype_indices = {}
    node_calibration_scores = {}
    node_to_prototype_index = {
        node: index for index, node in enumerate(prototype_nodes)
    }
    prototype_calibration_scores = []
    for node in prototype_nodes:
        node_index = hierarchy.id_node_list.index(node)
        selected = calibration_indices[
            sample_node_indices.index_select(0, calibration_indices)
            == node_index
        ]
        if int(selected.numel()) == 0:
            raise RuntimeError(
                f"No class-conditional calibration samples for {node!r}"
            )
        prototype = prototype_tensor[node_to_prototype_index[node]]
        scores = (
            features.index_select(0, selected) @ prototype
        ).sort().values
        prototype_calibration_scores.append(scores)
    candidate_nodes = sorted(
        set(hierarchy.parent2children) | set(prototype_nodes),
        key=lambda node: (len(hierarchy.node_ancestors.get(node, [])), node),
    )
    for node in candidate_nodes:
        descendant_indices = [
            index
            for index, leaf in enumerate(prototype_nodes)
            if node in _node_path(hierarchy, leaf)
        ]
        if not descendant_indices:
            continue
        prototype_index = torch.tensor(descendant_indices, dtype=torch.long)
        node_prototype_indices[node] = prototype_index
        calibration_sample_indices = torch.tensor(
            [
                index
                for index in calibration_indices.tolist()
                if node in _node_path(hierarchy, sample_nodes[index])
            ],
            dtype=torch.long,
        )
        if int(calibration_sample_indices.numel()) == 0:
            raise RuntimeError(f"No calibration samples were retained below {node!r}")
        query = features.index_select(0, calibration_sample_indices)
        bank = prototype_tensor.index_select(0, prototype_index)
        scores = (query @ bank.t()).max(dim=1).values.sort().values
        node_calibration_scores[node] = scores

    return HierarchicalSupportCalibration(
        prototype_nodes=tuple(prototype_nodes),
        prototypes=prototype_tensor,
        node_prototype_indices=node_prototype_indices,
        node_calibration_scores=node_calibration_scores,
        prototype_calibration_scores=tuple(
            prototype_calibration_scores
        ),
        reference_indices=reference_indices,
        calibration_indices=calibration_indices,
    )


def expand_to_reference_sample_prototypes(
    hierarchy,
    calibration: HierarchicalSupportCalibration,
    features: torch.Tensor,
    classes: list[str],
    targets: torch.Tensor,
) -> HierarchicalSupportCalibration:
    """Replace leaf centroids by every retained reference embedding."""
    features = F.normalize(features.detach().float().cpu(), dim=-1)
    targets = targets.detach().long().cpu()
    class_to_node = hierarchy.gen_ds2node_map(classes).long().cpu()
    sample_node_indices = class_to_node[targets]
    sample_nodes = [
        hierarchy.id_node_list[int(index)]
        for index in sample_node_indices.tolist()
    ]
    reference_indices = calibration.reference_indices.detach().long().cpu()
    prototype_nodes = tuple(
        sample_nodes[index] for index in reference_indices.tolist()
    )
    prototypes = features.index_select(0, reference_indices)
    candidate_nodes = sorted(
        set(hierarchy.parent2children) | set(prototype_nodes),
        key=lambda node: (
            len(hierarchy.node_ancestors.get(node, [])),
            node,
        ),
    )
    node_prototype_indices = {}
    for node in candidate_nodes:
        descendant_indices = [
            index
            for index, leaf in enumerate(prototype_nodes)
            if node in _node_path(hierarchy, leaf)
        ]
        if descendant_indices:
            node_prototype_indices[node] = torch.tensor(
                descendant_indices, dtype=torch.long
            )
    return HierarchicalSupportCalibration(
        prototype_nodes=prototype_nodes,
        prototypes=prototypes,
        node_prototype_indices=node_prototype_indices,
        node_calibration_scores=calibration.node_calibration_scores,
        prototype_calibration_scores=calibration.prototype_calibration_scores,
        reference_indices=calibration.reference_indices,
        calibration_indices=calibration.calibration_indices,
    )


def support_scores(
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
    node: str,
) -> torch.Tensor:
    if node not in calibration.node_prototype_indices:
        raise KeyError(f"No support prototypes exist for node {node!r}")
    query = F.normalize(features.detach().float().cpu(), dim=-1)
    indices = calibration.node_prototype_indices[node]
    bank = calibration.prototypes.index_select(0, indices)
    return (query @ bank.t()).max(dim=1).values


def conformal_p_values(
    scores: torch.Tensor,
    calibration_scores: torch.Tensor,
) -> torch.Tensor:
    """Return high p-values for queries well supported by ID calibration."""
    scores = scores.detach().float().cpu()
    calibration_scores = calibration_scores.detach().float().cpu().sort().values
    if int(calibration_scores.numel()) == 0:
        raise ValueError("At least one calibration score is required")
    counts = torch.searchsorted(calibration_scores, scores, right=True)
    return (1.0 + counts.float()) / (float(calibration_scores.numel()) + 1.0)


def node_support_p_values(
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
) -> dict[str, torch.Tensor]:
    return {
        node: conformal_p_values(
            support_scores(features, calibration, node),
            calibration.node_calibration_scores[node],
        )
        for node in calibration.node_calibration_scores
    }


def nearest_support_prototype_predictions(
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
    hierarchy,
) -> torch.Tensor:
    query = F.normalize(features.detach().float().cpu(), dim=-1)
    prototypes = F.normalize(calibration.prototypes.float().cpu(), dim=-1)
    winners = (query @ prototypes.t()).argmax(dim=1)
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    return torch.tensor(
        [
            node_to_index[calibration.prototype_nodes[int(index)]]
            for index in winners.tolist()
        ],
        dtype=torch.long,
    )


def mondrian_support_p_values(
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
) -> torch.Tensor:
    query = F.normalize(features.detach().float().cpu(), dim=-1)
    prototypes = F.normalize(calibration.prototypes.float().cpu(), dim=-1)
    similarities = query @ prototypes.t()
    winners = similarities.argmax(dim=1)
    values = torch.empty(int(query.shape[0]), dtype=torch.float32)
    for prototype_index in range(int(prototypes.shape[0])):
        mask = winners == prototype_index
        if not bool(mask.any()):
            continue
        values[mask] = conformal_p_values(
            similarities[mask, prototype_index],
            calibration.prototype_calibration_scores[prototype_index],
        )
    return values


def positive_route_conditionals(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    *,
    logit_scale: float,
) -> dict[str, torch.Tensor]:
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be positive")
    query = F.normalize(features.detach().float().cpu(), dim=-1)
    result = {}
    for parent, children in hierarchy.parent2children.items():
        local = semantic_index[parent]
        child_features = F.normalize(local.child_features.detach().float().cpu(), dim=-1)
        logits = float(logit_scale) * (query @ child_features.t())
        if int(logits.shape[1]) != len(children):
            raise RuntimeError(f"Routing feature count differs at parent {parent!r}")
        result[parent] = F.softmax(logits, dim=1)
    return result


def factorized_terminal_probabilities(
    hierarchy,
    route_conditionals: dict[str, torch.Tensor],
    support_p_values_by_node: dict[str, torch.Tensor],
    *,
    alpha: float = 0.05,
    gate: str = "conformal_ramp",
) -> torch.Tensor:
    """Build one normalized distribution over leaves and unknown parents."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if gate not in {"conformal_ramp", "hard"}:
        raise ValueError(f"Unsupported support gate: {gate}")
    if "root" not in route_conditionals:
        raise ValueError("route_conditionals must contain root")
    sample_count = int(route_conditionals["root"].shape[0])
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    terminal_probabilities = torch.zeros(
        sample_count,
        len(hierarchy.id_node_list),
        dtype=torch.float32,
    )
    reach = {"root": torch.ones(sample_count, dtype=torch.float32)}
    ordered_parents = sorted(
        hierarchy.parent2children,
        key=lambda node: (len(hierarchy.node_ancestors.get(node, [])), node),
    )
    for parent in ordered_parents:
        if parent not in reach:
            continue
        parent_reach = reach[parent]
        if parent == "root":
            continuation = torch.ones_like(parent_reach)
        else:
            p_values = support_p_values_by_node[parent].float().cpu()
            if gate == "hard":
                continuation = (p_values > float(alpha)).float()
            else:
                continuation = (p_values / float(alpha)).clamp(0.0, 1.0)
            terminal_probabilities[:, node_to_index[parent]] += (
                parent_reach * (1.0 - continuation)
            )
        route = route_conditionals[parent].float().cpu()
        children = list(hierarchy.parent2children[parent])
        for child_index, child in enumerate(children):
            child_reach = parent_reach * continuation * route[:, child_index]
            if child in hierarchy.parent2children:
                reach[child] = child_reach
            else:
                terminal_probabilities[:, node_to_index[child]] += child_reach
    probability_sums = terminal_probabilities.sum(dim=1)
    if not torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=1e-5):
        raise RuntimeError(
            "Factorized terminal probabilities do not sum to one: "
            f"range=({float(probability_sums.min())}, {float(probability_sums.max())})"
        )
    return terminal_probabilities


def expected_hierarchy_distance_predictions(
    terminal_probabilities: torch.Tensor,
    hierarchy_distance_matrix: torch.Tensor,
) -> torch.Tensor:
    probabilities = terminal_probabilities.detach().float().cpu()
    distances = hierarchy_distance_matrix.detach().float().cpu()
    expected_distances = probabilities @ distances.t()
    return expected_distances.argmin(dim=1)


def global_gate_route_stop_predictions(
    hierarchy,
    positive_leaf_predictions: torch.Tensor,
    support_p_values_by_node: dict[str, torch.Tensor],
    *,
    alpha: float = 0.05,
    localizer: str = "first_unsupported",
) -> tuple[torch.Tensor, dict]:
    """Binary root support gate followed by a positive-route stop localizer."""
    if localizer not in {"first_unsupported", "weakest_support", "deepest"}:
        raise ValueError(f"Unsupported rejection localizer: {localizer}")
    root_p = support_p_values_by_node["root"].float().cpu()
    reject = root_p <= float(alpha)
    positive_leaf_predictions = positive_leaf_predictions.detach().long().cpu()
    predictions = positive_leaf_predictions.clone()
    stop_nodes = []
    for sample_index, leaf_index in enumerate(positive_leaf_predictions.tolist()):
        leaf = hierarchy.id_node_list[int(leaf_index)]
        path = _node_path(hierarchy, leaf)
        parents = [
            node
            for node in path[:-1]
            if node != "root" and node in hierarchy.parent2children
        ]
        if not bool(reject[sample_index]) or not parents:
            stop_nodes.append(leaf)
            continue
        if localizer == "deepest":
            stop = parents[-1]
        elif localizer == "weakest_support":
            stop = min(
                parents,
                key=lambda node: float(
                    support_p_values_by_node[node][sample_index]
                ),
            )
        else:
            unsupported = [
                node
                for node in parents
                if float(support_p_values_by_node[node][sample_index]) <= float(alpha)
            ]
            stop = unsupported[0] if unsupported else min(
                parents,
                key=lambda node: float(
                    support_p_values_by_node[node][sample_index]
                ),
            )
        predictions[sample_index] = hierarchy.id_node_list.index(stop)
        stop_nodes.append(stop)
    return predictions, {
        "rejection_rate": float(reject.float().mean()),
        "rejected": int(reject.sum()),
        "stop_nodes": stop_nodes,
        "root_support_p_values": root_p,
    }
