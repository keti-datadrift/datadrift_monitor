from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MetricTerminalSpec:
    """A complete metric-decoding candidate in the hierarchy.

    A known leaf terminates at its final positive edge. An unknown candidate
    follows the positive route to ``unknown_parent`` and then terminates at the
    negative/unknown prompt owned by that parent.
    """

    node: str
    route_edges: tuple[tuple[str, str], ...]
    unknown_parent: str | None = None

    @property
    def kind(self) -> str:
        return "unknown" if self.unknown_parent is not None else "leaf"


def _node_path(hierarchy, node: str) -> list[str]:
    if node not in hierarchy.id_node_list:
        raise ValueError(f"Unknown hierarchy node: {node!r}")
    ancestors = [
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.node_ancestors.get(node, [])
    ]
    return ancestors + [node]


def _route_edges(hierarchy, node: str) -> tuple[tuple[str, str], ...]:
    path = _node_path(hierarchy, node)
    return tuple(zip(path[:-1], path[1:]))


def build_metric_terminal_specs(
    hierarchy,
    unknown_parents: list[str] | tuple[str, ...] = (),
    *,
    allow_root_unknown: bool = False,
) -> list[MetricTerminalSpec]:
    """Build the global leaf and enabled parent-unknown terminal space."""
    leaves = [
        node for node in hierarchy.id_node_list
        if node not in hierarchy.parent2children
    ]
    if not leaves:
        raise RuntimeError("Metric terminal decoder found no known leaf nodes")

    specs = [
        MetricTerminalSpec(node=leaf, route_edges=_route_edges(hierarchy, leaf))
        for leaf in leaves
    ]
    for parent in dict.fromkeys(unknown_parents):
        if parent not in hierarchy.parent2children:
            raise ValueError(
                f"Unknown terminal parent must be an internal hierarchy node: {parent!r}"
            )
        if parent == "root" and not allow_root_unknown:
            continue
        specs.append(
            MetricTerminalSpec(
                node=parent,
                route_edges=_route_edges(hierarchy, parent),
                unknown_parent=parent,
            )
        )
    return specs


def normalized_softmin(values: torch.Tensor, temperature: float, dim: int = -1) -> torch.Tensor:
    """Smooth minimum without a path-length-dependent ``log(n)`` offset."""
    if float(temperature) <= 0.0:
        raise ValueError("Soft-min temperature must be positive")
    count = int(values.shape[dim])
    if count <= 0:
        raise ValueError("Soft-min needs at least one route component")
    return -float(temperature) * (
        torch.logsumexp(-values / float(temperature), dim=dim) - math.log(count)
    )


def grouped_cosine_logmeanexp(
    image_features: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Count-invariant aggregation of several terminal prompt prototypes."""
    if float(temperature) <= 0.0:
        raise ValueError("Unknown aggregation temperature must be positive")
    if prototypes.dim() == 1:
        prototypes = prototypes.unsqueeze(0)
    if prototypes.dim() != 2 or int(prototypes.shape[0]) == 0:
        raise ValueError("Unknown prototypes must have shape [K, D] with K >= 1")
    images = F.normalize(image_features.float(), dim=-1)
    prompts = F.normalize(prototypes.to(images.device).float(), dim=-1)
    similarities = images @ prompts.t()
    count = int(similarities.shape[1])
    return float(temperature) * (
        torch.logsumexp(similarities / float(temperature), dim=1) - math.log(count)
    )


def metric_terminal_scores(
    image_features: torch.Tensor,
    positive_edge_features: dict[tuple[str, str], torch.Tensor],
    terminal_specs: list[MetricTerminalSpec],
    *,
    unknown_features_by_parent: dict[str, torch.Tensor] | None = None,
    terminal_weight: float = 0.5,
    bottleneck_temperature: float = 0.1,
    unknown_temperature: float = 0.07,
) -> dict:
    """Score all known-leaf and parent-unknown terminals in one metric space.

    The score combines terminal affinity with length-normalized route
    consistency. No local softmax is used, so scores remain comparable across
    parents and hierarchy depths.
    """
    if image_features.dim() != 2:
        raise ValueError("image_features must have shape [B, D]")
    if not 0.0 <= float(terminal_weight) <= 1.0:
        raise ValueError("terminal_weight must be in [0, 1]")
    if not terminal_specs:
        raise ValueError("At least one metric terminal is required")

    images = F.normalize(image_features.float(), dim=-1)
    device = images.device
    needed_edges = list(dict.fromkeys(
        edge for spec in terminal_specs for edge in spec.route_edges
    ))
    missing_edges = [edge for edge in needed_edges if edge not in positive_edge_features]
    if missing_edges:
        raise KeyError(f"Missing {len(missing_edges)} positive edge features: {missing_edges[:3]}")
    edge_matrix = torch.stack([
        positive_edge_features[edge].to(device).float() for edge in needed_edges
    ])
    edge_matrix = F.normalize(edge_matrix, dim=-1)
    edge_affinities = images @ edge_matrix.t()
    edge_to_column = {edge: index for index, edge in enumerate(needed_edges)}

    unknown_features_by_parent = unknown_features_by_parent or {}
    unknown_affinities = {}
    for spec in terminal_specs:
        if spec.unknown_parent is None or spec.unknown_parent in unknown_affinities:
            continue
        if spec.unknown_parent not in unknown_features_by_parent:
            raise KeyError(f"Missing unknown features for parent {spec.unknown_parent!r}")
        unknown_affinities[spec.unknown_parent] = grouped_cosine_logmeanexp(
            images,
            unknown_features_by_parent[spec.unknown_parent],
            unknown_temperature,
        )

    score_columns = []
    terminal_affinity_columns = []
    route_consistency_columns = []
    for spec in terminal_specs:
        components = [edge_affinities[:, edge_to_column[edge]] for edge in spec.route_edges]
        if spec.unknown_parent is not None:
            terminal_affinity = unknown_affinities[spec.unknown_parent]
            components.append(terminal_affinity)
        else:
            if not components:
                raise ValueError(f"Known terminal {spec.node!r} has an empty route")
            terminal_affinity = components[-1]
        component_matrix = torch.stack(components, dim=1)
        route_consistency = normalized_softmin(
            component_matrix,
            bottleneck_temperature,
            dim=1,
        )
        score_columns.append(
            float(terminal_weight) * terminal_affinity
            + (1.0 - float(terminal_weight)) * route_consistency
        )
        terminal_affinity_columns.append(terminal_affinity)
        route_consistency_columns.append(route_consistency)

    return {
        "score_matrix": torch.stack(score_columns, dim=1),
        "terminal_affinity_matrix": torch.stack(terminal_affinity_columns, dim=1),
        "route_consistency_matrix": torch.stack(route_consistency_columns, dim=1),
        "candidate_nodes": [spec.node for spec in terminal_specs],
        "candidate_kinds": [spec.kind for spec in terminal_specs],
    }


@torch.no_grad()
def predict_features_metric_terminal(
    image_features: torch.Tensor,
    hierarchy,
    positive_edge_features: dict[tuple[str, str], torch.Tensor],
    terminal_specs: list[MetricTerminalSpec],
    **score_kwargs,
) -> dict:
    scores = metric_terminal_scores(
        image_features,
        positive_edge_features,
        terminal_specs,
        **score_kwargs,
    )
    winner_indices = scores["score_matrix"].argmax(dim=1).detach().cpu().tolist()
    winner_nodes = [scores["candidate_nodes"][index] for index in winner_indices]
    winner_kinds = [scores["candidate_kinds"][index] for index in winner_indices]
    node_to_index = {node: index for index, node in enumerate(hierarchy.id_node_list)}
    preds = torch.tensor([node_to_index[node] for node in winner_nodes], dtype=torch.long)
    depth_counts = Counter(
        len(hierarchy.node_ancestors.get(node, [])) for node in winner_nodes
    )
    kind_counts = Counter(winner_kinds)
    return {
        "preds": preds,
        "diagnostics": {
            "stop_depth_counts": dict(sorted(depth_counts.items())),
            "stop_node_counts": dict(Counter(winner_nodes).most_common()),
            "candidate_type_counts": dict(kind_counts),
            "unknown_selection_rate": kind_counts.get("unknown", 0) / max(1, len(winner_nodes)),
        },
        "scores": scores,
    }
