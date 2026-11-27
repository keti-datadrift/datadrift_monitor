from __future__ import annotations

from collections import Counter
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metric_terminal import (
    MetricTerminalSpec,
    grouped_cosine_logmeanexp,
    normalized_softmin,
)
from .prompt_text import build_parent_text


class VirtualSiblingPromptLearner(nn.Module):
    """Shared learnable contexts for an augmented unknown child of each parent."""

    def __init__(
        self,
        positive_learner,
        *,
        num_unknown_prompts: int = 2,
        init_noise: float = 1e-2,
    ):
        super().__init__()
        self.positive_learner = positive_learner
        self.num_unknown_prompts = max(1, int(num_unknown_prompts))
        with torch.no_grad():
            context = positive_learner._context_for_parents(["root"])
        if context.ndim != 3 or int(context.shape[1]) == 0:
            raise ValueError(
                "Virtual sibling prompts require positive context tokens"
            )
        self.context_offsets = nn.Parameter(torch.empty(
            self.num_unknown_prompts,
            int(context.shape[1]),
            int(context.shape[2]),
            device=context.device,
            dtype=context.dtype,
        ))
        nn.init.uniform_(
            self.context_offsets,
            -float(init_noise),
            float(init_noise),
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.context_offsets]

    def encode_parents(self, parents: list[str]) -> torch.Tensor:
        if not parents:
            return torch.empty(
                0,
                self.num_unknown_prompts,
                self.positive_learner.projection_dim,
                device=self.context_offsets.device,
            )
        with torch.no_grad():
            base_context = self.positive_learner._context_for_parents(
                parents
            ).detach()
        contexts = (
            base_context.unsqueeze(1)
            + self.context_offsets.unsqueeze(0)
        )
        texts = [
            build_parent_text(
                self.positive_learner.dataset_name,
                self.positive_learner.hierarchy,
                parent,
            )
            for parent in parents
            for _ in range(self.num_unknown_prompts)
        ]
        flat_contexts = contexts.reshape(
            len(parents) * self.num_unknown_prompts,
            contexts.shape[2],
            contexts.shape[3],
        )
        features = (
            self.positive_learner.text_encoder.encode_with_context(
                texts,
                flat_contexts,
            )
        )
        return features.view(
            len(parents),
            self.num_unknown_prompts,
            -1,
        )


def _node_path(hierarchy, node: str) -> tuple[str, ...]:
    if node not in hierarchy.id_node_list:
        raise ValueError(f"Unknown hierarchy node: {node!r}")
    ancestors = tuple(
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.node_ancestors.get(node, [])
    )
    return ancestors + (node,)


def _tree_distance(
    first_path: tuple[str, ...],
    second_path: tuple[str, ...],
) -> int:
    shared = 0
    for first, second in zip(first_path, second_path):
        if first != second:
            break
        shared += 1
    return len(first_path) + len(second_path) - 2 * shared


def augmented_unknown_distance_matrix(
    hierarchy,
    unknown_parents: list[str],
    positive_nodes: list[str],
) -> torch.Tensor:
    """Distance from every virtual unknown child ``u_p`` to positive nodes."""
    positive_paths = [_node_path(hierarchy, node) for node in positive_nodes]
    rows = []
    for parent in unknown_parents:
        if parent not in hierarchy.parent2children:
            raise ValueError(
                f"Unknown parent must be internal: {parent!r}"
            )
        virtual_path = _node_path(hierarchy, parent) + (
            f"__unknown__:{parent}",
        )
        rows.append([
            _tree_distance(virtual_path, path)
            for path in positive_paths
        ])
    return torch.tensor(rows, dtype=torch.float32)


def leaf_unknown_distance_matrix(
    hierarchy,
    leaves: list[str],
    unknown_parents: list[str],
) -> torch.Tensor:
    """Distance from known leaves to augmented unknown-child terminals."""
    rows = []
    for leaf in leaves:
        leaf_path = _node_path(hierarchy, leaf)
        row = []
        for parent in unknown_parents:
            virtual_path = _node_path(hierarchy, parent) + (
                f"__unknown__:{parent}",
            )
            row.append(_tree_distance(leaf_path, virtual_path))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def tree_ordinal_prompt_loss(
    unknown_features: torch.Tensor,
    positive_node_features: torch.Tensor,
    tree_distances: torch.Tensor,
    *,
    margin_per_step: float = 0.02,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Make text similarity order agree with augmented-tree distance order."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if unknown_features.ndim != 3:
        raise ValueError("unknown_features must have shape [parents,K,D]")
    if positive_node_features.ndim != 2:
        raise ValueError(
            "positive_node_features must have shape [nodes,D]"
        )
    expected = (
        unknown_features.shape[0],
        positive_node_features.shape[0],
    )
    if tuple(tree_distances.shape) != expected:
        raise ValueError(
            f"tree_distances must have shape {expected}, got "
            f"{tuple(tree_distances.shape)}"
        )

    unknowns = F.normalize(unknown_features.float(), dim=-1)
    positives = F.normalize(positive_node_features.float(), dim=-1)
    similarities = torch.einsum("pkd,nd->pkn", unknowns, positives)
    distances = tree_distances.to(
        similarities.device,
        dtype=similarities.dtype,
    )
    # i is the nearer node and j is the farther node.
    distance_delta = (
        distances[:, None, None, :]
        - distances[:, None, :, None]
    )
    ordered = distance_delta > 0.0
    near_similarity = similarities.unsqueeze(-1)
    far_similarity = similarities.unsqueeze(-2)
    violation = (
        far_similarity
        - near_similarity
        + float(margin_per_step) * distance_delta
    )
    expanded_mask = ordered.expand(
        -1,
        unknown_features.shape[1],
        -1,
        -1,
    )
    if not bool(expanded_mask.any()):
        zero = similarities.sum() * 0.0
        return zero, {
            "tree_ordinal_loss": 0.0,
            "tree_order_violation_rate": 0.0,
            "tree_order_pairs": 0,
        }
    selected = violation[expanded_mask]
    loss = (
        float(temperature)
        * F.softplus(selected / float(temperature))
    ).mean()
    return loss, {
        "tree_ordinal_loss": float(loss.detach().cpu()),
        "tree_order_violation_rate": float(
            (selected > 0.0).float().mean().detach().cpu()
        ),
        "tree_order_pairs": int(selected.numel()),
    }


def virtual_sibling_shell_loss(
    unknown_features_by_parent: dict[str, torch.Tensor],
    child_features_by_parent: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    """Place ``u_p`` on the empirical sibling shell and prevent bank collapse."""
    shell_terms = []
    isotropy_terms = []
    diversity_terms = []
    target_values = []
    for parent, unknown_features in unknown_features_by_parent.items():
        if parent not in child_features_by_parent:
            raise KeyError(f"Missing child features for {parent!r}")
        children = F.normalize(
            child_features_by_parent[parent].float(),
            dim=-1,
        )
        unknowns = F.normalize(unknown_features.float(), dim=-1)
        if children.ndim != 2 or int(children.shape[0]) < 2:
            raise ValueError(
                "Every virtual sibling parent needs at least two children"
            )
        child_pairwise = children @ children.t()
        child_mask = ~torch.eye(
            children.shape[0],
            dtype=torch.bool,
            device=children.device,
        )
        sibling_target = child_pairwise[child_mask].mean().detach()
        unknown_child = unknowns @ children.t()
        shell_terms.append(
            (unknown_child.mean(dim=1) - sibling_target).square().mean()
        )
        isotropy_terms.append(unknown_child.var(dim=1, unbiased=False).mean())
        target_values.append(sibling_target)

        prompt_count = int(unknowns.shape[0])
        if prompt_count > 1:
            unknown_pairwise = unknowns @ unknowns.t()
            unknown_mask = ~torch.eye(
                prompt_count,
                dtype=torch.bool,
                device=unknowns.device,
            )
            diversity_terms.append(
                F.relu(
                    unknown_pairwise[unknown_mask] - sibling_target
                ).square().mean()
            )

    if not shell_terms:
        raise ValueError("At least one parent is required")
    shell = torch.stack(shell_terms).mean()
    isotropy = torch.stack(isotropy_terms).mean()
    diversity = (
        torch.stack(diversity_terms).mean()
        if diversity_terms
        else shell.new_zeros(())
    )
    target = torch.stack(target_values).mean()
    return shell + isotropy + diversity, {
        "virtual_sibling_shell_loss": float(shell.detach().cpu()),
        "virtual_sibling_isotropy_loss": float(
            isotropy.detach().cpu()
        ),
        "virtual_sibling_diversity_loss": float(
            diversity.detach().cpu()
        ),
        "virtual_sibling_target_cosine": float(target.detach().cpu()),
    }


def _normalized_logmeanexp(
    values: torch.Tensor,
    temperature: float,
    dim: int,
) -> torch.Tensor:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    count = int(values.shape[dim])
    return float(temperature) * (
        torch.logsumexp(values / float(temperature), dim=dim)
        - math.log(count)
    )


def tree_complement_terminal_scores(
    image_features: torch.Tensor,
    hierarchy,
    positive_edge_features: dict[tuple[str, str], torch.Tensor],
    terminal_specs: list[MetricTerminalSpec],
    unknown_features_by_parent: dict[str, torch.Tensor],
    *,
    excluded_children_by_parent: dict[str, set[str]] | None = None,
    terminal_weight: float = 0.75,
    bottleneck_temperature: float = 0.1,
    unknown_temperature: float = 0.07,
    child_temperature: float = 0.07,
    complement_weight: float = 0.5,
) -> dict:
    """Score leaves and virtual unknown children in one augmented-tree space.

    ``excluded_children_by_parent`` is used only for leave-one-child-out
    episodes. It removes the held-out child from the corresponding parent's
    known-child support while the caller supplies terminal specs pruned by the
    same hidden subtree.
    """
    if image_features.ndim != 2:
        raise ValueError("image_features must have shape [B,D]")
    if not 0.0 <= float(terminal_weight) <= 1.0:
        raise ValueError("terminal_weight must be in [0,1]")
    if not terminal_specs:
        raise ValueError("At least one terminal candidate is required")

    images = F.normalize(image_features.float(), dim=-1)
    device = images.device
    needed_edges = list(dict.fromkeys(
        edge for spec in terminal_specs for edge in spec.route_edges
    ))
    for spec in terminal_specs:
        if spec.unknown_parent is None:
            continue
        for child in hierarchy.parent2children[spec.unknown_parent]:
            edge = (spec.unknown_parent, child)
            if edge not in needed_edges:
                needed_edges.append(edge)
    missing = [
        edge for edge in needed_edges
        if edge not in positive_edge_features
    ]
    if missing:
        raise KeyError(f"Missing positive edges: {missing[:3]}")
    edge_matrix = F.normalize(torch.stack([
        positive_edge_features[edge].to(device).float()
        for edge in needed_edges
    ]), dim=-1)
    edge_affinities = images @ edge_matrix.t()
    edge_to_column = {
        edge: index for index, edge in enumerate(needed_edges)
    }

    excluded_children_by_parent = excluded_children_by_parent or {}
    unknown_affinities = {}
    child_supports = {}
    for spec in terminal_specs:
        parent = spec.unknown_parent
        if parent is None or parent in unknown_affinities:
            continue
        if parent not in unknown_features_by_parent:
            raise KeyError(f"Missing unknown features for {parent!r}")
        unknown_affinities[parent] = grouped_cosine_logmeanexp(
            images,
            unknown_features_by_parent[parent],
            unknown_temperature,
        )
        excluded = excluded_children_by_parent.get(parent, set())
        visible_children = [
            child
            for child in hierarchy.parent2children[parent]
            if child not in excluded
        ]
        if not visible_children:
            raise ValueError(
                f"Parent {parent!r} has no visible child support"
            )
        child_columns = [
            edge_to_column[(parent, child)]
            for child in visible_children
        ]
        child_supports[parent] = _normalized_logmeanexp(
            edge_affinities[:, child_columns],
            child_temperature,
            dim=1,
        )

    score_columns = []
    terminal_columns = []
    route_columns = []
    complement_columns = []
    for spec in terminal_specs:
        components = [
            edge_affinities[:, edge_to_column[edge]]
            for edge in spec.route_edges
        ]
        if spec.unknown_parent is None:
            if not components:
                raise ValueError(
                    f"Known leaf {spec.node!r} has an empty route"
                )
            terminal_affinity = components[-1]
            complement_gap = terminal_affinity.new_zeros(
                terminal_affinity.shape
            )
        else:
            parent = spec.unknown_parent
            terminal_affinity = unknown_affinities[parent]
            components.append(terminal_affinity)
            complement_gap = (
                terminal_affinity - child_supports[parent]
            )
        route_consistency = normalized_softmin(
            torch.stack(components, dim=1),
            bottleneck_temperature,
            dim=1,
        )
        base_score = (
            float(terminal_weight) * terminal_affinity
            + (1.0 - float(terminal_weight)) * route_consistency
        )
        score_columns.append(
            base_score + float(complement_weight) * complement_gap
        )
        terminal_columns.append(terminal_affinity)
        route_columns.append(route_consistency)
        complement_columns.append(complement_gap)

    return {
        "score_matrix": torch.stack(score_columns, dim=1),
        "terminal_affinity_matrix": torch.stack(
            terminal_columns, dim=1
        ),
        "route_consistency_matrix": torch.stack(route_columns, dim=1),
        "complement_gap_matrix": torch.stack(
            complement_columns, dim=1
        ),
        "candidate_nodes": [spec.node for spec in terminal_specs],
        "candidate_kinds": [spec.kind for spec in terminal_specs],
        "unknown_parents": [
            spec.unknown_parent
            for spec in terminal_specs
            if spec.unknown_parent is not None
        ],
    }


def decoder_aligned_hierarchical_id_loss(
    score_output: dict,
    target_leaves: list[str],
    leaf_unknown_distances: torch.Tensor,
    *,
    local_margin: float = 0.05,
    distance_margin: float = 0.02,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Keep each ID leaf above unknown terminals with tree-aware margins."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    scores = score_output["score_matrix"]
    candidate_nodes = score_output["candidate_nodes"]
    candidate_kinds = score_output["candidate_kinds"]
    if len(target_leaves) != int(scores.shape[0]):
        raise ValueError("target_leaves must match the image batch")
    leaf_indices = {
        node: index
        for index, (node, kind) in enumerate(
            zip(candidate_nodes, candidate_kinds)
        )
        if kind == "leaf"
    }
    unknown_indices = [
        index for index, kind in enumerate(candidate_kinds)
        if kind == "unknown"
    ]
    if not unknown_indices:
        raise ValueError("At least one unknown terminal is required")
    target_indices = torch.tensor(
        [leaf_indices[leaf] for leaf in target_leaves],
        dtype=torch.long,
        device=scores.device,
    )
    true_scores = scores.gather(
        1, target_indices.unsqueeze(1)
    ).squeeze(1)
    unknown_scores = scores[:, unknown_indices]
    if tuple(leaf_unknown_distances.shape) != tuple(
        unknown_scores.shape
    ):
        raise ValueError(
            "leaf_unknown_distances must have shape "
            f"{tuple(unknown_scores.shape)}"
        )
    distances = leaf_unknown_distances.to(
        scores.device,
        dtype=scores.dtype,
    )
    nearest = distances.min(dim=1, keepdim=True).values
    margins = (
        float(local_margin)
        + float(distance_margin) * (distances - nearest)
    )
    violations = (
        unknown_scores - true_scores.unsqueeze(1) + margins
    )
    smooth_hardest = float(temperature) * (
        torch.logsumexp(
            violations / float(temperature),
            dim=1,
        )
        - math.log(int(violations.shape[1]))
    )
    loss = (
        float(temperature)
        * F.softplus(smooth_hardest / float(temperature))
    ).mean()
    return loss, {
        "decoder_id_loss": float(loss.detach().cpu()),
        "decoder_id_unknown_win_rate": float(
            (
                unknown_scores.max(dim=1).values > true_scores
            ).float().mean().detach().cpu()
        ),
        "decoder_id_margin_violation_rate": float(
            (violations > 0.0).float().mean().detach().cpu()
        ),
        "decoder_true_leaf_score": float(
            true_scores.mean().detach().cpu()
        ),
        "decoder_best_unknown_score": float(
            unknown_scores.max(dim=1).values.mean().detach().cpu()
        ),
    }


def tree_virtual_unknown_loss(
    *,
    score_output: dict,
    target_leaves: list[str],
    leaf_unknown_distances: torch.Tensor,
    unknown_feature_tensor: torch.Tensor,
    positive_node_features: torch.Tensor,
    unknown_node_distances: torch.Tensor,
    unknown_features_by_parent: dict[str, torch.Tensor],
    child_features_by_parent: dict[str, torch.Tensor],
    lambda_id: float = 1.0,
    lambda_tree: float = 1.0,
    lambda_shell: float = 1.0,
    local_margin: float = 0.05,
    distance_margin: float = 0.02,
    id_temperature: float = 0.05,
    ordinal_margin_per_step: float = 0.02,
    ordinal_temperature: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Tree-Consistent Virtual-Sibling loss for parent-level unknown prompts."""
    id_loss, id_stats = decoder_aligned_hierarchical_id_loss(
        score_output,
        target_leaves,
        leaf_unknown_distances,
        local_margin=local_margin,
        distance_margin=distance_margin,
        temperature=id_temperature,
    )
    tree_loss, tree_stats = tree_ordinal_prompt_loss(
        unknown_feature_tensor,
        positive_node_features,
        unknown_node_distances,
        margin_per_step=ordinal_margin_per_step,
        temperature=ordinal_temperature,
    )
    shell_loss, shell_stats = virtual_sibling_shell_loss(
        unknown_features_by_parent,
        child_features_by_parent,
    )
    total = (
        float(lambda_id) * id_loss
        + float(lambda_tree) * tree_loss
        + float(lambda_shell) * shell_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        **id_stats,
        **tree_stats,
        **shell_stats,
    }


@torch.no_grad()
def predict_tree_complement_terminals(
    image_features: torch.Tensor,
    hierarchy,
    positive_edge_features: dict[tuple[str, str], torch.Tensor],
    terminal_specs: list[MetricTerminalSpec],
    unknown_features_by_parent: dict[str, torch.Tensor],
    unknown_threshold: float = 0.0,
    **score_kwargs,
) -> dict:
    scores = tree_complement_terminal_scores(
        image_features,
        hierarchy,
        positive_edge_features,
        terminal_specs,
        unknown_features_by_parent,
        **score_kwargs,
    )
    leaf_indices = [
        index
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is None
    ]
    unknown_indices = [
        index
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is not None
    ]
    if not leaf_indices or not unknown_indices:
        raise ValueError(
            "Tree-complement prediction needs leaves and unknown terminals"
        )
    score_matrix = scores["score_matrix"]
    leaf_scores, leaf_local = score_matrix[:, leaf_indices].max(dim=1)
    unknown_scores, unknown_local = score_matrix[
        :, unknown_indices
    ].max(dim=1)
    leaf_winners = torch.tensor(
        leaf_indices,
        dtype=torch.long,
        device=score_matrix.device,
    )[leaf_local]
    unknown_winners = torch.tensor(
        unknown_indices,
        dtype=torch.long,
        device=score_matrix.device,
    )[unknown_local]
    unknown_gap = unknown_scores - leaf_scores
    choose_unknown = unknown_gap >= float(unknown_threshold)
    winner_indices = torch.where(
        choose_unknown,
        unknown_winners,
        leaf_winners,
    ).cpu().tolist()
    nodes = [
        scores["candidate_nodes"][index] for index in winner_indices
    ]
    kinds = [
        scores["candidate_kinds"][index] for index in winner_indices
    ]
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    preds = torch.tensor(
        [node_to_index[node] for node in nodes],
        dtype=torch.long,
    )
    kind_counts = Counter(kinds)
    return {
        "preds": preds,
        "scores": scores,
        "diagnostics": {
            "candidate_type_counts": dict(kind_counts),
            "unknown_selection_rate": (
                kind_counts.get("unknown", 0) / max(1, len(nodes))
            ),
            "unknown_threshold": float(unknown_threshold),
            "unknown_gap": unknown_gap.detach().cpu(),
            "stop_node_counts": dict(Counter(nodes).most_common()),
        },
    }


def calibrate_unknown_gap_threshold(
    id_unknown_gaps: torch.Tensor,
    *,
    id_acceptance: float = 0.95,
) -> float:
    """Set an unknown-gap threshold using ID validation only."""
    if id_unknown_gaps.ndim != 1 or int(id_unknown_gaps.numel()) == 0:
        raise ValueError("ID unknown gaps must be a non-empty vector")
    if not 0.0 < float(id_acceptance) < 1.0:
        raise ValueError("id_acceptance must lie strictly between 0 and 1")
    return float(torch.quantile(
        id_unknown_gaps.float(),
        float(id_acceptance),
    ))
