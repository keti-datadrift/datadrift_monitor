from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from .metric_terminal import MetricTerminalSpec, metric_terminal_scores


def _prototype_diversity(
    prototypes: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    prototypes = F.normalize(prototypes.float(), dim=-1)
    if int(prototypes.shape[0]) <= 1:
        return prototypes.new_zeros(())
    similarities = prototypes @ prototypes.t()
    off_diagonal = ~torch.eye(
        prototypes.shape[0],
        dtype=torch.bool,
        device=prototypes.device,
    )
    return F.relu(
        similarities[off_diagonal] - float(margin)
    ).square().mean()


def global_metric_terminal_negprompt_loss(
    id_features: torch.Tensor,
    virtual_features: torch.Tensor,
    virtual_parents: list[str],
    positive_edge_features: dict[tuple[str, str], torch.Tensor],
    terminal_specs: list[MetricTerminalSpec],
    unknown_features_by_parent: dict[str, torch.Tensor],
    *,
    loss_temperature: float,
    terminal_weight: float,
    bottleneck_temperature: float,
    unknown_temperature: float,
    lambda_virtual: float,
    lambda_id_teacher: float,
    lambda_coverage: float,
    lambda_diversity: float,
    diversity_margin: float,
) -> tuple[torch.Tensor, dict]:
    """Fit unknown prompts in the same global terminal space used at inference.

    Positive edge features and image features are treated as frozen. Virtual
    sibling-mix features target their exact parent-unknown terminal. For ID
    images, a positive-only teacher distribution is preserved while the full
    student distribution additionally contains every parent-unknown terminal.
    """
    if float(loss_temperature) <= 0.0:
        raise ValueError("loss_temperature must be positive")
    if int(virtual_features.shape[0]) != len(virtual_parents):
        raise ValueError("virtual_features and virtual_parents must match")

    id_scores = metric_terminal_scores(
        id_features.detach(),
        positive_edge_features,
        terminal_specs,
        unknown_features_by_parent=unknown_features_by_parent,
        terminal_weight=terminal_weight,
        bottleneck_temperature=bottleneck_temperature,
        unknown_temperature=unknown_temperature,
    )["score_matrix"]
    virtual_scores = metric_terminal_scores(
        virtual_features.detach(),
        positive_edge_features,
        terminal_specs,
        unknown_features_by_parent=unknown_features_by_parent,
        terminal_weight=terminal_weight,
        bottleneck_temperature=bottleneck_temperature,
        unknown_temperature=unknown_temperature,
    )["score_matrix"]

    known_indices = [
        index for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is None
    ]
    unknown_index_by_parent = {
        spec.unknown_parent: index
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is not None
    }
    missing = sorted(set(virtual_parents) - set(unknown_index_by_parent))
    if missing:
        raise KeyError(f"Missing virtual target unknown terminals: {missing}")

    scale = 1.0 / float(loss_temperature)
    known_index_tensor = torch.tensor(
        known_indices, dtype=torch.long, device=id_scores.device
    )
    teacher_logits = id_scores.index_select(1, known_index_tensor) * scale
    teacher_probabilities = F.softmax(teacher_logits, dim=1).detach()
    student_log_probabilities = F.log_softmax(id_scores * scale, dim=1)
    id_teacher_loss = -(
        teacher_probabilities
        * student_log_probabilities.index_select(1, known_index_tensor)
    ).sum(dim=1).mean()
    teacher_entropy = -(
        teacher_probabilities
        * teacher_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1).mean()

    virtual_targets = torch.tensor(
        [unknown_index_by_parent[parent] for parent in virtual_parents],
        dtype=torch.long,
        device=virtual_scores.device,
    )
    virtual_loss = F.cross_entropy(virtual_scores * scale, virtual_targets)

    parent_to_virtual_indices: dict[str, list[int]] = defaultdict(list)
    for index, parent in enumerate(virtual_parents):
        parent_to_virtual_indices[parent].append(index)
    coverage_terms = []
    virtual_coverage_values = []
    prototype_coverage_values = []
    for parent, indices in parent_to_virtual_indices.items():
        points = F.normalize(
            virtual_features[indices].detach().float(), dim=-1
        )
        prototypes = F.normalize(
            unknown_features_by_parent[parent].float(), dim=-1
        )
        similarities = points @ prototypes.t()
        virtual_coverage = similarities.max(dim=1).values.mean()
        prototype_coverage = similarities.max(dim=0).values.mean()
        coverage_terms.append(
            1.0 - 0.5 * (virtual_coverage + prototype_coverage)
        )
        virtual_coverage_values.append(virtual_coverage)
        prototype_coverage_values.append(prototype_coverage)
    coverage_loss = torch.stack(coverage_terms).mean()

    diversity_loss = torch.stack([
        _prototype_diversity(bank, diversity_margin)
        for bank in unknown_features_by_parent.values()
    ]).mean()
    open_objective = (
        float(lambda_virtual) * virtual_loss
        + float(lambda_coverage) * coverage_loss
        + float(lambda_diversity) * diversity_loss
    )
    total = open_objective + float(lambda_id_teacher) * id_teacher_loss

    virtual_predictions = virtual_scores.argmax(dim=1)
    id_predictions = id_scores.argmax(dim=1)
    unknown_mask = torch.tensor(
        [spec.unknown_parent is not None for spec in terminal_specs],
        dtype=torch.bool,
        device=id_scores.device,
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "open_objective": float(open_objective.detach().cpu()),
        "virtual_loss": float(virtual_loss.detach().cpu()),
        "virtual_exact_parent_recall": float(
            (virtual_predictions == virtual_targets).float().mean().detach().cpu()
        ),
        "id_teacher_loss": float(id_teacher_loss.detach().cpu()),
        "id_teacher_excess": float(
            (id_teacher_loss - teacher_entropy).detach().cpu()
        ),
        "id_unknown_selection_rate": float(
            unknown_mask[id_predictions].float().mean().detach().cpu()
        ),
        "coverage_loss": float(coverage_loss.detach().cpu()),
        "virtual_coverage": float(
            torch.stack(virtual_coverage_values).mean().detach().cpu()
        ),
        "prototype_coverage": float(
            torch.stack(prototype_coverage_values).mean().detach().cpu()
        ),
        "diversity_loss": float(diversity_loss.detach().cpu()),
    }


def threshold_terminal_winner_indices(
    score_matrix: torch.Tensor,
    terminal_specs: list[MetricTerminalSpec],
    *,
    unknown_threshold: float,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose a global known leaf or parent-unknown with a calibrated margin.

    The best unknown is selected only when
    ``best_unknown_score - best_known_score >= unknown_threshold``.
    """
    if score_matrix.ndim != 2:
        raise ValueError("score_matrix must have shape [samples, terminals]")
    if int(score_matrix.shape[1]) != len(terminal_specs):
        raise ValueError("score_matrix and terminal_specs must match")

    device = score_matrix.device
    allowed = torch.ones_like(score_matrix, dtype=torch.bool)
    if candidate_mask is not None:
        candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
        if candidate_mask.ndim == 1:
            if int(candidate_mask.numel()) != int(score_matrix.shape[1]):
                raise ValueError("candidate_mask length must match terminal count")
            allowed &= candidate_mask.unsqueeze(0)
        elif candidate_mask.shape == score_matrix.shape:
            allowed &= candidate_mask
        else:
            raise ValueError(
                "candidate_mask must have shape [terminals] or [samples, terminals]"
            )

    known_mask = torch.tensor(
        [spec.unknown_parent is None for spec in terminal_specs],
        dtype=torch.bool,
        device=device,
    )
    unknown_mask = ~known_mask
    known_scores = score_matrix.masked_fill(
        ~(allowed & known_mask.unsqueeze(0)), -torch.inf
    )
    unknown_scores = score_matrix.masked_fill(
        ~(allowed & unknown_mask.unsqueeze(0)), -torch.inf
    )
    best_known_scores, best_known_indices = known_scores.max(dim=1)
    best_unknown_scores, best_unknown_indices = unknown_scores.max(dim=1)
    if torch.isneginf(best_known_scores).any():
        raise ValueError("Every sample must retain at least one known terminal")
    choose_unknown = (
        ~torch.isneginf(best_unknown_scores)
        & (
            best_unknown_scores - best_known_scores
            >= float(unknown_threshold)
        )
    )
    return torch.where(
        choose_unknown, best_unknown_indices, best_known_indices
    )


def _is_descendant_or_self(hierarchy, node: str, ancestor: str) -> bool:
    if node == ancestor:
        return True
    ancestor_indices = hierarchy.node_ancestors.get(node, [])
    return any(
        hierarchy.id_node_list[int(index)] == ancestor
        for index in ancestor_indices
    )


def leave_one_child_out_terminal_recall(
    score_matrix: torch.Tensor,
    terminal_specs: list[MetricTerminalSpec],
    hierarchy,
    retained_leaf_targets: list[str],
    *,
    unknown_threshold: float,
) -> dict:
    """Evaluate decoder calibration without treating ID children as train targets.

    For each non-root parent/child edge, the child's terminal subtree is hidden
    only at validation time. Its validation images should terminate at the
    parent's unknown terminal. Every hidden-child fold receives equal weight.
    """
    if int(score_matrix.shape[0]) != len(retained_leaf_targets):
        raise ValueError("score_matrix rows and retained_leaf_targets must match")
    unknown_index_by_parent = {
        spec.unknown_parent: index
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is not None
    }
    fold_rows = []
    parent_correct: dict[str, int] = defaultdict(int)
    parent_total: dict[str, int] = defaultdict(int)
    for parent in sorted(
        unknown_index_by_parent,
        key=lambda node: (
            len(hierarchy.node_ancestors.get(node, [])),
            node,
        ),
    ):
        for child in hierarchy.parent2children[parent]:
            sample_indices = [
                index
                for index, leaf in enumerate(retained_leaf_targets)
                if _is_descendant_or_self(hierarchy, leaf, child)
            ]
            if not sample_indices:
                continue
            candidate_mask = torch.tensor([
                not (
                    _is_descendant_or_self(hierarchy, spec.node, child)
                    if spec.unknown_parent is None
                    else _is_descendant_or_self(
                        hierarchy, spec.unknown_parent, child
                    )
                )
                for spec in terminal_specs
            ], dtype=torch.bool, device=score_matrix.device)
            fold_scores = score_matrix[sample_indices]
            winners = threshold_terminal_winner_indices(
                fold_scores,
                terminal_specs,
                unknown_threshold=unknown_threshold,
                candidate_mask=candidate_mask,
            )
            expected = int(unknown_index_by_parent[parent])
            correct = int((winners == expected).sum().item())
            total = len(sample_indices)
            recall = correct / total
            fold_rows.append({
                "parent": parent,
                "hidden_child": child,
                "samples": total,
                "correct": correct,
                "recall": recall,
            })
            parent_correct[parent] += correct
            parent_total[parent] += total

    if not fold_rows:
        raise RuntimeError("No leave-one-child-out validation folds were built")
    parent_recalls = {
        parent: parent_correct[parent] / parent_total[parent]
        for parent in parent_total
    }
    return {
        "fold_macro_recall": sum(row["recall"] for row in fold_rows) / len(fold_rows),
        "parent_macro_recall": sum(parent_recalls.values()) / len(parent_recalls),
        "sample_recall": (
            sum(row["correct"] for row in fold_rows)
            / sum(row["samples"] for row in fold_rows)
        ),
        "fold_count": len(fold_rows),
        "parent_count": len(parent_recalls),
        "parent_recalls": parent_recalls,
        "folds": fold_rows,
    }
