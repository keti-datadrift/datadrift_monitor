from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metric_terminal import MetricTerminalSpec
from .prompt_text import build_parent_text
from .tree_virtual_unknown import tree_complement_terminal_scores


class ParentSpecificVirtualSiblingPromptLearner(nn.Module):
    """K virtual unseen-child prompts with shared and parent-local offsets."""

    def __init__(
        self,
        positive_learner,
        parents: list[str],
        *,
        num_unknown_prompts: int = 4,
        shared_init_noise: float = 1e-2,
        parent_init_noise: float = 1e-3,
    ):
        super().__init__()
        if not parents:
            raise ValueError("At least one unknown parent is required")
        if len(parents) != len(set(parents)):
            raise ValueError("Unknown parents must be unique")
        self.positive_learner = positive_learner
        self.parents = tuple(parents)
        self.parent_to_index = {
            parent: index for index, parent in enumerate(self.parents)
        }
        self.num_unknown_prompts = max(1, int(num_unknown_prompts))
        with torch.no_grad():
            context = positive_learner._context_for_parents(
                [self.parents[0]]
            )
        if context.ndim != 3 or int(context.shape[1]) == 0:
            raise ValueError(
                "LOCO virtual siblings require positive context tokens"
            )
        shape = (
            self.num_unknown_prompts,
            int(context.shape[1]),
            int(context.shape[2]),
        )
        self.shared_offsets = nn.Parameter(torch.empty(
            shape,
            device=context.device,
            dtype=context.dtype,
        ))
        self.parent_offsets = nn.Parameter(torch.empty(
            len(self.parents),
            *shape,
            device=context.device,
            dtype=context.dtype,
        ))
        nn.init.uniform_(
            self.shared_offsets,
            -float(shared_init_noise),
            float(shared_init_noise),
        )
        nn.init.uniform_(
            self.parent_offsets,
            -float(parent_init_noise),
            float(parent_init_noise),
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.shared_offsets, self.parent_offsets]

    def prompt_state(self) -> dict[str, torch.Tensor]:
        return {
            "shared_offsets": self.shared_offsets.detach().cpu().clone(),
            "parent_offsets": self.parent_offsets.detach().cpu().clone(),
            "parents": list(self.parents),
            "num_unknown_prompts": self.num_unknown_prompts,
        }

    def load_prompt_state(self, state: dict) -> None:
        if tuple(state["parents"]) != self.parents:
            raise ValueError("Checkpoint unknown-parent ordering differs")
        if int(state["num_unknown_prompts"]) != self.num_unknown_prompts:
            raise ValueError("Checkpoint unknown-prompt count differs")
        with torch.no_grad():
            self.shared_offsets.copy_(state["shared_offsets"].to(
                self.shared_offsets.device,
                dtype=self.shared_offsets.dtype,
            ))
            self.parent_offsets.copy_(state["parent_offsets"].to(
                self.parent_offsets.device,
                dtype=self.parent_offsets.dtype,
            ))

    def parent_offset_regularizer(self) -> torch.Tensor:
        return self.parent_offsets.float().square().mean()

    def encode_parents(self, parents: list[str]) -> torch.Tensor:
        if not parents:
            return torch.empty(
                0,
                self.num_unknown_prompts,
                self.positive_learner.projection_dim,
                device=self.shared_offsets.device,
            )
        missing = [
            parent for parent in parents
            if parent not in self.parent_to_index
        ]
        if missing:
            raise KeyError(f"Unknown LOCO parents: {missing[:3]}")
        indices = torch.tensor(
            [self.parent_to_index[parent] for parent in parents],
            dtype=torch.long,
            device=self.parent_offsets.device,
        )
        with torch.no_grad():
            base_context = self.positive_learner._context_for_parents(
                parents
            ).detach()
        contexts = (
            base_context.unsqueeze(1)
            + self.shared_offsets.unsqueeze(0)
            + self.parent_offsets.index_select(0, indices)
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
    ancestors = tuple(
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.node_ancestors.get(node, [])
    )
    return ancestors + (node,)


def _is_descendant_or_self(hierarchy, node: str, ancestor: str) -> bool:
    return ancestor in _node_path(hierarchy, node)


def prune_terminal_specs_for_hidden_child(
    hierarchy,
    terminal_specs: list[MetricTerminalSpec],
    parent: str,
    hidden_child: str,
) -> list[MetricTerminalSpec]:
    """Remove terminals inside a child subtree while retaining unknown@parent."""
    if hidden_child not in hierarchy.parent2children.get(parent, []):
        raise ValueError(
            f"{hidden_child!r} is not a child of {parent!r}"
        )
    pruned = []
    for spec in terminal_specs:
        terminal_node = (
            spec.node
            if spec.unknown_parent is None
            else spec.unknown_parent
        )
        if _is_descendant_or_self(
            hierarchy,
            terminal_node,
            hidden_child,
        ):
            continue
        pruned.append(spec)
    if not any(
        spec.unknown_parent == parent for spec in pruned
    ):
        raise RuntimeError("LOCO pruning removed its target unknown terminal")
    return pruned


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


def _terminal_augmented_path(
    hierarchy,
    spec: MetricTerminalSpec,
) -> tuple[str, ...]:
    if spec.unknown_parent is None:
        return _node_path(hierarchy, spec.node)
    return _node_path(hierarchy, spec.unknown_parent) + (
        f"__unknown__:{spec.unknown_parent}",
    )


def loco_pseudo_unknown_loss(
    score_output: dict,
    hierarchy,
    terminal_specs: list[MetricTerminalSpec],
    target_parent: str,
    *,
    local_margin: float = 0.05,
    distance_margin: float = 0.02,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Make hidden-child images terminate at unknown@parent globally."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    scores = score_output["score_matrix"]
    if int(scores.shape[1]) != len(terminal_specs):
        raise ValueError("Scores and terminal specs must have equal columns")
    target_candidates = [
        index
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent == target_parent
    ]
    if len(target_candidates) != 1:
        raise ValueError(
            f"Expected one unknown terminal for {target_parent!r}"
        )
    target_index = target_candidates[0]
    competitor_indices = [
        index for index in range(len(terminal_specs))
        if index != target_index
    ]
    if not competitor_indices:
        raise ValueError("LOCO loss requires at least one competitor")

    target_path = _terminal_augmented_path(
        hierarchy,
        terminal_specs[target_index],
    )
    distances = torch.tensor(
        [
            _tree_distance(
                target_path,
                _terminal_augmented_path(
                    hierarchy,
                    terminal_specs[index],
                ),
            )
            for index in competitor_indices
        ],
        dtype=scores.dtype,
        device=scores.device,
    )
    nearest = distances.min()
    margins = (
        float(local_margin)
        + float(distance_margin) * (distances - nearest)
    )
    target_scores = scores[:, target_index]
    competitor_scores = scores[:, competitor_indices]
    violations = (
        competitor_scores
        - target_scores.unsqueeze(1)
        + margins.unsqueeze(0)
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
        "loco_pseudo_loss": float(loss.detach().cpu()),
        "loco_target_win_rate": float(
            (
                target_scores
                > competitor_scores.max(dim=1).values
            ).float().mean().detach().cpu()
        ),
        "loco_margin_violation_rate": float(
            (violations > 0.0).float().mean().detach().cpu()
        ),
        "loco_target_score": float(
            target_scores.mean().detach().cpu()
        ),
        "loco_best_competitor_score": float(
            competitor_scores.max(dim=1).values.mean().detach().cpu()
        ),
    }


def balanced_slot_assignment_loss(
    image_features: torch.Tensor,
    unknown_prototypes: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Specialize each image to one slot while using all K slots."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if image_features.ndim != 2 or unknown_prototypes.ndim != 2:
        raise ValueError("Expected image [B,D] and prototype [K,D] tensors")
    if int(image_features.shape[0]) == 0:
        raise ValueError("Slot assignment needs at least one image")
    count = int(unknown_prototypes.shape[0])
    if count == 1:
        zero = unknown_prototypes.sum() * 0.0
        return zero, {
            "slot_loss": 0.0,
            "slot_confidence_loss": 0.0,
            "slot_balance_loss": 0.0,
            "slot_effective_count": 1.0,
            "slot_max_usage": 1.0,
        }
    images = F.normalize(image_features.float(), dim=-1)
    prototypes = F.normalize(unknown_prototypes.float(), dim=-1)
    similarities = images @ prototypes.t()
    assignments = F.softmax(
        similarities / float(temperature),
        dim=1,
    )
    eps = torch.finfo(assignments.dtype).eps
    log_count = math.log(count)
    sample_entropy = -(
        assignments * assignments.clamp_min(eps).log()
    ).sum(dim=1).mean()
    usage = assignments.mean(dim=0)
    usage_entropy = -(
        usage * usage.clamp_min(eps).log()
    ).sum()
    confidence_loss = sample_entropy / log_count
    balance_loss = (log_count - usage_entropy) / log_count
    loss = confidence_loss + balance_loss
    effective_count = usage_entropy.exp()
    return loss, {
        "slot_loss": float(loss.detach().cpu()),
        "slot_confidence_loss": float(
            confidence_loss.detach().cpu()
        ),
        "slot_balance_loss": float(balance_loss.detach().cpu()),
        "slot_effective_count": float(
            effective_count.detach().cpu()
        ),
        "slot_max_usage": float(usage.max().detach().cpu()),
    }


@torch.no_grad()
def leave_one_child_out_global_recall(
    image_features: torch.Tensor,
    retained_leaf_targets: list[str],
    hierarchy,
    positive_edge_features: dict[tuple[str, str], torch.Tensor],
    terminal_specs: list[MetricTerminalSpec],
    unknown_features_by_parent: dict[str, torch.Tensor],
    **score_kwargs,
) -> dict:
    """Macro and sample recall over every non-root hidden-child fold."""
    if int(image_features.shape[0]) != len(retained_leaf_targets):
        raise ValueError("Feature rows and leaf targets must match")
    fold_rows = []
    parent_correct = defaultdict(int)
    parent_total = defaultdict(int)
    for parent in sorted(unknown_features_by_parent):
        for child in hierarchy.parent2children[parent]:
            sample_indices = [
                index
                for index, leaf in enumerate(retained_leaf_targets)
                if _is_descendant_or_self(hierarchy, leaf, child)
            ]
            if not sample_indices:
                continue
            pruned_specs = prune_terminal_specs_for_hidden_child(
                hierarchy,
                terminal_specs,
                parent,
                child,
            )
            batch = image_features.index_select(
                0,
                torch.tensor(
                    sample_indices,
                    dtype=torch.long,
                    device=image_features.device,
                ),
            )
            output = tree_complement_terminal_scores(
                batch,
                hierarchy,
                positive_edge_features,
                pruned_specs,
                unknown_features_by_parent,
                excluded_children_by_parent={parent: {child}},
                **score_kwargs,
            )
            target_index = next(
                index
                for index, spec in enumerate(pruned_specs)
                if spec.unknown_parent == parent
            )
            winners = output["score_matrix"].argmax(dim=1)
            correct = int((winners == target_index).sum().item())
            total = len(sample_indices)
            fold_rows.append({
                "parent": parent,
                "hidden_child": child,
                "samples": total,
                "correct": correct,
                "recall": correct / total,
            })
            parent_correct[parent] += correct
            parent_total[parent] += total
    if not fold_rows:
        raise RuntimeError("No LOCO folds were available")
    parent_recalls = {
        parent: parent_correct[parent] / parent_total[parent]
        for parent in parent_total
    }
    return {
        "fold_macro_recall": sum(
            row["recall"] for row in fold_rows
        ) / len(fold_rows),
        "parent_macro_recall": sum(parent_recalls.values()) / len(
            parent_recalls
        ),
        "sample_recall": sum(
            row["correct"] for row in fold_rows
        ) / sum(row["samples"] for row in fold_rows),
        "fold_count": len(fold_rows),
        "parent_count": len(parent_recalls),
        "parent_recalls": parent_recalls,
        "folds": fold_rows,
    }
