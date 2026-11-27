from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from sacil.anchors.hierarchical_anchor_bank import HierarchicalAnchorBank
from sacil.hierarchy.tree import HierarchyTree


@dataclass
class ConflictWeights:
    leaf_class_ids: tuple[int, ...]
    leaf_weights: Tensor
    leaf_activations: Tensor
    internal_node_ids: tuple[str, ...]
    internal_weights: Tensor
    internal_activations: Tensor
    max_leaf_scores: Tensor
    neighbors_per_new_class: int

    def state_dict(self) -> dict:
        return {
            "leaf_class_ids": list(self.leaf_class_ids),
            "leaf_weights": self.leaf_weights.detach().cpu(),
            "leaf_activations": self.leaf_activations.detach().cpu(),
            "internal_node_ids": list(self.internal_node_ids),
            "internal_weights": self.internal_weights.detach().cpu(),
            "internal_activations": self.internal_activations.detach().cpu(),
            "max_leaf_scores": self.max_leaf_scores.detach().cpu(),
            "neighbors_per_new_class": self.neighbors_per_new_class,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "ConflictWeights":
        return cls(
            leaf_class_ids=tuple(state["leaf_class_ids"]),
            leaf_weights=state["leaf_weights"],
            leaf_activations=state["leaf_activations"],
            internal_node_ids=tuple(state["internal_node_ids"]),
            internal_weights=state["internal_weights"],
            internal_activations=state["internal_activations"],
            max_leaf_scores=state["max_leaf_scores"],
            neighbors_per_new_class=int(state["neighbors_per_new_class"]),
        )


def _validate_hyperparameters(
    max_neighbors: int,
    old_class_ratio: float,
    temperature: float,
    min_preservation_weight: float,
    ancestor_decay: float,
) -> None:
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
    if not 0 < old_class_ratio <= 1:
        raise ValueError("old_class_ratio must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= min_preservation_weight <= 1:
        raise ValueError("min preservation weight must be in [0, 1]")
    if not 0 <= ancestor_decay <= 1:
        raise ValueError("ancestor_decay must be in [0, 1]")


def compute_conflict_weights(
    new_prototypes: Tensor,
    anchor_bank: HierarchicalAnchorBank,
    tree: HierarchyTree,
    *,
    max_neighbors: int = 5,
    old_class_ratio: float = 0.1,
    temperature: float = 0.05,
    min_preservation_weight: float = 0.1,
    ancestor_decay: float = 0.5,
) -> ConflictWeights:
    _validate_hyperparameters(
        max_neighbors,
        old_class_ratio,
        temperature,
        min_preservation_weight,
        ancestor_decay,
    )
    if new_prototypes.ndim != 2 or new_prototypes.shape[0] == 0:
        raise ValueError("at least one new-class prototype is required")
    if new_prototypes.shape[1] != anchor_bank.feature_dim:
        raise ValueError("prototype and anchor dimensions do not match")
    if tree.class_order != anchor_bank.leaf_class_ids:
        raise ValueError("tree and anchor leaf orders do not match")

    new_prototypes = F.normalize(new_prototypes.float(), dim=1)
    leaf_anchors = F.normalize(anchor_bank.leaf_anchors.float(), dim=1)
    scores = new_prototypes @ leaf_anchors.t()
    old_class_count = leaf_anchors.shape[0]
    neighbors = min(
        int(max_neighbors),
        max(1, int(math.ceil(old_class_ratio * old_class_count))),
    )
    top_values, top_indices = torch.topk(scores, k=neighbors, dim=1)
    thresholds = top_values[:, -1]
    activation_by_new = torch.zeros_like(scores)
    selected_activations = torch.sigmoid(
        (top_values - thresholds.unsqueeze(1)) / float(temperature)
    )
    activation_by_new.scatter_(1, top_indices, selected_activations)
    leaf_activations = activation_by_new.max(dim=0).values
    leaf_weights = 1.0 - (
        1.0 - float(min_preservation_weight)
    ) * leaf_activations

    leaf_activation_by_class = {
        class_id: leaf_activations[position]
        for position, class_id in enumerate(anchor_bank.leaf_class_ids)
    }
    internal_ids, _ = anchor_bank.internal_without_root()
    internal_activations = []
    for node_id in internal_ids:
        candidates = []
        for class_id in tree.descendants(node_id):
            distance = tree.distance_from_ancestor(
                node_id, tree.leaf_node_id(class_id)
            )
            candidates.append(
                (float(ancestor_decay) ** distance)
                * leaf_activation_by_class[class_id]
            )
        internal_activations.append(torch.stack(candidates).max())
    if internal_activations:
        internal_activation_tensor = torch.stack(internal_activations)
    else:
        internal_activation_tensor = torch.empty(
            0, dtype=leaf_activations.dtype
        )
    internal_weights = 1.0 - (
        1.0 - float(min_preservation_weight)
    ) * internal_activation_tensor

    return ConflictWeights(
        leaf_class_ids=anchor_bank.leaf_class_ids,
        leaf_weights=leaf_weights.detach().cpu(),
        leaf_activations=leaf_activations.detach().cpu(),
        internal_node_ids=internal_ids,
        internal_weights=internal_weights.detach().cpu(),
        internal_activations=internal_activation_tensor.detach().cpu(),
        max_leaf_scores=scores.max(dim=0).values.detach().cpu(),
        neighbors_per_new_class=neighbors,
    )


def global_preservation_weights(
    anchor_bank: HierarchicalAnchorBank,
) -> ConflictWeights:
    internal_ids, _ = anchor_bank.internal_without_root()
    return ConflictWeights(
        leaf_class_ids=anchor_bank.leaf_class_ids,
        leaf_weights=torch.ones(len(anchor_bank.leaf_class_ids)),
        leaf_activations=torch.zeros(len(anchor_bank.leaf_class_ids)),
        internal_node_ids=internal_ids,
        internal_weights=torch.ones(len(internal_ids)),
        internal_activations=torch.zeros(len(internal_ids)),
        max_leaf_scores=torch.zeros(len(anchor_bank.leaf_class_ids)),
        neighbors_per_new_class=0,
    )
