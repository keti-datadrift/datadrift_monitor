from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from sacil.hierarchy.tree import HierarchyTree

from .prototype_bank import PrototypeBank


class HierarchicalAnchorBank:
    """Post-hoc leaf and internal anchors associated with a taxonomy."""

    def __init__(
        self,
        leaf_class_ids: tuple[int, ...],
        leaf_anchors: Tensor,
        internal_node_ids: tuple[str, ...],
        internal_anchors: Tensor,
        root_id: str,
    ) -> None:
        self.leaf_class_ids = tuple(int(value) for value in leaf_class_ids)
        self.leaf_anchors = leaf_anchors.detach().cpu().float().clone()
        self.internal_node_ids = tuple(str(value) for value in internal_node_ids)
        self.internal_anchors = (
            internal_anchors.detach().cpu().float().clone()
        )
        self.root_id = str(root_id)
        self._validate()

    def _validate(self) -> None:
        if self.leaf_anchors.ndim != 2:
            raise ValueError("leaf anchors must be a matrix")
        if self.internal_anchors.ndim != 2:
            raise ValueError("internal anchors must be a matrix")
        if len(self.leaf_class_ids) != self.leaf_anchors.shape[0]:
            raise ValueError("leaf anchor count mismatch")
        if len(self.internal_node_ids) != self.internal_anchors.shape[0]:
            raise ValueError("internal anchor count mismatch")
        if (
            self.internal_anchors.shape[0] > 0
            and self.internal_anchors.shape[1] != self.leaf_anchors.shape[1]
        ):
            raise ValueError("anchor dimensions do not match")
        for anchors in (self.leaf_anchors, self.internal_anchors):
            if anchors.shape[0] == 0:
                continue
            norms = anchors.norm(dim=1)
            if not torch.allclose(
                norms, torch.ones_like(norms), atol=1e-5
            ):
                raise ValueError("anchors must be normalized")

    @property
    def feature_dim(self) -> int:
        return self.leaf_anchors.shape[1]

    @classmethod
    def from_tree(
        cls, prototypes: PrototypeBank, tree: HierarchyTree
    ) -> "HierarchicalAnchorBank":
        if prototypes.class_ids != tree.class_order:
            raise ValueError("prototype order and tree class order must match")
        internal_ids = tree.internal_node_ids(include_root=True)
        internal_anchors = []
        for node_id in internal_ids:
            descendant_prototypes = torch.stack(
                [
                    prototypes.for_class(class_id)
                    for class_id in tree.descendants(node_id)
                ],
                dim=0,
            )
            internal_anchors.append(
                F.normalize(
                    descendant_prototypes.mean(dim=0, keepdim=True), dim=1
                )[0]
            )
        if internal_anchors:
            internal_tensor = torch.stack(internal_anchors, dim=0)
        else:
            internal_tensor = torch.empty(
                0, prototypes.prototypes.shape[1]
            )
        return cls(
            leaf_class_ids=prototypes.class_ids,
            leaf_anchors=prototypes.prototypes,
            internal_node_ids=internal_ids,
            internal_anchors=internal_tensor,
            root_id=tree.root_id,
        )

    def internal_without_root(self) -> tuple[tuple[str, ...], Tensor]:
        positions = [
            position
            for position, node_id in enumerate(self.internal_node_ids)
            if node_id != self.root_id
        ]
        node_ids = tuple(self.internal_node_ids[position] for position in positions)
        if not positions:
            anchors = torch.empty(0, self.feature_dim)
        else:
            anchors = self.internal_anchors[positions]
        return node_ids, anchors

    def state_dict(self) -> dict:
        return {
            "leaf_class_ids": list(self.leaf_class_ids),
            "leaf_anchors": self.leaf_anchors.clone(),
            "internal_node_ids": list(self.internal_node_ids),
            "internal_anchors": self.internal_anchors.clone(),
            "root_id": self.root_id,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "HierarchicalAnchorBank":
        return cls(
            tuple(state["leaf_class_ids"]),
            state["leaf_anchors"],
            tuple(state["internal_node_ids"]),
            state["internal_anchors"],
            state["root_id"],
        )

