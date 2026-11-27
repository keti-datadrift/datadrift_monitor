from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sacil.anchors.affinity import anchor_affinity
from sacil.anchors.hierarchical_anchor_bank import HierarchicalAnchorBank
from sacil.hierarchy.tree import HierarchyTree, TreeNode


RELAXATION_MODES = frozenset({"none", "global_margin", "local_margin"})
PATH_SCOPES = frozenset({"parent_only", "all_ancestors"})
PATH_NORMALIZATIONS = frozenset({"relation_mean", "sample_mean"})


@dataclass(frozen=True)
class ConflictAssignment:
    new_class_id: int
    nearest_leaf_class_id: int
    insertion_target_node_id: str
    conflict_root_node_id: str | None
    angular_distance: float
    node_radius: float
    overlap_score: float
    relaxed_old_class_ids: tuple[int, ...]


@dataclass
class ConflictPlan:
    assignments: tuple[ConflictAssignment, ...]
    relaxed_margins: dict[tuple[int, str], float]
    old_class_budget: int
    relaxed_old_class_ids: tuple[int, ...]
    node_radii: dict[str, float]
    relaxation_mode: str

    def state_dict(self) -> dict:
        return {
            "assignments": [asdict(item) for item in self.assignments],
            "relaxed_margins": [
                {
                    "class_id": int(class_id),
                    "node_id": str(node_id),
                    "margin": float(margin),
                }
                for (class_id, node_id), margin in sorted(
                    self.relaxed_margins.items()
                )
            ],
            "old_class_budget": int(self.old_class_budget),
            "relaxed_old_class_ids": list(self.relaxed_old_class_ids),
            "node_radii": {
                str(node_id): float(radius)
                for node_id, radius in sorted(self.node_radii.items())
            },
            "relaxation_mode": self.relaxation_mode,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "ConflictPlan":
        assignments = tuple(
            ConflictAssignment(
                new_class_id=int(item["new_class_id"]),
                nearest_leaf_class_id=int(item["nearest_leaf_class_id"]),
                insertion_target_node_id=str(
                    item["insertion_target_node_id"]
                ),
                conflict_root_node_id=(
                    None
                    if item.get("conflict_root_node_id") is None
                    else str(item["conflict_root_node_id"])
                ),
                angular_distance=float(item["angular_distance"]),
                node_radius=float(item["node_radius"]),
                overlap_score=float(item["overlap_score"]),
                relaxed_old_class_ids=tuple(
                    int(value) for value in item["relaxed_old_class_ids"]
                ),
            )
            for item in state["assignments"]
        )
        margins = {
            (int(item["class_id"]), str(item["node_id"])): float(
                item["margin"]
            )
            for item in state.get("relaxed_margins", [])
        }
        return cls(
            assignments=assignments,
            relaxed_margins=margins,
            old_class_budget=int(state["old_class_budget"]),
            relaxed_old_class_ids=tuple(
                int(value) for value in state["relaxed_old_class_ids"]
            ),
            node_radii={
                str(key): float(value)
                for key, value in state["node_radii"].items()
            },
            relaxation_mode=str(state["relaxation_mode"]),
        )


def class_ancestor_path(
    tree: HierarchyTree,
    class_id: int,
    *,
    include_root: bool = False,
) -> tuple[str, ...]:
    """Return internal ancestors ordered from the leaf parent upward."""

    leaf_id = tree.leaf_node_id(int(class_id))
    if leaf_id not in tree.nodes:
        raise ValueError(f"class {class_id} is not present in the tree")
    path: list[str] = []
    current = leaf_id
    while True:
        parent = tree.parent(current)
        if parent is None:
            break
        if include_root or parent != tree.root_id:
            path.append(parent)
        current = parent
    return tuple(path)


def _internal_anchor_map(
    anchor_bank: HierarchicalAnchorBank,
) -> dict[str, Tensor]:
    return {
        node_id: anchor_bank.internal_anchors[position]
        for position, node_id in enumerate(anchor_bank.internal_node_ids)
    }


def compute_internal_node_radii(
    features: Tensor,
    original_targets: Tensor,
    anchor_bank: HierarchicalAnchorBank,
    tree: HierarchyTree,
    *,
    quantile: float = 0.9,
    minimum_radius: float = 0.0,
) -> dict[str, float]:
    """Estimate a robust angular support radius for every internal node."""

    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if features.shape[0] != original_targets.numel():
        raise ValueError("feature and target counts differ")
    if features.shape[1] != anchor_bank.feature_dim:
        raise ValueError("feature and anchor dimensions differ")
    if tree.class_order != anchor_bank.leaf_class_ids:
        raise ValueError("tree and anchor class orders differ")
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("radius quantile must be in (0, 1]")
    if not 0.0 <= float(minimum_radius) < math.pi:
        raise ValueError("minimum radius must be in [0, pi)")

    normalized = F.normalize(features.detach().float().cpu(), dim=1)
    targets = original_targets.detach().long().cpu()
    anchors = _internal_anchor_map(anchor_bank)
    radii: dict[str, float] = {}
    for node_id in tree.internal_node_ids(include_root=True):
        members = torch.tensor(tree.descendants(node_id), dtype=torch.long)
        mask = (targets.unsqueeze(1) == members.unsqueeze(0)).any(dim=1)
        if not bool(mask.any()):
            raise ValueError(f"node {node_id} has no supporting features")
        anchor = F.normalize(anchors[node_id].float(), dim=0)
        cosine = (normalized[mask] @ anchor).clamp(-1.0, 1.0)
        angles = torch.acos(cosine)
        radius = float(torch.quantile(angles, float(quantile)).item())
        radii[node_id] = max(float(minimum_radius), radius)
    return radii


def _margin_for_relation(
    tree: HierarchyTree,
    class_id: int,
    node_id: str,
    *,
    base_margin: float,
    ancestor_decay: float,
) -> float:
    distance = tree.distance_from_ancestor(
        node_id, tree.leaf_node_id(class_id)
    )
    return float(base_margin) * float(ancestor_decay) ** max(0, distance - 1)


def plan_conflict_relaxation(
    new_prototypes: Tensor,
    new_class_ids: Sequence[int],
    anchor_bank: HierarchicalAnchorBank,
    tree: HierarchyTree,
    node_radii: Mapping[str, float],
    *,
    relaxation_mode: str = "local_margin",
    path_scope: str = "all_ancestors",
    radius_slack: float = 0.05,
    minimum_overlap: float = 0.0,
    max_conflict_leaf_ratio: float = 0.1,
    relaxation_margin: float = 0.05,
    margin_ancestor_decay: float = 0.7,
    include_root: bool = False,
) -> ConflictPlan:
    """Build a globally bounded, radius-gated conflict-subtree plan.

    The incoming class first selects its nearest old leaf.  Only ancestors of
    that leaf are eligible conflict roots.  A root is active when the incoming
    prototype falls inside its robust angular radius plus ``radius_slack``.
    This implements the intended *nearest class -> ancestor anchors* routing
    without relaxing unrelated branches.
    """

    mode = str(relaxation_mode).lower().replace("-", "_")
    scope = str(path_scope).lower().replace("-", "_")
    if mode not in RELAXATION_MODES:
        raise ValueError(f"unknown relaxation mode: {relaxation_mode}")
    if scope not in PATH_SCOPES:
        raise ValueError(f"unknown path scope: {path_scope}")
    if new_prototypes.ndim != 2 or new_prototypes.shape[0] == 0:
        raise ValueError("at least one new prototype is required")
    if new_prototypes.shape[0] != len(new_class_ids):
        raise ValueError("new prototype and class counts differ")
    if new_prototypes.shape[1] != anchor_bank.feature_dim:
        raise ValueError("new prototype and anchor dimensions differ")
    if not 0.0 <= float(radius_slack) < math.pi:
        raise ValueError("radius slack must be in [0, pi)")
    if not 0.0 < float(max_conflict_leaf_ratio) <= 1.0:
        raise ValueError("max conflict leaf ratio must be in (0, 1]")
    if not 0.0 <= float(relaxation_margin) <= 2.0:
        raise ValueError("relaxation margin must be in [0, 2]")
    if not 0.0 <= float(margin_ancestor_decay) <= 1.0:
        raise ValueError("margin ancestor decay must be in [0, 1]")

    new_values = F.normalize(new_prototypes.detach().float().cpu(), dim=1)
    leaf_values = F.normalize(anchor_bank.leaf_anchors.float(), dim=1)
    internal_anchors = _internal_anchor_map(anchor_bank)
    budget = max(
        1,
        int(math.ceil(float(max_conflict_leaf_ratio) * tree.num_leaves)),
    )

    raw: list[tuple[int, int, list[tuple[str, float, float, float]]]] = []
    for row, class_id in enumerate(new_class_ids):
        nearest_position = int((new_values[row] @ leaf_values.t()).argmax())
        nearest_class = anchor_bank.leaf_class_ids[nearest_position]
        candidates: list[tuple[str, float, float, float]] = []
        for node_id in class_ancestor_path(
            tree, nearest_class, include_root=include_root
        ):
            if len(tree.descendants(node_id)) > budget:
                continue
            anchor = F.normalize(internal_anchors[node_id].float(), dim=0)
            distance = float(
                torch.acos((new_values[row] @ anchor).clamp(-1.0, 1.0)).item()
            )
            radius = float(node_radii[node_id])
            overlap = radius + float(radius_slack) - distance
            if overlap >= float(minimum_overlap):
                candidates.append((node_id, distance, radius, overlap))
        raw.append((int(class_id), int(nearest_class), candidates))

    # Classes with the strongest valid overlap claim the bounded old-class
    # budget first.  Reusing an already selected subtree consumes no new leaves.
    ordering = sorted(
        range(len(raw)),
        key=lambda index: (
            -max((item[3] for item in raw[index][2]), default=-float("inf")),
            raw[index][0],
        ),
    )
    used_old_classes: set[int] = set()
    selected: dict[int, tuple[str, float, float, float] | None] = {}
    for index in ordering:
        _, _, candidates = raw[index]
        choice = None
        # class_ancestor_path is deepest-first, which enforces locality.
        for candidate in candidates:
            candidate_classes = set(tree.descendants(candidate[0]))
            if len(used_old_classes | candidate_classes) <= budget:
                choice = candidate
                used_old_classes.update(candidate_classes)
                break
        selected[index] = choice

    assignments: list[ConflictAssignment] = []
    relaxed_margins: dict[tuple[int, str], float] = {}
    for index, (new_class_id, nearest_class, _) in enumerate(raw):
        choice = selected[index]
        if choice is None:
            insertion_target = tree.leaf_node_id(nearest_class)
            conflict_root = None
            distance = math.pi
            radius = 0.0
            overlap = -math.pi
            relaxed_classes: tuple[int, ...] = ()
        else:
            conflict_root, distance, radius, overlap = choice
            insertion_target = conflict_root
            relaxed_classes = tuple(tree.descendants(conflict_root))
        assignments.append(
            ConflictAssignment(
                new_class_id=new_class_id,
                nearest_leaf_class_id=nearest_class,
                insertion_target_node_id=insertion_target,
                conflict_root_node_id=conflict_root,
                angular_distance=distance,
                node_radius=radius,
                overlap_score=overlap,
                relaxed_old_class_ids=relaxed_classes,
            )
        )

        if mode != "local_margin" or conflict_root is None:
            continue
        conflict_members = set(tree.descendants(conflict_root))
        for old_class_id in relaxed_classes:
            path = class_ancestor_path(
                tree, old_class_id, include_root=include_root
            )
            if scope == "parent_only":
                path = path[:1]
            for node_id in path:
                if not set(tree.descendants(node_id)).issubset(conflict_members):
                    continue
                margin = _margin_for_relation(
                    tree,
                    old_class_id,
                    node_id,
                    base_margin=relaxation_margin,
                    ancestor_decay=margin_ancestor_decay,
                )
                key = (old_class_id, node_id)
                relaxed_margins[key] = max(
                    float(relaxed_margins.get(key, 0.0)), margin
                )

    if mode == "global_margin":
        for old_class_id in tree.class_order:
            path = class_ancestor_path(
                tree, old_class_id, include_root=include_root
            )
            if scope == "parent_only":
                path = path[:1]
            for node_id in path:
                relaxed_margins[(old_class_id, node_id)] = (
                    _margin_for_relation(
                        tree,
                        old_class_id,
                        node_id,
                        base_margin=relaxation_margin,
                        ancestor_decay=margin_ancestor_decay,
                    )
                )

    reported_relaxed_classes = (
        set(tree.class_order) if mode == "global_margin" else used_old_classes
    )
    return ConflictPlan(
        # Preserve the protocol's incoming class order.  Original CIFAR IDs are
        # intentionally not numerically sorted in standard CIL class orders.
        assignments=tuple(assignments),
        relaxed_margins=relaxed_margins,
        old_class_budget=budget,
        relaxed_old_class_ids=tuple(sorted(reported_relaxed_classes)),
        node_radii={str(key): float(value) for key, value in node_radii.items()},
        relaxation_mode=mode,
    )


def _next_internal_node_id(tree: HierarchyTree) -> str:
    maximum = -1
    for node_id in tree.internal_node_ids(include_root=True):
        suffix = node_id.split(":", maxsplit=1)[-1]
        try:
            maximum = max(maximum, int(suffix))
        except ValueError:
            continue
    return f"node:{maximum + 1:04d}"


def _recompute_members(
    nodes: dict[str, TreeNode], node_id: str
) -> tuple[int, ...]:
    node = nodes[node_id]
    if node.is_leaf:
        return node.members
    if node.left is None or node.right is None:
        raise ValueError("internal node has missing children")
    members = tuple(
        sorted(
            _recompute_members(nodes, node.left)
            + _recompute_members(nodes, node.right)
        )
    )
    nodes[node_id] = TreeNode(
        node_id=node.node_id,
        members=members,
        left=node.left,
        right=node.right,
    )
    return members


def insert_leaf_as_sibling(
    tree: HierarchyTree,
    target_node_id: str,
    class_id: int,
) -> tuple[HierarchyTree, str]:
    """Insert one new leaf locally without rebuilding any old subtree."""

    class_id = int(class_id)
    leaf_id = tree.leaf_node_id(class_id)
    if leaf_id in tree.nodes or class_id in tree.class_order:
        raise ValueError(f"class {class_id} already exists in the tree")
    if target_node_id not in tree.nodes:
        raise ValueError(f"insertion target is missing: {target_node_id}")

    nodes = dict(tree.nodes)
    nodes[leaf_id] = TreeNode(
        node_id=leaf_id,
        members=(class_id,),
        class_id=class_id,
    )
    inserted_id = _next_internal_node_id(tree)
    target = nodes[target_node_id]
    left_id, right_id = target_node_id, leaf_id
    if (class_id,) < target.members:
        left_id, right_id = leaf_id, target_node_id
    nodes[inserted_id] = TreeNode(
        node_id=inserted_id,
        members=tuple(sorted((*target.members, class_id))),
        left=left_id,
        right=right_id,
    )

    old_parent = tree.parent(target_node_id)
    root_id = tree.root_id
    if old_parent is None:
        root_id = inserted_id
    else:
        parent = nodes[old_parent]
        nodes[old_parent] = TreeNode(
            node_id=parent.node_id,
            members=parent.members,
            left=(inserted_id if parent.left == target_node_id else parent.left),
            right=(inserted_id if parent.right == target_node_id else parent.right),
        )
    _recompute_members(nodes, root_id)
    return (
        HierarchyTree(nodes, root_id, (*tree.class_order, class_id)),
        inserted_id,
    )


def insert_planned_classes(
    tree: HierarchyTree,
    plan: ConflictPlan,
) -> tuple[HierarchyTree, tuple[dict[str, object], ...]]:
    """Apply deterministic online insertions recorded before training."""

    current = tree
    replacement: dict[str, str] = {}
    logs: list[dict[str, object]] = []
    for assignment in plan.assignments:
        requested = assignment.insertion_target_node_id
        target = replacement.get(requested, requested)
        current, inserted = insert_leaf_as_sibling(
            current, target, assignment.new_class_id
        )
        replacement[requested] = inserted
        logs.append(
            {
                "new_class_id": assignment.new_class_id,
                "requested_target": requested,
                "effective_target": target,
                "inserted_parent": inserted,
            }
        )
    return current, tuple(logs)


class SACILV1PathLoss(nn.Module):
    """Co-moving ancestor-path preservation with structured margin relaxation.

    Every old exemplar is compared only with anchors on its own ancestor path.
    Conflict-local entries receive a tolerance margin; all other entries keep a
    zero margin. ``relation_mean`` averages all active relations, while
    ``sample_mean`` first averages each exemplar's path and then averages the
    exemplars so that tree depth does not change an exemplar's total weight.
    """

    def __init__(
        self,
        anchor_bank: HierarchicalAnchorBank,
        tree: HierarchyTree,
        conflict_plan: ConflictPlan,
        *,
        path_scope: str = "all_ancestors",
        path_normalization: str = "relation_mean",
        include_root: bool = False,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        scope = str(path_scope).lower().replace("-", "_")
        if scope not in PATH_SCOPES:
            raise ValueError(f"unknown path scope: {path_scope}")
        normalization = str(path_normalization).lower().replace("-", "_")
        if normalization not in PATH_NORMALIZATIONS:
            raise ValueError(
                f"unknown path normalization: {path_normalization}"
            )
        if anchor_bank.leaf_class_ids != tree.class_order:
            raise ValueError("tree and anchor class orders differ")
        internal_ids = tuple(
            node_id
            for node_id in anchor_bank.internal_node_ids
            if include_root or node_id != tree.root_id
        )
        internal_positions = {
            node_id: position
            for position, node_id in enumerate(anchor_bank.internal_node_ids)
        }
        anchors = torch.stack(
            [anchor_bank.internal_anchors[internal_positions[node_id]] for node_id in internal_ids]
        )
        class_ids = tuple(tree.class_order)
        path_mask = torch.zeros(len(class_ids), len(internal_ids), dtype=torch.bool)
        margins = torch.zeros(len(class_ids), len(internal_ids), dtype=torch.float32)
        node_positions = {node_id: index for index, node_id in enumerate(internal_ids)}
        for class_position, class_id in enumerate(class_ids):
            path = class_ancestor_path(tree, class_id, include_root=include_root)
            if scope == "parent_only":
                path = path[:1]
            for node_id in path:
                if node_id not in node_positions:
                    continue
                node_position = node_positions[node_id]
                path_mask[class_position, node_position] = True
                margins[class_position, node_position] = float(
                    conflict_plan.relaxed_margins.get((class_id, node_id), 0.0)
                )
        if not bool(path_mask.any()):
            raise ValueError("SACIL-v1 path loss has no active relations")

        self.class_ids = class_ids
        self.internal_node_ids = internal_ids
        self.path_scope = scope
        self.path_normalization = normalization
        self.include_root = bool(include_root)
        self.conflict_plan = conflict_plan
        self.epsilon = float(epsilon)
        self.register_buffer("reference_anchors", anchors.float().clone())
        self.register_buffer("current_anchors", anchors.float().clone())
        self.register_buffer("path_mask", path_mask)
        self.register_buffer("relaxation_margins", margins)

    @property
    def requires_current_anchor_refresh(self) -> bool:
        return True

    @torch.no_grad()
    def update_current_anchors(
        self, anchor_bank: HierarchicalAnchorBank
    ) -> None:
        if anchor_bank.leaf_class_ids != self.class_ids:
            raise ValueError("current anchor class order differs")
        position = {
            node_id: index
            for index, node_id in enumerate(anchor_bank.internal_node_ids)
        }
        if any(node_id not in position for node_id in self.internal_node_ids):
            raise ValueError("current anchor tree differs")
        values = torch.stack(
            [anchor_bank.internal_anchors[position[node_id]] for node_id in self.internal_node_ids]
        )
        if values.shape != self.current_anchors.shape:
            raise ValueError("current anchor shape differs")
        self.current_anchors.copy_(values.to(self.current_anchors))

    def _class_positions(self, original_targets: Tensor) -> Tensor:
        targets = original_targets.long()
        positions = torch.full_like(targets, -1)
        for position, class_id in enumerate(self.class_ids):
            positions = torch.where(
                targets.eq(int(class_id)),
                positions.new_full((), position),
                positions,
            )
        if bool(positions.lt(0).any()):
            unknown = sorted(set(targets[positions.lt(0)].detach().cpu().tolist()))
            raise ValueError(f"targets are absent from the old tree: {unknown}")
        return positions

    def forward(
        self,
        current_features: Tensor,
        reference_features: Tensor,
        original_targets: Tensor,
    ) -> Tensor:
        if current_features.shape != reference_features.shape:
            raise ValueError("current and reference feature shapes differ")
        if current_features.shape[0] != original_targets.numel():
            raise ValueError("feature and target counts differ")
        if current_features.shape[0] == 0:
            return current_features.sum() * 0.0
        class_positions = self._class_positions(original_targets)
        mask = self.path_mask[class_positions].to(current_features)
        margins = self.relaxation_margins[class_positions].to(current_features)
        with torch.no_grad():
            reference = anchor_affinity(
                reference_features, self.reference_anchors
            )
        current = anchor_affinity(current_features, self.current_anchors)
        deviation = (current - reference).abs()
        error = F.relu(deviation - margins).square() * mask
        if self.path_normalization == "sample_mean":
            per_sample = error.sum(dim=1) / mask.sum(dim=1).clamp_min(
                self.epsilon
            )
            return per_sample.mean()
        return error.sum() / mask.sum().clamp_min(self.epsilon)

    @torch.no_grad()
    def diagnostics(self) -> dict[str, object]:
        active = self.path_mask.sum().item()
        relaxed = (self.relaxation_margins > 0).logical_and(
            self.path_mask
        ).sum().item()
        return {
            "path_scope": self.path_scope,
            "path_normalization": self.path_normalization,
            "include_root": self.include_root,
            "active_relation_count": int(active),
            "relaxed_relation_count": int(relaxed),
            "relaxed_relation_fraction": float(relaxed / max(1, active)),
            "conflict_plan": self.conflict_plan.state_dict(),
        }
