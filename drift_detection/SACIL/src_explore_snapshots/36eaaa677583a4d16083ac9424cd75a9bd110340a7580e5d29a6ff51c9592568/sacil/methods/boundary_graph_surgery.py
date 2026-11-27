from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sacil.hierarchy import HierarchyTree


INSIDE, BOUNDARY, OUTSIDE = 0, 1, 2
PAIR_TYPE_NAMES = ("inside", "boundary", "outside")


@dataclass(frozen=True)
class BoundedAncestorBranchSelection:
    """Deterministic nearest-leaf branch selections for incoming classes.

    ``selected_branch_nodes`` may contain a leaf when the leaf's immediate
    parent already exceeds the cap.  ``primary_internal_nodes`` then records
    that immediate parent explicitly; it is metadata for the insertion
    contract and is not part of the bounded conflict region.
    """

    nearest_leaf_positions: tuple[int, ...]
    nearest_leaf_original_ids: tuple[int, ...]
    nearest_leaf_node_ids: tuple[str, ...]
    nearest_leaf_scores: Tensor
    selected_branch_nodes: tuple[str, ...]
    selected_branch_leaf_counts: tuple[int, ...]
    primary_internal_nodes: tuple[str, ...]


def effective_bounded_branch_cap(
    configured_max_branch_leaves: int,
    max_conflict_leaf_coverage: float,
    old_class_count: int,
    incoming_class_count: int,
) -> int:
    """Return the per-session cap whose union upper bound respects coverage."""

    configured = int(configured_max_branch_leaves)
    old_count = int(old_class_count)
    incoming_count = int(incoming_class_count)
    coverage = float(max_conflict_leaf_coverage)
    if configured < 2:
        raise ValueError("max_branch_leaves must be at least 2")
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("max_conflict_leaf_coverage must be in (0, 1)")
    if old_count <= 0 or incoming_count <= 0:
        raise ValueError("old and incoming class counts must be positive")
    coverage_cap = math.floor(coverage * old_count / incoming_count)
    return min(configured, max(2, coverage_cap))


def nearest_leaf_bounded_ancestor_branches(
    tree: HierarchyTree,
    old_class_ids: Sequence[int],
    old_leaf_prototypes: Tensor,
    incoming_teacher_prototypes: Tensor,
    *,
    max_branch_leaves: int,
) -> BoundedAncestorBranchSelection:
    """Choose the highest capped ancestor above each nearest old leaf.

    Ties in nearest-leaf cosine similarity use PyTorch's deterministic first
    argmax, which follows ``old_class_ids`` order.  If no internal ancestor is
    within the cap, the selected conflict region is the single nearest leaf.
    Its immediate parent is retained separately as the primary-internal
    metadata contract.
    """

    class_ids = tuple(int(value) for value in old_class_ids)
    cap = int(max_branch_leaves)
    if cap < 2:
        raise ValueError("bounded branch cap must be at least 2")
    if class_ids != tuple(tree.class_order):
        raise ValueError("old class IDs must follow the hierarchy leaf order")
    if old_leaf_prototypes.ndim != 2 or incoming_teacher_prototypes.ndim != 2:
        raise ValueError("bounded branch prototypes must be matrices")
    if old_leaf_prototypes.shape[0] != len(class_ids):
        raise ValueError("old leaf prototype count does not match old classes")
    if old_leaf_prototypes.shape[1] != incoming_teacher_prototypes.shape[1]:
        raise ValueError("old and incoming prototype dimensions do not match")
    if incoming_teacher_prototypes.shape[0] == 0:
        raise ValueError("bounded branch selection requires incoming classes")
    if not torch.isfinite(old_leaf_prototypes).all() or not torch.isfinite(
        incoming_teacher_prototypes
    ).all():
        raise ValueError("bounded branch prototypes must be finite")

    similarities = (
        F.normalize(incoming_teacher_prototypes.float(), dim=1)
        @ F.normalize(old_leaf_prototypes.float(), dim=1).T
    )
    nearest_positions_tensor = similarities.argmax(dim=1)
    nearest_positions = tuple(
        int(value) for value in nearest_positions_tensor.tolist()
    )
    nearest_scores = similarities.gather(
        1, nearest_positions_tensor[:, None]
    ).squeeze(1)
    nearest_class_ids = tuple(class_ids[position] for position in nearest_positions)
    nearest_nodes = tuple(
        tree.leaf_node_id(class_id) for class_id in nearest_class_ids
    )
    selected_nodes: list[str] = []
    selected_sizes: list[int] = []
    primary_internal_nodes: list[str] = []
    for leaf_node in nearest_nodes:
        selected = leaf_node
        while True:
            parent = tree.parent(selected)
            if parent is None or len(tree.descendants(parent)) > cap:
                break
            selected = parent
        selected_nodes.append(selected)
        selected_sizes.append(len(tree.descendants(selected)))
        if tree.nodes[selected].is_leaf:
            parent = tree.parent(selected)
            if parent is None:
                raise ValueError(
                    "single-leaf selection has no internal parent contract"
                )
            primary_internal_nodes.append(parent)
        else:
            primary_internal_nodes.append(selected)
    if any(size > cap for size in selected_sizes):
        raise AssertionError("bounded branch selection exceeded its cap")
    return BoundedAncestorBranchSelection(
        nearest_leaf_positions=nearest_positions,
        nearest_leaf_original_ids=nearest_class_ids,
        nearest_leaf_node_ids=nearest_nodes,
        nearest_leaf_scores=nearest_scores.detach().cpu(),
        selected_branch_nodes=tuple(selected_nodes),
        selected_branch_leaf_counts=tuple(selected_sizes),
        primary_internal_nodes=tuple(primary_internal_nodes),
    )


def bounded_conflict_union_diagnostics(
    tree: HierarchyTree,
    selected_nodes: Sequence[str],
    old_class_ids: Sequence[int],
    *,
    max_conflict_leaf_coverage: float,
) -> dict[str, float | int]:
    """Validate and summarize the realized bounded conflict union."""

    old_ids = tuple(int(value) for value in old_class_ids)
    if not old_ids:
        raise ValueError("bounded conflict union requires old classes")
    coverage_limit = float(max_conflict_leaf_coverage)
    if not math.isfinite(coverage_limit) or not 0 < coverage_limit < 1:
        raise ValueError("max_conflict_leaf_coverage must be in (0, 1)")
    union = set().union(
        *(set(tree.descendants(str(node))) for node in selected_nodes)
    )
    if not union <= set(old_ids):
        raise ValueError("bounded conflict union contains a non-old class")
    coverage = len(union) / len(old_ids)
    if coverage > coverage_limit + 1e-12:
        raise AssertionError(
            "bounded conflict union coverage exceeds max_conflict_leaf_coverage"
        )
    return {
        "conflict_union_leaf_count": len(union),
        "conflict_union_coverage": coverage,
    }


def canonical_regions(
    tree: HierarchyTree,
    raw_nodes: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    unique = tuple(dict.fromkeys(str(node) for node in raw_nodes))
    canonical = []
    for node in unique:
        members = set(tree.descendants(node))
        if not any(
            other != node and members < set(tree.descendants(other))
            for other in unique
        ):
            canonical.append(node)
    internal_order = {
        node_id: index
        for index, node_id in enumerate(tree.internal_node_ids())
    }
    leaf_order = {
        node_id: index for index, node_id in enumerate(tree.leaf_node_ids())
    }
    canonical = tuple(
        sorted(
            canonical,
            key=lambda value: (
                0,
                internal_order[value],
            )
            if value in internal_order
            else (1, leaf_order[value]),
        )
    )
    mapping = {}
    for node in unique:
        members = set(tree.descendants(node))
        containers = [q for q in canonical if members <= set(tree.descendants(q))]
        if len(containers) != 1:
            raise ValueError("raw branch does not map to one canonical region")
        mapping[node] = containers[0]
    descendant_sets = [set(tree.descendants(node)) for node in canonical]
    if any(a & b for i, a in enumerate(descendant_sets) for b in descendant_sets[i + 1 :]):
        raise ValueError("canonical regions are not disjoint")
    return canonical, mapping


def endpoint_regions(
    tree: HierarchyTree,
    canonical: Sequence[str],
    old_class_ids: Sequence[int],
    anchor_descendants: Sequence[Sequence[int]],
) -> tuple[tuple[str | None, ...], tuple[str | None, ...]]:
    regions = {node: set(tree.descendants(node)) for node in canonical}
    sample = tuple(
        next(
            (q for q, members in regions.items() if int(c) in members),
            None,
        )
        for c in old_class_ids
    )
    anchors = []
    for descendants in anchor_descendants:
        members = set(int(value) for value in descendants)
        anchors.append(next((q for q, region in regions.items() if members <= region), None))
    return sample, tuple(anchors)


def pair_types_and_weights(
    sample_regions: Sequence[str | None],
    anchor_regions: Sequence[str | None],
    *,
    inside_weight: float,
    boundary_weight: float,
    mask_mode: str,
) -> tuple[Tensor, Tensor]:
    pair_types = torch.empty(len(sample_regions), len(anchor_regions), dtype=torch.int8)
    weights = torch.empty_like(pair_types, dtype=torch.float32)
    for row, sample_region in enumerate(sample_regions):
        for column, anchor_region in enumerate(anchor_regions):
            inside = sample_region is not None and sample_region == anchor_region
            outside = sample_region is None and anchor_region is None
            pair_type = INSIDE if inside else OUTSIDE if outside else BOUNDARY
            pair_types[row, column] = pair_type
            if mask_mode == "incident":
                relaxed = sample_region is not None or anchor_region is not None
                weights[row, column] = (
                    inside_weight if relaxed else 1.0
                )
            else:
                weights[row, column] = (
                    inside_weight
                    if inside
                    else boundary_weight
                    if not outside
                    else 1.0
                )
    return pair_types, weights


def random_pair_seed(
    experiment_seed: int,
    session_id: int,
    group: str,
    original_class_id: int,
) -> int:
    payload = (
        "bgs_random_pair_v1|"
        f"{experiment_seed}|{session_id}|{group}|{original_class_id}"
    ).encode("utf-8")
    return (
        int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        % (2**63 - 1)
    )


def row_permuted_random_weights(
    weights: Tensor,
    old_class_ids: Sequence[int],
    *,
    experiment_seed: int,
    session_id: int,
    group: str,
) -> tuple[Tensor, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    randomized = torch.empty_like(weights)
    seeds, permutations = [], []
    for row, class_id in enumerate(old_class_ids):
        seed = random_pair_seed(experiment_seed, session_id, group, int(class_id))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(weights.shape[1], generator=generator)
        randomized[row] = weights[row, permutation]
        seeds.append(seed)
        permutations.append(tuple(int(value) for value in permutation))
    return randomized, tuple(seeds), tuple(permutations)


def tensor_sha256(value: Tensor) -> str:
    """Hash a frozen tensor together with its dtype and exact shape."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def pair_mask_summary(
    pair_types: Tensor,
    weights: Tensor,
    *,
    inside_weight: float,
    boundary_weight: float,
) -> dict:
    """Return static semantic and realized-mask diagnostics.

    ``pair_types`` always describes the learned structured partition.  The
    realized weights may differ for the incident and random-pair controls, so
    their mismatch is reported explicitly instead of relabeling randomized
    entries as structured inside/boundary/outside pairs.
    """

    if pair_types.shape != weights.shape:
        raise ValueError("pair types and realized weights must have one shape")
    total = int(pair_types.numel())
    expected = torch.ones_like(weights)
    expected[pair_types == INSIDE] = float(inside_weight)
    expected[pair_types == BOUNDARY] = float(boundary_weight)
    summary: dict[str, object] = {
        "pair_count": total,
        "mean_weight": float(weights.mean()) if total else None,
        "total_weight": float(weights.sum()),
        "relaxation_deficit": float((1.0 - weights).sum()),
        "realized_weight_sha256": tensor_sha256(weights),
        "pair_type_sha256": tensor_sha256(pair_types),
        "structured_weight_mismatch_count": int(
            (~torch.isclose(weights, expected, atol=0.0, rtol=0.0)).sum()
        ),
        "per_old_class_relaxed_pair_count": (
            (weights < 1.0).sum(dim=1).tolist()
        ),
        "per_old_class_relaxation_deficit": (
            (1.0 - weights).sum(dim=1).tolist()
        ),
    }
    for code, name in enumerate(PAIR_TYPE_NAMES):
        mask = pair_types == code
        count = int(mask.sum())
        summary[name] = {
            "pair_count": count,
            "pair_ratio": None if total == 0 else count / total,
            "mean_realized_weight": (
                None if count == 0 else float(weights[mask].mean())
            ),
            "total_realized_weight": float(weights[mask].sum()),
            "realized_relaxation_deficit": float(
                (1.0 - weights[mask]).sum()
            ),
        }
    return summary


def negative_candidate_positions(
    tree: HierarchyTree,
    old_class_ids: Sequence[int],
    raw_node: str,
    scope: str,
) -> tuple[int, ...]:
    if scope == "all_old":
        candidates = set(int(value) for value in old_class_ids)
    elif scope == "branch_local":
        candidates = set(tree.descendants(raw_node))
    else:
        raise ValueError(f"unsupported BGS negative scope: {scope}")
    return tuple(
        position
        for position, class_id in enumerate(old_class_ids)
        if int(class_id) in candidates
    )


@dataclass
class BoundaryGraphSurgeryReference:
    session_id: int
    old_original_ids: tuple[int, ...]
    new_original_ids: tuple[int, ...]
    raw_branch_nodes: tuple[str, ...]
    raw_branch_scores: Tensor
    canonical_nodes: tuple[str, ...]
    raw_to_canonical: dict[str, str]
    tree_state: dict
    anchor_state: dict
    leaf_pair_types: Tensor
    internal_pair_types: Tensor
    leaf_weights: Tensor
    internal_weights: Tensor
    random_seeds: dict
    random_permutations: dict
    incoming_teacher_prototypes: Tensor
    negative_class_positions: tuple[tuple[int, ...], ...]
    parent_thresholds: Tensor
    options: dict
    old_incremental_ids: tuple[int, ...] = ()
    new_incremental_ids: tuple[int, ...] = ()
    sample_region_ids: tuple[str | None, ...] = ()
    leaf_anchor_ids: tuple[str, ...] = ()
    leaf_anchor_region_ids: tuple[str | None, ...] = ()
    internal_anchor_ids: tuple[str, ...] = ()
    internal_anchor_region_ids: tuple[str | None, ...] = ()
    primary_internal_positions: tuple[int, ...] = ()
    mask_diagnostics: dict | None = None
    spec_version: str = "bgs_v1"

    def state_dict(self) -> dict:
        state = copy.deepcopy(self.__dict__)
        for key in (
            "raw_branch_scores",
            "leaf_pair_types",
            "internal_pair_types",
            "leaf_weights",
            "internal_weights",
            "incoming_teacher_prototypes",
            "parent_thresholds",
        ):
            state[key] = state[key].detach().cpu().clone()
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "BoundaryGraphSurgeryReference":
        return cls(**copy.deepcopy(dict(state)))


class BoundaryGraphSurgeryLoss(nn.Module):
    def __init__(self, reference: BoundaryGraphSurgeryReference) -> None:
        super().__init__()
        anchor_state = reference.anchor_state
        geometry = reference.options.get("geometry", {})
        use_leaf = bool(geometry.get("use_leaf", True))
        use_internal = bool(
            geometry.get("use_internal_without_root", True)
        )
        leaf_anchors = anchor_state["leaf_anchors"].float().clone()
        leaf_weights = reference.leaf_weights.float().clone()
        leaf_types = reference.leaf_pair_types.clone()
        if not use_leaf:
            leaf_anchors = leaf_anchors[:0]
            leaf_weights = leaf_weights[:, :0]
            leaf_types = leaf_types[:, :0]
        self.register_buffer("leaf_anchors", leaf_anchors)
        internal_ids = tuple(anchor_state["internal_node_ids"])
        positions = [i for i, node in enumerate(internal_ids) if node != anchor_state["root_id"]]
        internal_anchors = (
            anchor_state["internal_anchors"][positions].float().clone()
        )
        internal_weights = reference.internal_weights.float().clone()
        internal_types = reference.internal_pair_types.clone()
        if not use_internal:
            internal_anchors = internal_anchors[:0]
            internal_weights = internal_weights[:, :0]
            internal_types = internal_types[:, :0]
        self.register_buffer("internal_anchors", internal_anchors)
        self.register_buffer("leaf_weights", leaf_weights)
        self.register_buffer("internal_weights", internal_weights)
        self.register_buffer("leaf_pair_types", leaf_types)
        self.register_buffer("internal_pair_types", internal_types)

    @staticmethod
    def _group(
        current: Tensor,
        teacher: Tensor,
        anchors: Tensor,
        weights: Tensor,
        pair_types: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if anchors.shape[0] == 0:
            zero = current.sum() * 0.0
            stats = {
                "error": zero,
                "realized_weight_deficit": zero.detach(),
                "structured_weight_mismatch_count": zero.detach(),
            }
            for name in PAIR_TYPE_NAMES:
                stats[f"{name}_count"] = zero.detach()
                stats[f"{name}_drift"] = zero.detach()
                stats[f"{name}_weighted_contribution"] = zero.detach()
                stats[f"{name}_weight_deficit"] = zero.detach()
                stats[f"{name}_teacher_affinity_mean"] = zero.detach()
                stats[f"{name}_teacher_affinity_std"] = zero.detach()
                stats[f"{name}_current_affinity_mean"] = zero.detach()
                stats[f"{name}_current_affinity_std"] = zero.detach()
            return zero, stats
        current_affinity = F.normalize(current, dim=1) @ F.normalize(anchors, dim=1).T
        with torch.no_grad():
            teacher_affinity = F.normalize(teacher, dim=1) @ F.normalize(anchors, dim=1).T
        error = (current_affinity - teacher_affinity).square()
        denominator = current.shape[0] * anchors.shape[0]
        loss = (weights * error).sum() / denominator
        stats = {
            "error": error.mean().detach(),
            "realized_weight_deficit": (1.0 - weights).sum().detach(),
        }
        for code, name in enumerate(PAIR_TYPE_NAMES):
            mask = pair_types == code
            count = mask.sum()
            stats[f"{name}_count"] = count.detach()
            if int(count) == 0:
                zero = error.sum().detach() * 0.0
                stats[f"{name}_drift"] = zero
                stats[f"{name}_weighted_contribution"] = zero
                stats[f"{name}_weight_deficit"] = zero
                stats[f"{name}_teacher_affinity_mean"] = zero
                stats[f"{name}_teacher_affinity_std"] = zero
                stats[f"{name}_current_affinity_mean"] = zero
                stats[f"{name}_current_affinity_std"] = zero
                continue
            stats[f"{name}_drift"] = error[mask].mean().detach()
            stats[f"{name}_weighted_contribution"] = (
                (weights[mask] * error[mask]).sum() / denominator
            ).detach()
            stats[f"{name}_weight_deficit"] = (
                1.0 - weights[mask]
            ).sum().detach()
            teacher_values = teacher_affinity[mask]
            current_values = current_affinity[mask]
            stats[f"{name}_teacher_affinity_mean"] = (
                teacher_values.mean().detach()
            )
            stats[f"{name}_teacher_affinity_std"] = (
                teacher_values.std(unbiased=False).detach()
            )
            stats[f"{name}_current_affinity_mean"] = (
                current_values.mean().detach()
            )
            stats[f"{name}_current_affinity_std"] = (
                current_values.std(unbiased=False).detach()
            )
        return loss, stats

    def forward(
        self,
        current: Tensor,
        teacher: Tensor,
        old_incremental_targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if current.shape[0] == 0:
            zero = current.sum() * 0.0
            return zero, {"leaf": zero, "internal": zero}
        leaf_weights = self.leaf_weights[old_incremental_targets]
        internal_weights = self.internal_weights[old_incremental_targets]
        leaf_types = self.leaf_pair_types[old_incremental_targets]
        internal_types = self.internal_pair_types[old_incremental_targets]
        leaf, leaf_stats = self._group(
            current,
            teacher,
            self.leaf_anchors,
            leaf_weights,
            leaf_types,
        )
        internal, internal_stats = self._group(
            current,
            teacher,
            self.internal_anchors,
            internal_weights,
            internal_types,
        )
        valid = []
        if self.leaf_anchors.shape[0]:
            valid.append(leaf)
        if self.internal_anchors.shape[0]:
            valid.append(internal)
        loss = torch.stack(valid).mean() if valid else current.sum() * 0.0
        stats: dict[str, Tensor] = {"leaf": leaf, "internal": internal}
        stats.update({f"leaf_{key}": value for key, value in leaf_stats.items()})
        stats.update(
            {f"internal_{key}": value for key, value in internal_stats.items()}
        )
        return loss, stats


def bgs_insertion_loss(
    current_features: Tensor,
    new_class_indices: Tensor,
    positive_prototypes: Tensor,
    leaf_anchors: Tensor,
    internal_anchors: Tensor,
    negative_positions: Sequence[Sequence[int]],
    primary_internal_positions: Tensor,
    parent_thresholds: Tensor,
    *,
    temperature: float,
    separation_enabled: bool,
    parent_weight: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    if current_features.shape[0] == 0:
        zero = current_features.sum() * 0.0
        return zero, zero, zero, {
            "parent_active_ratio": zero,
            "positive_cosine": zero,
            "negative_cosine": zero,
        }
    queries = F.normalize(current_features, dim=1)
    separation_rows, parent_rows, pos_cosines, neg_cosines = [], [], [], []
    for query, class_index in zip(queries, new_class_indices.tolist()):
        positive = positive_prototypes[class_index].detach()
        pos = (query * positive).sum()
        negatives = leaf_anchors[list(negative_positions[class_index])].detach()
        neg = negatives @ query
        if separation_enabled:
            logits = torch.cat((pos.view(1), neg)) / temperature
            separation_rows.append(-F.log_softmax(logits, dim=0)[0])
        else:
            separation_rows.append(pos * 0.0)
        parent = internal_anchors[int(primary_internal_positions[class_index])].detach()
        parent_rows.append(
            F.relu(
                parent_thresholds[class_index].detach()
                - (query * parent).sum()
            )
        )
        pos_cosines.append(pos.detach())
        neg_cosines.append(neg.mean().detach())
    separation = torch.stack(separation_rows).mean()
    parent = torch.stack(parent_rows).mean()
    total = separation + float(parent_weight) * parent
    return total, separation, parent, {
        "parent_active_ratio": torch.stack([value > 0 for value in parent_rows]).float().mean(),
        "positive_cosine": torch.stack(pos_cosines).mean(),
        "negative_cosine": torch.stack(neg_cosines).mean(),
    }
