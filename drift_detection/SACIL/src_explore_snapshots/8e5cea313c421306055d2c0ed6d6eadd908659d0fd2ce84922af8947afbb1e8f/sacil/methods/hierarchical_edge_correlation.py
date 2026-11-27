from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EDGE_WEIGHTING_MODES = frozenset(
    {"global", "conflict_branch_incident", "conflict_subtree_inside"}
)
STRATIFIED_EDGE_GROUP_NAMES = ("stable", "boundary", "conflict")
STRATIFIED_EDGE_GROUP_STABLE = 0
STRATIFIED_EDGE_GROUP_BOUNDARY = 1
STRATIFIED_EDGE_GROUP_CONFLICT = 2


def pairwise_cosine_edge_vector(features: Tensor) -> Tensor:
    """Vectorize every undirected representative edge in upper-triangle order."""

    if features.ndim != 2:
        raise ValueError("representative features must have shape [R, D]")
    if features.shape[0] < 2:
        raise ValueError("edge topology requires at least two representatives")
    normalized = F.normalize(features.float(), dim=1)
    rows, columns = torch.triu_indices(
        normalized.shape[0],
        normalized.shape[0],
        offset=1,
        device=normalized.device,
    )
    return (normalized[rows] * normalized[columns]).sum(dim=1)


def global_edge_weights(
    representative_count: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    count = int(representative_count)
    if count < 2:
        raise ValueError("edge topology requires at least two representatives")
    return torch.ones(
        count * (count - 1) // 2,
        dtype=dtype,
        device=device,
    )


def incident_edge_weights(representative_weights: Tensor) -> Tensor:
    """Relax every edge incident to a low-preservation representative.

    The minimum endpoint weight implements the literal incident-edge rule: an
    edge is relaxed whenever either endpoint belongs to a conflict branch.
    """

    if representative_weights.ndim != 1:
        raise ValueError("representative weights must be a vector")
    if representative_weights.numel() < 2:
        raise ValueError("at least two representative weights are required")
    if not torch.isfinite(representative_weights).all():
        raise ValueError("representative weights must be finite")
    if bool((representative_weights < 0).any()):
        raise ValueError("representative weights must be non-negative")
    rows, columns = torch.triu_indices(
        representative_weights.numel(),
        representative_weights.numel(),
        offset=1,
        device=representative_weights.device,
    )
    return torch.minimum(
        representative_weights[rows], representative_weights[columns]
    )


def conflict_subtree_inside_edge_weights(
    conflict_membership: Tensor,
    *,
    min_edge_weight: float,
) -> Tensor:
    """Relax inside-subtree edges while preserving cut and outside edges.

    Output follows the same upper-triangle ordering as
    :func:`pairwise_cosine_edge_vector`.
    """

    if conflict_membership.ndim != 1:
        raise ValueError("conflict membership must be a vector")
    if conflict_membership.numel() < 2:
        raise ValueError("at least two representative memberships are required")
    if conflict_membership.dtype != torch.bool:
        raise ValueError("conflict membership must be boolean")
    minimum = float(min_edge_weight)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("min_edge_weight must be in [0, 1]")
    rows, columns = torch.triu_indices(
        conflict_membership.numel(),
        conflict_membership.numel(),
        offset=1,
        device=conflict_membership.device,
    )
    inside = conflict_membership[rows] & conflict_membership[columns]
    weights = torch.ones(
        rows.numel(), dtype=torch.float32, device=conflict_membership.device
    )
    return weights.masked_fill(inside, minimum)


def conflict_union_membership(
    representative_class_ids: Sequence[int],
    conflict_subtrees: Sequence[Sequence[int]],
) -> Tensor:
    """Return deterministic membership in the union of conflict subtrees.

    Repeated or overlapping descendants are collapsed by set union. The
    representative order is never changed, so the result remains aligned with
    the fixed upper-triangle edge order.
    """

    class_ids = tuple(int(value) for value in representative_class_ids)
    if len(class_ids) < 2:
        raise ValueError("at least two representative classes are required")
    conflict_classes = {
        int(class_id)
        for subtree in conflict_subtrees
        for class_id in subtree
    }
    return torch.tensor(
        [class_id in conflict_classes for class_id in class_ids],
        dtype=torch.bool,
    )


def stratified_edge_group_ids(conflict_membership: Tensor) -> Tensor:
    """Partition upper-triangle edges into stable, boundary, and conflict.

    Stable edges have both endpoints outside the union of selected conflict
    subtrees. Boundary-cut edges have exactly one endpoint inside. Conflict
    edges have both endpoints inside. The three groups are disjoint and cover
    every edge, including the deterministic all-outside/all-inside cases.
    """

    if conflict_membership.ndim != 1:
        raise ValueError("conflict membership must be a vector")
    if conflict_membership.numel() < 2:
        raise ValueError("at least two representative memberships are required")
    if conflict_membership.dtype != torch.bool:
        raise ValueError("conflict membership must be boolean")
    rows, columns = torch.triu_indices(
        conflict_membership.numel(),
        conflict_membership.numel(),
        offset=1,
        device=conflict_membership.device,
    )
    left = conflict_membership[rows]
    right = conflict_membership[columns]
    group_ids = torch.full(
        (rows.numel(),),
        STRATIFIED_EDGE_GROUP_STABLE,
        dtype=torch.long,
        device=conflict_membership.device,
    )
    group_ids[left ^ right] = STRATIFIED_EDGE_GROUP_BOUNDARY
    group_ids[left & right] = STRATIFIED_EDGE_GROUP_CONFLICT
    return group_ids


def weighted_global_edge_correlation_loss(
    current_edges: Tensor,
    reference_edges: Tensor,
    edge_weights: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """One weighted Pearson correlation over the complete topology vector.

    This is deliberately different from sample-to-anchor correlation.  The
    correlation axis is the single vector of all representative pairs, not an
    anchor dimension independently reduced for each training sample.
    """

    if current_edges.ndim != 1 or reference_edges.ndim != 1:
        raise ValueError("edge correlations require vectors")
    if current_edges.shape != reference_edges.shape:
        raise ValueError("current and reference edge vectors differ")
    if edge_weights.ndim != 1 or edge_weights.shape != current_edges.shape:
        raise ValueError("edge weight vector differs from topology vector")
    if current_edges.numel() < 2:
        raise ValueError("edge correlation requires at least two edges")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    weights = edge_weights.to(current_edges).clamp_min(0.0)
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError("edge weights must have positive finite mass")
    weights = weights / weights.sum()
    reference = reference_edges.detach().to(current_edges)
    difference = current_edges - reference
    # Pearson correlation is exactly one at the unchanged session boundary,
    # but float round-off around clamp(1) can leave a spurious gradient. Use a
    # zero-minimum quadratic only inside the numerical equality neighborhood.
    if bool(difference.detach().abs().max() <= 1e-7):
        return difference.square().mean()
    current_mean = (weights * current_edges).sum()
    reference_mean = (weights * reference).sum()
    current_centered = current_edges - current_mean
    reference_centered = reference - reference_mean
    covariance = (weights * current_centered * reference_centered).sum()
    current_variance = (weights * current_centered.square()).sum()
    reference_variance = (weights * reference_centered.square()).sum()
    if float(reference_variance) <= epsilon:
        return current_edges.sum() * 0.0
    denominator = (current_variance * reference_variance).clamp_min(0.0).sqrt()
    correlation = covariance / denominator.clamp_min(epsilon)
    correlation = correlation.clamp(-1.0, 1.0)
    return 1.0 - correlation


class HierarchicalEdgeCorrelationLoss(nn.Module):
    """TPCIL-style global representative topology correlation."""

    def __init__(
        self,
        reference_edges: Tensor,
        edge_weights: Tensor,
        *,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if reference_edges.ndim != 1:
            raise ValueError("reference edges must be a vector")
        if edge_weights.shape != reference_edges.shape:
            raise ValueError("edge weights and reference edges differ")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        reference = reference_edges.detach().float().clone()
        weights = edge_weights.detach().float().clone().clamp_min(0.0)
        if not torch.isfinite(reference).all():
            raise ValueError("reference edges must be finite")
        if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            raise ValueError("edge weights must have positive finite mass")
        weights = weights / weights.sum()
        reference_mean = (weights * reference).sum()
        reference_centered = reference - reference_mean
        reference_variance = (weights * reference_centered.square()).sum()
        self.register_buffer(
            "reference_edges", reference
        )
        self.register_buffer("edge_weights", weights)
        self.register_buffer("reference_centered", reference_centered)
        self.register_buffer("reference_variance", reference_variance)
        self.epsilon = float(epsilon)

    def forward(self, current_representative_features: Tensor) -> Tensor:
        current_edges = pairwise_cosine_edge_vector(
            current_representative_features
        )
        if float(self.reference_variance) <= self.epsilon:
            return current_edges.sum() * 0.0
        current_mean = (self.edge_weights * current_edges).sum()
        current_centered = current_edges - current_mean
        covariance = (
            self.edge_weights * current_centered * self.reference_centered
        ).sum()
        current_variance = (
            self.edge_weights * current_centered.square()
        ).sum()
        denominator = (
            current_variance * self.reference_variance
        ).clamp_min(0.0).sqrt()
        correlation = covariance / denominator.clamp_min(self.epsilon)
        return 1.0 - correlation.clamp(-1.0, 1.0)


@dataclass
class StratifiedEdgeCorrelationResult:
    """Loss and explicit per-stratum Pearson diagnostics."""

    loss: Tensor
    group_losses: Tensor
    group_counts: Tensor
    reference_variances: Tensor
    current_variances: Tensor
    reference_active_groups: Tensor
    current_active_groups: Tensor
    active_groups: Tensor

    def detached_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        for index, name in enumerate(STRATIFIED_EDGE_GROUP_NAMES):
            metrics[f"{name}_loss"] = float(
                self.group_losses[index].detach()
            )
            metrics[f"{name}_edge_count"] = int(
                self.group_counts[index].detach()
            )
            metrics[f"{name}_reference_variance"] = float(
                self.reference_variances[index].detach()
            )
            metrics[f"{name}_current_variance"] = float(
                self.current_variances[index].detach()
            )
            metrics[f"{name}_reference_active"] = int(
                bool(self.reference_active_groups[index].detach())
            )
            metrics[f"{name}_current_active"] = int(
                bool(self.current_active_groups[index].detach())
            )
            metrics[f"{name}_active"] = int(
                bool(self.active_groups[index].detach())
            )
        return metrics


class StratifiedHierarchicalEdgeCorrelationLoss(nn.Module):
    """Independent Pearson objectives for three hierarchy-defined strata."""

    def __init__(
        self,
        reference_edges: Tensor,
        edge_group_ids: Tensor,
        *,
        beta_boundary: float = 1.0,
        gamma_conflict: float = 0.1,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if reference_edges.ndim != 1:
            raise ValueError("reference edges must be a vector")
        if edge_group_ids.shape != reference_edges.shape:
            raise ValueError("edge groups and reference edges differ")
        if edge_group_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("edge group IDs must be integral")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        beta = float(beta_boundary)
        gamma = float(gamma_conflict)
        if (
            not torch.isfinite(torch.tensor([beta, gamma])).all()
            or beta < 0.0
            or gamma < 0.0
        ):
            raise ValueError("stratified coefficients must be finite and non-negative")
        reference = reference_edges.detach().float().clone()
        groups = edge_group_ids.detach().long().clone()
        if not torch.isfinite(reference).all():
            raise ValueError("reference edges must be finite")
        if bool((groups < 0).any()) or bool((groups > 2).any()):
            raise ValueError("edge group IDs must be stable, boundary, or conflict")

        counts: list[int] = []
        variances: list[Tensor] = []
        for group_id in range(len(STRATIFIED_EDGE_GROUP_NAMES)):
            values = reference[groups == group_id]
            counts.append(int(values.numel()))
            if values.numel() < 2:
                variances.append(reference.new_zeros(()))
            else:
                variances.append((values - values.mean()).square().mean())
        reference_variances = torch.stack(variances)
        reference_active = (
            torch.tensor(counts, dtype=torch.long) >= 2
        ) & (reference_variances > float(epsilon))

        self.register_buffer("reference_edges", reference)
        self.register_buffer("edge_group_ids", groups)
        self.register_buffer(
            "group_counts", torch.tensor(counts, dtype=torch.long)
        )
        self.register_buffer("reference_variances", reference_variances)
        self.register_buffer("reference_active", reference_active)
        self.register_buffer(
            "group_coefficients",
            torch.tensor([1.0, beta, gamma], dtype=torch.float32),
        )
        self.beta_boundary = beta
        self.gamma_conflict = gamma
        self.epsilon = float(epsilon)

    def forward(
        self, current_representative_features: Tensor
    ) -> StratifiedEdgeCorrelationResult:
        current_edges = pairwise_cosine_edge_vector(
            current_representative_features
        )
        if current_edges.shape != self.reference_edges.shape:
            raise ValueError("current and reference edge vectors differ")
        if not torch.isfinite(current_edges).all():
            raise ValueError("current edges must be finite")

        losses: list[Tensor] = []
        current_variances: list[Tensor] = []
        current_active_groups: list[bool] = []
        active_groups: list[bool] = []
        for group_id in range(len(STRATIFIED_EDGE_GROUP_NAMES)):
            mask = self.edge_group_ids == group_id
            current = current_edges[mask]
            reference = self.reference_edges[mask].to(current_edges)
            if current.numel() < 2:
                current_variance = current_edges.new_zeros(())
            else:
                current_variance = (current - current.mean()).square().mean()
            current_variances.append(current_variance)
            current_active = current.numel() >= 2 and bool(
                current_variance.detach() > self.epsilon
            )
            current_active_groups.append(current_active)
            active = bool(self.reference_active[group_id]) and current_active
            active_groups.append(active)
            if not active:
                losses.append(current_edges.sum() * 0.0)
                continue
            difference = current - reference
            if bool(difference.detach().abs().max() <= 1e-7):
                losses.append(difference.square().mean())
                continue
            current_centered = current - current.mean()
            reference_centered = reference - reference.mean()
            covariance = (current_centered * reference_centered).mean()
            denominator = (
                current_variance * self.reference_variances[group_id]
            ).clamp_min(0.0).sqrt()
            correlation = covariance / denominator.clamp_min(self.epsilon)
            losses.append(1.0 - correlation.clamp(-1.0, 1.0))

        group_losses = torch.stack(losses)
        return StratifiedEdgeCorrelationResult(
            loss=(self.group_coefficients * group_losses).sum(),
            group_losses=group_losses,
            group_counts=self.group_counts,
            reference_variances=self.reference_variances,
            current_variances=torch.stack(current_variances),
            reference_active_groups=self.reference_active,
            current_active_groups=torch.tensor(
                current_active_groups,
                dtype=torch.bool,
                device=current_edges.device,
            ),
            active_groups=torch.tensor(
                active_groups,
                dtype=torch.bool,
                device=current_edges.device,
            ),
        )


@dataclass
class HierarchicalEdgeReference:
    """Serializable session-local topology object used by the unified runner."""

    session_id: int
    representatives_per_class: int
    representative_indices: tuple[int, ...]
    representative_class_ids: tuple[int, ...]
    reference_edges: Tensor
    edge_weights: Tensor
    edge_weighting: str = "global"
    conflict_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.representative_indices)
        if count < 2 or len(self.representative_class_ids) != count:
            raise ValueError("invalid representative metadata")
        if self.representatives_per_class <= 0:
            raise ValueError("representatives_per_class must be positive")
        expected_edges = count * (count - 1) // 2
        if self.reference_edges.shape != (expected_edges,):
            raise ValueError("reference edge count does not match representatives")
        if self.edge_weights.shape != (expected_edges,):
            raise ValueError("edge weight count does not match representatives")
        if self.edge_weighting not in EDGE_WEIGHTING_MODES:
            raise ValueError("invalid edge weighting mode")
        if not torch.isfinite(self.reference_edges).all():
            raise ValueError("reference edges must be finite")
        if (
            not torch.isfinite(self.edge_weights).all()
            or bool((self.edge_weights < 0).any())
            or float(self.edge_weights.sum()) <= 0.0
        ):
            raise ValueError("edge weights must have positive finite mass")

    @property
    def representative_count(self) -> int:
        return len(self.representative_indices)

    @property
    def edge_count(self) -> int:
        return int(self.reference_edges.numel())

    def loss_module(self) -> HierarchicalEdgeCorrelationLoss:
        return HierarchicalEdgeCorrelationLoss(
            self.reference_edges, self.edge_weights
        )

    def state_dict(self) -> dict:
        return {
            "session_id": int(self.session_id),
            "representatives_per_class": int(self.representatives_per_class),
            "representative_indices": list(self.representative_indices),
            "representative_class_ids": list(self.representative_class_ids),
            "reference_edges": self.reference_edges.detach().cpu().clone(),
            "edge_weights": self.edge_weights.detach().cpu().clone(),
            "edge_weighting": self.edge_weighting,
            "conflict_node_ids": list(self.conflict_node_ids),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "HierarchicalEdgeReference":
        return cls(
            session_id=int(state["session_id"]),
            representatives_per_class=int(state["representatives_per_class"]),
            representative_indices=tuple(
                int(value) for value in state["representative_indices"]
            ),
            representative_class_ids=tuple(
                int(value) for value in state["representative_class_ids"]
            ),
            reference_edges=state["reference_edges"],
            edge_weights=state["edge_weights"],
            edge_weighting=str(state["edge_weighting"]),
            conflict_node_ids=tuple(
                str(value) for value in state.get("conflict_node_ids", ())
            ),
        )


@dataclass
class StratifiedHierarchicalEdgeReference:
    """Serializable fixed hierarchy partition for stratified edge Pearson."""

    session_id: int
    representatives_per_class: int
    representative_indices: tuple[int, ...]
    representative_class_ids: tuple[int, ...]
    reference_edges: Tensor
    edge_group_ids: Tensor
    beta_boundary: float = 1.0
    gamma_conflict: float = 0.1
    conflict_node_ids: tuple[str, ...] = ()
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        count = len(self.representative_indices)
        if count < 2 or len(self.representative_class_ids) != count:
            raise ValueError("invalid representative metadata")
        if self.representatives_per_class <= 0:
            raise ValueError("representatives_per_class must be positive")
        expected_edges = count * (count - 1) // 2
        if self.reference_edges.shape != (expected_edges,):
            raise ValueError("reference edge count does not match representatives")
        if self.edge_group_ids.shape != (expected_edges,):
            raise ValueError("edge group count does not match representatives")
        if not torch.isfinite(self.reference_edges).all():
            raise ValueError("reference edges must be finite")
        if self.edge_group_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("edge group IDs must be integral")
        if bool((self.edge_group_ids < 0).any()) or bool(
            (self.edge_group_ids > 2).any()
        ):
            raise ValueError("invalid stratified edge group")
        if self.conflict_node_ids != tuple(sorted(set(self.conflict_node_ids))):
            raise ValueError("conflict node IDs must be sorted and unique")
        coefficients = torch.tensor(
            [float(self.beta_boundary), float(self.gamma_conflict)]
        )
        if (
            not torch.isfinite(coefficients).all()
            or bool((coefficients < 0).any())
        ):
            raise ValueError("stratified coefficients must be finite and non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def representative_count(self) -> int:
        return len(self.representative_indices)

    @property
    def edge_count(self) -> int:
        return int(self.reference_edges.numel())

    @property
    def group_counts(self) -> dict[str, int]:
        return {
            name: int((self.edge_group_ids == group_id).sum())
            for group_id, name in enumerate(STRATIFIED_EDGE_GROUP_NAMES)
        }

    def loss_module(self) -> StratifiedHierarchicalEdgeCorrelationLoss:
        return StratifiedHierarchicalEdgeCorrelationLoss(
            self.reference_edges,
            self.edge_group_ids,
            beta_boundary=self.beta_boundary,
            gamma_conflict=self.gamma_conflict,
            epsilon=self.epsilon,
        )

    def state_dict(self) -> dict:
        return {
            "reference_type": "stratified_hierarchical_edge_correlation",
            "session_id": int(self.session_id),
            "representatives_per_class": int(self.representatives_per_class),
            "representative_indices": list(self.representative_indices),
            "representative_class_ids": list(self.representative_class_ids),
            "reference_edges": self.reference_edges.detach().cpu().clone(),
            "edge_group_ids": self.edge_group_ids.detach().cpu().clone(),
            "beta_boundary": float(self.beta_boundary),
            "gamma_conflict": float(self.gamma_conflict),
            "conflict_node_ids": list(self.conflict_node_ids),
            "epsilon": float(self.epsilon),
            "group_counts": self.group_counts,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping
    ) -> "StratifiedHierarchicalEdgeReference":
        return cls(
            session_id=int(state["session_id"]),
            representatives_per_class=int(state["representatives_per_class"]),
            representative_indices=tuple(
                int(value) for value in state["representative_indices"]
            ),
            representative_class_ids=tuple(
                int(value) for value in state["representative_class_ids"]
            ),
            reference_edges=state["reference_edges"],
            edge_group_ids=state["edge_group_ids"],
            beta_boundary=float(state.get("beta_boundary", 1.0)),
            gamma_conflict=float(state.get("gamma_conflict", 0.1)),
            conflict_node_ids=tuple(
                str(value) for value in state.get("conflict_node_ids", ())
            ),
            epsilon=float(state.get("epsilon", 1e-12)),
        )
