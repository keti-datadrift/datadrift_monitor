from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sacil.anchors.affinity import anchor_affinity
from sacil.anchors.hierarchical_anchor_bank import HierarchicalAnchorBank
from sacil.hierarchy.tree import HierarchyTree


ANCHOR_FRAME_MODES = frozenset({"fixed", "co_moving", "hybrid"})
ANCHOR_RELATION_OBJECTIVES = frozenset(
    {"mse", "correlation", "triplet_rank"}
)
ANCHOR_WEIGHT_NORMALIZATIONS = frozenset({"weight_sum", "anchor_count"})


def weighted_anchor_correlation_loss(
    current_affinity: Tensor,
    reference_affinity: Tensor,
    weights: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """Preserve each sample's anchor-relation pattern up to affine scale.

    This is the anchor-relative analogue of TPCIL's correlation constraint.
    Unlike element-wise MSE, it permits a common shift or positive rescaling
    of affinities while penalizing changes in their centered relation pattern.
    """

    if current_affinity.shape != reference_affinity.shape:
        raise ValueError("current and reference affinities differ")
    if current_affinity.ndim != 2:
        raise ValueError("affinities must have shape [N, A]")
    if weights.ndim != 1 or weights.numel() != current_affinity.shape[1]:
        raise ValueError("anchor weight count mismatch")
    if current_affinity.shape[1] < 2:
        return current_affinity.sum() * 0.0

    normalized_weights = weights.to(current_affinity)
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(
        epsilon
    )
    normalized_weights = normalized_weights.unsqueeze(0)
    reference = reference_affinity.detach().to(current_affinity)
    current_mean = (current_affinity * normalized_weights).sum(dim=1, keepdim=True)
    reference_mean = (reference * normalized_weights).sum(dim=1, keepdim=True)
    current_centered = current_affinity - current_mean
    reference_centered = reference - reference_mean
    covariance = (
        normalized_weights * current_centered * reference_centered
    ).sum(dim=1)
    current_variance = (
        normalized_weights * current_centered.square()
    ).sum(dim=1)
    reference_variance = (
        normalized_weights * reference_centered.square()
    ).sum(dim=1)
    denominator = (current_variance * reference_variance).clamp_min(0.0).sqrt()
    valid_reference = reference_variance > epsilon
    valid_current = current_variance > epsilon
    valid = valid_reference & valid_current
    correlation = torch.zeros_like(covariance)
    correlation = torch.where(
        valid,
        covariance / denominator.clamp_min(epsilon),
        correlation,
    ).clamp(-1.0, 1.0)
    per_sample = 1.0 - correlation
    # A constant teacher relation contains no ordering information to retain.
    per_sample = torch.where(
        valid_reference, per_sample, torch.zeros_like(per_sample)
    )
    return per_sample.mean()


def hierarchical_triplet_rank_loss(
    current_affinity: Tensor,
    reference_affinity: Tensor,
    weights: Tensor,
    *,
    margin_scale: float = 1.0,
    rank_tolerance: float = 1e-4,
    weight_normalization: str = "weight_sum",
    epsilon: float = 1e-12,
) -> Tensor:
    """Preserve teacher anchor orderings with a TCP-style triplet hinge.

    For every teacher relation ``a_pos > a_neg``, the student is required to
    retain the corresponding affinity margin.  Local SACIL relaxation enters
    through the geometric mean of the two involved anchor weights.
    """

    if current_affinity.shape != reference_affinity.shape:
        raise ValueError("current and reference affinities differ")
    if current_affinity.ndim != 2:
        raise ValueError("affinities must have shape [N, A]")
    if weights.ndim != 1 or weights.numel() != current_affinity.shape[1]:
        raise ValueError("anchor weight count mismatch")
    if not 0.0 < float(margin_scale) <= 1.0:
        raise ValueError("margin_scale must be in (0, 1]")
    if rank_tolerance < 0:
        raise ValueError("rank_tolerance must be non-negative")
    weight_normalization = str(weight_normalization).lower().replace(
        "-", "_"
    )
    weight_normalization = {
        "relative": "weight_sum",
        "absolute": "anchor_count",
    }.get(weight_normalization, weight_normalization)
    if weight_normalization not in ANCHOR_WEIGHT_NORMALIZATIONS:
        raise ValueError("invalid triplet weight normalization")
    if current_affinity.shape[1] < 2:
        return current_affinity.sum() * 0.0

    reference = reference_affinity.detach().to(current_affinity)
    reference_margin = reference.unsqueeze(2) - reference.unsqueeze(1)
    current_margin = current_affinity.unsqueeze(2) - current_affinity.unsqueeze(1)
    valid = reference_margin > float(rank_tolerance)
    anchor_weights = weights.to(current_affinity).clamp_min(0.0)
    pair_weights = torch.sqrt(
        anchor_weights.unsqueeze(1) * anchor_weights.unsqueeze(0)
    ).unsqueeze(0)
    active_weights = pair_weights * valid.to(pair_weights)
    hinge = F.relu(float(margin_scale) * reference_margin - current_margin)
    denominator = (
        active_weights.sum()
        if weight_normalization == "weight_sum"
        else valid.sum().to(active_weights)
    ).clamp_min(epsilon)
    return (hinge * active_weights).sum() / denominator


def inverse_angular_dispersion_reliability(
    features: Tensor,
    original_targets: Tensor,
    anchor_bank: HierarchicalAnchorBank,
    tree: HierarchyTree,
    *,
    power: float = 1.0,
    epsilon: float = 1e-4,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Return TOPIC-inspired reliability for leaf and internal anchors.

    TOPIC relaxes uncertain feature dimensions with inverse variance.  SACIL's
    geometry is expressed as scalar cosine affinity per anchor, so the faithful
    analogue is an inverse *angular dispersion* per anchor.  Compact anchors
    receive more relative weight; broad/noisy anchors receive less.  The
    weights are normalized independently within the leaf and internal groups.
    """

    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if features.shape[0] != original_targets.numel():
        raise ValueError("feature and target counts do not match")
    if features.shape[1] != anchor_bank.feature_dim:
        raise ValueError("feature and anchor dimensions do not match")
    if tree.class_order != anchor_bank.leaf_class_ids:
        raise ValueError("tree and anchor leaf orders do not match")
    if power < 0:
        raise ValueError("reliability power must be non-negative")
    if epsilon <= 0:
        raise ValueError("reliability epsilon must be positive")
    if not 0 < min_weight <= max_weight:
        raise ValueError("invalid reliability weight bounds")

    normalized = F.normalize(features.float(), dim=1)
    targets = original_targets.long().to(normalized.device)

    def _group_reliability(
        anchors: Tensor, class_groups: tuple[tuple[int, ...], ...]
    ) -> Tensor:
        if anchors.shape[0] == 0:
            return torch.empty(0, dtype=normalized.dtype)
        anchors = F.normalize(anchors.to(normalized), dim=1)
        dispersions = []
        for anchor, class_ids in zip(anchors, class_groups, strict=True):
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for class_id in class_ids:
                mask |= targets.eq(int(class_id))
            if not bool(mask.any()):
                raise ValueError(
                    f"no features found for anchor classes {class_ids}"
                )
            cosine = normalized[mask] @ anchor
            dispersions.append((1.0 - cosine).clamp_min(0.0).mean())
        dispersion = torch.stack(dispersions)
        reliability = (dispersion + float(epsilon)).pow(-float(power))
        reliability = reliability / reliability.mean().clamp_min(epsilon)
        return reliability.clamp(
            min=float(min_weight), max=float(max_weight)
        ).detach().cpu()

    leaf_groups = tuple((class_id,) for class_id in anchor_bank.leaf_class_ids)
    leaf_reliability = _group_reliability(
        anchor_bank.leaf_anchors, leaf_groups
    )
    internal_ids, internal_anchors = anchor_bank.internal_without_root()
    internal_groups = tuple(
        tuple(tree.descendants(node_id)) for node_id in internal_ids
    )
    internal_reliability = _group_reliability(
        internal_anchors, internal_groups
    )
    return leaf_reliability, internal_reliability


class AnchorGeometryLoss(nn.Module):
    """Group-normalized anchor-relation preservation.

    ``fixed`` exactly preserves the original SACIL implementation.  In
    ``co_moving`` mode, current features are compared with anchors rebuilt in
    the current model while reference features use the frozen old anchors.
    ``hybrid`` is a convex mixture of both errors.
    """

    def __init__(
        self,
        anchor_bank: HierarchicalAnchorBank,
        leaf_weights: Tensor,
        internal_weights: Tensor,
        use_internal_anchors: bool = True,
        anchor_frame: str = "fixed",
        fixed_mix: float = 0.5,
        objective: str = "mse",
        weight_normalization: str = "weight_sum",
        triplet_margin_scale: float = 1.0,
        rank_tolerance: float = 1e-4,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if leaf_weights.numel() != anchor_bank.leaf_anchors.shape[0]:
            raise ValueError("leaf weight count mismatch")
        internal_ids, internal_anchors = anchor_bank.internal_without_root()
        if internal_weights.numel() != len(internal_ids):
            raise ValueError("internal weight count mismatch")
        self.leaf_class_ids = anchor_bank.leaf_class_ids
        self.internal_node_ids = internal_ids
        self.register_buffer(
            "leaf_anchors", anchor_bank.leaf_anchors.clone()
        )
        self.register_buffer(
            "current_leaf_anchors", anchor_bank.leaf_anchors.clone()
        )
        self.register_buffer("leaf_weights", leaf_weights.float().clone())
        self.register_buffer("internal_anchors", internal_anchors.clone())
        self.register_buffer(
            "current_internal_anchors", internal_anchors.clone()
        )
        self.register_buffer(
            "internal_weights", internal_weights.float().clone()
        )
        self.use_internal_anchors = bool(use_internal_anchors)
        frame = str(anchor_frame).lower().replace("-", "_")
        if frame not in ANCHOR_FRAME_MODES:
            raise ValueError(
                f"unknown anchor frame {anchor_frame!r}; expected one of "
                f"{sorted(ANCHOR_FRAME_MODES)}"
            )
        if not 0.0 <= float(fixed_mix) <= 1.0:
            raise ValueError("fixed_mix must be in [0, 1]")
        self.anchor_frame = frame
        self.fixed_mix = (
            1.0
            if frame == "fixed"
            else 0.0
            if frame == "co_moving"
            else float(fixed_mix)
        )
        objective = str(objective).lower().replace("-", "_")
        objective = {
            "pearson": "correlation",
            "rank": "triplet_rank",
            "triplet": "triplet_rank",
        }.get(objective, objective)
        if objective not in ANCHOR_RELATION_OBJECTIVES:
            raise ValueError(
                f"unknown anchor relation objective {objective!r}; expected "
                f"one of {sorted(ANCHOR_RELATION_OBJECTIVES)}"
            )
        if not 0.0 < float(triplet_margin_scale) <= 1.0:
            raise ValueError("triplet_margin_scale must be in (0, 1]")
        if float(rank_tolerance) < 0.0:
            raise ValueError("rank_tolerance must be non-negative")
        self.objective = objective
        weight_normalization = str(weight_normalization).lower().replace(
            "-", "_"
        )
        weight_normalization = {
            "relative": "weight_sum",
            "absolute": "anchor_count",
        }.get(weight_normalization, weight_normalization)
        if weight_normalization not in ANCHOR_WEIGHT_NORMALIZATIONS:
            raise ValueError(
                "unknown anchor weight normalization "
                f"{weight_normalization!r}; expected one of "
                f"{sorted(ANCHOR_WEIGHT_NORMALIZATIONS)}"
            )
        self.weight_normalization = weight_normalization
        self.triplet_margin_scale = float(triplet_margin_scale)
        self.rank_tolerance = float(rank_tolerance)
        self.epsilon = float(epsilon)

    @property
    def requires_current_anchor_refresh(self) -> bool:
        return self.anchor_frame in {"co_moving", "hybrid"}

    @torch.no_grad()
    def update_current_anchors(
        self, anchor_bank: HierarchicalAnchorBank
    ) -> None:
        """Replace only the current coordinate frame, keeping IDs fixed."""

        if anchor_bank.leaf_class_ids != self.leaf_class_ids:
            raise ValueError("current leaf anchor ID/order mismatch")
        if anchor_bank.leaf_anchors.shape != self.leaf_anchors.shape:
            raise ValueError("current leaf anchor shape mismatch")
        internal_ids, internal_anchors = anchor_bank.internal_without_root()
        if internal_ids != self.internal_node_ids:
            raise ValueError("current internal anchor ID/order mismatch")
        if internal_anchors.shape != self.internal_anchors.shape:
            raise ValueError("current internal anchor shape mismatch")
        self.current_leaf_anchors.copy_(
            anchor_bank.leaf_anchors.to(self.current_leaf_anchors)
        )
        self.current_internal_anchors.copy_(
            internal_anchors.to(self.current_internal_anchors)
        )

    def _squared_error(
        self,
        current_features: Tensor,
        reference_features: Tensor,
        reference_anchors: Tensor,
        current_anchors: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            reference = anchor_affinity(reference_features, reference_anchors)
        fixed_error = (
            anchor_affinity(current_features, reference_anchors) - reference
        ).pow(2)
        if self.fixed_mix == 1.0:
            return fixed_error
        moving_error = (
            anchor_affinity(current_features, current_anchors) - reference
        ).pow(2)
        if self.fixed_mix == 0.0:
            return moving_error
        return self.fixed_mix * fixed_error + (
            1.0 - self.fixed_mix
        ) * moving_error

    def _group_loss(
        self,
        current_features: Tensor,
        reference_features: Tensor,
        reference_anchors: Tensor,
        current_anchors: Tensor,
        weights: Tensor,
    ) -> Tensor | None:
        if reference_anchors.shape[0] == 0:
            return None
        if self.objective == "mse":
            squared_error = self._squared_error(
                current_features,
                reference_features,
                reference_anchors,
                current_anchors,
            )
            # ``weight_sum`` is the legacy relative reweighting: lowering a
            # conflict weight redistributes the same total regularization to
            # unaffected anchors.  ``anchor_count`` implements absolute local
            # relaxation from the SACIL objective: a weight below one reduces
            # that anchor's constraint without amplifying the others.
            anchor_denominator = (
                weights.sum().clamp_min(self.epsilon)
                if self.weight_normalization == "weight_sum"
                else weights.new_tensor(weights.numel()).clamp_min(
                    self.epsilon
                )
            )
            denominator = current_features.shape[0] * anchor_denominator
            return (squared_error * weights.unsqueeze(0)).sum() / denominator

        with torch.no_grad():
            reference = anchor_affinity(reference_features, reference_anchors)

        def relation_loss(anchors: Tensor) -> Tensor:
            current = anchor_affinity(current_features, anchors)
            if self.objective == "correlation":
                return weighted_anchor_correlation_loss(
                    current,
                    reference,
                    weights,
                    epsilon=self.epsilon,
                )
            return hierarchical_triplet_rank_loss(
                current,
                reference,
                weights,
                margin_scale=self.triplet_margin_scale,
                rank_tolerance=self.rank_tolerance,
                weight_normalization=self.weight_normalization,
                epsilon=self.epsilon,
            )

        # Correlation/ranking objectives internally form weighted means.  In
        # absolute mode retain their relation statistic but reduce its group
        # strength by the mean preservation weight, matching the MSE mode's
        # no-amplification contract.
        relation_scale = (
            weights.mean()
            if self.weight_normalization == "anchor_count"
            and self.objective == "correlation"
            else weights.new_ones(())
        )
        fixed_loss = relation_scale * relation_loss(reference_anchors)
        if self.fixed_mix == 1.0:
            return fixed_loss
        moving_loss = relation_scale * relation_loss(current_anchors)
        if self.fixed_mix == 0.0:
            return moving_loss
        return self.fixed_mix * fixed_loss + (
            1.0 - self.fixed_mix
        ) * moving_loss

    def forward(
        self, current_features: Tensor, reference_features: Tensor
    ) -> Tensor:
        if current_features.shape != reference_features.shape:
            raise ValueError("current and reference feature shapes differ")
        if current_features.shape[0] == 0:
            return current_features.sum() * 0.0
        losses = []
        leaf_loss = self._group_loss(
            current_features,
            reference_features,
            self.leaf_anchors,
            self.current_leaf_anchors,
            self.leaf_weights,
        )
        if leaf_loss is not None:
            losses.append(leaf_loss)
        if self.use_internal_anchors:
            internal_loss = self._group_loss(
                current_features,
                reference_features,
                self.internal_anchors,
                self.current_internal_anchors,
                self.internal_weights,
            )
            if internal_loss is not None:
                losses.append(internal_loss)
        if not losses:
            return current_features.sum() * 0.0
        return torch.stack(losses).mean()

    @torch.no_grad()
    def per_anchor_drift(
        self, current_features: Tensor, reference_features: Tensor
    ) -> dict[str, Tensor]:
        result = {
            "leaf": self._squared_error(
                current_features,
                reference_features,
                self.leaf_anchors,
                self.current_leaf_anchors,
            ).mean(dim=0)
        }
        if self.internal_anchors.shape[0] > 0:
            result["internal"] = self._squared_error(
                current_features,
                reference_features,
                self.internal_anchors,
                self.current_internal_anchors,
            ).mean(dim=0)
        else:
            result["internal"] = torch.empty(
                0, device=current_features.device
            )
        return result
