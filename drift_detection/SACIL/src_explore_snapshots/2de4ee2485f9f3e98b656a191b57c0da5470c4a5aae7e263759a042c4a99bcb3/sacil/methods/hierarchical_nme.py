from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank
from sacil.hierarchy import GriffinPeronaGreedy, cosine_soft_confusion
from sacil.hierarchy.soft_confusion import symmetric_affinity


@dataclass(frozen=True)
class HierarchicalNMEShrinkageDiagnostics:
    shrinkage: tuple[float, ...]
    within_dispersion: tuple[float, ...]
    mean_variance: tuple[float, ...]
    parent_distance: tuple[float, ...]
    parent_node_ids: tuple[str, ...]
    max_shrinkage: float

    def to_dict(self) -> dict:
        return asdict(self)


def hierarchical_shrink_nme_means(
    class_means: Tensor,
    memory_features: Tensor,
    memory_targets: Tensor,
    class_ids: Sequence[int],
    *,
    taxonomy_temperature: float = 0.2,
    max_shrinkage: float = 0.35,
    epsilon: float = 1.0e-8,
) -> tuple[Tensor, HierarchicalNMEShrinkageDiagnostics, dict]:
    """Empirical-Bayes shrinkage of NME means toward their learned parent.

    The class means, hierarchy, parent anchors, and uncertainty are all
    recomputed in the current feature frame.  Consequently this is a post-hoc
    co-moving estimator; no old absolute feature coordinate is used.

    For class ``c``, the uncertainty of its finite-memory mean is estimated as
    within-class angular dispersion divided by its exemplar count.  The prior
    variance is the angular distance from the empirical class mean to its
    immediate parent anchor.  Their precision ratio gives a data-dependent
    shrinkage coefficient, capped to avoid collapsing sibling prototypes.
    """

    if class_means.ndim != 2 or memory_features.ndim != 2:
        raise ValueError("class means and memory features must be matrices")
    if memory_features.shape[0] != memory_targets.numel():
        raise ValueError("memory feature and target counts do not match")
    if class_means.shape[1] != memory_features.shape[1]:
        raise ValueError("feature dimensions do not match")
    if class_means.shape[0] != len(class_ids):
        raise ValueError("class ID and mean counts do not match")
    if not 0.0 <= float(max_shrinkage) <= 1.0:
        raise ValueError("max_shrinkage must be in [0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    means = F.normalize(class_means.detach().float(), dim=1).cpu()
    features = F.normalize(memory_features.detach().float(), dim=1).cpu()
    targets = memory_targets.detach().long().cpu()
    class_ids = tuple(int(value) for value in class_ids)
    expected_targets = set(range(len(class_ids)))
    observed_targets = set(int(value) for value in targets.unique().tolist())
    if observed_targets != expected_targets:
        raise ValueError(
            "memory targets must cover contiguous incremental labels"
        )

    confusion = cosine_soft_confusion(
        features,
        targets,
        means,
        temperature=float(taxonomy_temperature),
    )
    tree = GriffinPeronaGreedy().build(
        class_ids, symmetric_affinity(confusion)
    )
    anchors = HierarchicalAnchorBank.from_tree(
        PrototypeBank(class_ids, means), tree
    )
    internal_position = {
        node_id: position
        for position, node_id in enumerate(anchors.internal_node_ids)
    }

    shrunken: list[Tensor] = []
    shrinkage: list[float] = []
    within_values: list[float] = []
    mean_variances: list[float] = []
    parent_distances: list[float] = []
    parent_ids: list[str] = []
    for position, class_id in enumerate(class_ids):
        mask = targets == position
        count = int(mask.sum().item())
        if count <= 0:
            raise ValueError(f"class has no memory features: {class_id}")
        mean = means[position]
        within = (1.0 - features[mask] @ mean).clamp_min(0.0).mean()
        mean_variance = within / float(count)
        parent_id = tree.parent(tree.leaf_node_id(class_id))
        if parent_id is None:
            parent = mean
        else:
            parent = anchors.internal_anchors[internal_position[parent_id]]
        parent_distance = (1.0 - torch.dot(mean, parent)).clamp_min(0.0)
        raw = mean_variance / (mean_variance + parent_distance + epsilon)
        alpha = min(float(max_shrinkage), max(0.0, float(raw.item())))
        shrunken.append(
            F.normalize(
                ((1.0 - alpha) * mean + alpha * parent).unsqueeze(0),
                dim=1,
            )[0]
        )
        shrinkage.append(alpha)
        within_values.append(float(within.item()))
        mean_variances.append(float(mean_variance.item()))
        parent_distances.append(float(parent_distance.item()))
        parent_ids.append("" if parent_id is None else parent_id)

    diagnostics = HierarchicalNMEShrinkageDiagnostics(
        shrinkage=tuple(shrinkage),
        within_dispersion=tuple(within_values),
        mean_variance=tuple(mean_variances),
        parent_distance=tuple(parent_distances),
        parent_node_ids=tuple(parent_ids),
        max_shrinkage=float(max_shrinkage),
    )
    return torch.stack(shrunken, dim=0), diagnostics, tree.state_dict()
