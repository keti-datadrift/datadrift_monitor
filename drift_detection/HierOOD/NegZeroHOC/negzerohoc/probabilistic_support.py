from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_support import HierarchicalSupportCalibration, conformal_p_values


def validate_reference_only_split(
    targets: torch.Tensor,
    reference_indices,
    calibration_indices,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate that a saved reference/calibration split partitions training data."""
    targets = targets.detach().long().cpu()
    reference = torch.as_tensor(reference_indices, dtype=torch.long).flatten()
    calibration = torch.as_tensor(
        calibration_indices, dtype=torch.long
    ).flatten()
    sample_count = int(targets.numel())
    if sample_count == 0:
        raise ValueError("Reference-only split cannot validate an empty dataset")
    if int(reference.numel()) == 0 or int(calibration.numel()) == 0:
        raise ValueError("Reference and calibration partitions must be non-empty")
    combined = torch.cat([reference, calibration])
    if bool(((combined < 0) | (combined >= sample_count)).any()):
        raise ValueError("Reference-only split contains an out-of-range index")
    if int(torch.unique(reference).numel()) != int(reference.numel()):
        raise ValueError("Reference partition contains duplicate indices")
    if int(torch.unique(calibration).numel()) != int(calibration.numel()):
        raise ValueError("Calibration partition contains duplicate indices")
    if int(torch.unique(combined).numel()) != int(combined.numel()):
        raise ValueError("Reference and calibration partitions overlap")
    if not torch.equal(
        combined.sort().values,
        torch.arange(sample_count, dtype=torch.long),
    ):
        raise ValueError(
            "Reference and calibration partitions do not cover the dataset"
        )
    for target in sorted(set(targets.tolist())):
        if not bool((targets[reference] == int(target)).any()):
            raise ValueError(f"Reference partition omits class {target}")
        if not bool((targets[calibration] == int(target)).any()):
            raise ValueError(f"Calibration partition omits class {target}")
    return reference.sort().values, calibration.sort().values


def stratified_calibration_train_val_split(
    targets: torch.Tensor,
    calibration_indices: torch.Tensor,
    *,
    train_per_class: int = 8,
    val_per_class: int = 5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split only the saved calibration partition into fixed per-class counts."""
    if int(train_per_class) <= 0 or int(val_per_class) <= 0:
        raise ValueError("Per-class posterior split counts must be positive")
    targets = targets.detach().long().cpu()
    calibration = torch.as_tensor(
        calibration_indices, dtype=torch.long
    ).flatten()
    generator = torch.Generator().manual_seed(int(seed))
    train_indices = []
    val_indices = []
    required = int(train_per_class) + int(val_per_class)
    for target in sorted(set(targets.tolist())):
        class_indices = calibration[
            targets.index_select(0, calibration) == int(target)
        ]
        if int(class_indices.numel()) != required:
            raise ValueError(
                f"Calibration class {target} has {int(class_indices.numel())} "
                f"samples; exactly {required} are required for "
                f"{train_per_class}/{val_per_class}"
            )
        order = torch.randperm(
            int(class_indices.numel()), generator=generator
        )
        shuffled = class_indices.index_select(0, order)
        train_indices.extend(shuffled[:train_per_class].tolist())
        val_indices.extend(shuffled[train_per_class:].tolist())
    train = torch.tensor(sorted(train_indices), dtype=torch.long)
    val = torch.tensor(sorted(val_indices), dtype=torch.long)
    if not torch.equal(
        torch.cat([train, val]).sort().values,
        calibration.sort().values,
    ):
        raise RuntimeError("Posterior train/validation split changed calibration")
    return train, val


def _checkpoint_lineage(checkpoint: dict) -> dict:
    args = checkpoint.get("args") or {}
    return {
        "stage": checkpoint.get("stage"),
        "dataset": checkpoint.get("dataset"),
        "clip_model": checkpoint.get("clip_model"),
        "hierarchy": checkpoint.get("hierarchy"),
        "id_split": checkpoint.get("id_split"),
        "experiment_name": args.get("experiment_name"),
        "seed": args.get("seed"),
        "reference_only_training": args.get(
            "reference_only_training"
        ),
        "metric_proxy_classes": tuple(
            checkpoint.get("metric_proxy_classes") or ()
        ),
    }


def reference_only_partitions_from_checkpoints(
    support_checkpoint: dict,
    targets: torch.Tensor,
    *,
    expected_seed: int,
    metadata_checkpoint: dict | None = None,
    posterior_train_per_class: int = 8,
    posterior_val_per_class: int = 5,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Recover the original reference-only lineage and derive posterior splits."""
    support_lineage = _checkpoint_lineage(support_checkpoint)
    metadata_lineage = None
    if metadata_checkpoint is not None:
        metadata_lineage = _checkpoint_lineage(metadata_checkpoint)
        mismatches = {
            key: (support_lineage[key], metadata_lineage[key])
            for key in support_lineage
            if support_lineage[key] != metadata_lineage[key]
        }
        if mismatches:
            raise ValueError(
                "Split metadata checkpoint is from a different run: "
                f"{mismatches}"
            )
    support_split = (
        support_checkpoint.get("metrics") or {}
    ).get("training_split")
    metadata_split = None
    if metadata_checkpoint is not None:
        metadata_split = (
            metadata_checkpoint.get("metrics") or {}
        ).get("training_split")
    if support_split is not None and metadata_split is not None:
        if support_split != metadata_split:
            raise ValueError(
                "Support and metadata checkpoints disagree on training_split"
            )
    split = support_split or metadata_split
    if split is None:
        raise ValueError(
            "No training_split metadata exists; provide the finalized best "
            "checkpoint through split_metadata_checkpoint"
        )
    if not bool(split.get("reference_only")):
        raise ValueError("Support checkpoint was not trained reference-only")
    if int(split.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            f"Training split seed {split.get('seed')} differs from "
            f"runtime seed {expected_seed}"
        )
    if support_lineage["seed"] is not None and int(
        support_lineage["seed"]
    ) != int(expected_seed):
        raise ValueError(
            f"Support checkpoint seed {support_lineage['seed']} differs from "
            f"runtime seed {expected_seed}"
        )
    reference, original_calibration = validate_reference_only_split(
        targets,
        split.get("reference_indices", []),
        split.get("calibration_indices", []),
    )
    if int(split.get("reference_samples", -1)) != int(reference.numel()):
        raise ValueError("Saved reference sample count is inconsistent")
    if int(split.get("calibration_samples", -1)) != int(
        original_calibration.numel()
    ):
        raise ValueError("Saved calibration sample count is inconsistent")
    posterior_train, posterior_val = (
        stratified_calibration_train_val_split(
            targets,
            original_calibration,
            train_per_class=posterior_train_per_class,
            val_per_class=posterior_val_per_class,
            seed=expected_seed,
        )
    )
    partitions = {
        "reference": reference,
        "original_calibration": original_calibration,
        "posterior_train": posterior_train,
        "posterior_val": posterior_val,
    }
    lineage = {
        "support": support_lineage,
        "metadata": metadata_lineage,
        "training_split_source": (
            "support_checkpoint"
            if support_split is not None
            else "split_metadata_checkpoint"
        ),
        "reference_fraction": float(split.get("reference_fraction")),
        "posterior_train_per_class": int(posterior_train_per_class),
        "posterior_val_per_class": int(posterior_val_per_class),
    }
    return partitions, lineage


def stratified_four_way_split(
    targets: torch.Tensor,
    *,
    fractions: tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(fractions) != 4 or any(float(value) <= 0.0 for value in fractions):
        raise ValueError("Four positive split fractions are required")
    total = sum(float(value) for value in fractions)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to one")
    targets = targets.detach().long().cpu()
    generator = torch.Generator().manual_seed(int(seed))
    partitions = [[] for _ in range(4)]
    for target in sorted(set(targets.tolist())):
        indices = torch.nonzero(targets == int(target), as_tuple=False).flatten()
        if int(indices.numel()) < 8:
            raise ValueError(
                f"Class {target} needs at least eight samples for a four-way split"
            )
        indices = indices[torch.randperm(int(indices.numel()), generator=generator)]
        count = int(indices.numel())
        raw = [float(value) * count for value in fractions]
        counts = [int(value) for value in raw]
        remainder = count - sum(counts)
        order = sorted(
            range(4),
            key=lambda index: (raw[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        if any(value == 0 for value in counts):
            raise RuntimeError(f"Four-way split made an empty class partition for {target}")
        offset = 0
        for partition, partition_count in zip(partitions, counts):
            partition.extend(
                indices[offset:offset + partition_count].tolist()
            )
            offset += partition_count
    return tuple(
        torch.tensor(sorted(partition), dtype=torch.long)
        for partition in partitions
    )


def candidate_support_p_values(
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
    candidate_nodes: list[str],
) -> torch.Tensor:
    if not candidate_nodes:
        raise ValueError("At least one candidate node is required")
    query = F.normalize(features.detach().float().cpu(), dim=-1)
    columns = []
    for node in candidate_nodes:
        if node not in calibration.node_prototype_indices:
            raise KeyError(f"No support bank exists for candidate {node!r}")
        prototypes = calibration.prototypes.index_select(
            0, calibration.node_prototype_indices[node]
        )
        scores = (query @ prototypes.t()).max(dim=1).values
        columns.append(conformal_p_values(
            scores, calibration.node_calibration_scores[node]
        ))
    return torch.stack(columns, dim=1)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.detach().float().cpu()
    if int(probabilities.shape[1]) <= 1:
        return torch.zeros(int(probabilities.shape[0]), dtype=torch.float32)
    safe = probabilities.clamp_min(1e-12)
    entropy = -(safe * safe.log()).sum(dim=1)
    return entropy / torch.log(
        torch.tensor(float(probabilities.shape[1]))
    )


def support_evidence(
    child_support_p_values: torch.Tensor,
    child_route_probabilities: torch.Tensor,
) -> torch.Tensor:
    if child_support_p_values.shape != child_route_probabilities.shape:
        raise ValueError("Support and route tensors must have the same shape")
    max_support = child_support_p_values.float().max(dim=1).values
    nonconformity = -max_support.clamp_min(1e-8).log()
    entropy = normalized_entropy(child_route_probabilities)
    return torch.stack([nonconformity, entropy], dim=1)


class SharedSupportPosterior(nn.Module):
    """A global monotone support-to-unknown posterior with no node threshold."""

    def __init__(self, *, use_entropy: bool = True):
        super().__init__()
        self.use_entropy = bool(use_entropy)
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.raw_support_weight = nn.Parameter(torch.tensor(0.0))
        if self.use_entropy:
            self.raw_entropy_weight = nn.Parameter(torch.tensor(-1.0))
        else:
            self.register_parameter("raw_entropy_weight", None)

    def weights(self) -> dict[str, torch.Tensor]:
        result = {
            "support": F.softplus(self.raw_support_weight),
        }
        if self.use_entropy:
            result["entropy"] = F.softplus(self.raw_entropy_weight)
        return result

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 2 or int(evidence.shape[1]) != 2:
            raise ValueError("Support evidence must have shape [N, 2]")
        weights = self.weights()
        logits = self.bias + weights["support"] * evidence[:, 0]
        if self.use_entropy:
            logits = logits + weights["entropy"] * evidence[:, 1]
        return logits


class SharedMaskedSupportLikelihood(nn.Module):
    """Global vMF-mixture support likelihood shared across the whole tree."""

    def __init__(
        self,
        *,
        initial_concentration: float = 5.0,
        initial_base_energy: float = 2.5,
    ):
        super().__init__()
        if float(initial_concentration) <= 0.0:
            raise ValueError("initial_concentration must be positive")
        raw = torch.log(torch.expm1(torch.tensor(float(initial_concentration))))
        self.raw_concentration = nn.Parameter(raw)
        self.base_energy = nn.Parameter(
            torch.tensor(float(initial_base_energy))
        )

    @property
    def concentration(self) -> torch.Tensor:
        return F.softplus(self.raw_concentration).clamp_min(1e-4)

    def child_energies(
        self,
        child_similarities: torch.Tensor,
        prototype_mask: torch.Tensor,
        child_mask: torch.Tensor,
    ) -> torch.Tensor:
        if child_similarities.ndim != 3:
            raise ValueError(
                "child_similarities must have shape [N, children, prototypes]"
            )
        if prototype_mask.shape != child_similarities.shape:
            raise ValueError("prototype_mask shape differs from similarities")
        if child_mask.shape != child_similarities.shape[:2]:
            raise ValueError("child_mask shape differs from similarities")
        if bool((child_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every likelihood row needs a visible child")
        visible_proto_counts = prototype_mask.sum(dim=2)
        if bool(((visible_proto_counts == 0) & child_mask).any()):
            raise ValueError("A visible child has no prototype")
        scaled = self.concentration * child_similarities.float()
        scaled = scaled.masked_fill(~prototype_mask, float("-inf"))
        child_energy = torch.logsumexp(scaled, dim=2) - (
            visible_proto_counts.clamp_min(1).float().log()
        )
        child_energy = child_energy.masked_fill(
            ~child_mask, float("-inf")
        )
        return child_energy

    def known_energy(
        self,
        child_similarities: torch.Tensor,
        prototype_mask: torch.Tensor,
        child_mask: torch.Tensor,
    ) -> torch.Tensor:
        child_energy = self.child_energies(
            child_similarities, prototype_mask, child_mask
        )
        return torch.logsumexp(child_energy, dim=1) - (
            child_mask.sum(dim=1).float().log()
        )

    def forward(
        self,
        child_similarities: torch.Tensor,
        prototype_mask: torch.Tensor,
        child_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.base_energy - self.known_energy(
            child_similarities, prototype_mask, child_mask
        )


class SharedMaskedCategoricalLikelihood(SharedMaskedSupportLikelihood):
    """Global K+1 child/unknown mixture likelihood shared across the tree."""

    def categorical_logits(
        self,
        child_similarities: torch.Tensor,
        prototype_mask: torch.Tensor,
        child_mask: torch.Tensor,
    ) -> torch.Tensor:
        child_energy = self.child_energies(
            child_similarities, prototype_mask, child_mask
        )
        unknown = self.base_energy.expand(
            int(child_energy.shape[0]), 1
        )
        return torch.cat([child_energy, unknown], dim=1)


@dataclass
class EnergyEpisodes:
    child_similarities: torch.Tensor
    prototype_mask: torch.Tensor
    child_mask: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor
    parent_nodes: tuple[str, ...]
    true_children: tuple[str, ...]
    true_child_positions: torch.Tensor
    masked: torch.Tensor


def _episode_group_weights(
    metadata: list[tuple[str, str, bool, int]],
    *,
    weighting: str,
) -> torch.Tensor:
    if weighting not in {"uniform_terminal", "paired_view"}:
        raise ValueError(f"Unsupported episode weighting: {weighting}")
    counts = {}
    for parent, child, masked, _ in metadata:
        key = (parent, child, masked)
        counts[key] = counts.get(key, 0) + 1
    result = []
    for parent, child, masked, arity in metadata:
        if weighting == "paired_view":
            group_mass = 0.5 / float(arity)
        elif masked:
            group_mass = 1.0 / float(arity * (arity + 1))
        else:
            group_mass = 1.0 / float(arity + 1)
        result.append(
            group_mass / float(counts[(parent, child, masked)])
        )
    return torch.tensor(result, dtype=torch.float32)


def build_energy_episodes(
    hierarchy,
    features: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_leaf_nodes: list[str],
    calibration: HierarchicalSupportCalibration,
    *,
    weighting: str = "uniform_terminal",
) -> EnergyEpisodes:
    normalized = F.normalize(features.detach().float().cpu(), dim=-1)
    prototypes = F.normalize(calibration.prototypes.float().cpu(), dim=-1)
    all_similarities = normalized @ prototypes.t()
    records = []
    metadata = []
    for sample_index in sample_indices.detach().long().cpu().tolist():
        leaf = sample_leaf_nodes[sample_index]
        for parent, true_child in true_path_edges(hierarchy, leaf):
            children = list(hierarchy.parent2children[parent])
            true_position = children.index(true_child)
            child_banks = [
                calibration.node_prototype_indices[child]
                for child in children
            ]
            records.append((sample_index, child_banks, None))
            metadata.append((parent, true_child, False, len(children)))
            records.append((sample_index, child_banks, true_position))
            metadata.append((parent, true_child, True, len(children)))
    if not records:
        raise RuntimeError("No non-root energy episodes were constructed")
    max_children = max(len(banks) for _, banks, _ in records)
    max_prototypes = max(
        int(bank.numel())
        for _, banks, _ in records
        for bank in banks
    )
    similarities = torch.zeros(
        len(records), max_children, max_prototypes, dtype=torch.float32
    )
    prototype_mask = torch.zeros_like(similarities, dtype=torch.bool)
    child_mask = torch.zeros(
        len(records), max_children, dtype=torch.bool
    )
    for row, (sample_index, child_banks, hidden_position) in enumerate(records):
        for child_position, bank in enumerate(child_banks):
            count = int(bank.numel())
            similarities[row, child_position, :count] = (
                all_similarities[sample_index].index_select(0, bank)
            )
            prototype_mask[row, child_position, :count] = True
            child_mask[row, child_position] = (
                hidden_position != child_position
            )
    return EnergyEpisodes(
        child_similarities=similarities,
        prototype_mask=prototype_mask,
        child_mask=child_mask,
        targets=torch.tensor(
            [float(masked) for _, _, masked, _ in metadata],
            dtype=torch.float32,
        ),
        weights=_episode_group_weights(metadata, weighting=weighting),
        parent_nodes=tuple(parent for parent, _, _, _ in metadata),
        true_children=tuple(child for _, child, _, _ in metadata),
        true_child_positions=torch.tensor([
            list(hierarchy.parent2children[parent]).index(child)
            for parent, child, _, _ in metadata
        ], dtype=torch.long),
        masked=torch.tensor(
            [masked for _, _, masked, _ in metadata],
            dtype=torch.bool,
        ),
    )


def build_global_leaf_energy_episodes(
    hierarchy,
    features: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_leaf_nodes: list[str],
    calibration: HierarchicalSupportCalibration,
) -> EnergyEpisodes:
    """Build full/leave-one-leaf-out episodes for global knownness."""
    leaf_nodes = sorted(
        set(calibration.prototype_nodes),
        key=hierarchy.id_node_list.index,
    )
    if len(leaf_nodes) == 0:
        raise ValueError("Global knownness needs at least one leaf prototype")
    leaf_to_position = {
        leaf: position for position, leaf in enumerate(leaf_nodes)
    }
    missing = sorted(set(sample_leaf_nodes) - set(leaf_to_position))
    if missing:
        raise ValueError(
            f"Global prototype bank misses sample leaves: {missing[:3]}"
        )
    selected_features = features.detach().float().cpu().index_select(
        0, sample_indices.detach().long().cpu()
    )
    packed, packed_mask, packed_child_mask = (
        parent_child_similarity_tensors(
            selected_features, calibration, leaf_nodes
        )
    )
    sample_count = int(packed.shape[0])
    leaf_count = len(leaf_nodes)
    rows = sample_count * 2
    child_similarities = packed.repeat_interleave(2, dim=0)
    prototype_mask = packed_mask.repeat_interleave(2, dim=0)
    child_mask = packed_child_mask.repeat_interleave(2, dim=0)
    selected_indices = sample_indices.detach().long().cpu().tolist()
    true_positions = torch.tensor([
        leaf_to_position[sample_leaf_nodes[index]]
        for index in selected_indices
    ], dtype=torch.long).repeat_interleave(2)
    masked = torch.zeros(rows, dtype=torch.bool)
    masked[1::2] = True
    child_mask[
        torch.arange(1, rows, 2),
        true_positions[1::2],
    ] = False

    internal_count = sum(
        1
        for parent in hierarchy.parent2children
        if parent != "root"
    )
    if internal_count <= 0:
        raise ValueError("Global knownness needs internal unknown terminals")
    unknown_prior = internal_count / float(leaf_count + internal_count)
    counts = {}
    for position in true_positions[::2].tolist():
        counts[position] = counts.get(position, 0) + 1
    weights = torch.empty(rows, dtype=torch.float32)
    for row in range(0, rows, 2):
        position = int(true_positions[row])
        group_count = float(counts[position])
        weights[row] = (
            (1.0 - unknown_prior) / float(leaf_count) / group_count
        )
        weights[row + 1] = (
            unknown_prior / float(leaf_count) / group_count
        )
    true_children = tuple(
        sample_leaf_nodes[index]
        for index in selected_indices
        for _ in range(2)
    )
    return EnergyEpisodes(
        child_similarities=child_similarities,
        prototype_mask=prototype_mask,
        child_mask=child_mask,
        targets=masked.float(),
        weights=weights,
        parent_nodes=tuple("global" for _ in range(rows)),
        true_children=true_children,
        true_child_positions=true_positions,
        masked=masked,
    )


def parent_child_similarity_tensors(
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
    children: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = F.normalize(features.detach().float().cpu(), dim=-1)
    prototypes = F.normalize(calibration.prototypes.float().cpu(), dim=-1)
    similarities = normalized @ prototypes.t()
    banks = [
        calibration.node_prototype_indices[child] for child in children
    ]
    max_prototypes = max(int(bank.numel()) for bank in banks)
    packed = torch.zeros(
        int(features.shape[0]),
        len(children),
        max_prototypes,
        dtype=torch.float32,
    )
    prototype_mask = torch.zeros_like(packed, dtype=torch.bool)
    for child_index, bank in enumerate(banks):
        count = int(bank.numel())
        packed[:, child_index, :count] = similarities.index_select(1, bank)
        prototype_mask[:, child_index, :count] = True
    child_mask = torch.ones(
        int(features.shape[0]), len(children), dtype=torch.bool
    )
    return packed, prototype_mask, child_mask


@torch.no_grad()
def energy_unknown_probabilities_by_parent(
    model: SharedMaskedSupportLikelihood,
    hierarchy,
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
) -> dict[str, torch.Tensor]:
    model.eval()
    result = {}
    for parent, children in hierarchy.parent2children.items():
        if parent == "root":
            continue
        packed, prototype_mask, child_mask = (
            parent_child_similarity_tensors(
                features, calibration, list(children)
            )
        )
        model_device = next(model.parameters()).device
        result[parent] = torch.sigmoid(
            model(
                packed.to(model_device),
                prototype_mask.to(model_device),
                child_mask.to(model_device),
            )
        ).cpu()
    return result


@dataclass
class SupportEpisodes:
    evidence: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor
    parent_nodes: tuple[str, ...]
    masked: torch.Tensor


def true_path_edges(hierarchy, leaf: str) -> list[tuple[str, str]]:
    path = [
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.node_ancestors.get(leaf, [])
    ] + [leaf]
    return [
        (parent, child)
        for parent, child in zip(path[:-1], path[1:])
        if parent != "root"
    ]


def build_support_episodes(
    hierarchy,
    sample_indices: torch.Tensor,
    sample_leaf_nodes: list[str],
    node_support_p_values: dict[str, torch.Tensor],
    route_conditionals: dict[str, torch.Tensor],
) -> SupportEpisodes:
    evidence_rows = []
    targets = []
    weights = []
    parent_nodes = []
    masked_flags = []
    for sample_index in sample_indices.detach().long().cpu().tolist():
        leaf = sample_leaf_nodes[sample_index]
        for parent, true_child in true_path_edges(hierarchy, leaf):
            children = list(hierarchy.parent2children[parent])
            true_position = children.index(true_child)
            support = torch.stack(
                [node_support_p_values[child][sample_index] for child in children]
            ).unsqueeze(0)
            route = route_conditionals[parent][sample_index].unsqueeze(0)
            evidence_rows.append(support_evidence(support, route)[0])
            targets.append(0.0)
            weights.append(1.0)
            parent_nodes.append(parent)
            masked_flags.append(False)

            visible = [
                index for index in range(len(children))
                if index != true_position
            ]
            if not visible:
                continue
            visible_index = torch.tensor(visible, dtype=torch.long)
            masked_support = support.index_select(1, visible_index)
            masked_route = route.index_select(1, visible_index)
            masked_route = masked_route / masked_route.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            evidence_rows.append(
                support_evidence(masked_support, masked_route)[0]
            )
            targets.append(1.0)
            # One pseudo-unknown is a single additional terminal beside K
            # known children, so its aggregate class-balanced mass is 1/(K+1).
            weights.append(1.0 / float(len(children)))
            parent_nodes.append(parent)
            masked_flags.append(True)
    if not evidence_rows:
        raise RuntimeError("No non-root support episodes were constructed")
    return SupportEpisodes(
        evidence=torch.stack(evidence_rows),
        targets=torch.tensor(targets, dtype=torch.float32),
        weights=torch.tensor(weights, dtype=torch.float32),
        parent_nodes=tuple(parent_nodes),
        masked=torch.tensor(masked_flags, dtype=torch.bool),
    )


def weighted_binary_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    weights = weights.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def categorical_episode_targets(episodes: EnergyEpisodes) -> torch.Tensor:
    unknown_index = int(episodes.child_mask.shape[1])
    return torch.where(
        episodes.masked,
        torch.full_like(episodes.true_child_positions, unknown_index),
        episodes.true_child_positions,
    )


def weighted_categorical_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits, targets.long(), reduction="none"
    )
    weights = weights.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


@torch.no_grad()
def mixture_conditionals_by_parent(
    model: SharedMaskedCategoricalLikelihood,
    hierarchy,
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return coherent prototype routes and non-root unknown probabilities."""
    model.eval()
    model_device = next(model.parameters()).device
    routes = {}
    unknown_probabilities = {}
    for parent, children in hierarchy.parent2children.items():
        packed, prototype_mask, child_mask = (
            parent_child_similarity_tensors(
                features, calibration, list(children)
            )
        )
        packed = packed.to(model_device)
        prototype_mask = prototype_mask.to(model_device)
        child_mask = child_mask.to(model_device)
        if parent == "root":
            child_logits = model.child_energies(
                packed, prototype_mask, child_mask
            )
            routes[parent] = torch.softmax(
                child_logits, dim=1
            ).cpu()
            continue
        probabilities = torch.softmax(
            model.categorical_logits(
                packed, prototype_mask, child_mask
            ),
            dim=1,
        ).cpu()
        known = probabilities[:, :-1]
        unknown = probabilities[:, -1]
        routes[parent] = known / known.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        unknown_probabilities[parent] = unknown
    return routes, unknown_probabilities


def prior_corrected_product_unknown(
    hierarchy,
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    prior_mode: str,
) -> dict[str, torch.Tensor]:
    """Combine two local unknown posteriors while counting their prior once."""
    if prior_mode not in {"uniform_terminal", "paired_view"}:
        raise ValueError(f"Unsupported prior mode: {prior_mode}")
    expected = {
        parent
        for parent in hierarchy.parent2children
        if parent != "root"
    }
    if set(left) != expected or set(right) != expected:
        raise ValueError(
            "Both posterior dictionaries must cover every non-root parent"
        )
    result = {}
    for parent in sorted(expected):
        if prior_mode == "paired_view":
            prior = 0.5
        else:
            prior = 1.0 / (
                float(len(hierarchy.parent2children[parent])) + 1.0
            )
        left_logit = torch.logit(
            left[parent].float().clamp(1e-7, 1.0 - 1e-7)
        )
        right_logit = torch.logit(
            right[parent].float().clamp(1e-7, 1.0 - 1e-7)
        )
        prior_logit = torch.logit(torch.tensor(prior))
        result[parent] = torch.sigmoid(
            left_logit + right_logit - prior_logit
        )
    return result


@torch.no_grad()
def global_unknown_probability(
    model: SharedMaskedCategoricalLikelihood,
    hierarchy,
    features: torch.Tensor,
    calibration: HierarchicalSupportCalibration,
) -> torch.Tensor:
    """Evaluate the global leaf-mixture background posterior."""
    model.eval()
    model_device = next(model.parameters()).device
    leaf_nodes = sorted(
        set(calibration.prototype_nodes),
        key=hierarchy.id_node_list.index,
    )
    similarities, prototype_mask, child_mask = (
        parent_child_similarity_tensors(
            features, calibration, leaf_nodes
        )
    )
    probabilities = torch.softmax(
        model.categorical_logits(
            similarities.to(model_device),
            prototype_mask.to(model_device),
            child_mask.to(model_device),
        ),
        dim=1,
    )
    return probabilities[:, -1].cpu()


def latent_knownness_terminal_distribution(
    hierarchy,
    unknown_probability: torch.Tensor,
    leaf_conditional: torch.Tensor,
    parent_conditional: torch.Tensor,
) -> torch.Tensor:
    """Mix normalized known-leaf and internal-parent conditionals."""
    if leaf_conditional.shape != parent_conditional.shape:
        raise ValueError("Leaf and parent tensors must share a shape")
    if leaf_conditional.ndim != 2:
        raise ValueError("Terminal conditionals must be matrices")
    sample_count, node_count = leaf_conditional.shape
    if node_count != len(hierarchy.id_node_list):
        raise ValueError("Terminal width differs from hierarchy")
    unknown_probability = unknown_probability.float().cpu().flatten()
    if int(unknown_probability.numel()) != int(sample_count):
        raise ValueError("Knownness probability count differs from terminals")
    leaf_indices = torch.tensor([
        index
        for index, node in enumerate(hierarchy.id_node_list)
        if node not in hierarchy.parent2children
    ], dtype=torch.long)
    parent_indices = torch.tensor([
        index
        for index, node in enumerate(hierarchy.id_node_list)
        if node != "root" and node in hierarchy.parent2children
    ], dtype=torch.long)
    leaf = torch.zeros_like(leaf_conditional, dtype=torch.float32)
    parent = torch.zeros_like(parent_conditional, dtype=torch.float32)
    leaf[:, leaf_indices] = leaf_conditional.float().cpu()[
        :, leaf_indices
    ]
    parent[:, parent_indices] = parent_conditional.float().cpu()[
        :, parent_indices
    ]
    leaf = leaf / leaf.sum(dim=1, keepdim=True).clamp_min(1e-12)
    parent = parent / parent.sum(dim=1, keepdim=True).clamp_min(1e-12)
    result = (
        (1.0 - unknown_probability).unsqueeze(1) * leaf
        + unknown_probability.unsqueeze(1) * parent
    )
    sums = result.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise RuntimeError("Latent-knownness terminal distribution is invalid")
    return result


@torch.no_grad()
def masked_subtree_terminal_distribution(
    model: SharedMaskedCategoricalLikelihood,
    hierarchy,
    features: torch.Tensor,
    sample_leaf_nodes: list[str],
    calibration: HierarchicalSupportCalibration,
) -> dict:
    """Decode leave-one-child-out pseudo-OOD episodes as parent terminals."""
    features = features.detach().float().cpu()
    if int(features.shape[0]) != len(sample_leaf_nodes):
        raise ValueError("Feature and leaf-node counts differ")
    full_routes, full_unknown = mixture_conditionals_by_parent(
        model, hierarchy, features, calibration
    )
    groups: dict[tuple[str, str], list[int]] = {}
    for sample_index, leaf in enumerate(sample_leaf_nodes):
        for parent, true_child in true_path_edges(hierarchy, leaf):
            groups.setdefault((parent, true_child), []).append(sample_index)
    if not groups:
        raise RuntimeError("No non-root masked terminal episodes were built")

    model_device = next(model.parameters()).device
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    terminals = []
    targets = []
    max_masked_subtree_mass = 0.0
    max_normalization_error = 0.0
    episode_count = 0
    for (parent, hidden_child), sample_indices in groups.items():
        index = torch.tensor(sample_indices, dtype=torch.long)
        routes = {
            node: probabilities.index_select(0, index).clone()
            for node, probabilities in full_routes.items()
        }
        unknown = {
            node: probabilities.index_select(0, index).clone()
            for node, probabilities in full_unknown.items()
        }
        children = list(hierarchy.parent2children[parent])
        hidden_position = children.index(hidden_child)
        packed, prototype_mask, child_mask = (
            parent_child_similarity_tensors(
                features.index_select(0, index),
                calibration,
                children,
            )
        )
        child_mask[:, hidden_position] = False
        probabilities = torch.softmax(
            model.categorical_logits(
                packed.to(model_device),
                prototype_mask.to(model_device),
                child_mask.to(model_device),
            ),
            dim=1,
        ).cpu()
        known = probabilities[:, :-1]
        routes[parent] = known / known.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        unknown[parent] = probabilities[:, -1]
        terminal = probabilistic_terminal_distribution(
            hierarchy, routes, unknown
        )
        terminal_sums = terminal.sum(dim=1)
        max_normalization_error = max(
            max_normalization_error,
            float((terminal_sums - 1.0).abs().max()),
        )
        hidden_index = node_to_index[hidden_child]
        subtree_indices = torch.tensor([
            node_index
            for node_index, node in enumerate(hierarchy.id_node_list)
            if node_index == hidden_index
            or hidden_index in hierarchy.node_ancestors.get(node, [])
        ], dtype=torch.long)
        max_masked_subtree_mass = max(
            max_masked_subtree_mass,
            float(
                terminal.index_select(1, subtree_indices).sum(dim=1).max()
            ),
        )
        terminals.append(terminal)
        targets.append(torch.full(
            (len(sample_indices),),
            node_to_index[parent],
            dtype=torch.long,
        ))
        episode_count += len(sample_indices)
    return {
        "terminal": torch.cat(terminals, dim=0),
        "targets": torch.cat(targets, dim=0),
        "episodes": episode_count,
        "groups": len(groups),
        "max_masked_subtree_mass": max_masked_subtree_mass,
        "max_normalization_error": max_normalization_error,
    }


@torch.no_grad()
def unknown_probabilities_by_parent(
    model: SharedSupportPosterior,
    hierarchy,
    node_support_p_values: dict[str, torch.Tensor],
    route_conditionals: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    model.eval()
    result = {}
    for parent, children in hierarchy.parent2children.items():
        if parent == "root":
            continue
        support = torch.stack(
            [node_support_p_values[child] for child in children], dim=1
        )
        evidence = support_evidence(
            support, route_conditionals[parent]
        )
        result[parent] = torch.sigmoid(model(evidence)).cpu()
    return result


def probabilistic_terminal_distribution(
    hierarchy,
    route_conditionals: dict[str, torch.Tensor],
    unknown_probabilities: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "root" not in route_conditionals:
        raise ValueError("Root route probabilities are required")
    sample_count = int(route_conditionals["root"].shape[0])
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    terminal = torch.zeros(
        sample_count, len(hierarchy.id_node_list), dtype=torch.float32
    )
    reach = {"root": torch.ones(sample_count, dtype=torch.float32)}
    ordered_parents = sorted(
        hierarchy.parent2children,
        key=lambda node: (len(hierarchy.node_ancestors.get(node, [])), node),
    )
    for parent in ordered_parents:
        if parent not in reach:
            continue
        parent_reach = reach[parent]
        if parent == "root":
            continuation = torch.ones_like(parent_reach)
        else:
            unknown = unknown_probabilities[parent].float().cpu()
            terminal[:, node_to_index[parent]] += parent_reach * unknown
            continuation = 1.0 - unknown
        route = route_conditionals[parent].float().cpu()
        for child_index, child in enumerate(hierarchy.parent2children[parent]):
            child_reach = (
                parent_reach * continuation * route[:, child_index]
            )
            if child in hierarchy.parent2children:
                reach[child] = child_reach
            else:
                terminal[:, node_to_index[child]] += child_reach
    sums = terminal.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise RuntimeError(
            "Probabilistic terminal distribution is not normalized: "
            f"range=({float(sums.min())}, {float(sums.max())})"
        )
    return terminal
