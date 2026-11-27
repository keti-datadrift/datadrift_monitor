from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


MANIFEST_VERSION = 1
MANIFEST_ALGORITHM = "loco_nonroot_lexicographic_balanced_v2"


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hierarchy_topology_record(hierarchy) -> dict:
    """Return a canonical released-ProHOC hierarchy provenance record."""
    node_list = list(hierarchy.id_node_list)
    payload = {
        "node_list": node_list,
        "parent2children": {
            parent: list(hierarchy.parent2children[parent])
            for parent in sorted(hierarchy.parent2children)
        },
        "node_ancestors": {
            node: [
                int(index)
                for index in hierarchy.node_ancestors.get(node, [])
            ]
            for node in node_list
        },
    }
    return {
        **payload,
        "node_count": len(node_list),
        "topology_hash": canonical_hash(payload),
    }


def atomic_save_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(
                payload,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _leaf_path(hierarchy, leaf: str) -> list[str]:
    return [
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.node_ancestors.get(leaf, [])
    ] + [leaf]


def _mapped_retained_ancestor(
    hierarchy,
    leaf: str,
    retained_leaves: set[str],
) -> str | None:
    retained_child_branches: dict[str, set[str]] = {}
    for retained_leaf in retained_leaves:
        path = _leaf_path(hierarchy, retained_leaf)
        for parent, child in zip(path[:-1], path[1:]):
            retained_child_branches.setdefault(parent, set()).add(child)
    branching_nodes = {
        node for node, children in retained_child_branches.items()
        if len(children) >= 2
    }
    ancestors = _leaf_path(hierarchy, leaf)[:-1]
    for node in reversed(ancestors):
        if node != "root" and node in branching_nodes:
            return node
    return None


def mapped_retained_ancestor(
    hierarchy,
    leaf: str,
    retained_leaves: Sequence[str],
) -> str | None:
    """Public deterministic non-root mapping used by meta-LOCO stages."""
    return _mapped_retained_ancestor(
        hierarchy, str(leaf), set(str(value) for value in retained_leaves)
    )


def _fold_mappings(
    hierarchy,
    leaf_classes: Sequence[str],
    heldout_leaves: Sequence[str],
) -> dict[str, str] | None:
    heldout = set(heldout_leaves)
    retained_leaves = set(leaf_classes) - heldout
    mappings = {}
    for leaf in heldout_leaves:
        mapped = _mapped_retained_ancestor(
            hierarchy, leaf, retained_leaves
        )
        if mapped is None:
            return None
        mappings[leaf] = mapped
    return mappings


def build_topology_holdout_manifest(
    hierarchy,
    leaf_classes: Sequence[str],
    *,
    num_folds: int = 4,
    requested_fold_size: int = 16,
    seed: int = 0,
) -> dict:
    """Create the lexicographically first feasible balanced LOCO folds."""
    leaf_classes = sorted(str(value) for value in leaf_classes)
    if len(leaf_classes) != len(set(leaf_classes)):
        raise ValueError("Leaf class names must be unique")
    if int(num_folds) < 2:
        raise ValueError("At least two cross-fit folds are required")
    if int(requested_fold_size) < 1:
        raise ValueError("requested_fold_size must be positive")
    if int(seed) != 0:
        raise ValueError(
            "Lexicographic topology folds are seed-free; manifest seed must be 0"
        )
    missing = [
        leaf for leaf in leaf_classes
        if leaf not in hierarchy.id_node_list
    ]
    if missing:
        raise ValueError(f"Leaves are absent from the hierarchy: {missing}")

    loco_mapping = {}
    parent_depth = {}
    ineligible = {}
    for leaf in leaf_classes:
        mapped = _mapped_retained_ancestor(
            hierarchy, leaf, set(leaf_classes) - {leaf}
        )
        if mapped is None:
            ineligible[leaf] = "leave_one_out_maps_to_root"
        else:
            loco_mapping[leaf] = mapped
            parent_depth[mapped] = len(
                hierarchy.node_ancestors.get(mapped, [])
            )
    eligible = sorted(loco_mapping)
    if not eligible:
        raise ValueError(
            "Hierarchy has no feasible non-root class-holdout fold"
        )
    if int(num_folds) != 4:
        raise ValueError("The audited LOCO design requires exactly four folds")
    fold_sizes = [
        len(eligible) // int(num_folds)
        + (1 if fold < len(eligible) % int(num_folds) else 0)
        for fold in range(int(num_folds))
    ]
    parents: dict[str, list[str]] = {}
    for leaf in eligible:
        parents.setdefault(loco_mapping[leaf], []).append(leaf)
    parent_records = [
        (
            parent,
            sorted(leaves),
            parent_depth[parent],
        )
        for parent, leaves in sorted(parents.items())
    ]

    def balanced_options(count: int) -> list[tuple[int, ...]]:
        floor = count // int(num_folds)
        remainder = count % int(num_folds)
        options = []
        for high_folds in __import__("itertools").combinations(
            range(int(num_folds)), remainder
        ):
            high = set(high_folds)
            option = tuple(
                floor + (1 if fold in high else 0)
                for fold in range(int(num_folds))
            )
            options.append(option)
        # Filling lexicographically earlier leaves into smaller fold IDs first
        # defines a stable lexicographic feasible assignment.
        return sorted(
            options,
            key=lambda option: tuple(
                fold
                for fold, count_for_fold in enumerate(option)
                for _ in range(count_for_fold)
            ),
        )

    solution = None
    fold_totals = [0] * int(num_folds)
    depth2_totals = [0] * int(num_folds)
    chosen_options: list[tuple[int, ...]] = []

    def materialize() -> list[list[str]]:
        folds = [[] for _ in range(int(num_folds))]
        for (_, leaves, _), option in zip(
            parent_records, chosen_options
        ):
            offset = 0
            for fold, count_for_fold in enumerate(option):
                folds[fold].extend(
                    leaves[offset:offset + count_for_fold]
                )
                offset += count_for_fold
        return [sorted(fold) for fold in folds]

    def search(parent_index: int) -> bool:
        nonlocal solution
        if parent_index == len(parent_records):
            if fold_totals != fold_sizes or depth2_totals != [3] * 4:
                return False
            folds = materialize()
            mappings = [
                _fold_mappings(hierarchy, leaf_classes, fold)
                for fold in folds
            ]
            if any(mapping is None for mapping in mappings):
                return False
            for fold_index, mapping in enumerate(mappings):
                if any(
                    mapping[leaf] != loco_mapping[leaf]
                    for leaf in folds[fold_index]
                ):
                    return False
                unique_parents = set(mapping.values())
                depth1_parents = {
                    parent for parent in unique_parents
                    if parent_depth[parent] == 1
                }
                depth2_parents = {
                    parent for parent in unique_parents
                    if parent_depth[parent] == 2
                }
                if (
                    len(unique_parents) < 10
                    or len(depth1_parents) < 7
                    or len(depth2_parents) < 2
                ):
                    return False
            solution = (folds, mappings)
            return True
        _, _, depth = parent_records[parent_index]
        for option in balanced_options(
            len(parent_records[parent_index][1])
        ):
            if any(
                fold_totals[fold] + option[fold] > fold_sizes[fold]
                for fold in range(int(num_folds))
            ):
                continue
            if depth == 2 and any(
                depth2_totals[fold] + option[fold] > 3
                for fold in range(int(num_folds))
            ):
                continue
            chosen_options.append(option)
            for fold in range(int(num_folds)):
                fold_totals[fold] += option[fold]
                if depth == 2:
                    depth2_totals[fold] += option[fold]
            if search(parent_index + 1):
                return True
            for fold in range(int(num_folds)):
                fold_totals[fold] -= option[fold]
                if depth == 2:
                    depth2_totals[fold] -= option[fold]
            chosen_options.pop()
        return False

    if not search(0) or solution is None:
        raise RuntimeError(
            "No lexicographic assignment satisfies audited LOCO constraints"
        )
    assigned_folds, assigned_mappings = solution
    class_to_index = {
        leaf: index for index, leaf in enumerate(leaf_classes)
    }
    manifest = {
        "version": MANIFEST_VERSION,
        "algorithm": MANIFEST_ALGORITHM,
        "tie_break": "lexicographic",
        "seed": None,
        "num_folds": int(num_folds),
        "requested_fold_size": int(requested_fold_size),
        "feasible_fold_sizes": fold_sizes,
        "fold_size_reduced": True,
        "leaf_count": len(leaf_classes),
        "eligible_leaf_count": len(eligible),
        "ineligible_leaves": ineligible,
        "unassigned_eligible_leaves": [],
        "constraints": {
            "eligible_used_exactly_once": True,
            "parent_frequency_per_fold": "floor_or_ceil",
            "mapped_parent_unique_min": 10,
            "depth1_parent_unique_min": 7,
            "depth2_parent_unique_min": 2,
            "depth2_sample_count_exact": 3,
        },
        "folds": [
            (lambda record: {
                **record,
                "fold_hash": canonical_hash(record),
            })({
                "fold": fold_index,
                "heldout_leaves": fold,
                "heldout_original_class_indices": [
                    class_to_index[leaf] for leaf in fold
                ],
                "mapped_unknown_nodes": {
                    leaf: assigned_mappings[fold_index][leaf]
                    for leaf in fold
                },
                "mapped_parent_depths": {
                    leaf: parent_depth[
                        assigned_mappings[fold_index][leaf]
                    ]
                    for leaf in fold
                },
            })
            for fold_index, fold in enumerate(assigned_folds)
        ],
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    validate_topology_holdout_manifest(
        hierarchy, leaf_classes, manifest
    )
    return manifest


def validate_topology_holdout_manifest(
    hierarchy,
    leaf_classes: Sequence[str],
    manifest: dict,
) -> None:
    manifest_without_hash = dict(manifest)
    saved_hash = manifest_without_hash.pop("manifest_hash", None)
    if saved_hash != canonical_hash(manifest_without_hash):
        raise ValueError("Cross-fit manifest hash is invalid")
    if manifest.get("algorithm") != MANIFEST_ALGORITHM:
        raise ValueError("Unsupported cross-fit manifest algorithm")
    folds = manifest.get("folds") or []
    if len(folds) != int(manifest.get("num_folds", -1)):
        raise ValueError("Cross-fit manifest fold count is inconsistent")
    feasible_sizes = manifest.get("feasible_fold_sizes") or []
    if feasible_sizes != [13, 13, 12, 12]:
        raise ValueError(
            "Audited LOCO fold sizes must be exactly 13/13/12/12"
        )
    all_heldout = []
    parent_fold_counts: dict[str, list[int]] = {}
    for fold_index, fold in enumerate(folds):
        if int(fold.get("fold", -1)) != fold_index:
            raise ValueError("Cross-fit fold indices are not canonical")
        heldout = fold.get("heldout_leaves") or []
        if len(heldout) != feasible_sizes[fold_index]:
            raise ValueError("Cross-fit fold has an unexpected size")
        fold_without_hash = dict(fold)
        fold_hash = fold_without_hash.pop("fold_hash", None)
        if fold_hash != canonical_hash(fold_without_hash):
            raise ValueError("Cross-fit fold hash is invalid")
        mappings = _fold_mappings(hierarchy, leaf_classes, heldout)
        if mappings != fold.get("mapped_unknown_nodes"):
            raise ValueError("Cross-fit non-root mappings are inconsistent")
        depths = {
            leaf: len(hierarchy.node_ancestors[mappings[leaf]])
            for leaf in heldout
        }
        if depths != fold.get("mapped_parent_depths"):
            raise ValueError("Cross-fit mapped-parent depths are inconsistent")
        unique_parents = set(mappings.values())
        if len(unique_parents) < 10:
            raise ValueError("Cross-fit fold has fewer than 10 mapped parents")
        if sum(
            len(hierarchy.node_ancestors[parent]) == 1
            for parent in unique_parents
        ) < 7:
            raise ValueError("Cross-fit fold has fewer than 7 depth-1 parents")
        if sum(
            len(hierarchy.node_ancestors[parent]) == 2
            for parent in unique_parents
        ) < 2:
            raise ValueError("Cross-fit fold has fewer than 2 depth-2 parents")
        if sum(depth == 2 for depth in depths.values()) != 3:
            raise ValueError(
                "Cross-fit fold must contain exactly 3 depth-2 samples"
            )
        for parent in mappings.values():
            parent_fold_counts.setdefault(parent, [0] * len(folds))
            parent_fold_counts[parent][fold_index] += 1
        all_heldout.extend(heldout)
    if len(all_heldout) != len(set(all_heldout)):
        raise ValueError("Cross-fit heldout folds overlap")
    if len(all_heldout) != int(manifest.get("eligible_leaf_count", -1)):
        raise ValueError("Cross-fit folds do not use every eligible leaf")
    for counts in parent_fold_counts.values():
        if max(counts) - min(counts) > 1:
            raise ValueError(
                "Mapped-parent frequency is not floor/ceil balanced"
            )


def stratified_retained_image_split(
    targets: torch.Tensor,
    retained_class_indices: Sequence[int],
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Split retained images into representation, selection, and query sets."""
    if len(fractions) != 3 or any(float(value) <= 0 for value in fractions):
        raise ValueError("Three positive retained split fractions are required")
    if abs(sum(float(value) for value in fractions) - 1.0) > 1e-6:
        raise ValueError("Retained split fractions must sum to one")
    targets = targets.detach().long().cpu()
    retained = sorted(set(int(value) for value in retained_class_indices))
    if not retained:
        raise ValueError("At least one retained class is required")
    generator = torch.Generator().manual_seed(int(seed))
    partitions = [[] for _ in range(3)]
    for target in retained:
        indices = torch.nonzero(
            targets == target, as_tuple=False
        ).flatten()
        if int(indices.numel()) < 3:
            raise ValueError(
                f"Retained class {target} needs at least three images"
            )
        indices = indices[
            torch.randperm(int(indices.numel()), generator=generator)
        ]
        raw = [
            float(fraction) * int(indices.numel())
            for fraction in fractions
        ]
        counts = [int(value) for value in raw]
        remainder = int(indices.numel()) - sum(counts)
        order = sorted(
            range(3),
            key=lambda index: (raw[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        if any(count == 0 for count in counts):
            raise ValueError(
                f"Retained class {target} produced an empty partition"
            )
        offset = 0
        for partition, count in zip(partitions, counts):
            partition.extend(indices[offset:offset + count].tolist())
            offset += count
    names = ("representation_train", "model_selection", "known_query")
    result = {
        name: torch.tensor(sorted(values), dtype=torch.long)
        for name, values in zip(names, partitions)
    }
    combined = torch.cat(list(result.values()))
    retained_mask = torch.zeros_like(targets, dtype=torch.bool)
    for target in retained:
        retained_mask |= targets == target
    expected = torch.nonzero(retained_mask, as_tuple=False).flatten()
    if int(torch.unique(combined).numel()) != int(combined.numel()):
        raise RuntimeError("Retained image partitions overlap")
    if not torch.equal(combined.sort().values, expected):
        raise RuntimeError("Retained image partitions are incomplete")
    return result


def tensor_partitions_hash(partitions: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(partitions):
        digest.update(name.encode("utf-8"))
        digest.update(
            partitions[name].detach().long().cpu().numpy().tobytes()
        )
    return digest.hexdigest()


class RemappedSubset(Dataset):
    """Subset a dataset and remap sparse original class IDs to [0, C)."""

    def __init__(
        self,
        dataset,
        indices: Sequence[int] | torch.Tensor,
        original_to_compact: dict[int, int],
        classes: Sequence[str],
    ):
        self.dataset = dataset
        self.indices = [
            int(value)
            for value in torch.as_tensor(
                indices, dtype=torch.long
            ).flatten().tolist()
        ]
        self.original_to_compact = {
            int(key): int(value)
            for key, value in original_to_compact.items()
        }
        self.classes = list(classes)
        original_targets = getattr(dataset, "targets", None)
        if original_targets is None:
            raise ValueError("RemappedSubset requires dataset.targets")
        self.targets = []
        for index in self.indices:
            original_target = int(original_targets[index])
            if original_target not in self.original_to_compact:
                raise ValueError(
                    f"Subset index {index} belongs to excluded class "
                    f"{original_target}"
                )
            self.targets.append(
                self.original_to_compact[original_target]
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, original_target = self.dataset[self.indices[index]]
        compact_target = self.original_to_compact[int(original_target)]
        return image, compact_target
