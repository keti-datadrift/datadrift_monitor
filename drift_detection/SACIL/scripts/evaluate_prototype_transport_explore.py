from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_explore"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.anchors import compute_prototypes  # noqa: E402
from sacil.engine import UnifiedTable1Trainer  # noqa: E402
from sacil.engine.checkpoint import load_checkpoint  # noqa: E402
from sacil.engine.evaluator import evaluate_nme  # noqa: E402
from sacil.features import collect_features  # noqa: E402
from sacil.hierarchy import (  # noqa: E402
    GriffinPeronaGreedy,
    cosine_soft_confusion,
    symmetric_affinity,
)
from sacil.memory import ExemplarMemory  # noqa: E402
from sacil.methods.prototype_transport import (  # noqa: E402
    affine_ridge_transport,
    empirical_bayes_residual_transport,
    orthogonal_procrustes_transport,
    rigid_procrustes_transport,
    similarity_procrustes_transport,
    transport_class_prototypes,
    weighted_rigid_procrustes_transport,
)
from sacil.methods.boundary_graph_surgery import (  # noqa: E402
    effective_bounded_branch_cap,
    nearest_leaf_bounded_ancestor_branches,
    canonical_regions,
)
from sacil.provenance import build_exploration_provenance  # noqa: E402
from sacil.utils import dump_json  # noqa: E402


@torch.inference_mode()
def _collect_horizontal_flip_features(model, loader, device) -> torch.Tensor:
    was_training = model.training
    model.eval()
    values: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = model.extract_features(torch.flip(images, dims=(3,)))
        values.append(features.detach().cpu())
    if was_training:
        model.train()
    if not values:
        raise ValueError("cannot collect flipped features from an empty loader")
    return torch.cat(values, dim=0)


def _full_session_flip_prototypes(trainer, model, session_id: int) -> torch.Tensor:
    class_ids = trainer.protocol.classes_for_session(session_id)
    dataset = trainer.data.train_eval_dataset_for_classes(class_ids)
    loader = trainer._loader(
        dataset, shuffle=False, session_id=session_id + 13000
    )
    regular = collect_features(model, loader, trainer.device)
    flipped = _collect_horizontal_flip_features(model, loader, trainer.device)
    return compute_prototypes(
        torch.cat([regular.features, flipped], dim=0),
        torch.cat([regular.original_targets, regular.original_targets], dim=0),
        class_ids,
    ).cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIL-valid sequential prototype-transport checkpoint rescore"
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=(
            "global",
            "class",
            "procrustes_global",
            "procrustes_rigid",
            "procrustes_rigid_full_seen",
            "procrustes_rigid_eb_residual",
            "procrustes_rigid_memory_blend",
            "procrustes_rigid_weighted",
            "procrustes_similarity",
            "procrustes_rigid_centroid",
            "procrustes_rigid_flip",
            "procrustes_rigid_hierarchy_residual",
            "procrustes_rigid_conflict_blend",
            "affine_ridge",
            "hierarchy_parent",
            "hierarchy_conflict",
            "hierarchy_conflict_branch",
        ),
        default="global",
    )
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--taxonomy-temperature", type=float, default=0.2)
    parser.add_argument("--max-branch-leaves", type=int, default=8)
    parser.add_argument("--max-conflict-leaf-coverage", type=float, default=0.6)
    parser.add_argument("--conflict-memory-mix", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--center-strength", type=float, default=0.0)
    parser.add_argument("--query-horizontal-flip", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_model(trainer, checkpoint, seen: int):
    model = trainer._new_model(seen).to(trainer.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def main() -> int:
    args = parse_args()
    checkpoints = [load_checkpoint(path, map_location="cpu") for path in args.checkpoints]
    session_ids = [int(value["session_id"]) for value in checkpoints]
    if session_ids != list(range(len(checkpoints))):
        raise ValueError("checkpoints must be a contiguous sequence starting at S0")

    transported: torch.Tensor | None = None
    previous_checkpoint = None
    records: list[dict] = []
    for checkpoint_path, checkpoint in zip(args.checkpoints, checkpoints):
        session_id = int(checkpoint["session_id"])
        config = copy.deepcopy(checkpoint["config"])
        config["device"] = args.device
        config["output"]["directory"] = str(
            (PROJECT_ROOT / "outputs" / "explore" / "transport_rescore_tmp").resolve()
        )
        config["output"]["run_name"] = f"{args.mode}_transport_rescore"
        config["exploration_provenance"] = build_exploration_provenance(
            SOURCE_ROOT, PROJECT_ROOT / "src"
        )
        trainer = UnifiedTable1Trainer(
            config, PROJECT_ROOT, max_sessions=session_id + 1
        )
        seen = trainer.protocol.session(session_id).stop
        current_model = _load_model(trainer, checkpoint, seen)
        trainer.model = current_model
        trainer.memory = ExemplarMemory.from_state_dict(checkpoint["memory"])

        transport_diagnostics: dict = {"initialized": session_id == 0}
        conflict_positions: tuple[int, ...] = ()
        if session_id == 0:
            if args.mode == "procrustes_rigid_flip":
                transported = _full_session_flip_prototypes(
                    trainer, current_model, 0
                )
            else:
                transported = trainer._full_session_prototypes(0)
        else:
            assert previous_checkpoint is not None and transported is not None
            known = trainer.protocol.session(session_id).start
            previous_model = _load_model(trainer, previous_checkpoint, known)
            trainer.memory = ExemplarMemory.from_state_dict(
                previous_checkpoint["memory"]
            )
            support_loader = trainer._memory_loader(
                session_id - 1, augment=False
            )
            old_collection = collect_features(
                previous_model, support_loader, trainer.device
            )
            current_collection = collect_features(
                current_model, support_loader, trainer.device
            )
            if not torch.equal(old_collection.indices, current_collection.indices):
                raise RuntimeError("transport support rows are misaligned")
            old_class_ids = trainer.protocol.seen_classes(session_id - 1)
            tree = None
            if args.mode in {
                "hierarchy_parent",
                "hierarchy_conflict",
                "hierarchy_conflict_branch",
                "procrustes_rigid_hierarchy_residual",
                "procrustes_rigid_conflict_blend",
            }:
                old_memory_means = compute_prototypes(
                    old_collection.features,
                    old_collection.original_targets,
                    old_class_ids,
                )
                confusion = cosine_soft_confusion(
                    old_collection.features,
                    old_collection.targets,
                    old_memory_means,
                    temperature=args.taxonomy_temperature,
                )
                tree = GriffinPeronaGreedy().build(
                    old_class_ids, symmetric_affinity(confusion)
                )
            residual = None
            if args.mode == "affine_ridge":
                transported_old, _, residual = affine_ridge_transport(
                    transported,
                    old_collection.features,
                    current_collection.features,
                    ridge=args.ridge,
                )
                drift_norms = (transported_old - transported).norm(dim=1)
                support_counts = torch.full(
                    (known,), old_collection.features.shape[0], dtype=torch.long
                )
            elif args.mode == "procrustes_rigid_flip":
                old_flipped = _collect_horizontal_flip_features(
                    previous_model, support_loader, trainer.device
                )
                current_flipped = _collect_horizontal_flip_features(
                    current_model, support_loader, trainer.device
                )
                transported_old, _, _, residual = rigid_procrustes_transport(
                    transported,
                    torch.cat([old_collection.features, old_flipped], dim=0),
                    torch.cat(
                        [current_collection.features, current_flipped], dim=0
                    ),
                )
                drift_norms = (transported_old - transported).norm(dim=1)
                support_counts = torch.full(
                    (known,), 2 * old_collection.features.shape[0], dtype=torch.long
                )
            elif args.mode == "procrustes_rigid_centroid":
                old_centroids = compute_prototypes(
                    old_collection.features,
                    old_collection.original_targets,
                    old_class_ids,
                )
                current_centroids = compute_prototypes(
                    current_collection.features,
                    current_collection.original_targets,
                    old_class_ids,
                )
                transported_old, _, _, residual = rigid_procrustes_transport(
                    transported,
                    old_centroids,
                    current_centroids,
                )
                drift_norms = (transported_old - transported).norm(dim=1)
                support_counts = torch.full(
                    (known,), known, dtype=torch.long
                )
            elif args.mode == "procrustes_similarity":
                (
                    transported_old,
                    _,
                    _,
                    similarity_scale,
                    residual,
                ) = similarity_procrustes_transport(
                    transported,
                    old_collection.features,
                    current_collection.features,
                )
                drift_norms = (transported_old - transported).norm(dim=1)
                support_counts = torch.full(
                    (known,), old_collection.features.shape[0], dtype=torch.long
                )
            elif args.mode == "procrustes_rigid_weighted":
                (
                    transported_old,
                    _,
                    _,
                    residual,
                    representative_weights,
                ) = weighted_rigid_procrustes_transport(
                    transported,
                    old_collection.features,
                    current_collection.features,
                    old_collection.targets,
                    sigma=args.sigma,
                )
                drift_norms = (transported_old - transported).norm(dim=1)
                support_counts = torch.full(
                    (known,), old_collection.features.shape[0], dtype=torch.long
                )
            elif args.mode == "procrustes_rigid_eb_residual":
                (
                    transported_old,
                    _,
                    _,
                    residual,
                    eb_shrinkage,
                    _,
                ) = empirical_bayes_residual_transport(
                    transported,
                    old_collection.features,
                    current_collection.features,
                    old_collection.targets,
                )
                drift_norms = (transported_old - transported).norm(dim=1)
                support_counts = torch.full(
                    (known,),
                    old_collection.features.shape[0] // known,
                    dtype=torch.long,
                )
            elif args.mode in {
                "procrustes_global",
                "procrustes_rigid",
                "procrustes_rigid_full_seen",
                "procrustes_rigid_memory_blend",
                "procrustes_rigid_hierarchy_residual",
                "procrustes_rigid_conflict_blend",
            }:
                if args.mode == "procrustes_global":
                    transported_old, _, residual = (
                        orthogonal_procrustes_transport(
                            transported,
                            old_collection.features,
                            current_collection.features,
                        )
                    )
                else:
                    transported_old, rigid_rotation, rigid_translation, residual = (
                        rigid_procrustes_transport(
                            transported,
                            old_collection.features,
                            current_collection.features,
                        )
                    )
                    if args.mode in {
                        "procrustes_rigid_hierarchy_residual",
                        "procrustes_rigid_conflict_blend",
                    }:
                        assert tree is not None
                        incoming_ids = trainer.protocol.classes_for_session(
                            session_id
                        )
                        incoming_dataset = (
                            trainer.data.train_eval_dataset_for_classes(
                                incoming_ids
                            )
                        )
                        incoming_loader = trainer._loader(
                            incoming_dataset,
                            shuffle=False,
                            session_id=session_id + 14000,
                        )
                        incoming_teacher = collect_features(
                            previous_model, incoming_loader, trainer.device
                        )
                        incoming_means = compute_prototypes(
                            incoming_teacher.features,
                            incoming_teacher.original_targets,
                            incoming_ids,
                        )
                        cap = effective_bounded_branch_cap(
                            args.max_branch_leaves,
                            args.max_conflict_leaf_coverage,
                            known,
                            len(incoming_ids),
                        )
                        selected = nearest_leaf_bounded_ancestor_branches(
                            tree,
                            old_class_ids,
                            old_memory_means,
                            incoming_means,
                            max_branch_leaves=cap,
                        )
                        canonical, _ = canonical_regions(
                            tree, selected.selected_branch_nodes
                        )
                        conflict_original_ids = {
                            int(class_id)
                            for node_id in canonical
                            for class_id in tree.descendants(node_id)
                        }
                        conflict_positions = tuple(
                            position
                            for position, class_id in enumerate(old_class_ids)
                            if int(class_id) in conflict_original_ids
                        )
                        support_counts = torch.full(
                            (known,),
                            old_collection.features.shape[0],
                            dtype=torch.long,
                        )
                        if args.mode == "procrustes_rigid_hierarchy_residual":
                            old_normalized = torch.nn.functional.normalize(
                                old_collection.features.float(), dim=1
                            )
                            current_normalized = torch.nn.functional.normalize(
                                current_collection.features.float(), dim=1
                            )
                            aligned = (
                                old_normalized @ rigid_rotation
                                + rigid_translation
                            )
                            sample_residual = current_normalized - aligned
                            weights = torch.empty(
                                old_normalized.shape[0],
                                dtype=old_normalized.dtype,
                            )
                            for class_position in range(known):
                                class_mask = (
                                    old_collection.targets == class_position
                                )
                                centered = old_normalized[class_mask] - (
                                    old_normalized[class_mask].mean(
                                        dim=0, keepdim=True
                                    )
                                )
                                weights[class_mask] = torch.exp(
                                    -centered.square().sum(dim=1)
                                    / (2.0 * args.sigma**2)
                                )
                            class_position_by_id = {
                                int(class_id): position
                                for position, class_id in enumerate(old_class_ids)
                            }
                            raw_transported = (
                                torch.nn.functional.normalize(
                                    transported.float(), dim=1
                                )
                                @ rigid_rotation
                                + rigid_translation
                            )
                            for node_id in canonical:
                                positions = [
                                    class_position_by_id[int(class_id)]
                                    for class_id in tree.descendants(node_id)
                                ]
                                region_mask = (
                                    old_collection.targets[:, None]
                                    == torch.tensor(positions)[None, :]
                                ).any(dim=1)
                                branch_weights = weights[region_mask]
                                branch_residual = (
                                    branch_weights[:, None]
                                    * sample_residual[region_mask]
                                ).sum(dim=0) / branch_weights.sum().clamp_min(
                                    1.0e-12
                                )
                                raw_transported[positions] += branch_residual
                                support_counts[positions] = int(
                                    region_mask.sum()
                                )
                            transported_old = torch.nn.functional.normalize(
                                raw_transported, dim=1
                            )
                drift_norms = (transported_old - transported).norm(dim=1)
                if args.mode not in {
                    "procrustes_rigid_hierarchy_residual",
                    "procrustes_rigid_conflict_blend",
                }:
                    support_counts = torch.full(
                        (known,),
                        old_collection.features.shape[0],
                        dtype=torch.long,
                    )
            elif args.mode in {
                "hierarchy_conflict",
                "hierarchy_conflict_branch",
            }:
                assert tree is not None
                global_result = transport_class_prototypes(
                    transported,
                    old_collection.features,
                    current_collection.features,
                    old_collection.targets,
                    old_class_ids,
                    mode="global",
                    sigma=args.sigma,
                )
                local_result = transport_class_prototypes(
                    transported,
                    old_collection.features,
                    current_collection.features,
                    old_collection.targets,
                    old_class_ids,
                    mode="hierarchy_parent",
                    tree=tree,
                    sigma=args.sigma,
                )
                incoming_ids = trainer.protocol.classes_for_session(session_id)
                incoming_dataset = trainer.data.train_eval_dataset_for_classes(
                    incoming_ids
                )
                incoming_loader = trainer._loader(
                    incoming_dataset,
                    shuffle=False,
                    session_id=session_id + 14000,
                )
                incoming_teacher = collect_features(
                    previous_model, incoming_loader, trainer.device
                )
                incoming_means = compute_prototypes(
                    incoming_teacher.features,
                    incoming_teacher.original_targets,
                    incoming_ids,
                )
                cap = effective_bounded_branch_cap(
                    args.max_branch_leaves,
                    args.max_conflict_leaf_coverage,
                    known,
                    len(incoming_ids),
                )
                selected = nearest_leaf_bounded_ancestor_branches(
                    tree,
                    old_class_ids,
                    old_memory_means,
                    incoming_means,
                    max_branch_leaves=cap,
                )
                conflict_original_ids = {
                    int(class_id)
                    for node_id in selected.selected_branch_nodes
                    for class_id in tree.descendants(node_id)
                }
                conflict_positions = tuple(
                    position
                    for position, class_id in enumerate(old_class_ids)
                    if int(class_id) in conflict_original_ids
                )
                mixed_drifts = global_result.drifts.clone()
                mixed_support = global_result.support_counts.clone()
                if conflict_positions:
                    if args.mode == "hierarchy_conflict":
                        mixed_drifts[list(conflict_positions)] = (
                            local_result.drifts[list(conflict_positions)]
                        )
                        mixed_support[list(conflict_positions)] = (
                            local_result.support_counts[
                                list(conflict_positions)
                            ]
                        )
                    else:
                        canonical, _ = canonical_regions(
                            tree, selected.selected_branch_nodes
                        )
                        old_normalized = torch.nn.functional.normalize(
                            old_collection.features.float(), dim=1
                        )
                        current_normalized = torch.nn.functional.normalize(
                            current_collection.features.float(), dim=1
                        )
                        sample_drift = current_normalized - old_normalized
                        weights = torch.empty(
                            old_normalized.shape[0], dtype=old_normalized.dtype
                        )
                        for class_position in range(known):
                            class_mask = (
                                old_collection.targets == class_position
                            )
                            centered = old_normalized[class_mask] - (
                                old_normalized[class_mask].mean(
                                    dim=0, keepdim=True
                                )
                            )
                            weights[class_mask] = torch.exp(
                                -centered.square().sum(dim=1)
                                / (2.0 * args.sigma**2)
                            )
                        class_position_by_id = {
                            int(class_id): position
                            for position, class_id in enumerate(old_class_ids)
                        }
                        for node_id in canonical:
                            positions = [
                                class_position_by_id[int(class_id)]
                                for class_id in tree.descendants(node_id)
                            ]
                            region_mask = (
                                old_collection.targets[:, None]
                                == torch.tensor(positions)[None, :]
                            ).any(dim=1)
                            branch_weights = weights[region_mask]
                            branch_drift = (
                                branch_weights[:, None]
                                * sample_drift[region_mask]
                            ).sum(dim=0) / branch_weights.sum().clamp_min(1.0e-12)
                            mixed_drifts[positions] = branch_drift
                            mixed_support[positions] = int(region_mask.sum())
                transported_old = torch.nn.functional.normalize(
                    transported + mixed_drifts.cpu(), dim=1
                )
                drift_norms = mixed_drifts.norm(dim=1)
                support_counts = mixed_support
            else:
                result = transport_class_prototypes(
                    transported,
                    old_collection.features,
                    current_collection.features,
                    old_collection.targets,
                    old_class_ids,
                    mode=args.mode,
                    tree=tree,
                    sigma=args.sigma,
                )
                transported_old = result.prototypes.cpu()
                drift_norms = result.drift_norms
                support_counts = result.support_counts
                residual = None
            trainer.model = current_model
            if args.mode == "procrustes_rigid_flip":
                new_full = _full_session_flip_prototypes(
                    trainer, current_model, session_id
                )
            else:
                new_full = trainer._full_session_prototypes(session_id)
            transported = torch.cat(
                [transported_old, new_full.cpu()], dim=0
            )
            transport_diagnostics = {
                "initialized": False,
                "mean_drift_norm": float(drift_norms.mean()),
                "min_drift_norm": float(drift_norms.min()),
                "max_drift_norm": float(drift_norms.max()),
                "mean_support_count": float(
                    support_counts.float().mean()
                ),
                "conflict_class_count": len(conflict_positions),
                "procrustes_residual": residual,
                "empirical_bayes_shrinkage": (
                    eb_shrinkage
                    if args.mode == "procrustes_rigid_eb_residual"
                    else None
                ),
                "representative_effective_sample_size": (
                    float(1.0 / representative_weights.square().sum().item())
                    if args.mode == "procrustes_rigid_weighted"
                    else None
                ),
                "similarity_scale": (
                    similarity_scale
                    if args.mode == "procrustes_similarity"
                    else None
                ),
            }

        baseline_means = checkpoint["class_means"].detach().cpu().clone()
        hybrid_means = baseline_means.clone()
        old = trainer.protocol.session(session_id).start
        if old > 0:
            assert transported is not None
            hybrid_means[:old] = transported[:old]
            if args.mode == "procrustes_rigid_memory_blend":
                if not 0.0 <= args.conflict_memory_mix <= 1.0:
                    raise ValueError("conflict-memory-mix must be in [0, 1]")
                hybrid_means[:old] = torch.nn.functional.normalize(
                    (1.0 - args.conflict_memory_mix) * hybrid_means[:old]
                    + args.conflict_memory_mix * baseline_means[:old],
                    dim=1,
                )
            if args.mode == "procrustes_rigid_conflict_blend":
                if not 0.0 <= args.conflict_memory_mix <= 1.0:
                    raise ValueError("conflict-memory-mix must be in [0, 1]")
                if conflict_positions:
                    positions = list(conflict_positions)
                    hybrid_means[positions] = torch.nn.functional.normalize(
                        (1.0 - args.conflict_memory_mix)
                        * hybrid_means[positions]
                        + args.conflict_memory_mix
                        * baseline_means[positions],
                        dim=1,
                    )
        if args.mode == "procrustes_rigid_full_seen":
            assert transported is not None
            hybrid_means[old:] = transported[old:]
        test_dataset = trainer.data.cumulative_test_dataset(session_id)
        test_loader = trainer._loader(
            test_dataset, shuffle=False, session_id=session_id + 11000
        )
        baseline = evaluate_nme(
            current_model,
            test_loader,
            trainer.device,
            old,
            baseline_means,
            center_strength=args.center_strength,
            horizontal_flip_query=args.query_horizontal_flip,
        ).to_dict()
        transported_eval = evaluate_nme(
            current_model,
            test_loader,
            trainer.device,
            old,
            hybrid_means,
            center_strength=args.center_strength,
            horizontal_flip_query=args.query_horizontal_flip,
        ).to_dict()
        records.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "session_id": session_id,
                "baseline": baseline,
                "transported": transported_eval,
                "delta_accuracy": transported_eval["accuracy"]
                - baseline["accuracy"],
                "transport_diagnostics": transport_diagnostics,
            }
        )
        previous_checkpoint = checkpoint

    payload = {
        "method": "sequential prototype transport",
        "mode": args.mode,
        "sigma": args.sigma,
        "conflict_memory_mix": args.conflict_memory_mix,
        "ridge": args.ridge,
        "center_strength": args.center_strength,
        "query_horizontal_flip": args.query_horizontal_flip,
        "cil_valid_data_access": True,
        "test_labels_used_for_selection": False,
        "exploration_provenance": build_exploration_provenance(
            SOURCE_ROOT, PROJECT_ROOT / "src"
        ),
        "records": records,
    }
    dump_json(payload, args.output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
