from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank, compute_prototypes
from sacil.config import get_required
from sacil.data import ClassOrderProtocol, build_data_module
from sacil.engine.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from sacil.engine.evaluator import (
    compute_nme_class_means,
    evaluate,
    evaluate_nme,
)
from sacil.features import collect_features
from sacil.hierarchy import (
    GriffinPeronaGreedy,
    HierarchyTree,
    cosine_soft_confusion,
    symmetric_affinity,
)
from sacil.memory import ExemplarMemory, herding_select
from sacil.methods import (
    AnchorGeometryLoss,
    BranchMaskedKDReference,
    BoundaryGraphSurgeryLoss,
    BoundaryGraphSurgeryReference,
    HierarchicalEdgeCorrelationLoss,
    HierarchicalEdgeReference,
    STRATIFIED_EDGE_GROUP_NAMES,
    StratifiedEdgeCorrelationResult,
    StratifiedHierarchicalEdgeCorrelationLoss,
    StratifiedHierarchicalEdgeReference,
    afc_nca_loss,
    afc_pod_loss,
    branch_masked_pycil_icarl_kd_loss,
    bgs_insertion_loss,
    bounded_conflict_union_diagnostics,
    canonical_regions,
    casper_spectral_loss,
    compute_conflict_weights,
    conflict_union_membership,
    conflict_subtree_inside_edge_weights,
    controlled_transfer_loss,
    cosine_feature_distillation_loss,
    cosine_imprinted_weights,
    hierarchy_routed_feature_sample_weights,
    normalized_cosine_classifier_logits,
    create_classification_loss,
    create_contrastive_loss,
    create_kd_loss,
    cross_space_clustering_loss,
    fgp_graph_preservation_loss,
    geodesic_distillation_loss,
    endpoint_regions,
    effective_bounded_branch_cap,
    global_preservation_weights,
    global_edge_weights,
    icarl_bce_loss,
    inverse_angular_dispersion_reliability,
    incident_edge_weights,
    old_logit_kl_loss,
    negative_candidate_positions,
    nearest_leaf_bounded_ancestor_branches,
    pair_mask_summary,
    parameter_l2_regularization,
    pairwise_cosine_edge_vector,
    pair_types_and_weights,
    pod_flat_loss,
    pod_spatial_loss,
    podnet_nca_loss,
    prototype_cross_entropy,
    project_insertion_gradient,
    pycil_finetune_loss,
    pycil_icarl_kd_loss,
    replay_cross_entropy,
    reconstruction_confidence_weights,
    row_permuted_random_weights,
    scheduled_afc_factor,
    scheduled_fgp_weight,
    selective_pycil_icarl_kd_loss,
    stratified_edge_group_ids,
    tensor_sha256,
    SUPPORTED_UNIFIED_METHODS,
    unified_method_contract,
    validate_annotation1_config,
    validate_annotation1_protocol,
)
from sacil.metrics import CILMetricsTracker
from sacil.models import (
    AFCIncrementalNet,
    CREATEIncrementalNet,
    CSCCTIncrementalNet,
    ExpandableLinearNet,
    FGPIncrementalNet,
    PyCILPODNet,
    kmeans_imprinted_weights,
)
from sacil.utils import (
    dump_json,
    ensure_dir,
    git_commit,
    make_generator,
    resolved_device,
    seed_worker,
    set_seed,
)


SUPPORTED_TABLE1_METHODS = SUPPORTED_UNIFIED_METHODS
GEOMETRY_MODES = frozenset({"none", "global", "flat", "sacil"})
GEOMETRY_SUBSTRATES = frozenset({"icarl", "afc", "sacil"})
GEOMETRY_RELIABILITY_MODES = frozenset(
    {"uniform", "inverse_angular_dispersion"}
)
GEOMETRY_OBJECTIVES = frozenset({"mse", "correlation", "triplet_rank"})
CASPER_SUBSTRATES = frozenset({"icarl", "casper"})
EDGE_TOPOLOGY_OBJECTIVE = "hierarchical_edge_correlation"
STRATIFIED_EDGE_TOPOLOGY_OBJECTIVE = (
    "stratified_hierarchical_edge_correlation"
)
EDGE_TOPOLOGY_OBJECTIVES = frozenset(
    {EDGE_TOPOLOGY_OBJECTIVE, STRATIFIED_EDGE_TOPOLOGY_OBJECTIVE}
)
EDGE_TOPOLOGY_SUBSTRATES = frozenset({"replay", "icarl"})
BRANCH_MASKED_KD_SUBSTRATES = frozenset({"icarl"})
SELECTIVE_KD_SUBSTRATES = frozenset({"icarl"})


def resolve_geometry_mode(
    method_name: str, method_config: dict[str, Any]
) -> str:
    """Resolve the geometry ablation without changing the CIL substrate.

    Older pure-SACIL configs exposed two boolean switches.  Explicit
    ``geometry_mode`` takes precedence, while the legacy switches remain a
    backward-compatible spelling for the same Global/Flat/SACIL variants.
    """

    method = str(method_name).lower()
    explicit = method_config.get("geometry_mode")
    if explicit is None:
        if method != "sacil":
            return "none"
        if not bool(method_config.get("local_relaxation", True)):
            return "global"
        if not bool(method_config.get("use_internal_anchors", True)):
            return "flat"
        return "sacil"
    mode = str(explicit).lower().replace("-", "_")
    aliases = {
        "off": "none",
        "global_hap": "global",
        "flat_lrhap": "flat",
        "full": "sacil",
    }
    mode = aliases.get(mode, mode)
    if mode not in GEOMETRY_MODES:
        raise ValueError(
            f"unknown geometry mode {explicit!r}; expected one of "
            f"{sorted(GEOMETRY_MODES)}"
        )
    if mode != "none" and method not in GEOMETRY_SUBSTRATES:
        raise ValueError(
            f"geometry mode {mode!r} is unsupported for substrate {method!r}"
        )
    return mode


def resolve_geometry_options(method_config: dict[str, Any]) -> dict[str, Any]:
    """Validate anchor-frame ablation options with legacy-safe defaults."""

    raw = method_config.get("geometry", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.geometry must be a mapping")
    frame = str(raw.get("anchor_frame", "fixed")).lower().replace("-", "_")
    frame = {"moving": "co_moving", "comoving": "co_moving"}.get(
        frame, frame
    )
    if frame not in {"fixed", "co_moving", "hybrid"}:
        raise ValueError(
            "method.geometry.anchor_frame must be fixed, co_moving, or hybrid"
        )
    reliability = str(raw.get("reliability", "uniform")).lower().replace(
        "-", "_"
    )
    reliability = {
        "none": "uniform",
        "inverse_dispersion": "inverse_angular_dispersion",
    }.get(reliability, reliability)
    if reliability not in GEOMETRY_RELIABILITY_MODES:
        raise ValueError(
            "method.geometry.reliability must be uniform or "
            "inverse_angular_dispersion"
        )
    requested_fixed_mix = float(raw.get("fixed_mix", 0.5))
    refresh = int(raw.get("refresh_interval_epochs", 1))
    power = float(raw.get("reliability_power", 1.0))
    epsilon = float(raw.get("reliability_epsilon", 1e-4))
    minimum = float(raw.get("reliability_min_weight", 0.25))
    maximum = float(raw.get("reliability_max_weight", 4.0))
    objective = str(raw.get("objective", "mse")).lower().replace("-", "_")
    objective = {
        "pearson": "correlation",
        "rank": "triplet_rank",
        "triplet": "triplet_rank",
    }.get(objective, objective)
    weight_normalization = str(
        raw.get("weight_normalization", "weight_sum")
    ).lower().replace("-", "_")
    weight_normalization = {
        "relative": "weight_sum",
        "absolute": "anchor_count",
    }.get(weight_normalization, weight_normalization)
    triplet_margin_scale = float(raw.get("triplet_margin_scale", 1.0))
    rank_tolerance = float(raw.get("rank_tolerance", 1e-4))
    if not 0.0 <= requested_fixed_mix <= 1.0:
        raise ValueError("method.geometry.fixed_mix must be in [0, 1]")
    if refresh <= 0:
        raise ValueError(
            "method.geometry.refresh_interval_epochs must be positive"
        )
    if power < 0 or epsilon <= 0:
        raise ValueError("invalid geometry reliability power/epsilon")
    if not 0 < minimum <= maximum:
        raise ValueError("invalid geometry reliability weight bounds")
    if objective not in GEOMETRY_OBJECTIVES:
        raise ValueError(
            "method.geometry.objective must be mse, correlation, or "
            "triplet_rank"
        )
    if weight_normalization not in {"weight_sum", "anchor_count"}:
        raise ValueError(
            "method.geometry.weight_normalization must be weight_sum or "
            "anchor_count"
        )
    if not 0.0 < triplet_margin_scale <= 1.0:
        raise ValueError(
            "method.geometry.triplet_margin_scale must be in (0, 1]"
        )
    if rank_tolerance < 0.0:
        raise ValueError(
            "method.geometry.rank_tolerance must be non-negative"
        )
    fixed_mix = (
        1.0
        if frame == "fixed"
        else 0.0
        if frame == "co_moving"
        else requested_fixed_mix
    )
    return {
        "anchor_frame": frame,
        "fixed_mix": fixed_mix,
        "refresh_interval_epochs": refresh,
        "reliability": reliability,
        "reliability_power": power,
        "reliability_epsilon": epsilon,
        "reliability_min_weight": minimum,
        "reliability_max_weight": maximum,
        "objective": objective,
        "weight_normalization": weight_normalization,
        "triplet_margin_scale": triplet_margin_scale,
        "rank_tolerance": rank_tolerance,
    }


def resolve_edge_topology_options(
    method_name: str, method_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the independent representative-edge topology plug-in.

    Controlled screens support both Replay-CE and the complete iCaRL CE +
    old-logit KD substrate. The term is inactive in session 0 because no old
    teacher exists.
    """

    method = str(method_name).lower()
    raw = method_config.get("edge_topology", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.edge_topology must be a mapping")
    enabled = bool(raw.get("enabled", False))
    objective = str(
        raw.get("objective", EDGE_TOPOLOGY_OBJECTIVE)
    ).lower().replace("-", "_")
    representatives = int(raw.get("representatives_per_class", 2))
    weight = float(raw.get("lambda_edge", 5.0))
    weighting = str(raw.get("edge_weighting", "global")).lower().replace(
        "-", "_"
    )
    update_interval = int(raw.get("update_interval_steps", 1))
    min_edge_weight = float(raw.get("min_edge_weight", 0.1))
    branches = int(raw.get("conflict_branches_per_new_class", 1))
    beta_boundary = float(raw.get("beta_boundary", 1.0))
    gamma_conflict = float(raw.get("gamma_conflict", 0.1))
    if enabled and method not in EDGE_TOPOLOGY_SUBSTRATES:
        raise ValueError(
            "hierarchical edge correlation requires the Replay-CE or iCaRL "
            "CE + old-logit KD substrate"
        )
    if objective not in EDGE_TOPOLOGY_OBJECTIVES:
        raise ValueError(
            "method.edge_topology.objective must be hierarchical_edge_correlation "
            "or stratified_hierarchical_edge_correlation"
        )
    if representatives not in {2, 4, 20}:
        raise ValueError("representatives_per_class must be 2, 4, or 20")
    if weight < 0:
        raise ValueError("lambda_edge must be non-negative")
    if objective == EDGE_TOPOLOGY_OBJECTIVE:
        if weighting not in {
            "global",
            "conflict_branch_incident",
            "conflict_subtree_inside",
        }:
            raise ValueError(
                "edge_weighting must be global, conflict_branch_incident, or "
                "conflict_subtree_inside"
            )
    elif weighting != "stratified_hierarchy":
        raise ValueError(
            "stratified edge correlation requires "
            "edge_weighting=stratified_hierarchy"
        )
    if update_interval <= 0:
        raise ValueError("update_interval_steps must be positive")
    if not 0.0 <= min_edge_weight <= 1.0:
        raise ValueError("min_edge_weight must be in [0, 1]")
    if branches <= 0:
        raise ValueError("conflict_branches_per_new_class must be positive")
    options = {
        "enabled": enabled,
        "objective": objective,
        "representatives_per_class": representatives,
        "lambda_edge": weight,
        "edge_weighting": weighting,
        "update_interval_steps": update_interval,
        "min_edge_weight": min_edge_weight,
        "conflict_branches_per_new_class": branches,
    }
    if objective == STRATIFIED_EDGE_TOPOLOGY_OBJECTIVE:
        if (
            not math.isfinite(weight)
            or not math.isfinite(beta_boundary)
            or not math.isfinite(gamma_conflict)
            or beta_boundary < 0.0
            or gamma_conflict < 0.0
        ):
            raise ValueError(
                "stratified edge coefficients must be finite and non-negative"
            )
        options.update(
            beta_boundary=beta_boundary,
            gamma_conflict=gamma_conflict,
        )
    return options


def resolve_branch_masked_kd_options(
    method_name: str, method_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve new-sample-only hierarchy masking for PyCIL iCaRL KD."""

    method = str(method_name).lower()
    raw = method_config.get("branch_masked_kd", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.branch_masked_kd must be a mapping")
    enabled = bool(raw.get("enabled", False))
    v_min = float(raw.get("v_min", 0.25))
    if enabled and method not in BRANCH_MASKED_KD_SUBSTRATES:
        raise ValueError("branch-masked KD requires the iCaRL substrate")
    if not 0.0 < v_min <= 1.0:
        raise ValueError("method.branch_masked_kd.v_min must be in (0, 1]")
    return {
        "enabled": enabled,
        "v_min": v_min,
        "top_internal_branches": 1,
    }


def resolve_selective_kd_options(
    method_name: str, method_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve SRIL-style analytic gradient-alignment KD routing."""

    method = str(method_name).lower()
    raw = method_config.get("selective_kd", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.selective_kd must be a mapping")
    enabled = bool(raw.get("enabled", False))
    threshold = float(raw.get("alignment_threshold", 0.0))
    if enabled and method not in SELECTIVE_KD_SUBSTRATES:
        raise ValueError("selective KD requires the iCaRL substrate")
    if not math.isfinite(threshold):
        raise ValueError("selective KD alignment_threshold must be finite")
    return {"enabled": enabled, "alignment_threshold": threshold}


def resolve_icarl_kd_weight(
    method_name: str, method_config: dict[str, Any]
) -> float:
    """Resolve the incremental old-logit KD coefficient for iCaRL screens.

    The default stays exactly one.  This opt-in coefficient exists so a
    shared session-0 checkpoint can support a matched KD-off diagnostic
    without changing any other iCaRL training or evaluation contract.
    """

    if str(method_name).lower() != "icarl":
        return 1.0
    weight = float(method_config.get("kd_weight", 1.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("iCaRL kd_weight must be finite and non-negative")
    return weight


def resolve_geodesic_distillation_options(
    method_name: str, method_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the equation-level GeoDL control used in exploration."""

    raw = method_config.get("geodesic_distillation", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.geodesic_distillation must be a mapping")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    if str(method_name).lower() != "icarl":
        raise ValueError("geodesic distillation requires the iCaRL substrate")
    weight = float(raw.get("lambda", 10.0))
    rank = int(raw.get("subspace_rank", 32))
    epsilon = float(raw.get("epsilon", 1e-7))
    adaptive = bool(raw.get("adaptive_new_over_old", True))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("GeoDL lambda must be finite and non-negative")
    if rank <= 0:
        raise ValueError("GeoDL subspace_rank must be positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("GeoDL epsilon must be finite and positive")
    return {
        "enabled": True,
        "lambda": weight,
        "subspace_rank": rank,
        "epsilon": epsilon,
        "adaptive_new_over_old": adaptive,
        "implementation": "equation_level_cvpr2021_control",
    }


def resolve_feature_cosine_distillation_options(
    method_name: str, method_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the paper-aligned LUCIR cosine-feature control."""

    raw = method_config.get("feature_cosine_distillation", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.feature_cosine_distillation must be a mapping")
    enabled = bool(raw.get("enabled", False))
    routing_raw = raw.get("hierarchy_routing")
    if routing_raw is not None and not isinstance(routing_raw, dict):
        raise ValueError("feature cosine hierarchy_routing must be a mapping")
    routing_enabled = bool(
        isinstance(routing_raw, dict) and routing_raw.get("enabled", False)
    )
    if not enabled:
        if routing_enabled:
            raise ValueError(
                "feature cosine hierarchy routing requires feature distillation"
            )
        return {"enabled": False}
    if str(method_name).lower() != "icarl":
        raise ValueError("cosine feature distillation requires iCaRL")
    weight = float(raw.get("lambda", 6.0))
    legacy_adaptive = bool(raw.get("adaptive_new_over_old", True))
    adaptive_mode = str(
        raw.get(
            "adaptive_mode",
            "new_over_old" if legacy_adaptive else "none",
        )
    ).lower().replace("-", "_")
    epsilon = float(raw.get("epsilon", 1e-12))
    training_classifier = str(
        raw.get("training_classifier", "linear")
    ).lower().replace("-", "_")
    cosine_scale = float(raw.get("cosine_scale", 10.0))
    imprint_new_weights = bool(raw.get("imprint_new_weights", False))
    freeze_old_weights = bool(raw.get("freeze_old_weights", False))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("feature cosine lambda must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("feature cosine epsilon must be finite and positive")
    if adaptive_mode not in {"none", "new_over_old", "old_over_new"}:
        raise ValueError(
            "feature cosine adaptive_mode must be none, new_over_old, or old_over_new"
        )
    if training_classifier not in {"linear", "normalized_cosine"}:
        raise ValueError(
            "feature cosine training_classifier must be linear or normalized_cosine"
        )
    if not math.isfinite(cosine_scale) or cosine_scale <= 0:
        raise ValueError("feature cosine classifier scale must be positive")
    if training_classifier == "linear" and (
        imprint_new_weights or freeze_old_weights
    ):
        raise ValueError(
            "imprinting/freezing old rows requires normalized_cosine classifier"
        )
    options = {
        "enabled": True,
        "lambda": weight,
        "adaptive_new_over_old": adaptive_mode == "new_over_old",
        "adaptive_mode": adaptive_mode,
        "epsilon": epsilon,
        "training_classifier": training_classifier,
        "cosine_scale": cosine_scale,
        "imprint_new_weights": imprint_new_weights,
        "freeze_old_weights": freeze_old_weights,
        "implementation": "lucir_cosine_embedding_control",
    }
    if routing_raw is not None:
        routing = {"enabled": routing_enabled}
        if routing_enabled:
            bgs_raw = method_config.get("boundary_graph_surgery", {})
            if not isinstance(bgs_raw, dict) or not bool(
                bgs_raw.get("enabled", False)
            ):
                raise ValueError(
                    "feature cosine hierarchy routing requires enabled BGS"
                )
            weights = {
                "old_conflict_weight": float(
                    routing_raw.get("old_conflict_weight", 0.1)
                ),
                "old_outside_weight": float(
                    routing_raw.get("old_outside_weight", 1.0)
                ),
                "new_weight": float(routing_raw.get("new_weight", 0.1)),
            }
            for name, value in weights.items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"feature cosine hierarchy {name} must be finite and non-negative"
                    )
            if sum(weights.values()) <= 0:
                raise ValueError(
                    "feature cosine hierarchy routing needs positive configured mass"
                )
            routing.update(weights)
            routing["partition_source"] = "bgs_reference.sample_region_ids"
        options["hierarchy_routing"] = routing
    return options


def resolve_bgs_options(method_name: str, method_config: dict[str, Any]) -> dict[str, Any]:
    raw = method_config.get("boundary_graph_surgery", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("boundary_graph_surgery must be a mapping")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    if str(method_name).lower() != "icarl":
        raise ValueError("BGS requires iCaRL")
    geometry = raw.get("geometry")
    insertion = raw.get("insertion")
    if not isinstance(geometry, dict) or not isinstance(insertion, dict):
        raise ValueError("enabled BGS requires explicit geometry and insertion")
    for key in (
        "lambda",
        "objective",
        "inside_weight",
        "boundary_weight",
        "denominator",
        "use_leaf",
        "use_internal_without_root",
        "mask_mode",
    ):
        if key not in geometry:
            raise ValueError(f"BGS geometry missing required field {key}")
    for key in (
        "enabled",
        "lambda",
        "negatives_per_class",
        "temperature",
        "prototype_refresh",
        "separation_enabled",
        "parent_weight",
        "parent_slack",
        "negative_scope",
    ):
        if key not in insertion:
            raise ValueError(f"BGS insertion missing required field {key}")
    for key in ("spec_version", "branch_source", "branches_per_new_class"):
        if key not in raw:
            raise ValueError(f"BGS missing required field {key}")
    if str(raw["spec_version"]) != "bgs_v1":
        raise ValueError("BGS supports only spec_version=bgs_v1")
    branch_source = str(raw["branch_source"])
    if branch_source not in {
        "i2_teacher_internal_top1",
        "nearest_leaf_bounded_ancestor",
    }:
        raise ValueError("unsupported BGS-v1 branch_source")
    branches_per_new_class = int(raw["branches_per_new_class"])
    if branches_per_new_class != 1:
        raise ValueError("BGS-v1 currently implements exactly one branch per new class")
    inside = float(geometry["inside_weight"])
    boundary = float(geometry["boundary_weight"])
    mode = str(geometry["mask_mode"])
    if not 0 <= inside <= boundary <= 1:
        raise ValueError("BGS requires 0 <= inside <= boundary <= 1")
    if mode not in {"global", "structured", "random_pair", "incident"}:
        raise ValueError("invalid BGS mask mode")
    if str(geometry["objective"]) != "fixed_anchor_mse":
        raise ValueError("BGS-v1 geometry objective must be fixed_anchor_mse")
    if str(geometry["denominator"]) != "old_sample_count_x_anchor_count":
        raise ValueError("BGS-v1 requires the absolute denominator")
    use_leaf = bool(geometry["use_leaf"])
    use_internal = bool(geometry["use_internal_without_root"])
    if not use_leaf and not use_internal:
        raise ValueError("BGS geometry must enable at least one anchor group")
    geometry_lambda = float(geometry["lambda"])
    insertion_lambda = float(insertion["lambda"])
    temperature = float(insertion["temperature"])
    negatives_per_class = int(insertion["negatives_per_class"])
    parent_weight = float(insertion["parent_weight"])
    parent_slack = float(insertion["parent_slack"])
    gradient_projection = bool(insertion.get("gradient_projection", False))
    projection_epsilon = float(insertion.get("projection_epsilon", 1e-12))
    if geometry_lambda < 0 or insertion_lambda < 0 or parent_weight < 0:
        raise ValueError("BGS loss weights must be nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("BGS insertion temperature must be positive")
    if negatives_per_class <= 0:
        raise ValueError("BGS requires at least one negative per class")
    if not math.isfinite(parent_slack) or parent_slack < 0:
        raise ValueError("BGS parent_slack must be finite and nonnegative")
    if gradient_projection and not bool(insertion["enabled"]):
        raise ValueError("BGS gradient projection requires insertion")
    if not math.isfinite(projection_epsilon) or projection_epsilon <= 0:
        raise ValueError("BGS projection_epsilon must be finite and positive")
    if str(insertion["prototype_refresh"]) != "epoch_start_full_new_unaugmented":
        raise ValueError("BGS-v1 requires epoch-start full unaugmented prototypes")
    negative_scope = str(insertion["negative_scope"])
    if negative_scope not in {"branch_local", "all_old"}:
        raise ValueError("invalid BGS negative_scope")
    options = {
        "enabled": True,
        "spec_version": str(raw["spec_version"]),
        "branch_source": branch_source,
        "branches_per_new_class": branches_per_new_class,
        "geometry": {
            "lambda": geometry_lambda,
            "inside_weight": inside,
            "boundary_weight": boundary,
            "mask_mode": mode,
            "use_leaf": use_leaf,
            "use_internal_without_root": use_internal,
            "denominator": str(geometry["denominator"]),
            "objective": str(geometry["objective"]),
        },
        "insertion": {
            "enabled": bool(insertion["enabled"]),
            "lambda": insertion_lambda,
            "negatives_per_class": negatives_per_class,
            "temperature": temperature,
            "prototype_refresh": str(insertion["prototype_refresh"]),
            "separation_enabled": bool(insertion["separation_enabled"]),
            "parent_weight": parent_weight,
            "parent_slack": parent_slack,
            "negative_scope": negative_scope,
            "gradient_projection": gradient_projection,
            "projection_epsilon": projection_epsilon,
        },
    }
    if branch_source == "nearest_leaf_bounded_ancestor":
        max_branch_leaves = int(raw.get("max_branch_leaves", 8))
        max_coverage = float(raw.get("max_conflict_leaf_coverage", 0.60))
        if max_branch_leaves < 2:
            raise ValueError("BGS max_branch_leaves must be at least 2")
        if not math.isfinite(max_coverage) or not 0 < max_coverage < 1:
            raise ValueError(
                "BGS max_conflict_leaf_coverage must be in (0, 1)"
            )
        options["max_branch_leaves"] = max_branch_leaves
        options["max_conflict_leaf_coverage"] = max_coverage
    return options


def resolve_casper_options(
    method_name: str, method_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the CaSpeR regularizer without replacing its CIL substrate.

    ``method=casper`` retains the author-style standalone adapter.  The
    controlled comparison instead uses ``method=icarl`` with
    ``method.casper.enabled=true`` so the model, CE/KD objective, optimizer,
    epochs, memory, and evaluator remain exactly those of the iCaRL control.
    """

    method = str(method_name).lower()
    raw = method_config.get("casper", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("method.casper must be a mapping")
    enabled = method == "casper" or bool(raw.get("enabled", False))
    weight = float(raw.get("weight", 0.01))
    knn = int(raw.get("knn", 10))
    classes = int(raw.get("classes_per_graph", 5))
    batch_size = int(raw.get("replay_batch_size", 64))
    solver = str(raw.get("solver", "xitorch")).lower()
    wd_reg = float(raw.get("wd_reg", 1e-5 if method == "casper" else 0.0))

    if enabled and method not in CASPER_SUBSTRATES:
        raise ValueError(
            f"CaSpeR regularization is unsupported for substrate {method!r}"
        )
    if weight < 0 or wd_reg < 0:
        raise ValueError("CaSpeR weights must be non-negative")
    if knn <= 0 or classes <= 0 or batch_size <= 0:
        raise ValueError("CaSpeR k, class count, and batch size must be positive")
    if enabled and batch_size <= max(knn, classes + 1):
        raise ValueError(
            "CaSpeR replay_batch_size must exceed both knn and "
            "classes_per_graph + 1"
        )
    if solver not in {"xitorch", "partial"}:
        raise ValueError("CaSpeR solver must use the author xitorch path")
    if method == "icarl" and wd_reg != 0.0:
        raise ValueError(
            "the controlled iCaRL+CaSpeR plug-in requires wd_reg=0 so "
            "only the spectral term differs from the iCaRL substrate"
        )
    return {
        "enabled": enabled,
        "weight": weight,
        "knn": knn,
        "classes_per_graph": classes,
        "replay_batch_size": batch_size,
        "solver": solver,
        "wd_reg": wd_reg,
    }


def geometry_preservation_component(
    geometry: AnchorGeometryLoss | None,
    current_features: Tensor,
    reference_features: Tensor | None,
    replay_mask: Tensor,
    *,
    weight: float,
) -> Tensor | None:
    """Return the weighted replay-only geometry term, when applicable."""

    if weight < 0:
        raise ValueError("geometry weight must be non-negative")
    if (
        geometry is None
        or reference_features is None
        or not bool(replay_mask.any())
    ):
        return None
    return float(weight) * geometry(
        current_features[replay_mask], reference_features[replay_mask]
    )


def base_recipe_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Return every configuration field that can affect session 0."""

    method = copy.deepcopy(config.get("method", {}))
    for key in (
        "geometry_mode",
        "lambda_geo",
        "hierarchy",
        "conflict",
        "geometry",
        "local_relaxation",
        "use_internal_anchors",
    ):
        method.pop(key, None)
    # In the controlled iCaRL plug-in, CaSpeR is inactive in S0 and therefore
    # must not prevent reuse of the exact iCaRL base checkpoint.  The separate
    # method=casper adapter keeps this field because its explicit L2 term can
    # affect S0.
    if str(method.get("name", "")).lower() == "icarl":
        method.pop("casper", None)
        method.pop("branch_masked_kd", None)
        method.pop("selective_kd", None)
        method.pop("boundary_graph_surgery", None)
        method.pop("kd_weight", None)
        method.pop("geodesic_distillation", None)
        method.pop("feature_cosine_distillation", None)
    if str(method.get("name", "")).lower() in {"replay", "icarl"}:
        method.pop("edge_topology", None)
    training = copy.deepcopy(config.get("training", {}))
    training.pop("incremental", None)
    debug = copy.deepcopy(config.get("debug", {}))
    debug.pop("max_sessions", None)
    return {
        "seed": int(config.get("seed", 1)),
        "deterministic": bool(config.get("deterministic", True)),
        "data": copy.deepcopy(config.get("data", {})),
        "model": copy.deepcopy(config.get("model", {})),
        "memory": copy.deepcopy(config.get("memory", {})),
        "evaluation": copy.deepcopy(config.get("evaluation", {})),
        "training": training,
        "method": method,
        "debug": debug,
    }


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Draw a fixed number of classes before drawing replay samples.

    This is the sampling contract used by CaSpeR.  It intentionally differs
    from ordinary class-balanced weighting: every graph contains exactly
    ``classes_per_batch`` old classes.
    """

    def __init__(
        self,
        labels: list[int],
        *,
        batch_size: int,
        classes_per_batch: int,
        batches: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.classes_per_batch = int(classes_per_batch)
        self.batches = int(batches)
        self.seed = int(seed)
        positions: dict[int, list[int]] = defaultdict(list)
        for position, label in enumerate(labels):
            positions[int(label)].append(position)
        self.positions = dict(positions)
        if self.classes_per_batch > len(self.positions):
            raise ValueError("CaSpeR requests more classes than memory contains")
        if self.classes_per_batch <= 0 or self.batch_size <= self.classes_per_batch:
            raise ValueError("invalid CaSpeR balanced batch dimensions")
        self._iteration = 0

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        classes = np.asarray(sorted(self.positions), dtype=np.int64)
        base, remainder = divmod(self.batch_size, self.classes_per_batch)
        for _ in range(self.batches):
            selected = rng.choice(
                classes, size=self.classes_per_batch, replace=False
            )
            batch: list[int] = []
            for offset, class_id in enumerate(selected.tolist()):
                count = base + int(offset < remainder)
                choices = rng.choice(
                    self.positions[int(class_id)], size=count, replace=True
                )
                batch.extend(int(value) for value in choices.tolist())
            rng.shuffle(batch)
            yield batch


class UnifiedTable1Trainer:
    """One in-repo Table-1 engine with method-specific adapters.

    Upstream repositories are algorithm references only.  This class has no
    runtime import or subprocess path into ``ref_codes``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        project_root: str | Path,
        *,
        max_sessions: int | None = None,
        base_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.project_root = Path(project_root).resolve()
        self.method = str(get_required(config, "method.name")).lower()
        if self.method not in SUPPORTED_TABLE1_METHODS:
            raise ValueError(f"unsupported unified method: {self.method}")
        validate_annotation1_config(self.config)
        self.geometry_mode = resolve_geometry_mode(
            self.method, self.config["method"]
        )
        self.geometry_options = resolve_geometry_options(
            self.config["method"]
        )
        self.casper_options = resolve_casper_options(
            self.method, self.config["method"]
        )
        self.edge_topology_options = resolve_edge_topology_options(
            self.method, self.config["method"]
        )
        self.branch_masked_kd_options = resolve_branch_masked_kd_options(
            self.method, self.config["method"]
        )
        self.selective_kd_options = resolve_selective_kd_options(
            self.method, self.config["method"]
        )
        self.icarl_kd_weight = resolve_icarl_kd_weight(
            self.method, self.config["method"]
        )
        self.geodesic_distillation_options = (
            resolve_geodesic_distillation_options(
                self.method, self.config["method"]
            )
        )
        self.feature_cosine_distillation_options = (
            resolve_feature_cosine_distillation_options(
                self.method, self.config["method"]
            )
        )
        self.bgs_options = resolve_bgs_options(self.method, self.config["method"])
        if (
            self.geodesic_distillation_options["enabled"]
            and self.icarl_kd_weight != 0.0
        ):
            raise ValueError("GeoDL control replaces old-logit KD; set kd_weight=0")
        if (
            self.feature_cosine_distillation_options["enabled"]
            and self.icarl_kd_weight != 0.0
        ):
            raise ValueError(
                "cosine feature control replaces old-logit KD; set kd_weight=0"
            )
        if (
            self.feature_cosine_distillation_options["enabled"]
            and self.geodesic_distillation_options["enabled"]
        ):
            raise ValueError("GeoDL and cosine feature controls are exclusive")
        if (
            self.branch_masked_kd_options["enabled"]
            and not self.edge_topology_options["enabled"]
        ):
            raise ValueError("branch-masked KD requires H-TPL edge topology")
        if self.selective_kd_options["enabled"] and not self.edge_topology_options[
            "enabled"
        ]:
            raise ValueError("selective KD requires H-TPL edge topology")
        if (
            self.selective_kd_options["enabled"]
            and self.branch_masked_kd_options["enabled"]
        ):
            raise ValueError("selective KD and branch-masked KD are exclusive")
        self.base_checkpoint_path = (
            None
            if base_checkpoint is None
            else self._project_path(base_checkpoint)
        )
        self.start_session = 0
        self.method_contract = unified_method_contract(self.method)
        self.seed = int(config.get("seed", 1))
        set_seed(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.device = resolved_device(str(config.get("device", "cuda:0")))
        self.protocol = ClassOrderProtocol.from_json(
            self._project_path(get_required(config, "data.protocol"))
        )
        validate_annotation1_protocol(self.protocol)
        self.data = build_data_module(
            str(config["data"].get("name", "cifar100")),
            self._project_path(get_required(config, "data.root")),
            self.protocol,
            download=bool(config["data"].get("download", False)),
            color_jitter=bool(config["data"].get("color_jitter", True)),
        )
        initial_limit = self._memory_limit(self.protocol.session(0).stop)
        self.memory = ExemplarMemory(initial_limit)
        self.metrics = CILMetricsTracker()
        self.model: nn.Module | None = None
        self.class_means: Tensor | None = None
        self.sacil_tree: HierarchyTree | None = None
        self.sacil_prototypes: PrototypeBank | None = None
        self.sacil_anchors: HierarchicalAnchorBank | None = None
        self.edge_topology_reference: (
            HierarchicalEdgeReference
            | StratifiedHierarchicalEdgeReference
            | None
        ) = None
        self._edge_topology_images: Tensor | None = None
        self._edge_topology_loss: (
            HierarchicalEdgeCorrelationLoss
            | StratifiedHierarchicalEdgeCorrelationLoss
            | None
        ) = None
        self._last_edge_topology_stats: dict[str, float | int] = {}
        self.branch_masked_kd_reference: BranchMaskedKDReference | None = None
        self._last_branch_kd_stats: dict[str, float | int] = {}
        self._last_selective_kd_stats: dict[str, float | int] = {}
        self.bgs_reference: BoundaryGraphSurgeryReference | None = None
        self._bgs_loss: BoundaryGraphSurgeryLoss | None = None
        self._bgs_positive_prototypes: Tensor | None = None
        self._bgs_reference_path: Path | None = None
        self._bgs_reference_sha256: str | None = None
        self._last_bgs_stats: dict[str, float | int] = {}
        self._last_feature_routing_stats: dict[str, float | int] = {}
        configured_max_sessions = self.config.get("debug", {}).get(
            "max_sessions"
        )
        requested_max_sessions = (
            configured_max_sessions if max_sessions is None else max_sessions
        )
        if requested_max_sessions is None:
            self.max_sessions = self.protocol.num_sessions
        else:
            requested_max_sessions = int(requested_max_sessions)
            if requested_max_sessions <= 0:
                raise ValueError("max_sessions must be positive")
            self.max_sessions = min(
                requested_max_sessions, self.protocol.num_sessions
            )
        output = config["output"]
        self.run_dir = ensure_dir(
            self._project_path(output["directory"])
            / str(output["run_name"])
            / f"seed_{self.seed}"
        )
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        dump_json(
            {
                "config": self.config,
                "framework": "sacil-unified",
                "pycil_used": False,
                "reference_code_executed": False,
                "method_contract": self.method_contract.as_dict(),
                "geometry_mode": self.geometry_mode,
                "geometry_options": self.geometry_options,
                "casper_options": self.casper_options,
                "edge_topology_options": self.edge_topology_options,
                "branch_masked_kd_options": self.branch_masked_kd_options,
                "selective_kd_options": self.selective_kd_options,
                "icarl_kd_weight": self.icarl_kd_weight,
                "geodesic_distillation_options": (
                    self.geodesic_distillation_options
                ),
                "feature_cosine_distillation_options": (
                    self.feature_cosine_distillation_options
                ),
                "bgs_options": self.bgs_options,
                "exploration_provenance": copy.deepcopy(
                    self.config.get("exploration_provenance")
                ),
                "base_checkpoint": (
                    None
                    if self.base_checkpoint_path is None
                    else str(self.base_checkpoint_path)
                ),
                "protocol_id": self.protocol.protocol_id,
                "device": str(self.device),
                "git_commit": git_commit(self.project_root),
            },
            self.run_dir / "resolved_config.json",
        )

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def _memory_limit(self, seen_classes: int) -> int:
        memory = self.config["memory"]
        mode = str(memory.get("mode", "per_class"))
        if mode == "per_class":
            return int(memory.get("exemplars_per_class", 20))
        if mode == "fixed_total":
            return max(1, int(memory.get("capacity", 2000)) // int(seen_classes))
        raise ValueError(f"unknown memory mode: {mode}")

    @property
    def debug_train_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("train_samples_per_class")
        return None if value is None else int(value)

    @property
    def debug_test_samples_per_class(self) -> int | None:
        value = self.config.get("debug", {}).get("test_samples_per_class")
        return None if value is None else int(value)

    def _loader(
        self,
        dataset,
        *,
        shuffle: bool,
        session_id: int,
        batch_size: int | None = None,
        batch_sampler=None,
    ) -> DataLoader:
        training = self.config["training"]
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "num_workers": int(training.get("num_workers", 0)),
            "pin_memory": bool(training.get("pin_memory", True)),
            "worker_init_fn": seed_worker,
            "generator": make_generator(self.seed * 1000 + session_id),
        }
        workers = kwargs["num_workers"]
        if workers > 0:
            kwargs["persistent_workers"] = bool(
                training.get("persistent_workers", False)
            )
        if batch_sampler is not None:
            kwargs["batch_sampler"] = batch_sampler
        else:
            kwargs.update(
                batch_size=int(batch_size or training.get("batch_size", 128)),
                shuffle=bool(shuffle),
            )
        return DataLoader(**kwargs)

    def _new_model(self, num_classes: int) -> nn.Module:
        model = self.config.get("model", {})
        if self.method == "podnet":
            return PyCILPODNet(
                num_classes,
                proxies_per_class=int(model.get("proxies_per_class", 10)),
            )
        if self.method == "afc":
            return AFCIncrementalNet(
                num_classes,
                initial_size=self.protocol.session(0).size,
                increment_size=self.protocol.session(1).size,
                proxies_per_class=int(model.get("proxies_per_class", 10)),
                classifier_scale=float(model.get("nca_scale", 1.0)),
                distance_scale=float(model.get("distance_scale", 3.0)),
            )
        if self.method == "create":
            return CREATEIncrementalNet(
                num_classes,
                hidden_layers=tuple(model.get("hidden_layers", [])),
                latent_features=int(model.get("latent_features", 32)),
                reconstruction_scale=float(model.get("reconstruction_scale", 0.1)),
                backbone=str(model.get("backbone", "resnet32")),
            )
        if self.method == "fgp":
            return FGPIncrementalNet(num_classes)
        if self.method == "cscct":
            return CSCCTIncrementalNet(num_classes)
        return ExpandableLinearNet(
            num_classes, backbone=str(model.get("backbone", "resnet32"))
        )

    def _phase(self, session_id: int) -> dict[str, Any]:
        name = "base" if session_id == 0 else "incremental"
        return dict(self.config["training"][name])

    @staticmethod
    def _scheduler(optimizer: SGD, phase: dict[str, Any], epochs: int):
        kind = str(phase.get("scheduler", "multistep"))
        if kind == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=float(phase.get("eta_min", 0.0)),
            )
        if kind == "multistep":
            return MultiStepLR(
                optimizer,
                milestones=[int(v) for v in phase.get("milestones", [])],
                gamma=float(phase.get("lr_decay", 0.1)),
            )
        raise ValueError(f"unknown scheduler: {kind}")

    def _training_dataset(self, session_id: int, old_indices: list[int]):
        if self.method == "joint":
            return self.data.train_dataset_for_classes(
                self.protocol.seen_classes(session_id),
                augment=True,
                samples_per_class=self.debug_train_samples_per_class,
            )
        if self.method == "finetune":
            return self.data.new_train_dataset(
                session_id,
                augment=True,
                samples_per_class=self.debug_train_samples_per_class,
            )
        return self.data.training_dataset(
            session_id,
            old_indices,
            samples_per_class=self.debug_train_samples_per_class,
        )

    def _prototype_dataset(self, session_id: int, old_indices: list[int]):
        new = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        if not old_indices:
            return new
        return ConcatDataset(
            [new, self.data.replay_dataset(old_indices, augment=False)]
        )

    @torch.inference_mode()
    def _incoming_collection(self, session_id: int):
        if self.model is None:
            raise RuntimeError("model is missing")
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        return collect_features(
            self.model,
            self._loader(dataset, shuffle=False, session_id=session_id + 4000),
            self.device,
        )

    def _expand_model(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("cannot expand a missing model")
        total = self.protocol.session(session_id).stop
        if isinstance(self.model, PyCILPODNet):
            # Stock PyCIL initializes the new proxy chunk through
            # SplitCosineLinear.reset_parameters; PODNet does not use AFC's
            # K-means weight imprinting.
            self.model.expand_classes(total)
        elif isinstance(self.model, AFCIncrementalNet):
            collection = self._incoming_collection(session_id)
            class_features = [
                collection.features[
                    collection.original_targets == int(class_id)
                ]
                for class_id in self.protocol.classes_for_session(session_id)
            ]
            weights = kmeans_imprinted_weights(
                class_features,
                self.model.classifier.weights,
                proxies_per_class=self.model.classifier.proxies_per_class,
                random_state=self.seed * 1000 + session_id * 100,
            )
            self.model.expand_classes(total, weights.to(self.device))
        elif isinstance(self.model, CSCCTIncrementalNet):
            collection = self._incoming_collection(session_id)
            old_norm = self.model.classifier.weight.detach().norm(dim=1).mean()
            values = []
            for class_id in self.protocol.classes_for_session(session_id):
                feature = collection.features[
                    collection.original_targets == int(class_id)
                ].mean(dim=0, keepdim=True)
                values.append(F.normalize(feature, dim=1) * old_norm.cpu())
            self.model.expand_classes(torch.cat(values).to(self.device))
        elif isinstance(self.model, ExpandableLinearNet) and (
            self.feature_cosine_distillation_options.get("enabled", False)
            and self.feature_cosine_distillation_options.get(
                "training_classifier"
            )
            == "normalized_cosine"
        ):
            known = self.model.num_classes
            collection = self._incoming_collection(session_id)
            old_weights = self.model.classifier.weight.detach().clone()
            self.model.expand_classes(total)
            if self.feature_cosine_distillation_options[
                "imprint_new_weights"
            ]:
                class_features = [
                    collection.features[
                        collection.original_targets == int(class_id)
                    ]
                    for class_id in self.protocol.classes_for_session(session_id)
                ]
                imprinted = cosine_imprinted_weights(
                    class_features,
                    old_weights.cpu(),
                    epsilon=float(
                        self.feature_cosine_distillation_options["epsilon"]
                    ),
                )
                with torch.no_grad():
                    self.model.classifier.weight[known:total].copy_(
                        imprinted.to(self.device)
                    )
                    self.model.classifier.bias[known:total].zero_()
            if self.feature_cosine_distillation_options[
                "freeze_old_weights"
            ]:
                self.model.classifier.weight.register_hook(
                    lambda gradient, boundary=known: torch.cat(
                        (
                            torch.zeros_like(gradient[:boundary]),
                            gradient[boundary:],
                        ),
                        dim=0,
                    )
                )
        elif hasattr(self.model, "expand_classes"):
            self.model.expand_classes(total)
        else:
            raise TypeError("model has no expansion contract")

    def _source_base_record(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        if self.base_checkpoint_path is None:
            raise RuntimeError("base checkpoint path is missing")
        session_log = self.base_checkpoint_path.parent.parent / "sessions.jsonl"
        if session_log.exists():
            for line in session_log.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if int(record.get("session_id", -1)) == 0:
                    return record
        embedded = checkpoint.get("session_record")
        if embedded is not None:
            return copy.deepcopy(embedded)
        records = checkpoint.get("metrics", {}).get("records", [])
        if len(records) != 1:
            raise ValueError(
                "base checkpoint has no recoverable session-0 record"
            )
        return {
            "session_id": 0,
            "method": self.method,
            "geometry_mode": self.geometry_mode,
            "seen_class_count": self.protocol.session(0).stop,
            "memory_size": len(self.memory),
            "training": {
                "source": "shared_base_checkpoint",
                "epoch_logs": [],
            },
            "post_training": None,
            "evaluation": copy.deepcopy(records[0]),
        }

    def _validate_base_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("framework") != "sacil-unified":
            raise ValueError("base checkpoint is not from the unified runner")
        if int(checkpoint.get("session_id", -1)) != 0:
            raise ValueError("base checkpoint must be session_00.pt")
        if checkpoint.get("protocol_id") != self.protocol.protocol_id:
            raise ValueError("base checkpoint protocol does not match config")
        if str(checkpoint.get("method", "")).lower() != self.method:
            raise ValueError(
                "base checkpoint substrate does not match target config"
            )
        source_config = checkpoint.get("config")
        if not isinstance(source_config, dict):
            raise ValueError("base checkpoint has no resolved config")
        if base_recipe_signature(source_config) != base_recipe_signature(
            self.config
        ):
            raise ValueError(
                "base checkpoint session-0 recipe differs from target config"
            )
        records = checkpoint.get("metrics", {}).get("records", [])
        if len(records) != 1 or int(records[0].get("session_id", -1)) != 0:
            raise ValueError("base checkpoint must contain exactly one S0 metric")
        class_means = checkpoint.get("class_means")
        expected_classes = self.protocol.session(0).stop
        if (
            not isinstance(class_means, Tensor)
            or class_means.shape[0] != expected_classes
        ):
            raise ValueError("base checkpoint has invalid S0 class means")

    def validate_base_checkpoint(self) -> dict[str, Any] | None:
        """Validate a configured shared base without mutating trainer state."""

        if self.base_checkpoint_path is None:
            return None
        checkpoint = load_checkpoint(self.base_checkpoint_path, map_location="cpu")
        self._validate_base_checkpoint(checkpoint)
        return {
            "path": str(self.base_checkpoint_path),
            "method": checkpoint["method"],
            "session_id": int(checkpoint["session_id"]),
            "seed": int(checkpoint["config"].get("seed", 1)),
        }

    def _bootstrap_from_base_checkpoint(self) -> dict[str, Any]:
        if self.base_checkpoint_path is None:
            raise RuntimeError("base checkpoint path is missing")
        checkpoint = load_checkpoint(self.base_checkpoint_path, map_location="cpu")
        self._validate_base_checkpoint(checkpoint)
        base_classes = self.protocol.session(0).stop
        model = self._new_model(base_classes)
        if type(model).__name__ != str(checkpoint.get("model_type", "")):
            raise ValueError("base checkpoint model type does not match config")
        model.load_state_dict(checkpoint["model"], strict=True)
        self.model = model.to(self.device)
        self.memory = ExemplarMemory.from_state_dict(checkpoint["memory"])
        self.metrics = CILMetricsTracker.from_state_dict(checkpoint["metrics"])
        self.class_means = checkpoint["class_means"].detach().cpu().clone()

        source_hierarchy = (
            checkpoint.get("config", {})
            .get("method", {})
            .get("hierarchy", {})
        )
        target_hierarchy = self.config["method"].get("hierarchy", {})
        has_artifacts = all(
            key in checkpoint for key in ("tree", "prototypes", "anchors")
        )
        if self.geometry_mode == "none":
            self.sacil_tree = None
            self.sacil_prototypes = None
            self.sacil_anchors = None
        elif has_artifacts and source_hierarchy == target_hierarchy:
            self.sacil_tree = HierarchyTree.from_state_dict(checkpoint["tree"])
            self.sacil_prototypes = PrototypeBank.from_state_dict(
                checkpoint["prototypes"]
            )
            self.sacil_anchors = HierarchicalAnchorBank.from_state_dict(
                checkpoint["anchors"]
            )
            dump_json(
                self.sacil_tree.state_dict(),
                self.run_dir / "tree_session_00.json",
            )
        else:
            self._build_sacil_artifacts(0)

        if "rng_state" not in checkpoint:
            raise ValueError("base checkpoint has no RNG state")
        restore_rng_state(checkpoint["rng_state"])
        self.start_session = 1

        record = copy.deepcopy(self._source_base_record(checkpoint))
        record.update(
            method=self.method,
            geometry_mode=self.geometry_mode,
            geometry_options=copy.deepcopy(self.geometry_options),
            casper_options=copy.deepcopy(self.casper_options),
            branch_masked_kd_options=copy.deepcopy(
                self.branch_masked_kd_options
            ),
            selective_kd_options=copy.deepcopy(self.selective_kd_options),
            bgs_options=copy.deepcopy(self.bgs_options),
            checkpoint=str(self.checkpoint_dir / "session_00.pt"),
            base_reused=True,
            shared_from_checkpoint=str(self.base_checkpoint_path),
        )
        self._save_checkpoint(0, session_record=record)
        return record

    def run(self) -> dict[str, Any]:
        session_log = self.run_dir / "sessions.jsonl"
        metrics_path = self.run_dir / "metrics.json"
        existing_checkpoints = tuple(self.checkpoint_dir.glob("session_*.pt"))
        if session_log.exists() or metrics_path.exists() or existing_checkpoints:
            raise FileExistsError(
                "the unified run directory already contains training "
                f"artifacts: {self.run_dir}; choose a new --run-name"
            )
        if self.base_checkpoint_path is not None:
            base_record = self._bootstrap_from_base_checkpoint()
            with session_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(base_record, ensure_ascii=False) + "\n")
        for session_id in range(self.start_session, self.max_sessions):
            started = time.time()
            record = self._run_session(session_id)
            record["elapsed_seconds"] = time.time() - started
            with session_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = self.metrics.summary()
        dump_json(
            {"summary": summary, "sessions": self.metrics.records},
            metrics_path,
        )
        return summary

    def _run_session(self, session_id: int) -> dict[str, Any]:
        session = self.protocol.session(session_id)
        old_indices = self.memory.all_indices(self.protocol.class_order)
        if session_id == 0 or self.method == "joint":
            self.model = self._new_model(session.stop).to(self.device)
            teacher = None
        else:
            if self.model is None:
                raise RuntimeError("incremental session has no model")
            teacher = copy.deepcopy(self.model).to(self.device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self._expand_model(session_id)
            self.model.to(self.device)

        geometry = self._prepare_sacil_geometry(session_id, teacher)
        self._prepare_edge_topology(session_id, teacher)
        self._prepare_branch_masked_kd(session_id, teacher)
        self._prepare_bgs(session_id, teacher)
        train_dataset = self._training_dataset(session_id, old_indices)
        train_loader = self._loader(
            train_dataset, shuffle=True, session_id=session_id
        )
        prototype_loader = (
            self._loader(
                self._prototype_dataset(session_id, old_indices),
                shuffle=False,
                session_id=session_id + 5000,
            )
            if self.method == "sacil"
            else None
        )
        training = self._train_session(
            session_id,
            train_loader,
            teacher,
            geometry,
            prototype_loader,
        )
        # PODNet, AFC, and CREATE construct a temporary balanced memory for
        # their classifier fine-tuning stage.  Their released lifecycles then
        # rebuild the incoming-class exemplars with the final, fine-tuned
        # representation before evaluation/after_task.
        self._update_memory(session_id)
        post = self._post_training(session_id, train_loader, teacher)
        if session_id > 0 and self.method in {"podnet", "afc", "create"}:
            self._update_memory(session_id)
        self._build_evaluation_state(session_id)
        evaluation = self._evaluate_session(session_id)
        metric = self.metrics.update(session_id, evaluation)
        record = {
            "session_id": session_id,
            "method": self.method,
            "geometry_mode": self.geometry_mode,
            "geometry_options": copy.deepcopy(self.geometry_options),
            "casper_options": copy.deepcopy(self.casper_options),
            "edge_topology_options": copy.deepcopy(
                self.edge_topology_options
            ),
            "branch_masked_kd_options": copy.deepcopy(
                self.branch_masked_kd_options
            ),
            "selective_kd_options": copy.deepcopy(
                self.selective_kd_options
            ),
            "bgs_options": copy.deepcopy(self.bgs_options),
            "bgs_reference_path": (
                None
                if self._bgs_reference_path is None
                else str(self._bgs_reference_path)
            ),
            "bgs_reference_sha256": self._bgs_reference_sha256,
            "bgs_mask_diagnostics": (
                None
                if self.bgs_reference is None
                else copy.deepcopy(self.bgs_reference.mask_diagnostics)
            ),
            "exploration_provenance": copy.deepcopy(
                self.config.get("exploration_provenance")
            ),
            "branch_masked_kd_mapping": (
                None
                if self.branch_masked_kd_reference is None
                else {
                    "branch_node_ids": list(
                        self.branch_masked_kd_reference.branch_node_ids
                    ),
                    "masked_class_ratio": float(
                        self.branch_masked_kd_reference.branch_class_mask
                        .float()
                        .mean()
                    ),
                }
            ),
            "seen_class_count": session.stop,
            "memory_size": len(self.memory),
            "training": training,
            "post_training": post,
            "evaluation": metric,
        }
        checkpoint = self._save_checkpoint(
            session_id, session_record=record
        )
        record["checkpoint"] = str(checkpoint)
        return record

    def _prepare_sacil_geometry(
        self, session_id: int, teacher: nn.Module | None
    ) -> AnchorGeometryLoss | None:
        if self.geometry_mode == "none" or session_id == 0:
            return None
        if (
            teacher is None
            or self.sacil_anchors is None
            or self.sacil_tree is None
        ):
            raise RuntimeError("SACIL incremental session lacks old anchors")
        conflict = self.config["method"].get("conflict", {})
        if self.geometry_mode == "global":
            weights = global_preservation_weights(self.sacil_anchors)
        else:
            collection = self._incoming_collection(session_id)
            incoming = compute_prototypes(
                collection.features,
                collection.original_targets,
                self.protocol.classes_for_session(session_id),
            )
            weights = compute_conflict_weights(
                incoming,
                self.sacil_anchors,
                self.sacil_tree,
                max_neighbors=int(conflict.get("max_neighbors", 5)),
                old_class_ratio=float(conflict.get("old_class_ratio", 0.1)),
                temperature=float(conflict.get("temperature", 0.05)),
                min_preservation_weight=float(
                    conflict.get("min_preservation_weight", 0.1)
                ),
                ancestor_decay=float(conflict.get("ancestor_decay", 0.5)),
            )
        if self.geometry_options["reliability"] == "inverse_angular_dispersion":
            memory_loader = self._memory_loader(session_id, augment=False)
            old_collection = collect_features(teacher, memory_loader, self.device)
            leaf_reliability, internal_reliability = (
                inverse_angular_dispersion_reliability(
                    old_collection.features,
                    old_collection.original_targets,
                    self.sacil_anchors,
                    self.sacil_tree,
                    power=self.geometry_options["reliability_power"],
                    epsilon=self.geometry_options["reliability_epsilon"],
                    min_weight=self.geometry_options[
                        "reliability_min_weight"
                    ],
                    max_weight=self.geometry_options[
                        "reliability_max_weight"
                    ],
                )
            )
            weights.leaf_weights = weights.leaf_weights * leaf_reliability
            weights.internal_weights = (
                weights.internal_weights * internal_reliability
            )
        return AnchorGeometryLoss(
            self.sacil_anchors,
            weights.leaf_weights,
            weights.internal_weights,
            use_internal_anchors=self.geometry_mode != "flat",
            anchor_frame=self.geometry_options["anchor_frame"],
            fixed_mix=self.geometry_options["fixed_mix"],
            objective=self.geometry_options["objective"],
            weight_normalization=self.geometry_options[
                "weight_normalization"
            ],
            triplet_margin_scale=self.geometry_options[
                "triplet_margin_scale"
            ],
            rank_tolerance=self.geometry_options["rank_tolerance"],
        ).to(self.device)

    def _edge_representative_batch(
        self, session_id: int
    ) -> tuple[Tensor, tuple[int, ...], tuple[int, ...]]:
        """Load the fixed herding-prefix representatives in one global batch."""

        count = int(self.edge_topology_options["representatives_per_class"])
        indices: list[int] = []
        class_ids: list[int] = []
        for class_id in self.protocol.old_classes(session_id):
            candidates = self.memory.indices_for_class(class_id)
            if len(candidates) < count:
                raise RuntimeError(
                    f"class {class_id} has only {len(candidates)} exemplars; "
                    f"edge topology requires {count}"
                )
            selected = candidates[:count]
            indices.extend(selected)
            class_ids.extend([int(class_id)] * count)
        if len(indices) < 2:
            raise RuntimeError("edge topology found fewer than two representatives")
        dataset = self.data.replay_dataset(indices, augment=False)
        loader = self._loader(
            dataset,
            shuffle=False,
            session_id=session_id + 12000,
            batch_size=len(indices),
        )
        batches = list(loader)
        if len(batches) != 1:
            raise RuntimeError("representative loader must form one global batch")
        batch = batches[0]
        loaded_indices = tuple(int(value) for value in batch["index"].tolist())
        loaded_classes = tuple(
            int(value) for value in batch["original_target"].tolist()
        )
        if loaded_indices != tuple(indices) or loaded_classes != tuple(class_ids):
            raise RuntimeError("representative loader changed the fixed ordering")
        return batch["image"].cpu(), loaded_indices, loaded_classes

    def _conflict_branch_edge_weights(
        self,
        session_id: int,
        teacher_features: Tensor,
        representative_class_ids: tuple[int, ...],
    ) -> tuple[Tensor, tuple[str, ...]]:
        """Relax edges incident to the closest old internal branch."""

        old_class_ids = self.protocol.old_classes(session_id)
        targets = torch.tensor(representative_class_ids, dtype=torch.long)
        prototypes = compute_prototypes(
            teacher_features.cpu(), targets, old_class_ids
        )
        incremental_targets = torch.tensor(
            [
                self.protocol.incremental_label(class_id)
                for class_id in representative_class_ids
            ],
            dtype=torch.long,
        )
        confusion = cosine_soft_confusion(
            teacher_features.cpu(),
            incremental_targets,
            prototypes,
            temperature=float(
                self.config["method"]
                .get("edge_topology", {})
                .get("taxonomy_temperature", 0.2)
            ),
        )
        tree = GriffinPeronaGreedy().build(
            old_class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(
            PrototypeBank(old_class_ids, prototypes), tree
        )
        internal_ids, internal_anchors = anchors.internal_without_root()
        if not internal_ids:
            return global_edge_weights(len(representative_class_ids)), ()

        incoming_collection = self._incoming_collection(session_id)
        incoming = compute_prototypes(
            incoming_collection.features,
            incoming_collection.original_targets,
            self.protocol.classes_for_session(session_id),
        )
        similarities = F.normalize(incoming.float(), dim=1) @ F.normalize(
            internal_anchors.float(), dim=1
        ).T
        per_class = min(
            int(self.edge_topology_options["conflict_branches_per_new_class"]),
            len(internal_ids),
        )
        selected_positions = torch.topk(
            similarities, k=per_class, dim=1
        ).indices.flatten()
        selected_node_ids = tuple(
            sorted({internal_ids[int(position)] for position in selected_positions})
        )
        conflict_classes: set[int] = set()
        for node_id in selected_node_ids:
            conflict_classes.update(tree.descendants(node_id))
        minimum = float(self.edge_topology_options["min_edge_weight"])
        conflict_membership = torch.tensor(
            [class_id in conflict_classes for class_id in representative_class_ids],
            dtype=torch.bool,
        )
        if self.edge_topology_options["edge_weighting"] == "conflict_subtree_inside":
            edge_weights = conflict_subtree_inside_edge_weights(
                conflict_membership, min_edge_weight=minimum
            )
        else:
            representative_weights = torch.where(
                conflict_membership,
                torch.full(conflict_membership.shape, minimum),
                torch.ones(conflict_membership.shape),
            )
            edge_weights = incident_edge_weights(representative_weights)
        return edge_weights, selected_node_ids

    def _stratified_conflict_membership(
        self,
        session_id: int,
        teacher_features: Tensor,
        representative_class_ids: tuple[int, ...],
    ) -> tuple[Tensor, tuple[str, ...]]:
        """Freeze a deterministic union of incoming-nearest old subtrees."""

        old_class_ids = self.protocol.old_classes(session_id)
        targets = torch.tensor(representative_class_ids, dtype=torch.long)
        prototypes = compute_prototypes(
            teacher_features.cpu(), targets, old_class_ids
        )
        incremental_targets = torch.tensor(
            [
                self.protocol.incremental_label(class_id)
                for class_id in representative_class_ids
            ],
            dtype=torch.long,
        )
        confusion = cosine_soft_confusion(
            teacher_features.cpu(),
            incremental_targets,
            prototypes,
            temperature=float(
                self.config["method"]
                .get("edge_topology", {})
                .get("taxonomy_temperature", 0.2)
            ),
        )
        tree = GriffinPeronaGreedy().build(
            old_class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(
            PrototypeBank(old_class_ids, prototypes), tree
        )
        internal_ids, internal_anchors = anchors.internal_without_root()
        if not internal_ids:
            return torch.zeros(
                len(representative_class_ids), dtype=torch.bool
            ), ()

        incoming_collection = self._incoming_collection(session_id)
        incoming = compute_prototypes(
            incoming_collection.features,
            incoming_collection.original_targets,
            self.protocol.classes_for_session(session_id),
        )
        similarities = F.normalize(incoming.float(), dim=1) @ F.normalize(
            internal_anchors.float(), dim=1
        ).T
        if not torch.isfinite(similarities).all():
            raise RuntimeError("conflict-branch similarities must be finite")
        per_class = min(
            int(self.edge_topology_options["conflict_branches_per_new_class"]),
            len(internal_ids),
        )
        selected: set[str] = set()
        # Stable score-descending, node-ID-ascending ranking makes exact ties
        # deterministic across repeated construction. Overlapping selections
        # are intentionally collapsed before descendant union.
        for row in similarities.detach().cpu():
            ranked_positions = sorted(
                range(len(internal_ids)),
                key=lambda position: (
                    -float(row[position]),
                    str(internal_ids[position]),
                ),
            )
            selected.update(
                str(internal_ids[position])
                for position in ranked_positions[:per_class]
            )
        selected_node_ids = tuple(sorted(selected))
        conflict_subtrees = tuple(
            tuple(int(value) for value in tree.descendants(node_id))
            for node_id in selected_node_ids
        )
        membership = conflict_union_membership(
            representative_class_ids, conflict_subtrees
        )
        return membership, selected_node_ids

    def _prepare_edge_topology(
        self, session_id: int, teacher: nn.Module | None
    ) -> None:
        """Cache one old-representative teacher topology for this session."""

        self.edge_topology_reference = None
        self._edge_topology_images = None
        self._edge_topology_loss = None
        self._last_edge_topology_stats = {}
        if not self.edge_topology_options["enabled"] or session_id == 0:
            return
        if teacher is None:
            raise RuntimeError("edge topology requires an old teacher")
        images, indices, class_ids = self._edge_representative_batch(session_id)
        with torch.inference_mode():
            teacher_features = teacher.extract_features(
                images.to(self.device, non_blocking=True)
            ).detach()
        reference_edges = pairwise_cosine_edge_vector(teacher_features).cpu()
        objective = str(self.edge_topology_options["objective"])
        weighting = str(self.edge_topology_options["edge_weighting"])
        if objective == STRATIFIED_EDGE_TOPOLOGY_OBJECTIVE:
            conflict_membership, conflict_node_ids = (
                self._stratified_conflict_membership(
                    session_id, teacher_features, class_ids
                )
            )
            reference = StratifiedHierarchicalEdgeReference(
                session_id=session_id,
                representatives_per_class=int(
                    self.edge_topology_options["representatives_per_class"]
                ),
                representative_indices=indices,
                representative_class_ids=class_ids,
                reference_edges=reference_edges,
                edge_group_ids=stratified_edge_group_ids(
                    conflict_membership
                ).cpu(),
                beta_boundary=float(
                    self.edge_topology_options["beta_boundary"]
                ),
                gamma_conflict=float(
                    self.edge_topology_options["gamma_conflict"]
                ),
                conflict_node_ids=conflict_node_ids,
            )
        else:
            if weighting == "global":
                edge_weights = global_edge_weights(len(indices))
                conflict_node_ids = ()
            else:
                edge_weights, conflict_node_ids = (
                    self._conflict_branch_edge_weights(
                        session_id, teacher_features, class_ids
                    )
                )
            reference = HierarchicalEdgeReference(
                session_id=session_id,
                representatives_per_class=int(
                    self.edge_topology_options["representatives_per_class"]
                ),
                representative_indices=indices,
                representative_class_ids=class_ids,
                reference_edges=reference_edges,
                edge_weights=edge_weights,
                edge_weighting=weighting,
                conflict_node_ids=conflict_node_ids,
            )
        self.edge_topology_reference = reference
        # The representative set is small (at most 190 CIFAR images for K=2)
        # and reused throughout the session.  Cache it on-device to avoid one
        # host-to-device copy on every scheduled topology update.
        self._edge_topology_images = images.to(
            self.device, non_blocking=True
        )
        self._edge_topology_loss = reference.loss_module().to(self.device)

    def _prepare_branch_masked_kd(
        self, session_id: int, teacher: nn.Module | None
    ) -> None:
        """Freeze each incoming class's top-1 old visual hierarchy branch."""

        self.branch_masked_kd_reference = None
        if not self.branch_masked_kd_options["enabled"] or session_id == 0:
            return
        if teacher is None:
            raise RuntimeError("branch-masked KD requires an old teacher")

        old_class_ids = self.protocol.old_classes(session_id)
        old_collection = collect_features(
            teacher,
            self._memory_loader(session_id, augment=False),
            self.device,
        )
        old_prototypes = compute_prototypes(
            old_collection.features,
            old_collection.original_targets,
            old_class_ids,
        )
        taxonomy_temperature = float(
            self.config["method"]
            .get("hierarchy", {})
            .get("taxonomy_temperature", 0.2)
        )
        confusion = cosine_soft_confusion(
            old_collection.features,
            old_collection.targets,
            old_prototypes,
            temperature=taxonomy_temperature,
        )
        tree = GriffinPeronaGreedy().build(
            old_class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(
            PrototypeBank(old_class_ids, old_prototypes), tree
        )
        internal_ids, internal_anchors = anchors.internal_without_root()
        if not internal_ids:
            raise RuntimeError("old hierarchy has no non-root internal branch")

        incoming_dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        incoming_collection = collect_features(
            teacher,
            self._loader(
                incoming_dataset,
                shuffle=False,
                session_id=session_id + 13000,
            ),
            self.device,
        )
        new_original_ids = self.protocol.classes_for_session(session_id)
        incoming_prototypes = compute_prototypes(
            incoming_collection.features,
            incoming_collection.original_targets,
            new_original_ids,
        )
        similarities = F.normalize(incoming_prototypes.float(), dim=1) @ F.normalize(
            internal_anchors.float(), dim=1
        ).T
        positions = similarities.argmax(dim=1).tolist()
        branch_node_ids = tuple(internal_ids[int(value)] for value in positions)
        branch_mask = torch.tensor(
            [
                [
                    old_class_id in tree.descendants(node_id)
                    for old_class_id in old_class_ids
                ]
                for node_id in branch_node_ids
            ],
            dtype=torch.bool,
        )
        known = self.protocol.session(session_id).start
        reference = BranchMaskedKDReference(
            session_id=session_id,
            known_classes=known,
            new_incremental_labels=tuple(
                range(known, self.protocol.session(session_id).stop)
            ),
            new_original_class_ids=tuple(new_original_ids),
            branch_node_ids=branch_node_ids,
            branch_class_mask=branch_mask,
            teacher_tree_state=tree.state_dict(),
        )
        self.branch_masked_kd_reference = reference
        dump_json(
            {
                "session_id": session_id,
                "new_incremental_labels": list(
                    reference.new_incremental_labels
                ),
                "new_original_class_ids": list(
                    reference.new_original_class_ids
                ),
                "branch_node_ids": list(reference.branch_node_ids),
                "branch_class_mask": reference.branch_class_mask.tolist(),
                "teacher_tree_state": reference.teacher_tree_state,
            },
            self.run_dir / f"branch_kd_mapping_session_{session_id:02d}.json",
        )

    def _prepare_bgs(self, session_id: int, teacher: nn.Module | None) -> None:
        self.bgs_reference = None
        self._bgs_loss = None
        self._bgs_positive_prototypes = None
        self._bgs_reference_path = None
        self._bgs_reference_sha256 = None
        if not self.bgs_options["enabled"] or session_id == 0:
            return
        if teacher is None:
            raise RuntimeError("BGS requires an old teacher")
        old_ids = self.protocol.old_classes(session_id)
        old_collection = collect_features(
            teacher,
            self._memory_loader(session_id, augment=False),
            self.device,
        )
        old_prototypes = compute_prototypes(
            old_collection.features,
            old_collection.original_targets,
            old_ids,
        )
        confusion = cosine_soft_confusion(
            old_collection.features,
            old_collection.targets,
            old_prototypes,
            temperature=0.2,
        )
        tree = GriffinPeronaGreedy().build(
            old_ids,
            symmetric_affinity(confusion),
        )
        anchors = HierarchicalAnchorBank.from_tree(
            PrototypeBank(old_ids, old_prototypes),
            tree,
        )
        internal_ids, internal_anchors = anchors.internal_without_root()
        incoming = collect_features(
            teacher,
            self._loader(
                self.data.new_train_dataset(
                    session_id,
                    augment=False,
                    samples_per_class=self.debug_train_samples_per_class,
                ),
                shuffle=False,
                session_id=session_id + 14000,
            ),
            self.device,
        )
        new_ids = self.protocol.classes_for_session(session_id)
        incoming_prototypes = compute_prototypes(
            incoming.features,
            incoming.original_targets,
            new_ids,
        )
        branch_source = self.bgs_options["branch_source"]
        branch_selection_diagnostics: dict[str, Any] | None = None
        if branch_source == "i2_teacher_internal_top1":
            # Preserve the legacy BGS-v1 selection path exactly.
            scores = (
                F.normalize(incoming_prototypes, dim=1)
                @ F.normalize(internal_anchors, dim=1).T
            )
            positions = scores.argmax(dim=1)
            raw_nodes = tuple(internal_ids[int(value)] for value in positions)
            raw_scores = scores.gather(1, positions[:, None]).squeeze(1)
            primary_internal_nodes = raw_nodes
            primary_positions = tuple(int(value) for value in positions)
            threshold_scores = raw_scores
        else:
            configured_cap = int(self.bgs_options["max_branch_leaves"])
            max_coverage = float(
                self.bgs_options["max_conflict_leaf_coverage"]
            )
            effective_cap = effective_bounded_branch_cap(
                configured_cap,
                max_coverage,
                len(old_ids),
                len(new_ids),
            )
            selection = nearest_leaf_bounded_ancestor_branches(
                tree,
                old_ids,
                anchors.leaf_anchors,
                incoming_prototypes,
                max_branch_leaves=effective_cap,
            )
            raw_nodes = selection.selected_branch_nodes
            primary_internal_nodes = selection.primary_internal_nodes
            all_internal_positions = {
                node_id: position
                for position, node_id in enumerate(
                    anchors.internal_node_ids
                )
            }
            non_root_positions = {
                node_id: position
                for position, node_id in enumerate(internal_ids)
            }
            leaf_positions = {
                tree.leaf_node_id(class_id): position
                for position, class_id in enumerate(
                    anchors.leaf_class_ids
                )
            }
            selected_anchors = torch.stack(
                [
                    anchors.leaf_anchors[leaf_positions[node_id]]
                    if tree.nodes[node_id].is_leaf
                    else anchors.internal_anchors[
                        all_internal_positions[node_id]
                    ]
                    for node_id in raw_nodes
                ],
                dim=0,
            )
            primary_anchors = torch.stack(
                [
                    anchors.internal_anchors[
                        all_internal_positions[node_id]
                    ]
                    for node_id in primary_internal_nodes
                ],
                dim=0,
            )
            normalized_incoming = F.normalize(
                incoming_prototypes.float(), dim=1
            )
            raw_scores = (
                normalized_incoming
                * F.normalize(selected_anchors.float(), dim=1)
            ).sum(dim=1)
            threshold_scores = (
                normalized_incoming
                * F.normalize(primary_anchors.float(), dim=1)
            ).sum(dim=1)
            primary_positions = tuple(
                non_root_positions.get(node_id, -1)
                for node_id in primary_internal_nodes
            )
            branch_selection_diagnostics = {
                "configured_max_branch_leaves": configured_cap,
                "effective_max_branch_leaves": effective_cap,
                "max_conflict_leaf_coverage": max_coverage,
                "nearest_old_leaf_positions": list(
                    selection.nearest_leaf_positions
                ),
                "nearest_old_leaf_original_ids": list(
                    selection.nearest_leaf_original_ids
                ),
                "nearest_old_leaf_node_ids": list(
                    selection.nearest_leaf_node_ids
                ),
                "nearest_old_leaf_scores": (
                    selection.nearest_leaf_scores.tolist()
                ),
                "selected_branch_nodes": list(raw_nodes),
                "selected_branch_leaf_counts": list(
                    selection.selected_branch_leaf_counts
                ),
                "selected_branch_is_leaf": [
                    bool(tree.nodes[node_id].is_leaf)
                    for node_id in raw_nodes
                ],
                "primary_internal_nodes": list(primary_internal_nodes),
                "primary_internal_positions": list(primary_positions),
            }
        canonical, raw_to_canonical = canonical_regions(tree, raw_nodes)
        leaf_descendants = [(int(class_id),) for class_id in anchors.leaf_class_ids]
        internal_descendants = [tree.descendants(node) for node in internal_ids]
        sample_regions, leaf_regions = endpoint_regions(
            tree,
            canonical,
            old_ids,
            leaf_descendants,
        )
        _, internal_regions = endpoint_regions(
            tree,
            canonical,
            old_ids,
            internal_descendants,
        )
        geometry = self.bgs_options["geometry"]
        structured_mode = (
            "structured"
            if geometry["mask_mode"] == "random_pair"
            else geometry["mask_mode"]
        )
        leaf_types, leaf_weights = pair_types_and_weights(
            sample_regions,
            leaf_regions,
            inside_weight=geometry["inside_weight"],
            boundary_weight=geometry["boundary_weight"],
            mask_mode=structured_mode,
        )
        internal_types, internal_weights = pair_types_and_weights(
            sample_regions,
            internal_regions,
            inside_weight=geometry["inside_weight"],
            boundary_weight=geometry["boundary_weight"],
            mask_mode=structured_mode,
        )
        if geometry["mask_mode"] == "global":
            leaf_weights.fill_(1.0)
            internal_weights.fill_(1.0)
        leaf_row_deficit = (1.0 - leaf_weights).sum(1).clone()
        internal_row_deficit = (1.0 - internal_weights).sum(1).clone()
        random_seeds, random_permutations = {}, {}
        if geometry["mask_mode"] == "random_pair":
            leaf_weights, leaf_seeds, leaf_permutations = (
                row_permuted_random_weights(
                    leaf_weights,
                    old_ids,
                    experiment_seed=self.seed,
                    session_id=session_id,
                    group="leaf",
                )
            )
            internal_weights, internal_seeds, internal_permutations = (
                row_permuted_random_weights(
                    internal_weights,
                    old_ids,
                    experiment_seed=self.seed,
                    session_id=session_id,
                    group="internal",
                )
            )
            random_seeds = {"leaf": leaf_seeds, "internal": internal_seeds}
            random_permutations = {
                "leaf": leaf_permutations,
                "internal": internal_permutations,
            }
        insertion = self.bgs_options["insertion"]
        if (
            branch_selection_diagnostics is not None
            and insertion["enabled"]
            and any(position < 0 for position in primary_positions)
        ):
            raise RuntimeError(
                "bounded leaf-only branch with root parent cannot use BGS insertion"
            )
        negative_positions = []
        for new_index, node in enumerate(raw_nodes):
            candidate_positions = negative_candidate_positions(
                tree,
                old_ids,
                node,
                insertion["negative_scope"],
            )
            similarity = (
                F.normalize(
                    incoming_prototypes[new_index : new_index + 1],
                    dim=1,
                )
                @ F.normalize(
                    anchors.leaf_anchors[list(candidate_positions)],
                    dim=1,
                ).T
            )
            k = min(insertion["negatives_per_class"], len(candidate_positions))
            chosen = similarity.topk(k).indices[0].tolist()
            negative_positions.append(
                tuple(candidate_positions[int(value)] for value in chosen)
            )
        thresholds = (
            threshold_scores - insertion["parent_slack"]
        ).clamp(-1, 1)
        leaf_diagnostics = pair_mask_summary(
            leaf_types,
            leaf_weights,
            inside_weight=geometry["inside_weight"],
            boundary_weight=geometry["boundary_weight"],
        )
        internal_diagnostics = pair_mask_summary(
            internal_types,
            internal_weights,
            inside_weight=geometry["inside_weight"],
            boundary_weight=geometry["boundary_weight"],
        )
        conflict_leaves = set().union(
            *(set(tree.descendants(node)) for node in canonical)
        )
        if branch_selection_diagnostics is not None:
            union_diagnostics = bounded_conflict_union_diagnostics(
                tree,
                canonical,
                old_ids,
                max_conflict_leaf_coverage=float(
                    self.bgs_options["max_conflict_leaf_coverage"]
                ),
            )
            if max(
                branch_selection_diagnostics[
                    "selected_branch_leaf_counts"
                ]
            ) > int(
                branch_selection_diagnostics[
                    "effective_max_branch_leaves"
                ]
            ):
                raise AssertionError(
                    "bounded BGS branch exceeded the effective cap"
                )
            branch_selection_diagnostics.update(union_diagnostics)
        tree_state = tree.state_dict()
        tree_hash = hashlib.sha256(
            json.dumps(tree_state, sort_keys=True).encode("utf-8")
        ).hexdigest()
        anchor_hash = hashlib.sha256(
            (
                tensor_sha256(anchors.leaf_anchors)
                + tensor_sha256(anchors.internal_anchors)
            ).encode("utf-8")
        ).hexdigest()
        random_pair_budget_equal = bool(
            torch.allclose(
                (1.0 - leaf_weights).sum(1),
                leaf_row_deficit,
                atol=1e-6,
                rtol=0.0,
            )
            and torch.allclose(
                (1.0 - internal_weights).sum(1),
                internal_row_deficit,
                atol=1e-6,
                rtol=0.0,
            )
        )
        mask_diagnostics = {
            "mask_mode": geometry["mask_mode"],
            "canonical_component_count": len(canonical),
            "conflict_leaf_count": len(conflict_leaves),
            "conflict_leaf_coverage": len(conflict_leaves) / len(old_ids),
            "leaf": leaf_diagnostics,
            "internal_without_root": internal_diagnostics,
            "tree_sha256": tree_hash,
            "anchor_bank_sha256": anchor_hash,
            "random_pair_budget_equal_by_row": random_pair_budget_equal,
        }
        if branch_selection_diagnostics is not None:
            mask_diagnostics["bounded_branch_selection"] = copy.deepcopy(
                branch_selection_diagnostics
            )
        if geometry["mask_mode"] == "random_pair" and not mask_diagnostics[
            "random_pair_budget_equal_by_row"
        ]:
            raise AssertionError("random-pair control changed a row budget")
        reference = BoundaryGraphSurgeryReference(
            session_id=session_id,
            old_original_ids=tuple(old_ids),
            new_original_ids=tuple(new_ids),
            raw_branch_nodes=raw_nodes,
            raw_branch_scores=raw_scores.cpu(),
            canonical_nodes=canonical,
            raw_to_canonical=raw_to_canonical,
            tree_state=tree_state,
            anchor_state=anchors.state_dict(),
            leaf_pair_types=leaf_types,
            internal_pair_types=internal_types,
            leaf_weights=leaf_weights,
            internal_weights=internal_weights,
            random_seeds=random_seeds,
            random_permutations=random_permutations,
            incoming_teacher_prototypes=incoming_prototypes.cpu(),
            negative_class_positions=tuple(negative_positions),
            parent_thresholds=thresholds.cpu(),
            options=copy.deepcopy(self.bgs_options),
            old_incremental_ids=tuple(range(len(old_ids))),
            new_incremental_ids=tuple(
                range(
                    self.protocol.session(session_id).start,
                    self.protocol.session(session_id).stop,
                )
            ),
            sample_region_ids=sample_regions,
            leaf_anchor_ids=tree.leaf_node_ids(),
            leaf_anchor_region_ids=leaf_regions,
            internal_anchor_ids=internal_ids,
            internal_anchor_region_ids=internal_regions,
            primary_internal_positions=primary_positions,
            mask_diagnostics=mask_diagnostics,
        )
        self.bgs_reference = reference
        self._bgs_loss = BoundaryGraphSurgeryLoss(reference).to(self.device)
        reference_path = (
            self.run_dir / f"bgs_reference_session_{session_id:02d}.json"
        )
        dump_json(
            {
                "spec_version": self.bgs_options["spec_version"],
                "session_id": session_id,
                "old_original_ids": list(old_ids),
                "new_original_ids": list(new_ids),
                "old_incremental_ids": list(reference.old_incremental_ids),
                "new_incremental_ids": list(reference.new_incremental_ids),
                "raw_branch_nodes": list(raw_nodes),
                "raw_branch_scores": raw_scores.tolist(),
                "primary_internal_positions": list(primary_positions),
                "canonical_nodes": list(canonical),
                "raw_to_canonical": raw_to_canonical,
                "sample_region_ids": list(sample_regions),
                "leaf_anchor_ids": list(reference.leaf_anchor_ids),
                "leaf_anchor_region_ids": list(leaf_regions),
                "internal_anchor_ids": list(internal_ids),
                "internal_anchor_region_ids": list(internal_regions),
                "leaf_pair_types": leaf_types.tolist(),
                "internal_pair_types": internal_types.tolist(),
                "leaf_weights": leaf_weights.tolist(),
                "internal_weights": internal_weights.tolist(),
                "random_seeds": random_seeds,
                "random_permutations": random_permutations,
                "incoming_teacher_prototypes": incoming_prototypes.tolist(),
                "negative_class_positions": negative_positions,
                "negative_original_class_ids": [
                    [old_ids[position] for position in row]
                    for row in negative_positions
                ],
                "parent_thresholds": thresholds.tolist(),
                "options": self.bgs_options,
                "mask_diagnostics": mask_diagnostics,
                **(
                    {}
                    if branch_selection_diagnostics is None
                    else {
                        "bounded_branch_selection": (
                            branch_selection_diagnostics
                        )
                    }
                ),
            },
            reference_path,
        )
        self._bgs_reference_path = reference_path
        self._bgs_reference_sha256 = hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest()

    def _refresh_bgs_positives(self, session_id: int) -> None:
        if self.bgs_reference is None or not self.bgs_options["insertion"]["enabled"]:
            self._bgs_positive_prototypes = None
            return
        collection = self._incoming_collection(session_id)
        self._bgs_positive_prototypes = compute_prototypes(
            collection.features,
            collection.original_targets,
            self.protocol.classes_for_session(session_id),
        ).to(self.device).detach()

    def _edge_topology_component(self, step: int) -> Tensor | None:
        """Evaluate the whole fixed topology only on scheduled optimizer steps.

        No detached loss is reused between updates: skipped steps receive no
        topology term and optimize the unchanged iCaRL objective alone.
        """

        self._last_edge_topology_stats = {}
        if (
            self._edge_topology_loss is None
            or self._edge_topology_images is None
            or step
            % int(self.edge_topology_options["update_interval_steps"])
            != 0
        ):
            return None
        if self.model is None:
            raise RuntimeError("edge topology has no current model")
        was_training = self.model.training
        self.model.eval()
        features = self.model.extract_features(
            self._edge_topology_images
        )
        if was_training:
            self.model.train()
        result = self._edge_topology_loss(features)
        if isinstance(result, StratifiedEdgeCorrelationResult):
            self._last_edge_topology_stats = result.detached_metrics()
            edge_loss = result.loss
        else:
            edge_loss = result
        return float(self.edge_topology_options["lambda_edge"]) * edge_loss

    def _refresh_geometry_current_anchors(
        self,
        session_id: int,
        geometry: AnchorGeometryLoss,
    ) -> None:
        """Rebuild current anchors on old memory without changing the tree."""

        if not geometry.requires_current_anchor_refresh:
            return
        if (
            self.model is None
            or self.sacil_tree is None
            or self.sacil_anchors is None
        ):
            raise RuntimeError("current-anchor refresh lacks SACIL state")
        collection = collect_features(
            self.model,
            self._memory_loader(session_id, augment=False),
            self.device,
        )
        class_ids = self.sacil_anchors.leaf_class_ids
        prototypes = PrototypeBank(
            class_ids,
            compute_prototypes(
                collection.features,
                collection.original_targets,
                class_ids,
            ),
        )
        current_anchors = HierarchicalAnchorBank.from_tree(
            prototypes, self.sacil_tree
        )
        geometry.update_current_anchors(current_anchors)

    def _train_session(
        self,
        session_id: int,
        loader: DataLoader,
        teacher: nn.Module | None,
        geometry: AnchorGeometryLoss | None,
        prototype_loader: DataLoader | None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("training has no model")
        phase = self._phase(session_id)
        parameters = self._main_parameters()
        if self.method == "afc":
            # AFC author code clips every trainable parameter gradient through
            # per-parameter hooks registered at the beginning of each task.
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda gradient: torch.clamp(gradient, -5.0, 5.0)
                    )
        optimizer = SGD(
            parameters,
            lr=float(phase["lr"]),
            momentum=float(phase.get("momentum", 0.9)),
            weight_decay=float(phase.get("weight_decay", 5e-4)),
            nesterov=bool(phase.get("nesterov", False)),
        )
        epochs = int(phase["epochs"])
        scheduler = self._scheduler(optimizer, phase, epochs)
        fusion_loader = self._cscct_balanced_loader(session_id)
        fusion_optimizer = (
            SGD(
                list(self.model.fusion.parameters()),
                lr=float(self.config["method"].get("fusion_lr", 1e-8)),
                momentum=float(phase.get("momentum", 0.9)),
                weight_decay=float(phase.get("weight_decay", 5e-4)),
            )
            if isinstance(self.model, CSCCTIncrementalNet)
            and fusion_loader is not None
            else None
        )
        fusion_scheduler = (
            self._scheduler(fusion_optimizer, phase, epochs)
            if fusion_optimizer is not None
            else None
        )
        casper_loader = self._casper_loader(session_id, max(1, len(loader)))
        max_batches_value = self.config.get("debug", {}).get(
            "max_batches_per_epoch"
        )
        max_batches = (
            None if max_batches_value is None else int(max_batches_value)
        )
        record_edge_gradient = bool(
            self.config.get("debug", {}).get(
                "record_edge_gradient_norm", False
            )
        )
        logs = []
        geometry_anchor_refreshes = 0
        edge_topology_updates = 0
        optimizer_step = 0
        progress = tqdm(
            range(epochs), disable=bool(self.config.get("disable_tqdm", False))
        )
        for epoch in progress:
            self._refresh_bgs_positives(session_id)
            if (
                geometry is not None
                and geometry.requires_current_anchor_refresh
                and epoch
                % int(self.geometry_options["refresh_interval_epochs"])
                == 0
            ):
                self._refresh_geometry_current_anchors(session_id, geometry)
                geometry_anchor_refreshes += 1
            if self.method == "cscct":
                # The CSCCT release advances both schedulers at the start of
                # every epoch, including the zeroth phase.  Preserve that
                # ordering instead of forcing the shared end-of-epoch order.
                scheduler.step()
                if fusion_scheduler is not None:
                    fusion_scheduler.step()
            prototypes = (
                self._refresh_training_prototypes(prototype_loader)
                if prototype_loader is not None
                else None
            )
            self.model.train()
            totals: dict[str, float] = defaultdict(float)
            count = batches = correct = 0
            epoch_edge_topology_updates = 0
            epoch_edge_gradient_l1 = 0.0
            epoch_edge_gradient_l2 = 0.0
            epoch_cil_gradient_l2 = 0.0
            branch_old_kd_sum = 0.0
            branch_new_kd_sum = 0.0
            branch_teacher_mass_sum = 0.0
            branch_student_mass_sum = 0.0
            branch_masked_ratio_sum = 0.0
            branch_old_count = 0
            branch_new_count = 0
            selective_old_kd_sum = 0.0
            selective_new_kd_sum = 0.0
            selective_keep_sum = 0.0
            selective_alignment_sum = 0.0
            selective_positive_sum = 0.0
            selective_old_count = 0
            selective_new_count = 0
            feature_routing_batch_logs: list[dict[str, float | int]] = []
            feature_routing_counts = {
                name: 0 for name in ("old_conflict", "old_outside", "new")
            }
            feature_routing_weight_sums = {
                name: 0.0 for name in ("old_conflict", "old_outside", "new")
            }
            feature_routing_sample_count = 0
            feature_routing_weight_sum = 0.0
            bgs_old_count = bgs_new_count = 0
            bgs_leaf_sum = bgs_internal_sum = 0.0
            bgs_separation_sum = bgs_parent_sum = 0.0
            bgs_positive_sum = bgs_negative_sum = bgs_parent_active_sum = 0.0
            pcli_updates = pcli_conflicts = 0
            pcli_cosine_sum = pcli_retained_ratio_sum = 0.0
            pcli_projection_coefficient_sum = 0.0
            bgs_partition_aggregates: dict[str, dict[str, float]] = {
                f"{group}_{pair_type}": defaultdict(float)
                for group in ("leaf", "internal")
                for pair_type in ("inside", "boundary", "outside")
            }
            stratified_diagnostic_updates = 0
            stratified_loss_sums = {
                name: 0.0 for name in STRATIFIED_EDGE_GROUP_NAMES
            }
            stratified_current_variance_sums = {
                name: 0.0 for name in STRATIFIED_EDGE_GROUP_NAMES
            }
            stratified_active_sums = {
                name: 0 for name in STRATIFIED_EDGE_GROUP_NAMES
            }
            stratified_current_active_sums = {
                name: 0 for name in STRATIFIED_EDGE_GROUP_NAMES
            }
            stratified_reference_active = {
                name: 0 for name in STRATIFIED_EDGE_GROUP_NAMES
            }
            stratified_group_counts = {
                name: 0 for name in STRATIFIED_EDGE_GROUP_NAMES
            }
            stratified_reference_variances = {
                name: 0.0 for name in STRATIFIED_EDGE_GROUP_NAMES
            }
            casper_iterator = None if casper_loader is None else iter(casper_loader)
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True).long()
                replay_mask = batch["is_replay"].to(self.device).bool()
                replay_images = None
                if casper_iterator is not None:
                    try:
                        replay_batch = next(casper_iterator)
                    except StopIteration:
                        casper_iterator = iter(casper_loader)
                        replay_batch = next(casper_iterator)
                    replay_images = replay_batch["image"].to(
                        self.device, non_blocking=True
                    )
                # Measure the session-start topology before the training-batch
                # forward mutates BatchNorm running statistics. With an
                # unchanged student/teacher backbone, the first scheduled
                # edge loss must therefore be numerically zero.
                edge_topology = self._edge_topology_component(optimizer_step)
                components, prediction = self._loss_components(
                    session_id,
                    images,
                    targets,
                    replay_mask,
                    teacher,
                    geometry,
                    prototypes,
                    replay_images,
                )
                if self._last_branch_kd_stats:
                    stats = self._last_branch_kd_stats
                    old_count = int(stats["old_count"])
                    new_count = int(stats["new_count"])
                    branch_old_count += old_count
                    branch_new_count += new_count
                    branch_old_kd_sum += float(stats["old_kd"]) * old_count
                    branch_new_kd_sum += float(stats["new_kd"]) * new_count
                    branch_teacher_mass_sum += (
                        float(stats["teacher_retained_mass"]) * new_count
                    )
                    branch_student_mass_sum += (
                        float(stats["student_retained_mass"]) * new_count
                    )
                    branch_masked_ratio_sum += (
                        float(stats["masked_class_ratio"]) * new_count
                    )
                if self._last_selective_kd_stats:
                    stats = self._last_selective_kd_stats
                    old_count = int(stats["old_count"])
                    new_count = int(stats["new_count"])
                    selective_old_count += old_count
                    selective_new_count += new_count
                    selective_old_kd_sum += float(stats["old_kd"]) * old_count
                    selective_new_kd_sum += float(stats["new_kd"]) * new_count
                    selective_keep_sum += (
                        float(stats["new_keep_ratio"]) * new_count
                    )
                    selective_alignment_sum += (
                        float(stats["alignment_mean"]) * new_count
                    )
                    selective_positive_sum += (
                        float(stats["alignment_positive_ratio"]) * new_count
                    )
                if self._last_feature_routing_stats:
                    stats = dict(self._last_feature_routing_stats)
                    stats["batch_index"] = batch_index
                    feature_routing_batch_logs.append(stats)
                    feature_routing_sample_count += int(stats["sample_count"])
                    feature_routing_weight_sum += float(stats["weight_sum"])
                    for group_name in (
                        "old_conflict",
                        "old_outside",
                        "new",
                    ):
                        feature_routing_counts[group_name] += int(
                            stats[f"{group_name}_count"]
                        )
                        feature_routing_weight_sums[group_name] += float(
                            stats[f"{group_name}_weight_sum"]
                        )
                if self._last_bgs_stats:
                    stats = self._last_bgs_stats
                    old_count = int(stats["old_count"])
                    new_count = int(stats["new_count"])
                    bgs_old_count += old_count
                    bgs_new_count += new_count
                    bgs_leaf_sum += float(stats["leaf"]) * old_count
                    bgs_internal_sum += float(stats["internal"]) * old_count
                    bgs_separation_sum += (
                        float(stats["separation"]) * new_count
                    )
                    bgs_parent_sum += float(stats["parent"]) * new_count
                    bgs_positive_sum += (
                        float(stats.get("positive_cosine", 0.0)) * new_count
                    )
                    bgs_negative_sum += (
                        float(stats.get("negative_cosine", 0.0)) * new_count
                    )
                    bgs_parent_active_sum += (
                        float(stats.get("parent_active_ratio", 0.0))
                        * new_count
                    )
                    for group in ("leaf", "internal"):
                        for pair_type in ("inside", "boundary", "outside"):
                            prefix = f"{group}_{pair_type}"
                            pair_count = int(stats.get(f"{prefix}_count", 0))
                            aggregate = bgs_partition_aggregates[prefix]
                            aggregate["count"] += pair_count
                            aggregate["drift_sum"] += (
                                float(stats.get(f"{prefix}_drift", 0.0))
                                * pair_count
                            )
                            aggregate["weighted_contribution_sum"] += (
                                float(
                                    stats.get(
                                        f"{prefix}_weighted_contribution",
                                        0.0,
                                    )
                                )
                                * old_count
                            )
                            aggregate["weight_deficit"] += float(
                                stats.get(f"{prefix}_weight_deficit", 0.0)
                            )
                            for frame in ("teacher", "current"):
                                mean = float(
                                    stats.get(
                                        f"{prefix}_{frame}_affinity_mean",
                                        0.0,
                                    )
                                )
                                std = float(
                                    stats.get(
                                        f"{prefix}_{frame}_affinity_std",
                                        0.0,
                                    )
                                )
                                aggregate[f"{frame}_sum"] += mean * pair_count
                                aggregate[f"{frame}_square_sum"] += (
                                    std * std + mean * mean
                                ) * pair_count
                if edge_topology is not None:
                    edge_component_name = str(
                        self.edge_topology_options["objective"]
                    )
                    components[edge_component_name] = edge_topology
                    edge_topology_updates += 1
                    epoch_edge_topology_updates += 1
                    if self._last_edge_topology_stats:
                        stats = self._last_edge_topology_stats
                        stratified_diagnostic_updates += 1
                        for group_name in STRATIFIED_EDGE_GROUP_NAMES:
                            stratified_loss_sums[group_name] += float(
                                stats[f"{group_name}_loss"]
                            )
                            stratified_current_variance_sums[
                                group_name
                            ] += float(
                                stats[f"{group_name}_current_variance"]
                            )
                            stratified_active_sums[group_name] += int(
                                stats[f"{group_name}_active"]
                            )
                            stratified_current_active_sums[
                                group_name
                            ] += int(stats[f"{group_name}_current_active"])
                            stratified_reference_active[group_name] = int(
                                stats[f"{group_name}_reference_active"]
                            )
                            stratified_group_counts[group_name] = int(
                                stats[f"{group_name}_edge_count"]
                            )
                            stratified_reference_variances[
                                group_name
                            ] = float(
                                stats[f"{group_name}_reference_variance"]
                            )
                    if record_edge_gradient:
                        cil_loss = sum(
                            value
                            for name, value in components.items()
                            if name != edge_component_name
                        )
                        edge_gradients = torch.autograd.grad(
                            edge_topology,
                            parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        cil_gradients = torch.autograd.grad(
                            cil_loss,
                            parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        epoch_edge_gradient_l1 += sum(
                            float(gradient.detach().abs().sum())
                            for gradient in edge_gradients
                            if gradient is not None
                        )
                        edge_l2 = math.sqrt(
                            sum(
                                float(gradient.detach().float().square().sum())
                                for gradient in edge_gradients
                                if gradient is not None
                            )
                        )
                        cil_l2 = math.sqrt(
                            sum(
                                float(gradient.detach().float().square().sum())
                                for gradient in cil_gradients
                                if gradient is not None
                            )
                        )
                        if not math.isfinite(edge_l2) or not math.isfinite(cil_l2):
                            raise RuntimeError("non-finite debug gradient norm")
                        epoch_edge_gradient_l2 += edge_l2
                        epoch_cil_gradient_l2 += cil_l2
                loss = sum(components.values())
                optimizer.zero_grad(set_to_none=True)
                insertion_options = self.bgs_options.get("insertion", {})
                projected_insertion = bool(
                    insertion_options.get("gradient_projection", False)
                    and "bgs_insertion" in components
                )
                if projected_insertion:
                    insertion_loss = components["bgs_insertion"]
                    stability_loss = sum(
                        value
                        for name, value in components.items()
                        if name != "bgs_insertion"
                    )
                    stability_gradients = torch.autograd.grad(
                        stability_loss,
                        parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    insertion_gradients = torch.autograd.grad(
                        insertion_loss,
                        parameters,
                        allow_unused=True,
                    )
                    projection = project_insertion_gradient(
                        stability_gradients,
                        insertion_gradients,
                        epsilon=float(insertion_options["projection_epsilon"]),
                    )
                    for parameter, gradient in zip(
                        parameters, projection.gradients
                    ):
                        parameter.grad = gradient
                    pcli_updates += 1
                    pcli_conflicts += int(projection.conflict)
                    pcli_cosine_sum += projection.cosine
                    pcli_retained_ratio_sum += projection.insertion_retained_ratio
                    pcli_projection_coefficient_sum += (
                        projection.projection_coefficient
                    )
                else:
                    loss.backward()
                optimizer.step()
                optimizer_step += 1
                batch_count = targets.numel()
                count += batch_count
                batches += 1
                correct += int(prediction.argmax(1).eq(targets).sum().item())
                totals["loss"] += float(loss.detach()) * batch_count
                for name, value in components.items():
                    totals[name] += float(value.detach()) * batch_count
            if self.method != "cscct":
                scheduler.step()
            if fusion_optimizer is not None and fusion_loader is not None:
                self._update_cscct_fusion(fusion_loader, fusion_optimizer)
                if fusion_scheduler is not None and self.method != "cscct":
                    fusion_scheduler.step()
            if count == 0:
                raise RuntimeError("unified training processed no samples")
            stratified_record: dict[str, float | int] = {}
            if (
                self.edge_topology_options["objective"]
                == STRATIFIED_EDGE_TOPOLOGY_OBJECTIVE
            ):
                stratified_record["stratified_diagnostic_updates"] = (
                    stratified_diagnostic_updates
                )
                for group_name in STRATIFIED_EDGE_GROUP_NAMES:
                    stratified_record.update(
                        {
                            f"stratified_{group_name}_loss": (
                                stratified_loss_sums[group_name]
                                / max(1, stratified_diagnostic_updates)
                            ),
                            f"stratified_{group_name}_edge_count": (
                                stratified_group_counts[group_name]
                            ),
                            f"stratified_{group_name}_reference_variance": (
                                stratified_reference_variances[group_name]
                            ),
                            f"stratified_{group_name}_current_variance": (
                                stratified_current_variance_sums[group_name]
                                / max(1, stratified_diagnostic_updates)
                            ),
                            f"stratified_{group_name}_reference_active": (
                                stratified_reference_active[group_name]
                            ),
                            f"stratified_{group_name}_current_active_ratio": (
                                stratified_current_active_sums[group_name]
                                / max(1, stratified_diagnostic_updates)
                            ),
                            f"stratified_{group_name}_active_ratio": (
                                stratified_active_sums[group_name]
                                / max(1, stratified_diagnostic_updates)
                            ),
                        }
                    )
            bgs_partition_record: dict[str, float | int | None] = {}
            for prefix, aggregate in bgs_partition_aggregates.items():
                pair_count = int(aggregate["count"])
                bgs_partition_record[f"bgs_{prefix}_pair_count"] = pair_count
                bgs_partition_record[f"bgs_{prefix}_weight_deficit"] = float(
                    aggregate["weight_deficit"]
                )
                bgs_partition_record[f"bgs_{prefix}_unweighted_drift"] = (
                    None
                    if pair_count == 0
                    else aggregate["drift_sum"] / pair_count
                )
                bgs_partition_record[
                    f"bgs_{prefix}_weighted_contribution"
                ] = (
                    None
                    if bgs_old_count == 0
                    else aggregate["weighted_contribution_sum"]
                    / bgs_old_count
                )
                for frame in ("teacher", "current"):
                    mean = (
                        None
                        if pair_count == 0
                        else aggregate[f"{frame}_sum"] / pair_count
                    )
                    variance = (
                        None
                        if pair_count == 0
                        else max(
                            0.0,
                            aggregate[f"{frame}_square_sum"] / pair_count
                            - float(mean) ** 2,
                        )
                    )
                    bgs_partition_record[
                        f"bgs_{prefix}_{frame}_affinity_mean"
                    ] = mean
                    bgs_partition_record[
                        f"bgs_{prefix}_{frame}_affinity_std"
                    ] = None if variance is None else math.sqrt(variance)
            feature_routing_record: dict[str, Any] = {}
            routing_options = self.feature_cosine_distillation_options.get(
                "hierarchy_routing", {"enabled": False}
            )
            if routing_options.get("enabled", False):
                feature_routing_record = {
                    "feature_routing_batches": len(
                        feature_routing_batch_logs
                    ),
                    "feature_routing_sample_count": (
                        feature_routing_sample_count
                    ),
                    "feature_routing_weight_sum": feature_routing_weight_sum,
                    "feature_routing_mean_weight": (
                        feature_routing_weight_sum
                        / max(1, feature_routing_sample_count)
                    ),
                    "feature_routing_batch_logs": (
                        feature_routing_batch_logs
                    ),
                }
                for group_name in (
                    "old_conflict",
                    "old_outside",
                    "new",
                ):
                    group_count = feature_routing_counts[group_name]
                    group_weight_sum = feature_routing_weight_sums[group_name]
                    feature_routing_record.update(
                        {
                            f"feature_routing_{group_name}_count": group_count,
                            f"feature_routing_{group_name}_mean_weight": (
                                0.0
                                if group_count == 0
                                else group_weight_sum / group_count
                            ),
                            f"feature_routing_{group_name}_effective_weight": (
                                0.0
                                if feature_routing_weight_sum == 0.0
                                else group_weight_sum
                                / feature_routing_weight_sum
                            ),
                        }
                    )
            record = {
                "epoch": epoch + 1,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "batches": batches,
                "icarl_kd_weight": self.icarl_kd_weight,
                "geodesic_effective_weight": (
                    0.0
                    if not self.geodesic_distillation_options["enabled"]
                    else float(self.geodesic_distillation_options["lambda"])
                    * (
                        math.sqrt(
                            float(self.protocol.session(session_id).size)
                            / float(max(1, self.protocol.session(session_id).start))
                        )
                        if self.geodesic_distillation_options[
                            "adaptive_new_over_old"
                        ]
                        else 1.0
                    )
                ),
                "feature_cosine_effective_weight": (
                    0.0
                    if not self.feature_cosine_distillation_options["enabled"]
                    else float(self.feature_cosine_distillation_options["lambda"])
                    * (
                        math.sqrt(
                            float(self.protocol.session(session_id).size)
                            / float(max(1, self.protocol.session(session_id).start))
                        )
                        if self.feature_cosine_distillation_options[
                            "adaptive_mode"
                        ] == "new_over_old"
                        else math.sqrt(
                            float(max(1, self.protocol.session(session_id).start))
                            / float(self.protocol.session(session_id).size)
                        )
                        if self.feature_cosine_distillation_options[
                            "adaptive_mode"
                        ] == "old_over_new"
                        else 1.0
                    )
                ),
                "edge_topology_updates": epoch_edge_topology_updates,
                "edge_topology_gradient_l1": epoch_edge_gradient_l1,
                "edge_topology_gradient_global_l2": (
                    epoch_edge_gradient_l2
                    / max(1, epoch_edge_topology_updates)
                ),
                "cil_gradient_global_l2": (
                    epoch_cil_gradient_l2
                    / max(1, epoch_edge_topology_updates)
                ),
                "edge_to_cil_gradient_l2_ratio": (
                    epoch_edge_gradient_l2
                    / max(epoch_cil_gradient_l2, 1e-12)
                ),
                "branch_kd_old": (
                    branch_old_kd_sum / max(1, branch_old_count)
                ),
                "branch_kd_new": (
                    branch_new_kd_sum / max(1, branch_new_count)
                ),
                "branch_kd_teacher_retained_mass": (
                    branch_teacher_mass_sum / max(1, branch_new_count)
                ),
                "branch_kd_student_retained_mass": (
                    branch_student_mass_sum / max(1, branch_new_count)
                ),
                "branch_kd_masked_class_ratio": (
                    branch_masked_ratio_sum / max(1, branch_new_count)
                ),
                "branch_kd_old_samples": branch_old_count,
                "branch_kd_new_samples": branch_new_count,
                "selective_kd_old": (
                    selective_old_kd_sum / max(1, selective_old_count)
                ),
                "selective_kd_new": (
                    selective_new_kd_sum / max(1, selective_new_count)
                ),
                "selective_kd_new_keep_ratio": (
                    selective_keep_sum / max(1, selective_new_count)
                ),
                "selective_kd_alignment_mean": (
                    selective_alignment_sum / max(1, selective_new_count)
                ),
                "selective_kd_alignment_positive_ratio": (
                    selective_positive_sum / max(1, selective_new_count)
                ),
                "selective_kd_old_samples": selective_old_count,
                "selective_kd_new_samples": selective_new_count,
                "bgs_leaf": bgs_leaf_sum / max(1, bgs_old_count),
                "bgs_internal": bgs_internal_sum / max(1, bgs_old_count),
                "bgs_separation": bgs_separation_sum / max(1, bgs_new_count),
                "bgs_parent": bgs_parent_sum / max(1, bgs_new_count),
                "bgs_positive_cosine": bgs_positive_sum / max(1, bgs_new_count),
                "bgs_negative_cosine": bgs_negative_sum / max(1, bgs_new_count),
                "bgs_parent_active_ratio": bgs_parent_active_sum / max(1, bgs_new_count),
                "bgs_old_samples": bgs_old_count,
                "bgs_new_samples": bgs_new_count,
                "pcli_updates": pcli_updates,
                "pcli_conflict_rate": pcli_conflicts / max(1, pcli_updates),
                "pcli_gradient_cosine": pcli_cosine_sum / max(1, pcli_updates),
                "pcli_insertion_retained_ratio": (
                    pcli_retained_ratio_sum / max(1, pcli_updates)
                ),
                "pcli_projection_coefficient": (
                    pcli_projection_coefficient_sum / max(1, pcli_updates)
                ),
                **bgs_partition_record,
                **stratified_record,
                **feature_routing_record,
                "train_accuracy": correct / count,
                **{key: value / count for key, value in totals.items()},
            }
            logs.append(record)
            progress.set_description(
                f"{self.method} s{session_id} e{epoch + 1}/{epochs} "
                f"loss={record['loss']:.4f} acc={record['train_accuracy']:.3f}"
            )
        return {
            "epochs": epochs,
            "samples": len(loader.dataset),
            "geometry_anchor_refreshes": geometry_anchor_refreshes,
            "edge_topology_updates": edge_topology_updates,
            "edge_topology_representatives": (
                0
                if self.edge_topology_reference is None
                else self.edge_topology_reference.representative_count
            ),
            "edge_topology_edges": (
                0
                if self.edge_topology_reference is None
                else self.edge_topology_reference.edge_count
            ),
            "epoch_logs": logs,
        }

    def _main_parameters(self) -> list[nn.Parameter]:
        if self.model is None:
            raise RuntimeError("model is missing")
        if isinstance(self.model, PyCILPODNet):
            return self.model.main_trainable_parameters()
        if isinstance(self.model, AFCIncrementalNet):
            for parameter in self.model.classifier.old_weights:
                parameter.requires_grad_(False)
            return [
                *[p for p in self.model.backbone.parameters() if p.requires_grad],
                self.model.classifier.new_weights,
            ]
        if isinstance(self.model, CSCCTIncrementalNet):
            for parameter in self.model.classifier.old_weights:
                parameter.requires_grad_(False)
            return self.model.main_parameters()
        if self.method == "sacil" and isinstance(
            self.model, ExpandableLinearNet
        ):
            for parameter in self.model.classifier.parameters():
                parameter.requires_grad_(False)
            return list(self.model.backbone.parameters())
        return [p for p in self.model.parameters() if p.requires_grad]

    def _add_geometry_component(
        self,
        components: dict[str, Tensor],
        current_features: Tensor,
        reference_features: Tensor | None,
        replay_mask: Tensor,
        geometry: AnchorGeometryLoss | None,
    ) -> None:
        value = geometry_preservation_component(
            geometry,
            current_features,
            reference_features,
            replay_mask,
            weight=float(self.config["method"].get("lambda_geo", 1.0)),
        )
        if value is not None:
            components["geometry"] = value

    def _add_casper_component(
        self,
        components: dict[str, Tensor],
        replay_images: Tensor | None,
    ) -> None:
        """Append the author CaSpeR spectral term on a balanced replay graph."""

        if not self.casper_options["enabled"] or replay_images is None:
            return
        if self.model is None:
            raise RuntimeError("CaSpeR regularization has no model")
        replay_features = self.model.extract_features(replay_images)
        components["spectral"] = float(
            self.casper_options["weight"]
        ) * casper_spectral_loss(
            replay_features,
            num_classes=int(self.casper_options["classes_per_graph"]),
            k=int(self.casper_options["knn"]),
            solver=str(self.casper_options["solver"]),
        )

    def _loss_components(
        self,
        session_id: int,
        images: Tensor,
        targets: Tensor,
        replay_mask: Tensor,
        teacher: nn.Module | None,
        geometry: AnchorGeometryLoss | None,
        prototypes: Tensor | None,
        replay_images: Tensor | None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        if self.model is None:
            raise RuntimeError("loss has no model")
        self._last_branch_kd_stats = {}
        self._last_selective_kd_stats = {}
        self._last_bgs_stats = {}
        self._last_feature_routing_stats = {}
        known = self.protocol.session(session_id).start
        total = self.protocol.session(session_id).stop
        new_count = self.protocol.session(session_id).size

        if isinstance(self.model, ExpandableLinearNet):
            output = self.model.forward_detailed(images)
            if self.method == "joint":
                return {"classification": F.cross_entropy(output.logits, targets)}, output.logits
            if self.method == "replay":
                return {
                    "classification": replay_cross_entropy(
                        output.logits, targets
                    )
                }, output.logits
            if self.method == "finetune":
                return {
                    "classification": pycil_finetune_loss(
                        output.logits,
                        targets,
                        known_classes=known,
                    )
                }, output.logits
            reference = (
                None
                if teacher is None
                else teacher.forward_detailed(images)
            )
            if self.method == "icarl":
                classification_logits = output.logits
                feature_cosine_options = getattr(
                    self,
                    "feature_cosine_distillation_options",
                    {"enabled": False},
                )
                if (
                    feature_cosine_options.get("enabled", False)
                    and feature_cosine_options.get("training_classifier")
                    == "normalized_cosine"
                ):
                    classification_logits = normalized_cosine_classifier_logits(
                        output.features,
                        self.model.classifier.weight,
                        scale=float(feature_cosine_options["cosine_scale"]),
                        epsilon=float(feature_cosine_options["epsilon"]),
                    )
                components = {
                    "classification": F.cross_entropy(
                        classification_logits, targets
                    )
                }
                if reference is not None:
                    if self.icarl_kd_weight == 0.0:
                        components["kd"] = output.logits[:, :known].sum() * 0.0
                    else:
                        temperature = float(
                            self.config["method"].get("kd_temperature", 2.0)
                        )
                        if self.selective_kd_options["enabled"]:
                            result = selective_pycil_icarl_kd_loss(
                                output.logits,
                                reference.logits,
                                targets,
                                replay_mask,
                                self.model.classifier.weight,
                                temperature=temperature,
                                alignment_threshold=float(
                                    self.selective_kd_options[
                                        "alignment_threshold"
                                    ]
                                ),
                            )
                            components["kd"] = self.icarl_kd_weight * result.loss
                            self._last_selective_kd_stats = (
                                result.detached_metrics()
                            )
                        elif self.branch_masked_kd_options["enabled"]:
                            branch_reference = self.branch_masked_kd_reference
                            if branch_reference is None:
                                raise RuntimeError(
                                    "branch-masked KD mapping is missing"
                                )
                            result = branch_masked_pycil_icarl_kd_loss(
                                output.logits[:, :known],
                                reference.logits,
                                replay_mask,
                                targets[~replay_mask] - known,
                                branch_reference.branch_class_mask,
                                temperature=temperature,
                                v_min=float(
                                    self.branch_masked_kd_options["v_min"]
                                ),
                            )
                            components["kd"] = self.icarl_kd_weight * result.loss
                            self._last_branch_kd_stats = (
                                result.detached_metrics()
                            )
                        else:
                            components["kd"] = (
                                self.icarl_kd_weight
                                * pycil_icarl_kd_loss(
                                    output.logits[:, :known],
                                    reference.logits,
                                    temperature=temperature,
                                )
                            )
                if (
                    self.geodesic_distillation_options["enabled"]
                    and reference is not None
                ):
                    geodesic_weight = float(
                        self.geodesic_distillation_options["lambda"]
                    )
                    if self.geodesic_distillation_options[
                        "adaptive_new_over_old"
                    ]:
                        geodesic_weight *= math.sqrt(
                            float(new_count) / float(max(1, known))
                        )
                    components["geodesic_distillation"] = (
                        geodesic_weight
                        * geodesic_distillation_loss(
                            output.features,
                            reference.features,
                            subspace_rank=int(
                                self.geodesic_distillation_options[
                                    "subspace_rank"
                                ]
                            ),
                            epsilon=float(
                                self.geodesic_distillation_options["epsilon"]
                            ),
                        )
                    )
                if (
                    feature_cosine_options["enabled"]
                    and reference is not None
                ):
                    feature_weight = float(feature_cosine_options["lambda"])
                    if feature_cosine_options["adaptive_mode"] == "new_over_old":
                        feature_weight *= math.sqrt(
                            float(new_count) / float(max(1, known))
                        )
                    elif feature_cosine_options["adaptive_mode"] == "old_over_new":
                        feature_weight *= math.sqrt(
                            float(max(1, known)) / float(new_count)
                        )
                    sample_weights = None
                    routing_options = feature_cosine_options.get(
                        "hierarchy_routing", {"enabled": False}
                    )
                    if routing_options.get("enabled", False):
                        if not self.bgs_options.get("enabled", False):
                            raise RuntimeError(
                                "feature cosine hierarchy routing requires enabled BGS"
                            )
                        if self.bgs_reference is None:
                            raise RuntimeError(
                                "feature cosine hierarchy routing requires a BGS reference"
                            )
                        expected_old_ids = tuple(range(known))
                        if (
                            tuple(self.bgs_reference.old_incremental_ids)
                            != expected_old_ids
                        ):
                            raise RuntimeError(
                                "BGS reference old-class mapping is incompatible with routing"
                            )
                        routed = hierarchy_routed_feature_sample_weights(
                            targets,
                            replay_mask,
                            known_classes=known,
                            sample_region_ids=(
                                self.bgs_reference.sample_region_ids
                            ),
                            old_conflict_weight=float(
                                routing_options["old_conflict_weight"]
                            ),
                            old_outside_weight=float(
                                routing_options["old_outside_weight"]
                            ),
                            new_weight=float(routing_options["new_weight"]),
                        )
                        sample_weights = routed.sample_weights
                        self._last_feature_routing_stats = (
                            routed.detached_metrics()
                        )
                    components["feature_cosine_distillation"] = (
                        feature_weight
                        * cosine_feature_distillation_loss(
                            output.features,
                            reference.features,
                            sample_weights=sample_weights,
                            epsilon=float(feature_cosine_options["epsilon"]),
                        )
                    )
                if self.bgs_options["enabled"] and reference is not None:
                    if self._bgs_loss is None or self.bgs_reference is None:
                        raise RuntimeError("BGS reference is missing")
                    bgs_loss, bgs_stats = self._bgs_loss(
                        output.features[replay_mask],
                        reference.features[replay_mask],
                        targets[replay_mask],
                    )
                    components["bgs_geometry"] = float(
                        self.bgs_options["geometry"]["lambda"]
                    ) * bgs_loss
                    insertion = self.bgs_options["insertion"]
                    insert_total = output.features.sum() * 0.0
                    separation = insert_total
                    parent = insert_total
                    insert_stats = {}
                    if insertion["enabled"]:
                        if self._bgs_positive_prototypes is None:
                            raise RuntimeError("BGS positives are missing")
                        anchor_state = self.bgs_reference.anchor_state
                        internal_ids = tuple(anchor_state["internal_node_ids"])
                        primary_positions = torch.tensor(
                            self.bgs_reference.primary_internal_positions,
                            device=self.device,
                        )
                        internal_positions = [
                            index for index, node in enumerate(internal_ids)
                            if node != anchor_state["root_id"]
                        ]
                        insert_total, separation, parent, insert_stats = (
                            bgs_insertion_loss(
                            output.features[~replay_mask],
                            targets[~replay_mask] - known,
                            self._bgs_positive_prototypes,
                            anchor_state["leaf_anchors"].to(self.device),
                            anchor_state["internal_anchors"][internal_positions]
                            .to(self.device),
                            self.bgs_reference.negative_class_positions,
                            primary_positions,
                            self.bgs_reference.parent_thresholds.to(self.device),
                            temperature=insertion["temperature"],
                            separation_enabled=insertion["separation_enabled"],
                            parent_weight=insertion["parent_weight"],
                        )
                        )
                        components["bgs_insertion"] = (
                            float(insertion["lambda"]) * insert_total
                        )
                    self._last_bgs_stats = {
                        "leaf": float(bgs_stats["leaf"].detach()),
                        "internal": float(bgs_stats["internal"].detach()),
                        "separation": float(separation.detach()),
                        "parent": float(parent.detach()),
                        "old_count": int(replay_mask.sum()),
                        "new_count": int((~replay_mask).sum()),
                        **{
                            key: float(value.detach())
                            for key, value in bgs_stats.items()
                            if key not in {"leaf", "internal"}
                        },
                        **{
                            key: float(value.detach())
                            for key, value in insert_stats.items()
                        },
                    }
                self._add_geometry_component(
                    components,
                    output.features,
                    None if reference is None else reference.features,
                    replay_mask,
                    geometry,
                )
                self._add_casper_component(components, replay_images)
                return components, classification_logits
            if self.method == "casper":
                loss = icarl_bce_loss(
                    output.logits,
                    targets,
                    old_logits=(None if reference is None else reference.logits),
                    known_classes=known,
                )
                components = {"classification": loss}
                components["regularization"] = parameter_l2_regularization(
                    self.model,
                    float(self.casper_options["wd_reg"]),
                )
                self._add_casper_component(components, replay_images)
                return components, output.logits
            if self.method == "sacil":
                if prototypes is None:
                    raise RuntimeError("SACIL training prototypes are missing")
                classification, prediction = prototype_cross_entropy(
                    output.features,
                    targets,
                    prototypes,
                    temperature=float(
                        self.config["method"].get("prototype_temperature", 0.1)
                    ),
                )
                components = {"classification": classification}
                self._add_geometry_component(
                    components,
                    output.features,
                    None if reference is None else reference.features,
                    replay_mask,
                    geometry,
                )
                return components, prediction
            raise RuntimeError(f"unhandled linear method: {self.method}")

        if isinstance(self.model, PyCILPODNet):
            output = self.model.forward_detailed(images)
            components = {
                "classification": podnet_nca_loss(
                    output.logits,
                    targets,
                    scale=1.0,
                )
            }
            if teacher is not None:
                if not isinstance(teacher, PyCILPODNet):
                    raise TypeError("PODNet teacher has the wrong model type")
                reference = teacher.forward_detailed(images)
                factor = math.sqrt(total / new_count)
                pod = self.config["method"].get("podnet", {})
                components["pod_flat"] = float(
                    pod.get("flat_weight", 1.0)
                ) * factor * pod_flat_loss(
                    output.features, reference.features
                )
                components["pod_spatial"] = float(
                    pod.get("spatial_weight", 5.0)
                ) * factor * pod_spatial_loss(
                    output.attentions, reference.attentions
                )
            return components, output.logits

        if isinstance(self.model, AFCIncrementalNet):
            output = self.model.forward_detailed(images)
            components = {
                "classification": afc_nca_loss(
                    output.logits, targets, 1.0
                )
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                afc = self.config["method"].get("afc", {})
                factor = scheduled_afc_factor(
                    total, new_count, float(afc.get("pod_base_factor", 4.0))
                )
                components["afc"] = factor * afc_pod_loss(
                    reference.attentions,
                    output.attentions,
                    reference.importance[:-1],
                )
                self._add_geometry_component(
                    components,
                    output.features,
                    reference.features,
                    replay_mask,
                    geometry,
                )
            return components, output.logits

        if isinstance(self.model, CREATEIncrementalNet):
            output = self.model.forward_detailed(images)
            create = self.config["method"].get("create", {})
            components = {
                "classification": create_classification_loss(
                    output["logits"], targets
                )
            }
            weights = reconstruction_confidence_weights(
                output["error_logits"],
                alpha=float(create.get("confidence_alpha", 2.0)),
            )
            components["contrastive"] = float(
                create.get("contrastive_weight", 1.0)
            ) * create_contrastive_loss(
                output["latents"], targets, sample_weights=weights
            )
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                components["kd"] = float(create.get("kd_weight", 1.0)) * create_kd_loss(
                    output["error_logits"][:, :known],
                    reference["error_logits"],
                    temperature=float(create.get("kd_temperature", 2.0)),
                )
            return components, output["logits"]

        if isinstance(self.model, FGPIncrementalNet):
            output = self.model.forward_detailed(images)
            one_hot = F.one_hot(targets, num_classes=total).to(output.logits)
            components = {
                "classification": F.binary_cross_entropy_with_logits(
                    output.logits, one_hot
                )
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                weight = scheduled_fgp_weight(
                    known,
                    total,
                    base_weight=float(
                        self.config["method"].get("fgp_weight", 0.1)
                    ),
                )
                components["graph"] = weight * fgp_graph_preservation_loss(
                    output.features,
                    reference.features,
                    self.model.classifier.weight[:known],
                    teacher.classifier.weight,
                )
            return components, output.logits

        if isinstance(self.model, CSCCTIncrementalNet):
            output = self.model.forward_detailed(images)
            components = {
                "classification": F.cross_entropy(output.logits, targets)
            }
            if teacher is not None:
                reference = teacher.forward_detailed(images)
                method = self.config["method"]
                components["kd"] = float(method.get("kd_weight", 0.25)) * old_logit_kl_loss(
                    output.logits[:, :known],
                    reference.logits,
                    temperature=float(method.get("kd_temperature", 2.0)),
                )
                components["csc"] = float(
                    method.get("csc_weight", 3.0)
                ) * cross_space_clustering_loss(
                    output.features,
                    reference.features,
                    targets,
                )
                components["ct"] = float(
                    method.get("ct_weight", 1.5)
                ) * controlled_transfer_loss(
                    output.features,
                    reference.features,
                    targets,
                    known_classes=known,
                    temperature=float(method.get("ct_temperature", 2.0)),
                )
            return components, output.logits
        raise TypeError(f"unsupported model type: {type(self.model).__name__}")

    def _refresh_training_prototypes(self, loader: DataLoader | None) -> Tensor:
        if loader is None or self.model is None:
            raise RuntimeError("prototype refresh lacks model or loader")
        collection = collect_features(self.model, loader, self.device)
        prototypes = compute_prototypes(
            collection.features,
            collection.targets,
            range(self.model.num_classes),
        )
        return prototypes.to(self.device)

    def _casper_loader(
        self, session_id: int, batches: int
    ) -> DataLoader | None:
        if not self.casper_options["enabled"] or session_id == 0:
            return None
        indices = self.memory.all_indices(self.protocol.class_order)
        dataset = self.data.replay_dataset(indices, augment=True)
        labels = [
            self.protocol.incremental_label(self.data.train_aug.targets[index])
            for index in indices
        ]
        sampler = BalancedClassBatchSampler(
            labels,
            batch_size=int(self.casper_options["replay_batch_size"]),
            classes_per_batch=int(self.casper_options["classes_per_graph"]),
            batches=batches,
            seed=self.seed * 10000 + session_id * 100,
        )
        return self._loader(
            dataset,
            shuffle=False,
            session_id=session_id + 6000,
            batch_sampler=sampler,
        )

    def _cscct_balanced_loader(self, session_id: int) -> DataLoader | None:
        if self.method != "cscct" or session_id == 0:
            return None
        old_indices = self.memory.all_indices(self.protocol.class_order)
        # CSCCT's official fusion update draws ``increment * exemplars`` new
        # images with replacement, then concatenates them with the complete
        # old memory (base_trainer.py::gen_balanced_loader).
        new_candidates = self.data.new_train_dataset(
            session_id,
            augment=True,
            samples_per_class=self.debug_train_samples_per_class,
        )
        rng = np.random.default_rng(self.seed * 10000 + session_id * 100 + 71)
        new_sample_count = (
            self.protocol.session(session_id).size
            * self._memory_limit(self.protocol.session(session_id).stop)
        )
        new_positions = rng.choice(
            len(new_candidates), size=new_sample_count, replace=True
        )
        new_indices = [
            new_candidates.indices[int(position)]
            for position in new_positions.tolist()
        ]
        dataset = ConcatDataset(
            [
                self.data.replay_dataset(old_indices, augment=True),
                self.data.train_dataset_from_indices(
                    new_indices, augment=True, is_replay=False
                ),
            ]
        )
        return self._loader(
            dataset, shuffle=True, session_id=session_id + 7000
        )

    def _update_cscct_fusion(
        self, loader: DataLoader, optimizer: SGD
    ) -> None:
        if not isinstance(self.model, CSCCTIncrementalNet):
            return
        self.model.eval()
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device).long()
            loss = F.cross_entropy(self.model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def _update_memory(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("memory update has no model")
        limit = self._memory_limit(self.protocol.session(session_id).stop)
        self.memory.resize_limit(limit)
        dataset = self.data.new_train_dataset(
            session_id,
            augment=False,
            samples_per_class=self.debug_train_samples_per_class,
        )
        collection = collect_features(
            self.model,
            self._loader(dataset, shuffle=False, session_id=session_id + 8000),
            self.device,
        )
        selection = str(self.config["memory"].get("selection", "icarl_herding"))
        for class_id in self.protocol.classes_for_session(session_id):
            mask = collection.original_targets == int(class_id)
            # The controlled Table-1 contract uses one selector for every
            # replay-based method.  ``icarl_herding`` means the running-mean
            # greedy procedure implemented by stock PyCIL BaseLearner, not
            # AFC's direction-update variant.
            function = herding_select
            if selection not in {"icarl_herding", "running_mean"}:
                raise ValueError(f"unsupported controlled herding: {selection}")
            selected = function(
                collection.features[mask],
                collection.indices[mask].tolist(),
                limit,
            )
            self.memory.set_class_indices(class_id, selected)

    def _post_training(
        self,
        session_id: int,
        train_loader: DataLoader,
        teacher: nn.Module | None,
    ) -> dict[str, Any] | None:
        if self.method == "podnet":
            return {"finetuning": self._podnet_finetune(session_id, teacher)}
        if self.method == "afc":
            return {
                "finetuning": self._afc_finetune(session_id),
                "importance": self._afc_importance(train_loader),
            }
        if self.method == "create":
            return {
                "finetuning": self._create_finetune(session_id, teacher)
            }
        return None

    def _podnet_finetune(
        self, session_id: int, teacher: nn.Module | None
    ) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, PyCILPODNet):
            return None
        config = self.config["method"].get("podnet", {}).get(
            "finetuning", {}
        )
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        parameters = self._main_parameters()
        optimizer = SGD(
            parameters,
            lr=float(config.get("lr", 0.005)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        loader = self._memory_loader(session_id, augment=True)
        losses: list[float] = []
        for _ in range(epochs):
            self.model.train()
            total_loss = 0.0
            count = 0
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device).long()
                components, _ = self._loss_components(
                    session_id,
                    images,
                    targets,
                    batch["is_replay"].to(self.device).bool(),
                    teacher,
                    None,
                    None,
                    None,
                )
                loss = sum(components.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * targets.numel()
                count += targets.numel()
            scheduler.step()
            losses.append(total_loss / count)
        return {"epochs": epochs, "losses": losses}

    def _memory_loader(self, session_id: int, *, augment: bool) -> DataLoader:
        indices = self.memory.all_indices(self.protocol.class_order)
        return self._loader(
            self.data.replay_dataset(indices, augment=augment),
            shuffle=augment,
            session_id=session_id + 9000 + int(augment),
        )

    def _afc_finetune(self, session_id: int) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, AFCIncrementalNet):
            return None
        config = self.config["method"].get("afc", {}).get("finetuning", {})
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(True)
        optimizer = SGD(
            self.model.classifier.parameters(),
            lr=float(config.get("lr", 0.05)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        loader = self._memory_loader(session_id, augment=True)
        losses = []
        for _ in range(epochs):
            total = count = 0
            self.model.train()
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).long()
                loss = afc_nca_loss(self.model(images), targets, 1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * targets.numel()
                count += targets.numel()
            losses.append(total / count)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        return {"epochs": epochs, "losses": losses}

    def _afc_importance(self, loader: DataLoader) -> dict[str, Any]:
        if not isinstance(self.model, AFCIncrementalNet):
            raise TypeError("AFC importance needs AFCIncrementalNet")
        self.model.backbone.reset_importance()
        self.model.backbone.start_importance_collection()
        self.model.train()
        batches = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device).long()
            self.model.zero_grad(set_to_none=True)
            loss = afc_nca_loss(self.model(images), targets, 1.0)
            loss.backward()
            batches += 1
        self.model.backbone.stop_importance_collection()
        self.model.backbone.normalize_importance()
        return {
            "batches": batches,
            "layer_means": [
                float(layer.importance.mean())
                for layer in self.model.backbone.importance_layers
            ],
        }

    def _create_finetune(
        self,
        session_id: int,
        teacher: nn.Module | None,
    ) -> dict[str, Any] | None:
        if session_id == 0 or not isinstance(self.model, CREATEIncrementalNet):
            return None
        config = self.config["method"].get("create", {}).get("finetuning", {})
        epochs = int(config.get("epochs", 20))
        if epochs <= 0:
            return None
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        optimizer = SGD(
            self.model.classifier.parameters(),
            lr=float(config.get("lr", 0.005)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        loader = self._memory_loader(session_id, augment=True)
        losses = []
        for _ in range(epochs):
            total = count = 0
            self.model.train()
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).long()
                components, _ = self._loss_components(
                    session_id,
                    images,
                    targets,
                    batch["is_replay"].to(self.device).bool(),
                    teacher,
                    None,
                    None,
                    None,
                )
                loss = sum(components.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * targets.numel()
                count += targets.numel()
            scheduler.step()
            losses.append(total / count)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(True)
        return {"epochs": epochs, "losses": losses}

    def _build_evaluation_state(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("evaluation state has no model")
        if self.method == "create":
            self.class_means = None
        else:
            if self.method == "joint":
                dataset = self.data.train_eval_dataset_for_classes(
                    self.protocol.seen_classes(session_id),
                    samples_per_class=self.debug_train_samples_per_class,
                )
                loader = self._loader(
                    dataset, shuffle=False, session_id=session_id + 10000
                )
            else:
                loader = self._memory_loader(session_id, augment=False)
            self.class_means = compute_nme_class_means(
                self.model,
                loader,
                self.device,
                self.protocol.session(session_id).stop,
                horizontal_flip=bool(
                    self.config.get("evaluation", {}).get(
                        "horizontal_flip", True
                    )
                ),
            ).cpu()
        if self.geometry_mode != "none":
            self._build_sacil_artifacts(session_id)

    def _build_sacil_artifacts(self, session_id: int) -> None:
        if self.model is None:
            raise RuntimeError("SACIL artifacts have no model")
        loader = self._memory_loader(session_id, augment=False)
        collection = collect_features(self.model, loader, self.device)
        class_ids = self.protocol.seen_classes(session_id)
        values = compute_prototypes(
            collection.features, collection.original_targets, class_ids
        )
        prototypes = PrototypeBank(class_ids, values)
        hierarchy = self.config["method"].get("hierarchy", {})
        confusion = cosine_soft_confusion(
            collection.features,
            collection.targets,
            values,
            temperature=float(hierarchy.get("taxonomy_temperature", 0.2)),
        )
        tree = GriffinPeronaGreedy().build(
            class_ids, symmetric_affinity(confusion)
        )
        anchors = HierarchicalAnchorBank.from_tree(prototypes, tree)
        self.sacil_tree = tree
        self.sacil_prototypes = prototypes
        self.sacil_anchors = anchors
        dump_json(
            tree.state_dict(), self.run_dir / f"tree_session_{session_id:02d}.json"
        )

    def _evaluate_session(self, session_id: int):
        if self.model is None:
            raise RuntimeError("evaluation has no model")
        dataset = self.data.cumulative_test_dataset(
            session_id, samples_per_class=self.debug_test_samples_per_class
        )
        loader = self._loader(
            dataset, shuffle=False, session_id=session_id + 11000
        )
        old = self.protocol.session(session_id).start
        if self.method == "create":
            return evaluate(self.model, loader, self.device, old)
        if self.class_means is None:
            raise RuntimeError("NME evaluation has no class means")
        return evaluate_nme(
            self.model, loader, self.device, old, self.class_means
        )

    def _save_checkpoint(
        self,
        session_id: int,
        *,
        session_record: dict[str, Any] | None = None,
    ) -> Path:
        if self.model is None:
            raise RuntimeError("checkpoint has no model")
        path = self.checkpoint_dir / f"session_{session_id:02d}.pt"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "framework": "sacil-unified",
            "pycil_used": False,
            "reference_code_executed": False,
            "method_contract": self.method_contract.as_dict(),
            "method": self.method,
            "geometry_mode": self.geometry_mode,
            "casper_options": copy.deepcopy(self.casper_options),
            "edge_topology_options": copy.deepcopy(
                self.edge_topology_options
            ),
            "branch_masked_kd_options": copy.deepcopy(
                self.branch_masked_kd_options
            ),
            "selective_kd_options": copy.deepcopy(
                self.selective_kd_options
            ),
            "bgs_options": copy.deepcopy(self.bgs_options),
            "bgs_reference_path": (
                None
                if self._bgs_reference_path is None
                else str(self._bgs_reference_path)
            ),
            "bgs_reference_sha256": self._bgs_reference_sha256,
            "exploration_provenance": copy.deepcopy(
                self.config.get("exploration_provenance")
            ),
            "session_id": session_id,
            "protocol_id": self.protocol.protocol_id,
            "config": self.config,
            "model": self.model.state_dict(),
            "model_type": type(self.model).__name__,
            "memory": self.memory.state_dict(),
            "metrics": self.metrics.state_dict(),
            "class_means": self.class_means,
            "session_record": (
                None
                if session_record is None
                else copy.deepcopy(session_record)
            ),
        }
        if self.sacil_tree is not None:
            payload.update(
                tree=self.sacil_tree.state_dict(),
                prototypes=self.sacil_prototypes.state_dict(),
                anchors=self.sacil_anchors.state_dict(),
            )
        if self.edge_topology_reference is not None:
            payload["edge_topology_reference"] = (
                self.edge_topology_reference.state_dict()
            )
        if self.branch_masked_kd_reference is not None:
            payload["branch_masked_kd_reference"] = (
                self.branch_masked_kd_reference.state_dict()
            )
        if self.bgs_reference is not None:
            payload["bgs_reference"] = self.bgs_reference.state_dict()
            payload["bgs_positive_prototypes"] = (
                None
                if self._bgs_positive_prototypes is None
                else self._bgs_positive_prototypes.detach().cpu().clone()
            )
        save_checkpoint(payload, path)
        return path


# Backward-compatible import name for old analysis utilities.  New experiment
# code and documentation use UnifiedTable1Trainer.
StandaloneTable1Trainer = UnifiedTable1Trainer
