from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from negzerohoc.cf_rpep import macro_terminal_weights


EVIDENCE_EPS = 1e-7
MASS_EPS = 1e-12
PROBABILITY_EPS = 1e-12
STD_FLOOR = 1e-8
LOCKED_MAX_ITER = 100
LOCKED_LINE_SEARCH = "strong_wolfe"
LOCKED_INITIAL_SCALARS = {
    "beta_0": 0.0,
    "beta_E": 1.0,
    "beta_H": 1.0,
    "gamma_H": 1.0,
    "gamma_E": 1.0,
    "gamma_D": 0.0,
}


@dataclass(frozen=True)
class FeatureNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    raw_std: torch.Tensor
    sample_count: int

    def as_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "raw_std": self.raw_std.detach().cpu().tolist(),
            "sample_count": int(self.sample_count),
            "variance_estimator": "population_unbiased_false",
            "minimum_std": STD_FLOOR,
            "fit_scope": "inner_calibration_episodes_only",
        }


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def rejection_features(
    leaf_probabilities: torch.Tensor,
    parent_mass: torch.Tensor,
    entcomp_unknown: torch.Tensor,
) -> torch.Tensor:
    """Return the two locked CF-FSHP global rejection features."""
    leaf = leaf_probabilities.double()
    mass = parent_mass.to(dtype=torch.float64, device=leaf.device)
    evidence = entcomp_unknown.to(dtype=torch.float64, device=leaf.device)
    if leaf.ndim != 2 or mass.ndim != 2 or evidence.shape != mass.shape:
        raise ValueError("CF-FSHP probabilities must be aligned matrices")
    if int(leaf.shape[0]) != int(mass.shape[0]):
        raise ValueError("CF-FSHP sample counts differ")
    if int(leaf.shape[1]) < 2:
        raise ValueError("CF-FSHP needs at least two retained leaves")
    if not torch.allclose(
        leaf.sum(dim=1),
        torch.ones(
            int(leaf.shape[0]), dtype=leaf.dtype, device=leaf.device
        ),
        atol=1e-5,
    ):
        raise ValueError("Leaf posterior is not normalized")
    leaf = leaf / leaf.sum(dim=1, keepdim=True)
    clipped_evidence = evidence.clamp(
        EVIDENCE_EPS, 1.0 - EVIDENCE_EPS
    )
    log_mass = mass.clamp_min(MASS_EPS).log()
    log_terms = log_mass + torch.logit(clipped_evidence)
    entcomp_aggregate = (
        torch.logsumexp(log_terms, dim=1)
        - torch.logsumexp(log_mass, dim=1)
    )
    safe_leaf = leaf.clamp_min(PROBABILITY_EPS)
    normalized_entropy = -(
        safe_leaf * safe_leaf.log()
    ).sum(dim=1) / math.log(int(leaf.shape[1]))
    result = torch.stack(
        [entcomp_aggregate, normalized_entropy], dim=1
    )
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("CF-FSHP rejection features are non-finite")
    return result


def fit_feature_normalization(
    calibration_features: torch.Tensor,
) -> FeatureNormalization:
    features = calibration_features.detach().double()
    if features.ndim != 2 or int(features.shape[1]) != 2:
        raise ValueError("CF-FSHP normalization expects two features")
    if int(features.shape[0]) < 2:
        raise ValueError("CF-FSHP normalization needs two episodes")
    mean = features.mean(dim=0)
    raw_std = features.std(dim=0, unbiased=False)
    std = raw_std.clamp_min(STD_FLOOR)
    return FeatureNormalization(
        mean=mean,
        std=std,
        raw_std=raw_std,
        sample_count=int(features.shape[0]),
    )


def normalize_rejection_features(
    features: torch.Tensor,
    normalization: FeatureNormalization,
) -> torch.Tensor:
    values = features.to(
        dtype=normalization.mean.dtype,
        device=normalization.mean.device,
    )
    return (values - normalization.mean) / normalization.std


def normalized_parent_depths(
    hierarchy,
    parent_node_indices,
    *,
    global_node_names: list[str] | None = None,
) -> torch.Tensor:
    indices = torch.as_tensor(parent_node_indices, dtype=torch.long)
    node_names = (
        list(hierarchy.id_node_list)
        if global_node_names is None
        else list(global_node_names)
    )
    max_depth = max(
        len(hierarchy.node_ancestors.get(node, []))
        for node in hierarchy.id_node_list
    )
    if max_depth <= 0:
        raise ValueError("Hierarchy has no non-root depth")
    values = [
        len(
            hierarchy.node_ancestors.get(
                node_names[int(index)], []
            )
        )
        / float(max_depth)
        for index in indices.tolist()
    ]
    return torch.tensor(values, dtype=torch.float64)


def augmented_tree_membership(
    hierarchy,
    leaf_node_indices: torch.Tensor,
    parent_node_indices: torch.Tensor,
    *,
    global_node_names: list[str] | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Build structural and explicit internal-unknown augmented-tree edges."""
    leaf_indices = [
        int(value)
        for value in torch.as_tensor(
            leaf_node_indices, dtype=torch.long
        ).tolist()
    ]
    parent_indices = [
        int(value)
        for value in torch.as_tensor(
            parent_node_indices, dtype=torch.long
        ).tolist()
    ]
    terminal_indices = leaf_indices + parent_indices
    if len(terminal_indices) != len(set(terminal_indices)):
        raise ValueError("Leaf and parent terminal indices overlap")
    output_node_names = (
        list(hierarchy.id_node_list)
        if global_node_names is None
        else list(global_node_names)
    )
    node_count = len(output_node_names)
    if any(index < 0 or index >= node_count for index in terminal_indices):
        raise ValueError("Terminal node indices are outside the hierarchy")
    root_name = "root"
    if any(
        output_node_names[index] == root_name
        for index in terminal_indices
    ):
        raise ValueError("Root cannot be a CF-FSHP terminal")
    fold_nodes = set(hierarchy.id_node_list)
    terminal_names = [
        output_node_names[index] for index in terminal_indices
    ]
    missing = sorted(set(terminal_names) - fold_nodes)
    if missing:
        raise ValueError(
            f"Fold-pruned hierarchy misses terminal nodes: {missing[:3]}"
        )

    rows = []
    audit = []
    for fold_node_index, node in enumerate(hierarchy.id_node_list):
        if node == root_name:
            continue
        row = torch.zeros(node_count, dtype=torch.float64)
        for terminal_index, terminal_node in zip(
            terminal_indices, terminal_names
        ):
            ancestor_names = {
                hierarchy.id_node_list[int(index)]
                for index in hierarchy.node_ancestors.get(
                    terminal_node, []
                )
            }
            if (
                terminal_node == node
                or node in ancestor_names
            ):
                row[terminal_index] = 1.0
        if not bool(row.any()):
            raise RuntimeError(
                f"Fold-pruned structural edge {node!r} has no terminal"
            )
        rows.append(row)
        audit.append({
            "edge_kind": "structural",
            "node": node,
            "fold_node_index": fold_node_index,
            "output_node_index": output_node_names.index(node),
        })
    for parent_index in parent_indices:
        row = torch.zeros(node_count, dtype=torch.float64)
        row[parent_index] = 1.0
        rows.append(row)
        audit.append({
            "edge_kind": "unknown_terminal",
            "node": output_node_names[parent_index],
            "output_node_index": parent_index,
        })
    if not rows:
        raise ValueError("Augmented hierarchy has no active edge")
    membership = torch.stack(rows, dim=0)
    expected_edge_count = (
        len(hierarchy.id_node_list) - 1 + len(parent_indices)
    )
    if int(membership.shape[0]) != expected_edge_count:
        raise RuntimeError(
            "Fold-pruned augmented-tree edge count is inconsistent"
        )
    return membership, audit


def tree_brier_loss(
    terminal_probabilities: torch.Tensor,
    target_node_indices: torch.Tensor,
    membership: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    probabilities = terminal_probabilities
    targets = torch.as_tensor(
        target_node_indices, dtype=torch.long, device=probabilities.device
    )
    edges = membership.to(
        dtype=probabilities.dtype, device=probabilities.device
    )
    weights = sample_weights.to(
        dtype=probabilities.dtype, device=probabilities.device
    )
    if probabilities.ndim != 2:
        raise ValueError("Terminal probabilities must be a matrix")
    if int(probabilities.shape[0]) != int(targets.numel()):
        raise ValueError("Tree-Brier targets and probabilities differ")
    if int(edges.shape[1]) != int(probabilities.shape[1]):
        raise ValueError("Tree-Brier membership has the wrong node count")
    if int(weights.numel()) != int(targets.numel()):
        raise ValueError("Tree-Brier sample weights differ")
    if bool((targets < 0).any()) or bool(
        (targets >= int(probabilities.shape[1])).any()
    ):
        raise ValueError("Tree-Brier target is out of range")
    cumulative = probabilities @ edges.t()
    desired = edges.index_select(1, targets).t()
    per_sample = (cumulative - desired).square().mean(dim=1)
    return (weights * per_sample).sum()


def factorized_terminal_posterior(
    leaf_probabilities: torch.Tensor,
    parent_mass: torch.Tensor,
    entcomp_unknown: torch.Tensor,
    *,
    normalized_features: torch.Tensor,
    normalized_depths: torch.Tensor,
    leaf_node_indices: torch.Tensor,
    parent_node_indices: torch.Tensor,
    node_count: int,
    scalars: dict[str, torch.Tensor | float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the locked six-scalar CF-FSHP terminal posterior."""
    leaf = leaf_probabilities.double()
    leaf = leaf / leaf.sum(dim=1, keepdim=True)
    mass = parent_mass.to(dtype=torch.float64, device=leaf.device)
    evidence = entcomp_unknown.to(dtype=torch.float64, device=leaf.device)
    features = normalized_features.to(
        dtype=leaf.dtype, device=leaf.device
    )
    depths = normalized_depths.to(
        dtype=leaf.dtype, device=leaf.device
    )
    if features.shape != (int(leaf.shape[0]), 2):
        raise ValueError("CF-FSHP normalized features have the wrong shape")
    if int(depths.numel()) != int(mass.shape[1]):
        raise ValueError("CF-FSHP parent depths have the wrong shape")

    def scalar(name):
        return torch.as_tensor(
            scalars[name], dtype=leaf.dtype, device=leaf.device
        )

    beta_e = scalar("beta_E")
    beta_h = scalar("beta_H")
    gamma_h = scalar("gamma_H")
    gamma_e = scalar("gamma_E")
    if bool(beta_e <= 0) or bool(beta_h <= 0):
        raise ValueError("CF-FSHP rejection slopes must be positive")
    if bool(gamma_h <= 0) or bool(gamma_e <= 0):
        raise ValueError("CF-FSHP localization slopes must be positive")
    rejection_logit = (
        scalar("beta_0")
        + beta_e * features[:, 0]
        + beta_h * features[:, 1]
    )
    q_unknown = torch.sigmoid(rejection_logit)
    localization_logits = (
        gamma_h * mass.clamp_min(MASS_EPS).log()
        + gamma_e * torch.logit(
            evidence.clamp(EVIDENCE_EPS, 1.0 - EVIDENCE_EPS)
        )
        + scalar("gamma_D") * depths.unsqueeze(0)
    )
    parent_distribution = F.softmax(localization_logits, dim=1)
    leaf_terminal = (1.0 - q_unknown).unsqueeze(1) * leaf
    parent_terminal = q_unknown.unsqueeze(1) * parent_distribution
    terminal = torch.zeros(
        int(leaf.shape[0]),
        int(node_count),
        dtype=leaf.dtype,
        device=leaf.device,
    )
    terminal = terminal.index_copy(
        1, leaf_node_indices.to(leaf.device), leaf_terminal
    )
    terminal = terminal.index_copy(
        1, parent_node_indices.to(leaf.device), parent_terminal
    )
    if not torch.allclose(
        terminal.sum(dim=1),
        torch.ones_like(q_unknown),
        atol=1e-6,
    ):
        raise RuntimeError("CF-FSHP terminal posterior is not normalized")
    return terminal, q_unknown, parent_distribution


def known_favoring_map(
    terminal_probabilities: torch.Tensor,
    leaf_node_indices: torch.Tensor,
    parent_node_indices: torch.Tensor,
) -> torch.Tensor:
    """MAP with the locked exact-tie preference for a known leaf."""
    terminal = terminal_probabilities
    leaf_indices = leaf_node_indices.to(terminal.device)
    parent_indices = parent_node_indices.to(terminal.device)
    leaf_values = terminal.index_select(1, leaf_indices)
    parent_values = terminal.index_select(1, parent_indices)
    leaf_max, leaf_local = leaf_values.max(dim=1)
    parent_max, parent_local = parent_values.max(dim=1)
    leaf_prediction = leaf_indices.index_select(0, leaf_local)
    parent_prediction = parent_indices.index_select(0, parent_local)
    return torch.where(
        parent_max > leaf_max,
        parent_prediction,
        leaf_prediction,
    )


def _actual_scalars(parameters: dict[str, torch.Tensor]) -> dict:
    return {
        "beta_0": parameters["beta_0"],
        "beta_E": F.softplus(parameters["raw_beta_E"]),
        "beta_H": F.softplus(parameters["raw_beta_H"]),
        "gamma_H": F.softplus(parameters["raw_gamma_H"]),
        "gamma_E": F.softplus(parameters["raw_gamma_E"]),
        "gamma_D": parameters["gamma_D"],
    }


def fit_cf_fshp(
    bundle: dict,
    hierarchy,
    *,
    global_node_names: list[str] | None = None,
    max_iter: int = LOCKED_MAX_ITER,
) -> dict:
    """Fit exactly six shared scalars on inner-calibration episodes."""
    if int(max_iter) != LOCKED_MAX_ITER:
        raise ValueError("CF-FSHP max_iter is locked to 100")
    leaf = bundle["leaf_probabilities"].double()
    mass = bundle["parent_mass"].double()
    evidence = bundle["entcomp_unknown"].double()
    targets = bundle["target_node_indices"].long()
    features = rejection_features(leaf, mass, evidence)
    normalization = fit_feature_normalization(features)
    normalized = normalize_rejection_features(
        features, normalization
    )
    depths = normalized_parent_depths(
        hierarchy,
        bundle["parent_node_indices"],
        global_node_names=global_node_names,
    )
    membership, edge_audit = augmented_tree_membership(
        hierarchy,
        bundle["leaf_node_indices"],
        bundle["parent_node_indices"],
        global_node_names=global_node_names,
    )
    weights = macro_terminal_weights(
        bundle["kinds"], bundle["target_groups"]
    )
    raw_one = _inverse_softplus(1.0)
    parameters = {
        "beta_0": torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)),
        "raw_beta_E": torch.nn.Parameter(
            torch.tensor(raw_one, dtype=torch.float64)
        ),
        "raw_beta_H": torch.nn.Parameter(
            torch.tensor(raw_one, dtype=torch.float64)
        ),
        "raw_gamma_H": torch.nn.Parameter(
            torch.tensor(raw_one, dtype=torch.float64)
        ),
        "raw_gamma_E": torch.nn.Parameter(
            torch.tensor(raw_one, dtype=torch.float64)
        ),
        "gamma_D": torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)),
    }
    optimizer = torch.optim.LBFGS(
        list(parameters.values()),
        lr=1.0,
        max_iter=LOCKED_MAX_ITER,
        line_search_fn=LOCKED_LINE_SEARCH,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def objective(backward: bool) -> torch.Tensor:
        terminal, _, _ = factorized_terminal_posterior(
            leaf,
            mass,
            evidence,
            normalized_features=normalized,
            normalized_depths=depths,
            leaf_node_indices=bundle["leaf_node_indices"],
            parent_node_indices=bundle["parent_node_indices"],
            node_count=bundle["node_count"],
            scalars=_actual_scalars(parameters),
        )
        loss = tree_brier_loss(
            terminal, targets, membership, weights
        )
        if backward:
            loss.backward()
        return loss

    initial_loss = float(objective(False).detach())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        return objective(True)

    optimizer.step(closure)
    final_loss = float(objective(False).detach())
    actual = {
        key: float(value.detach())
        for key, value in _actual_scalars(parameters).items()
    }
    return {
        "scalars": actual,
        "initial_scalars": dict(LOCKED_INITIAL_SCALARS),
        "initial_tree_brier": initial_loss,
        "final_tree_brier": final_loss,
        "feature_normalization": normalization.as_dict(),
        "optimizer": "LBFGS",
        "line_search": LOCKED_LINE_SEARCH,
        "max_iter": LOCKED_MAX_ITER,
        "loss": "augmented_tree_brier",
        "parameter_count": 6,
        "edge_count": int(membership.shape[0]),
        "edge_audit": edge_audit,
        "known_weight_sum": float(weights[
            torch.tensor([kind == "known" for kind in bundle["kinds"]])
        ].sum()),
        "pseudo_weight_sum": float(weights[
            torch.tensor([kind == "pseudo" for kind in bundle["kinds"]])
        ].sum()),
    }


def apply_cf_fshp(
    bundle: dict,
    hierarchy,
    fit: dict,
    *,
    global_node_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalization_payload = fit["feature_normalization"]
    normalization = FeatureNormalization(
        mean=torch.tensor(
            normalization_payload["mean"], dtype=torch.float64
        ),
        std=torch.tensor(
            normalization_payload["std"], dtype=torch.float64
        ),
        raw_std=torch.tensor(
            normalization_payload["raw_std"], dtype=torch.float64
        ),
        sample_count=int(normalization_payload["sample_count"]),
    )
    leaf = bundle["leaf_probabilities"].double()
    mass = bundle["parent_mass"].double()
    evidence = bundle["entcomp_unknown"].double()
    features = rejection_features(leaf, mass, evidence)
    normalized = normalize_rejection_features(
        features, normalization
    )
    depths = normalized_parent_depths(
        hierarchy,
        bundle["parent_node_indices"],
        global_node_names=global_node_names,
    )
    return factorized_terminal_posterior(
        leaf,
        mass,
        evidence,
        normalized_features=normalized,
        normalized_depths=depths,
        leaf_node_indices=bundle["leaf_node_indices"],
        parent_node_indices=bundle["parent_node_indices"],
        node_count=bundle["node_count"],
        scalars=fit["scalars"],
    )
