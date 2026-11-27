from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch

from negzerohoc.cf_fshp import (
    LOCKED_INITIAL_SCALARS,
    apply_cf_fshp,
    augmented_tree_membership,
    factorized_terminal_posterior,
    fit_cf_fshp,
    fit_feature_normalization,
    known_favoring_map,
    rejection_features,
    tree_brier_loss,
)
from scripts.train_cf_fshp_oof import (
    FGVC_EXPECTED_AUGMENTED_EDGE_COUNTS,
    load_config,
    method_development_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def toy_hierarchy_with_nonzero_root():
    # Parent p deliberately occupies output index zero. This mirrors the
    # released FGVC hierarchy, whose root is not the first sorted node.
    return SimpleNamespace(
        id_node_list=["p", "a", "b", "q", "root", "c", "d"],
        parent2children={
            "root": ["p", "q"],
            "p": ["a", "b"],
            "q": ["c", "d"],
        },
        node_ancestors={
            "root": [],
            "p": [4],
            "a": [4, 0],
            "b": [4, 0],
            "q": [4],
            "c": [4, 3],
            "d": [4, 3],
        },
    )


LEAF_INDICES = torch.tensor([1, 2, 5, 6])
PARENT_INDICES = torch.tensor([0, 3])


def test_route_mass_mean_feature_is_float64_and_scale_invariant():
    leaf = torch.tensor([[0.6, 0.2, 0.1, 0.1]])
    mass = torch.tensor([[0.8, 0.2]])
    evidence = torch.tensor([[0.2, 0.8]])
    features = rejection_features(leaf, mass, evidence)
    odds = evidence.double() / (1.0 - evidence.double())
    expected_entcomp = torch.log(
        (mass.double() * odds).sum(dim=1) / mass.double().sum(dim=1)
    )
    expected_entropy = -(
        leaf.double() * leaf.double().log()
    ).sum(dim=1) / math.log(4)
    assert features.dtype == torch.float64
    assert torch.allclose(features[:, 0], expected_entcomp)
    assert torch.allclose(features[:, 1], expected_entropy)
    scaled = rejection_features(leaf, mass * 17.0, evidence)
    assert torch.allclose(features, scaled)


def test_calibration_normalization_uses_population_std_and_locked_floor():
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 2.0]], dtype=torch.float64
    )
    normalization = fit_feature_normalization(features)
    assert torch.allclose(
        normalization.mean, torch.tensor([2.0, 2.0], dtype=torch.float64)
    )
    assert torch.allclose(
        normalization.raw_std,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        normalization.std,
        torch.tensor([1.0, 1e-8], dtype=torch.float64),
    )
    assert normalization.as_dict()["variance_estimator"] == (
        "population_unbiased_false"
    )


def test_augmented_tree_has_fold_edges_plus_explicit_unknown_edges():
    hierarchy = toy_hierarchy_with_nonzero_root()
    membership, audit = augmented_tree_membership(
        hierarchy, LEAF_INDICES, PARENT_INDICES
    )
    # Six original non-root edges and two virtual p->u_p/q->u_q edges.
    assert membership.shape == (8, 7)
    assert sum(row["edge_kind"] == "structural" for row in audit) == 6
    assert sum(
        row["edge_kind"] == "unknown_terminal" for row in audit
    ) == 2
    p_unknown_row = next(
        index for index, row in enumerate(audit)
        if row["edge_kind"] == "unknown_terminal"
        and row["node"] == "p"
    )
    assert torch.equal(
        membership[p_unknown_row],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     dtype=torch.float64),
    )


def test_tree_brier_penalizes_far_branch_more_than_near_sibling():
    hierarchy = toy_hierarchy_with_nonzero_root()
    membership, _ = augmented_tree_membership(
        hierarchy, LEAF_INDICES, PARENT_INDICES
    )
    target = torch.tensor([1])  # leaf a
    near = torch.zeros(1, 7, dtype=torch.float64)
    near[0, 2] = 1.0  # sibling leaf b under p
    far = torch.zeros(1, 7, dtype=torch.float64)
    far[0, 5] = 1.0  # leaf c under the other root child q
    weight = torch.ones(1, dtype=torch.float64)
    near_loss = tree_brier_loss(near, target, membership, weight)
    far_loss = tree_brier_loss(far, target, membership, weight)
    assert near_loss < far_loss


def test_factorized_posterior_preserves_leaf_odds_and_total_unknown():
    leaf = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    mass = torch.tensor([[0.7, 0.3]])
    evidence = torch.tensor([[0.2, 0.7]])
    terminal, unknown, parent = factorized_terminal_posterior(
        leaf,
        mass,
        evidence,
        normalized_features=torch.zeros(1, 2),
        normalized_depths=torch.tensor([0.5, 0.5]),
        leaf_node_indices=LEAF_INDICES,
        parent_node_indices=PARENT_INDICES,
        node_count=7,
        scalars=LOCKED_INITIAL_SCALARS,
    )
    assert terminal.dtype == torch.float64
    assert torch.allclose(terminal.sum(dim=1), torch.ones(1).double())
    assert torch.allclose(
        terminal[:, PARENT_INDICES].sum(dim=1), unknown
    )
    assert torch.allclose(parent.sum(dim=1), torch.ones(1).double())
    assert torch.allclose(
        terminal[0, 1] / terminal[0, 2],
        leaf[0, 0].double() / leaf[0, 1].double(),
    )


def test_known_favoring_map_uses_strict_unknown_comparison_and_stable_ties():
    terminal = torch.zeros(3, 7)
    terminal[0, 1] = 0.4
    terminal[0, 2] = 0.4
    terminal[0, 0] = 0.4  # exact leaf/unknown tie -> first leaf
    terminal[1, 1] = 0.4
    terminal[1, 0] = 0.400001  # strictly greater unknown
    terminal[2, 0] = 0.5
    terminal[2, 3] = 0.5  # parent tie -> first parent position
    predictions = known_favoring_map(
        terminal, LEAF_INDICES, PARENT_INDICES
    )
    assert predictions.tolist() == [1, 0, 0]


def synthetic_bundle():
    leaf = torch.tensor([
        [0.80, 0.10, 0.05, 0.05],
        [0.10, 0.80, 0.05, 0.05],
        [0.05, 0.05, 0.80, 0.10],
        [0.05, 0.05, 0.10, 0.80],
        [0.45, 0.45, 0.05, 0.05],
        [0.40, 0.40, 0.10, 0.10],
        [0.05, 0.05, 0.45, 0.45],
        [0.10, 0.10, 0.40, 0.40],
    ])
    mass = torch.stack(
        [leaf[:, :2].sum(dim=1), leaf[:, 2:].sum(dim=1)],
        dim=1,
    )
    evidence = torch.tensor([
        [0.05, 0.05],
        [0.05, 0.05],
        [0.05, 0.05],
        [0.05, 0.05],
        [0.90, 0.10],
        [0.80, 0.20],
        [0.10, 0.90],
        [0.20, 0.80],
    ])
    return {
        "leaf_probabilities": leaf,
        "parent_mass": mass,
        "entcomp_unknown": evidence,
        "leaf_node_indices": LEAF_INDICES,
        "parent_node_indices": PARENT_INDICES,
        "node_count": 7,
        "target_node_indices": torch.tensor([1, 2, 5, 6, 0, 0, 3, 3]),
        "kinds": ["known"] * 4 + ["pseudo"] * 4,
        "target_groups": ["a", "b", "c", "d", "p", "p", "q", "q"],
    }


def test_locked_fit_uses_six_scalars_calibration_norm_and_fold_topology():
    hierarchy = toy_hierarchy_with_nonzero_root()
    bundle = synthetic_bundle()
    fit = fit_cf_fshp(bundle, hierarchy)
    assert fit["parameter_count"] == 6
    assert set(fit["scalars"]) == set(LOCKED_INITIAL_SCALARS)
    assert fit["initial_scalars"] == LOCKED_INITIAL_SCALARS
    assert fit["max_iter"] == 100
    assert fit["line_search"] == "strong_wolfe"
    assert fit["edge_count"] == 8
    assert fit["feature_normalization"]["sample_count"] == 8
    assert fit["final_tree_brier"] <= fit["initial_tree_brier"] + 1e-10
    terminal, unknown, _ = apply_cf_fshp(bundle, hierarchy, fit)
    assert terminal.shape == (8, 7)
    assert unknown.shape == (8,)


def test_config_and_artifact_semantics_are_locked_actual_ood_free():
    args = load_config(
        REPO_ROOT
        / "configs/22_cf_fshp/fgvc_aircraft_oof_screen_gpu0.yaml"
    )
    assert args.experiment_name.startswith("ideaV-")
    assert args.max_iter == 100
    assert Path(args.cf_rpep_checkpoint).name.endswith(".pt")
    assert not any(
        "actual_ood" in key.lower() or "official_ood" in key.lower()
        for key in args.raw_config["cf_fshp"]
    )
    metadata = method_development_metadata()
    assert metadata["idea"] == "V"
    assert metadata["primary_decoder"] == "categorical_map"
    assert metadata["strict_confirmatory_gate"] is False
    assert metadata["may_unlock_official_ood"] is False
    assert FGVC_EXPECTED_AUGMENTED_EDGE_COUNTS == {
        0: 119, 1: 117, 2: 114, 3: 118
    }
