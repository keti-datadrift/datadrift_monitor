from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.density import gaussian_bump, gaussian_bump_integrals, gaussian_logpdf
from core.hierarchy_inference import hierarchical_node_probabilities


class TinyHierarchy:
    def __init__(self):
        self.id_node_list = ["root", "parent", "child_a", "child_b"]
        self.node_ancestors = {
            "root": [],
            "parent": [0],
            "child_a": [0, 1],
            "child_b": [0, 1],
        }
        self.parent2children = {
            "root": ["parent"],
            "parent": ["child_a", "child_b"],
        }
        self.ood_train_classes = ["unknown_child"]

    def gen_ds2node_map(self, classes):
        assert classes == self.ood_train_classes
        return torch.tensor([1], dtype=torch.long)


def build_density_payload():
    means = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    variances = torch.tensor(
        [
            [9.0, 9.0],
            [4.0, 1.0],
            [0.1, 0.1],
            [0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    return {
        "means": means,
        "variances": variances,
        "covariance_type": "diag",
        "counts": torch.tensor([100, 100, 50, 50], dtype=torch.long),
    }


def main():
    hierarchy = TinyHierarchy()
    density = build_density_payload()
    features = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    logpdf = gaussian_logpdf(
        features,
        density["means"],
        density["variances"],
        covariance_type="diag",
    )
    assert logpdf.shape == (3, 4)

    bumps = gaussian_bump(
        features,
        density["means"],
        density["variances"],
        covariance_type="diag",
        node_indices=[2, 3],
    )
    assert bumps.shape == (3, 2)
    assert torch.all(bumps > 0)
    assert torch.all(bumps <= 1.0)
    assert torch.isclose(bumps[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(bumps[2, 1], torch.tensor(1.0), atol=1e-6)

    integrals = gaussian_bump_integrals(
        1,
        [2, 3],
        density["means"],
        density["variances"],
        covariance_type="diag",
    )
    assert integrals.shape == (2,)
    assert torch.all(integrals > 0)
    assert torch.all(integrals <= 1.0)

    final_probs, debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "normalize_ood_pdf": True,
            "eps": 1e-12,
        },
    )
    assert final_probs.shape == (3, 4)
    assert torch.allclose(final_probs.sum(dim=1), torch.ones(3), atol=1e-6)

    parent_local = debug["local_info"]["parent"]
    local_sum = parent_local["child_probs"].sum(dim=1) + parent_local["ood_prob"]
    assert torch.allclose(local_sum, torch.ones(3), atol=1e-6)
    assert parent_local["ood_prob"][1] > parent_local["ood_prob"][0]
    assert parent_local["ood_prob"][1] > parent_local["ood_prob"][2]

    multiscale_probs, multiscale_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "multiscale_parent_mask",
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "ood_prior": 1.0,
            "parent_covariance_scales": [1.0, 2.0, 4.0],
            "parent_scale_weights": [0.2, 0.3, 0.5],
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.allclose(multiscale_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    multiscale_parent = multiscale_debug["local_info"]["parent"]
    assert torch.all(multiscale_parent["parent_scale_normalizers"] > 0)
    assert torch.isclose(
        multiscale_parent["parent_scale_weights"].sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )

    child_mixture_probs, child_mixture_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "child_mixture_mask",
            "ood_base_cov_scale": 2.0,
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "ood_prior": 1.0,
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.allclose(child_mixture_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    child_mixture_parent = child_mixture_debug["local_info"]["parent"]
    assert child_mixture_parent["normalizer"] > 0
    assert child_mixture_parent["bump_integrals"].shape == (2, 2)

    random_effects_probs, random_effects_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "random_effects_parent",
            "between_cov_estimator": "empirical_child_means",
            "between_cov_scale": 0.5,
            "mask_cov_scale": 4.0,
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "candidate_prior": "uniform",
            "ood_prior": 1.0,
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.allclose(random_effects_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    random_effects_parent = random_effects_debug["local_info"]["parent"]
    assert random_effects_parent["between_cov_estimator"] == "empirical_child_means"
    assert torch.allclose(
        random_effects_parent["ood_base_mean"],
        torch.tensor([0.0, 0.0]),
        atol=1e-6,
    )
    assert random_effects_parent["normalizer"] > 0

    shrunk_probs, shrunk_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "random_effects_parent",
            "between_cov_estimator": "shrunk_child_means",
            "between_cov_shrinkage_strength": 3.0,
            "between_cov_scale": 0.5,
            "mask_cov_scale": 4.0,
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "candidate_prior": "uniform",
            "ood_prior": 1.0,
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.allclose(shrunk_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    shrunk_parent = shrunk_debug["local_info"]["parent"]
    assert shrunk_parent["between_cov_estimator"] == "shrunk_child_means"
    assert shrunk_parent["between_cov_shrinkage_strength"] == 3.0

    product_probs, product_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "random_effects_parent",
            "between_cov_estimator": "shrunk_child_means",
            "between_cov_shrinkage_strength": 3.0,
            "between_cov_scale": 0.5,
            "mask_cov_scale": 4.0,
            "mask_type": "product",
            "lambda": 0.9,
            "product_mask_samples": 256,
            "child_weight": "uniform",
            "candidate_prior": "uniform",
            "ood_prior": 1.0,
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.allclose(product_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    product_parent = product_debug["local_info"]["parent"]
    assert product_parent["normalizer"] > 0
    assert product_parent["normalizer"] <= 1

    balanced_prior_probs, balanced_prior_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "random_effects_parent",
            "between_cov_estimator": "empirical_child_means",
            "between_cov_scale": 0.5,
            "mask_cov_scale": 4.0,
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "candidate_prior": "balanced_terminal",
            "ood_prior": 1.0,
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.allclose(
        balanced_prior_probs.sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )
    balanced_parent = balanced_prior_debug["local_info"]["parent"]
    assert torch.allclose(
        balanced_parent["child_candidate_priors"],
        torch.tensor([1.0 / 3.0, 1.0 / 3.0]),
        atol=1e-6,
    )
    assert torch.isclose(
        balanced_parent["ood_candidate_prior"],
        torch.tensor(1.0 / 3.0),
        atol=1e-6,
    )

    mixed_prior_cfg = {
        **balanced_prior_debug["cgm"],
        "candidate_prior": "mixed_balanced_terminal",
    }
    mixed_prior_probs, mixed_prior_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg=mixed_prior_cfg,
    )
    assert torch.allclose(mixed_prior_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    mixed_parent = mixed_prior_debug["local_info"]["parent"]
    assert torch.allclose(
        mixed_parent["child_candidate_priors"],
        torch.tensor([0.25, 0.25]),
        atol=1e-6,
    )
    assert torch.isclose(
        mixed_parent["ood_candidate_prior"],
        torch.tensor(0.5),
        atol=1e-6,
    )

    positive_residual_probs, positive_residual_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "positive_density_residual",
            "child_weight": "uniform",
            "candidate_prior": "uniform",
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.isfinite(positive_residual_probs).all()
    assert torch.allclose(
        positive_residual_probs.sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )
    positive_residual_parent = positive_residual_debug["local_info"]["parent"]
    assert positive_residual_parent["candidate_prior_mode"] == "induced_residual"
    assert positive_residual_parent["ood_prior"] == "induced_residual_mass"

    binary_probs, binary_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "ood_density": "random_effects_parent",
            "between_cov_estimator": "empirical_child_means",
            "between_cov_scale": 1.0,
            "mask_type": "sum",
            "lambda": 0.0,
            "child_weight": "uniform",
            "candidate_prior": "uniform",
            "normalize_ood_pdf": True,
            "local_mode": "binary_model_selection",
            "eps": 1e-12,
        },
    )
    assert torch.isfinite(binary_probs).all()
    assert torch.allclose(binary_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    binary_parent = binary_debug["local_info"]["parent"]
    assert binary_parent["candidate_prior_mode"] == "binary_symmetric"
    assert binary_parent["ood_prior"] == "binary_symmetric"

    vmf_probs, vmf_debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density,
        score_type="gaussian_loglik",
        temperature=1.0,
        cgm_cfg={
            "enabled": True,
            "strict_pdf": True,
            "density_family": "vmf",
            "ood_density": "vmf_parent_mask",
            "vmf_kappa_scale": 1.0,
            "vmf_ood_kappa_scale": 1.0,
            "vmf_mask_kappa": 20.0,
            "mask_type": "sum",
            "lambda": 0.9,
            "child_weight": "uniform",
            "candidate_prior": "balanced_terminal",
            "ood_prior": 1.0,
            "normalize_ood_pdf": True,
            "local_mode": "density_softmax",
            "eps": 1e-12,
        },
    )
    assert torch.allclose(vmf_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    vmf_parent = vmf_debug["local_info"]["parent"]
    assert vmf_parent["ood_density"] == "vmf_parent_mask"
    assert vmf_parent["normalizer"] > 0

    print("CGM smoke checks passed")


if __name__ == "__main__":
    main()
