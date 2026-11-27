import unittest

import torch

from libs.utils import score_util


class ScoreUtilTest(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.children_map = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        self.group_sizes = torch.tensor([2.0, 2.0])
        self.p = torch.tensor([[0.6, 0.2, 0.1, 0.05]], dtype=torch.float32)
        self.n_samples = self.p.size(0)
        self.n_parents = 2

    def test_normalized_entropy_range(self):
        group_sums, entropy, entropy_norm, _ = score_util._compute_score_terms(
            self.p,
            self.children_map,
            self.group_sizes,
            self.n_samples,
            self.n_parents,
            device=self.device,
        )
        self.assertTrue(torch.all(entropy >= 0.0))
        self.assertTrue(torch.all(entropy_norm >= 0.0))
        self.assertTrue(torch.all(entropy_norm <= 1.0))
        self.assertTrue(torch.all(group_sums <= 1.0))

    def test_complementary_probability_range(self):
        group_sums = score_util.compute_group_sums(
            self.p, self.children_map, self.n_samples, self.n_parents, device=self.device
        )
        p_comp = score_util.compute_complementary_probability(group_sums)
        self.assertTrue(torch.all(p_comp >= 0.0))
        self.assertTrue(torch.all(p_comp <= 1.0))

    def test_local_conditional_sums_to_one(self):
        methods = [
            score_util.compprob,
            score_util.entcompprob,
            score_util.legacy_entcompprob,
            score_util.normentropy_compprob,
            score_util.depth_weighted_raw,
            score_util.depth_weighted_norm,
            score_util.fixedbeta_norm,
            score_util.node_weighted_norm,
        ]

        kwargs = {
            "depth": 0,
            "depth_alpha": [1.0],
            "depth_beta": [1.0],
            "beta_rule": "inv_log",
            "num_classes": self.p.size(1),
            "node_alpha_by_depth": [torch.ones(self.n_parents)],
            "node_beta_by_depth": [torch.ones(self.n_parents)],
        }

        for method in methods:
            with self.subTest(method=method.__name__):
                result, p_ood = method(
                    self.p,
                    self.children_map,
                    self.group_sizes,
                    self.n_samples,
                    self.n_parents,
                    device=self.device,
                    **kwargs,
                )
                result_sums = score_util.compute_group_sums(
                    result,
                    self.children_map,
                    self.n_samples,
                    self.n_parents,
                    device=self.device,
                )
                self.assertTrue(torch.allclose(result_sums + p_ood,
                                               torch.ones_like(p_ood),
                                               atol=1e-4))

    def test_uniform_distribution_normalized_entropy_is_one(self):
        p = torch.tensor([[0.25, 0.25]], dtype=torch.float32)
        children_map = torch.tensor([0, 0], dtype=torch.long)
        group_sizes = torch.tensor([2.0])
        _, _, entropy_norm, _ = score_util._compute_score_terms(
            p, children_map, group_sizes, 1, 1, device=self.device
        )
        self.assertTrue(torch.allclose(entropy_norm, torch.ones_like(entropy_norm), atol=1e-4))

    def test_one_hot_distribution_normalized_entropy_is_zero(self):
        p = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        children_map = torch.tensor([0, 0], dtype=torch.long)
        group_sizes = torch.tensor([2.0])
        _, _, entropy_norm, _ = score_util._compute_score_terms(
            p, children_map, group_sizes, 1, 1, device=self.device
        )
        self.assertTrue(torch.allclose(entropy_norm, torch.zeros_like(entropy_norm), atol=1e-4))

    def test_baseline_matches_reference_formula(self):
        result, p_ood = score_util.entcompprob(
            self.p,
            self.children_map,
            self.group_sizes,
            self.n_samples,
            self.n_parents,
            device=self.device,
        )

        eps = 1e-12
        group_sums = torch.zeros(self.n_samples, self.n_parents)
        group_sums.scatter_add_(1, self.children_map.expand(self.n_samples, -1), self.p + eps)
        p_norm = (self.p + eps) / (group_sums[:, self.children_map] + eps)

        entropy = torch.zeros(self.n_samples, self.n_parents)
        entropy.scatter_add_(1, self.children_map.expand(self.n_samples, -1), -1.0 * p_norm * torch.log(p_norm + eps))

        p_comp = 1.0 - group_sums
        score = entropy + p_comp
        total_sums = group_sums + score
        ref_result = self.p / total_sums[:, self.children_map]
        ref_p_ood = score / total_sums

        self.assertTrue(torch.allclose(result, ref_result, atol=1e-6))
        self.assertTrue(torch.allclose(p_ood, ref_p_ood, atol=1e-6))

    def test_depth_weighted_raw_all_ones_matches_entcompprob(self):
        baseline_result, baseline_p_ood = score_util.entcompprob(
            self.p,
            self.children_map,
            self.group_sizes,
            self.n_samples,
            self.n_parents,
            device=self.device,
        )

        weighted_result, weighted_p_ood = score_util.depth_weighted_raw(
            self.p,
            self.children_map,
            self.group_sizes,
            self.n_samples,
            self.n_parents,
            device=self.device,
            depth=0,
            depth_alpha=[1.0],
            depth_beta=[1.0],
        )

        self.assertTrue(torch.allclose(baseline_result, weighted_result, atol=1e-6))
        self.assertTrue(torch.allclose(baseline_p_ood, weighted_p_ood, atol=1e-6))

    def test_legacy_entcompprob_matches_original_reference_formula(self):
        result, p_ood = score_util.legacy_entcompprob(
            self.p,
            self.children_map,
            self.group_sizes,
            self.n_samples,
            self.n_parents,
            device=self.device,
        )

        eps = 1e-12
        group_sums = torch.zeros(self.n_samples, self.n_parents)
        group_sums.scatter_add_(1, self.children_map.expand(self.n_samples, -1), self.p + eps)
        p_norm = (self.p + eps) / (group_sums[:, self.children_map] + eps)

        entropy = torch.zeros(self.n_samples, self.n_parents)
        entropy.scatter_add_(1, self.children_map.expand(self.n_samples, -1), -1.0 * p_norm * torch.log(p_norm + eps))

        total_sums = group_sums + entropy
        ref_result = self.p / total_sums[:, self.children_map]
        ref_p_ood = entropy / total_sums

        self.assertTrue(torch.allclose(result, ref_result, atol=1e-6))
        self.assertTrue(torch.allclose(p_ood, ref_p_ood, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
