import math
import unittest

import torch

from negzerohoc.hc_negprompt import (
    global_decoder_safety_loss,
    hierarchy_constrained_negprompt_loss,
)
from negzerohoc.unknown_scoring import grouped_unknown_logits


def hc_loss(images, positives, negatives, *, route_target=0, **overrides):
    defaults = dict(
        tau=0.1,
        hierarchy_tau=0.1,
        safety_margin=0.0,
        lambda_hnis=1.0,
        lambda_safe=1.0,
        lambda_shell=0.1,
        lambda_diversity=0.05,
        lambda_route=0.5,
        lambda_balance=0.1,
    )
    defaults.update(overrides)
    return hierarchy_constrained_negprompt_loss(
        images,
        positives,
        negatives,
        torch.zeros(images.shape[0], dtype=torch.long),
        torch.tensor([1.0, 0.0, 0.0]),
        [(torch.eye(3)[:2], route_target)],
        **defaults,
    )


class NegativeMassScoringTest(unittest.TestCase):
    def test_logsumexp_matches_total_negative_mass(self):
        images = torch.tensor([[1.0, 0.0]])
        known = torch.tensor([[0.0, 1.0]])
        unknown = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        logits = grouped_unknown_logits(
            images,
            known,
            unknown,
            logit_scale=2.0,
            aggregation="logsumexp",
        )
        expected = torch.logsumexp(torch.tensor([2.0, 0.0]), dim=0)
        torch.testing.assert_close(logits[0, -1], expected)

    def test_logsumexp_preserves_prototype_mass(self):
        images = torch.tensor([[1.0, 0.0]])
        known = torch.tensor([[0.0, 1.0]])
        unknown = torch.tensor([[0.6, 0.8]])
        single = grouped_unknown_logits(
            images,
            known,
            unknown,
            logit_scale=3.0,
            aggregation="logsumexp",
        )
        repeated = grouped_unknown_logits(
            images,
            known,
            unknown.repeat(4, 1),
            logit_scale=3.0,
            aggregation="logsumexp",
        )
        torch.testing.assert_close(
            repeated[0, -1],
            single[0, -1] + math.log(4),
        )


class HierarchyConstrainedLossTest(unittest.TestCase):
    def test_global_decoder_hinge_uses_best_leaf_and_unknown_terminals(self):
        scores = torch.tensor(
            [[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
            requires_grad=True,
        )
        loss, stats = global_decoder_safety_loss(
            scores,
            ["leaf", "leaf", "unknown"],
            margin=0.5,
            safety_mode="squared_hinge",
        )

        self.assertAlmostEqual(float(loss), 1.125)
        self.assertEqual(stats["global_safety_violation_rate"], 0.5)
        self.assertEqual(stats["global_unknown_win_rate"], 0.5)
        loss.backward()
        self.assertEqual(float(scores.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(scores.grad[1].abs().sum()), 0.0)

    def test_full_loss_is_finite_and_updates_negative_prompts(self):
        images = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        positives = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        negatives = torch.tensor(
            [
                [[0.8, 0.6, 0.0], [0.8, 0.0, 0.6]],
                [[0.6, 0.8, 0.0], [0.0, 0.8, 0.6]],
            ],
            requires_grad=True,
        )
        loss, stats = hc_loss(images, positives, negatives)

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("safe_loss", stats)
        self.assertIn("route_loss", stats)
        self.assertIn("diversity_loss", stats)
        loss.backward()
        self.assertIsNotNone(negatives.grad)
        self.assertGreater(float(negatives.grad.norm()), 0.0)

    def test_safety_penalizes_negative_mass_near_id_image(self):
        images = torch.tensor([[1.0, 0.0, 0.0]])
        positives = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        far = torch.tensor([[[-1.0, 0.0, 0.0]], [[0.0, -1.0, 0.0]]])
        near = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
        weights = dict(
            lambda_hnis=0.0,
            lambda_safe=1.0,
            lambda_shell=0.0,
            lambda_diversity=0.0,
            lambda_route=0.0,
            lambda_balance=0.0,
        )

        far_loss, _ = hc_loss(images, positives, far, **weights)
        near_loss, _ = hc_loss(images, positives, near, **weights)
        self.assertLess(float(far_loss), float(near_loss))

    def test_squared_hinge_stops_penalizing_satisfied_id_constraint(self):
        images = torch.tensor([[1.0, 0.0, 0.0]])
        positives = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        safe_negatives = torch.tensor(
            [[[-1.0, 0.0, 0.0]], [[0.0, -1.0, 0.0]]],
            requires_grad=True,
        )
        violating_negatives = torch.tensor(
            [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            requires_grad=True,
        )
        weights = dict(
            safety_mode="squared_hinge",
            lambda_hnis=0.0,
            lambda_safe=1.0,
            lambda_shell=0.0,
            lambda_diversity=0.0,
            lambda_route=0.0,
            lambda_balance=0.0,
        )

        safe_loss, safe_stats = hc_loss(
            images, positives, safe_negatives, **weights
        )
        violating_loss, violating_stats = hc_loss(
            images, positives, violating_negatives, **weights
        )

        self.assertEqual(float(safe_loss), 0.0)
        self.assertEqual(safe_stats["safety_violation_rate"], 0.0)
        self.assertGreater(float(violating_loss), 0.0)
        self.assertEqual(violating_stats["safety_violation_rate"], 1.0)
        safe_loss.backward()
        torch.testing.assert_close(
            safe_negatives.grad,
            torch.zeros_like(safe_negatives),
        )

    def test_rejects_unknown_safety_mode(self):
        images = torch.tensor([[1.0, 0.0, 0.0]])
        positives = torch.tensor([[1.0, 0.0, 0.0]])
        negatives = torch.tensor([[[0.0, 1.0, 0.0]]])

        with self.assertRaisesRegex(ValueError, "Unsupported safety_mode"):
            hc_loss(
                images,
                positives,
                negatives,
                safety_mode="not-a-loss",
            )

    def test_route_loss_places_negative_bank_under_correct_ancestor(self):
        images = torch.tensor([[1.0, 0.0, 0.0]])
        positives = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        correct_route = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
        wrong_route = torch.tensor([[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]])
        weights = dict(
            lambda_hnis=0.0,
            lambda_safe=0.0,
            lambda_shell=0.0,
            lambda_diversity=0.0,
            lambda_route=1.0,
            lambda_balance=0.0,
        )

        correct_loss, _ = hc_loss(images, positives, correct_route, **weights)
        wrong_loss, _ = hc_loss(images, positives, wrong_route, **weights)
        self.assertLess(float(correct_loss), float(wrong_loss))

    def test_tangent_diversity_separates_directions_around_positive(self):
        images = torch.tensor([[1.0, 0.0, 0.0]])
        positives = torch.tensor([[1.0, 0.0, 0.0]])
        collapsed = torch.tensor([[[0.8, 0.6, 0.0], [0.8, 0.6, 0.0]]])
        diverse = torch.tensor([[[0.8, 0.6, 0.0], [0.8, 0.0, 0.6]]])
        weights = dict(
            lambda_hnis=0.0,
            lambda_safe=0.0,
            lambda_shell=0.0,
            lambda_diversity=1.0,
            lambda_route=0.0,
            lambda_balance=0.0,
        )

        collapsed_loss, collapsed_stats = hc_loss(
            images, positives, collapsed, **weights
        )
        diverse_loss, diverse_stats = hc_loss(images, positives, diverse, **weights)
        self.assertLess(float(diverse_loss), float(collapsed_loss))
        self.assertLess(
            diverse_stats["diversity_loss"],
            collapsed_stats["diversity_loss"],
        )


if __name__ == "__main__":
    unittest.main()
