import random
import unittest

import torch

from negzerohoc.hier_negprompt import hierarchical_negprompt_loss
from negzerohoc.training_data import EdgeExample, sample_parent_known_episode


class ParentKnownEpisodeTest(unittest.TestCase):
    def test_keeps_every_child_as_known(self):
        child_examples = {
            child: [
                EdgeExample(
                    image_index=index,
                    parent="parent",
                    child=child,
                    leaf=child,
                    parent_depth=1,
                )
            ]
            for index, child in enumerate(("A", "B", "C"))
        }
        episode = sample_parent_known_episode(
            "parent",
            child_examples,
            max_examples=12,
            rng=random.Random(0),
        )

        self.assertEqual(episode.children, ["A", "B", "C"])
        self.assertEqual(sorted(episode.labels), ["A", "B", "C"])
        self.assertNotIn("__unknown__", episode.labels)

    def test_root_is_not_trained_as_unknown(self):
        self.assertIsNone(
            sample_parent_known_episode(
                "root",
                {"A": []},
                max_examples=8,
                rng=random.Random(0),
            )
        )


class HierarchicalNegPromptLossTest(unittest.TestCase):
    def test_loss_is_finite_and_updates_negative_features(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        negatives = torch.tensor(
            [[[0.6, 0.8], [0.8, 0.6]], [[-0.6, 0.8], [0.6, -0.8]]],
            requires_grad=True,
        )
        loss, stats = hierarchical_negprompt_loss(
            images,
            positives,
            negatives,
            torch.tensor([0, 1]),
            torch.tensor([0.707, 0.707]),
            tau=0.07,
            lambda_nis=1.0,
            lambda_npd=0.1,
            lambda_nnd=0.05,
            lambda_stop=1.0,
            lambda_parent=0.1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("nis_loss", stats)
        self.assertIn("stop_loss", stats)
        loss.backward()
        self.assertIsNotNone(negatives.grad)
        self.assertGreater(float(negatives.grad.norm()), 0.0)

    def test_stop_loss_rewards_known_children_over_unknown_bank(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        good_negatives = torch.tensor(
            [[[-1.0, 0.0], [-0.8, -0.2]], [[0.0, -1.0], [-0.2, -0.8]]]
        )
        bad_negatives = torch.tensor(
            [[[1.0, 0.0], [0.8, 0.2]], [[0.0, 1.0], [0.2, 0.8]]]
        )
        kwargs = dict(
            target_indices=torch.tensor([0, 1]),
            parent_feature=torch.tensor([0.707, 0.707]),
            tau=0.07,
            lambda_nis=0.0,
            lambda_npd=0.0,
            lambda_nnd=0.0,
            lambda_stop=1.0,
            lambda_parent=0.0,
        )

        good_loss, _ = hierarchical_negprompt_loss(
            images, positives, good_negatives, **kwargs
        )
        bad_loss, _ = hierarchical_negprompt_loss(
            images, positives, bad_negatives, **kwargs
        )
        self.assertLess(float(good_loss), float(bad_loss))


if __name__ == "__main__":
    unittest.main()
