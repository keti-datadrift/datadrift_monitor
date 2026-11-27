import random
import unittest

import torch

from negzerohoc.virtual_open_negprompt import (
    refine_virtual_open_features,
    spherical_sibling_mixup,
    virtual_open_negprompt_loss,
)


class VirtualOpenNegPromptTest(unittest.TestCase):
    def test_sibling_mixup_is_normalized_and_uses_multiple_classes(self):
        features = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ])
        targets = torch.tensor([0, 0, 1, 1])
        virtual = spherical_sibling_mixup(
            features,
            targets,
            num_samples=8,
            mix_min=0.5,
            mix_max=0.5,
            rng=random.Random(0),
        )
        self.assertEqual(tuple(virtual.shape), (8, 3))
        self.assertTrue(torch.allclose(virtual.norm(dim=1), torch.ones(8), atol=1e-5))
        self.assertTrue((virtual[:, 0] > 0.0).all())
        self.assertTrue((virtual[:, 1] > 0.0).all())

    def test_refinement_reduces_child_energy(self):
        virtual = torch.tensor([[0.7, 0.7, 0.1]])
        children = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        parent = torch.tensor([0.0, 0.0, 1.0])
        _, stats = refine_virtual_open_features(
            virtual,
            children,
            parent,
            steps=10,
            step_size=0.1,
            child_temperature=0.1,
            parent_weight=0.5,
            anchor_weight=0.1,
        )
        self.assertLess(
            stats["virtual_refined_child_energy"],
            stats["virtual_initial_child_energy"],
        )

    def test_loss_updates_only_unknown_features(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        children = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        virtual = torch.tensor([[0.7, 0.7], [0.6, 0.8]])
        unknowns = torch.tensor([[0.5, 0.5], [-0.5, 0.5]], requires_grad=True)
        loss, stats = virtual_open_negprompt_loss(
            images,
            children,
            unknowns,
            virtual,
            tau=0.1,
            lambda_virtual=1.0,
            lambda_id_teacher=1.0,
            lambda_coverage=0.1,
            lambda_diversity=0.1,
            diversity_margin=0.2,
        )
        loss.backward()
        self.assertIsNotNone(unknowns.grad)
        self.assertGreater(float(unknowns.grad.abs().sum()), 0.0)
        self.assertGreaterEqual(stats["virtual_unknown_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
