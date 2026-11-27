import unittest
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from negzerohoc.image_metric import (
    GloballyBalancedPKBatchSampler,
    HierarchyPKBatchSampler,
    MSLossPKBatchSampler,
    PKBatchSampler,
    batch_hard_hierarchical_triplet_loss,
    cosine_proxy_loss,
    supervised_contrastive_loss,
)
from negzerohoc.losses import dual_weihims_positive_loss
from negzerohoc.virtual_open_negprompt import joint_virtual_open_prompt_loss


class ImageMetricTest(unittest.TestCase):
    def test_pk_sampler_has_p_classes_and_k_examples(self):
        targets = [0] * 4 + [1] * 4 + [2] * 4
        sampler = PKBatchSampler(
            targets, classes_per_batch=2, examples_per_class=3, seed=7
        )
        batch = next(iter(sampler))
        selected = [targets[index] for index in batch]
        counts = {target: selected.count(target) for target in set(selected)}
        self.assertEqual(len(batch), 6)
        self.assertEqual(sorted(counts.values()), [3, 3])

    def test_hierarchy_pk_sampler_selects_near_and_far_classes(self):
        targets = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
        paths = {
            0: ("root", "a", "x"),
            1: ("root", "a", "y"),
            2: ("root", "b", "z"),
            3: ("root", "c", "w"),
        }
        sampler = HierarchyPKBatchSampler(
            targets,
            class_paths=paths,
            classes_per_batch=3,
            examples_per_class=2,
            seed=3,
        )
        selected = sampler._select_classes(random.Random(5))
        anchor = selected[0]
        candidate_distances = [
            sampler.class_distances[(anchor, target)]
            for target in sampler.classes if target != anchor
        ]
        selected_distances = [
            sampler.class_distances[(anchor, target)] for target in selected[1:]
        ]
        self.assertIn(min(candidate_distances), selected_distances)
        self.assertIn(max(candidate_distances), selected_distances)

    def test_ms_loss_sampler_consumes_each_full_queue_once(self):
        targets = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
        sampler = MSLossPKBatchSampler(
            targets,
            classes_per_batch=2,
            examples_per_class=2,
            seed=11,
        )
        sampler.set_epoch(1)
        batches = list(iter(sampler))
        counts_after_first_iteration = list(sampler.sample_counts)
        self.assertEqual(batches, list(iter(sampler)))
        self.assertEqual(counts_after_first_iteration, sampler.sample_counts)
        self.assertEqual(len(batches), 4)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(16)))
        for batch in batches:
            labels = [targets[index] for index in batch]
            self.assertEqual(sorted(labels.count(label) for label in set(labels)), [2, 2])

    def test_ms_loss_sampler_resume_preserves_next_batch(self):
        targets = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5
        sampler = MSLossPKBatchSampler(
            targets,
            classes_per_batch=2,
            examples_per_class=2,
            seed=13,
        )
        sampler.set_epoch(1)
        list(iter(sampler))
        state = sampler.state_dict()
        resumed = MSLossPKBatchSampler(
            targets,
            classes_per_batch=2,
            examples_per_class=2,
            seed=999,
        )
        resumed.load_state_dict(state)
        sampler.set_epoch(2)
        resumed.set_epoch(2)
        self.assertEqual(list(iter(sampler)), list(iter(resumed)))

    def test_globally_balanced_sampler_equalizes_sample_exposure(self):
        targets = [0] * 5 + [1] * 7 + [2] * 9 + [3] * 6
        sampler = GloballyBalancedPKBatchSampler(
            targets,
            classes_per_batch=2,
            examples_per_class=2,
            seed=17,
        )
        for epoch in range(1, 31):
            sampler.set_epoch(epoch)
            for batch in sampler:
                labels = [targets[index] for index in batch]
                self.assertEqual(
                    sorted(labels.count(label) for label in set(labels)),
                    [2, 2],
                )
        counts_after_training = list(sampler.sample_counts)
        list(iter(sampler))
        self.assertEqual(counts_after_training, sampler.sample_counts)
        self.assertLessEqual(
            max(sampler.sample_counts) - min(sampler.sample_counts),
            1,
        )

    def test_globally_balanced_sampler_resume_preserves_next_batch(self):
        targets = [0] * 5 + [1] * 7 + [2] * 9 + [3] * 6
        sampler = GloballyBalancedPKBatchSampler(
            targets,
            classes_per_batch=2,
            examples_per_class=2,
            seed=19,
        )
        sampler.set_epoch(1)
        list(iter(sampler))
        state = sampler.state_dict()
        resumed = GloballyBalancedPKBatchSampler(
            targets,
            classes_per_batch=2,
            examples_per_class=2,
            seed=999,
        )
        resumed.load_state_dict(state)
        sampler.set_epoch(2)
        resumed.set_epoch(2)
        self.assertEqual(list(iter(sampler)), list(iter(resumed)))

    def test_stateful_sampler_is_invariant_to_multiworker_prefetch(self):
        targets = [target for target in range(8) for _ in range(8)]
        for sampler_type in (MSLossPKBatchSampler, GloballyBalancedPKBatchSampler):
            sampler = sampler_type(
                targets,
                classes_per_batch=4,
                examples_per_class=2,
                seed=23,
            )
            sampler.set_epoch(1)
            expected_exposures = len(sampler) * 4 * 2
            loader = DataLoader(
                TensorDataset(torch.arange(len(targets))),
                batch_sampler=sampler,
                num_workers=2,
                persistent_workers=True,
            )
            self.assertEqual(sum(1 for _ in loader), len(sampler))
            self.assertEqual(sum(sampler.sample_counts), expected_exposures)
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()

    def test_metric_losses_backpropagate(self):
        features = torch.nn.Parameter(torch.tensor([
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
        ]))
        proxies = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        targets = torch.tensor([0, 0, 1, 1])
        supcon, _ = supervised_contrastive_loss(
            features, targets, temperature=0.1
        )
        triplet, _ = batch_hard_hierarchical_triplet_loss(
            features,
            targets,
            torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
            base_margin=0.1,
            hierarchy_margin=0.1,
        )
        proxy, _ = cosine_proxy_loss(
            features, proxies, targets, temperature=0.1, margin=0.05
        )
        loss = supcon + triplet + proxy
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(proxies.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_clustered_features_have_lower_supcon_loss(self):
        targets = torch.tensor([0, 0, 1, 1])
        clustered = torch.tensor([
            [1.0, 0.0], [0.99, 0.01],
            [0.0, 1.0], [0.01, 0.99],
        ])
        mixed = torch.tensor([
            [1.0, 0.0], [0.0, 1.0],
            [0.99, 0.01], [0.01, 0.99],
        ])
        clustered_loss, _ = supervised_contrastive_loss(
            clustered, targets, temperature=0.1
        )
        mixed_loss, _ = supervised_contrastive_loss(
            mixed, targets, temperature=0.1
        )
        self.assertLess(float(clustered_loss), float(mixed_loss))

    def test_joint_prompt_loss_updates_positive_and_negative_features(self):
        children = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        unknowns = torch.nn.Parameter(torch.tensor([[0.7, 0.7], [0.6, 0.8]]))
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        virtual = torch.tensor([[0.7, 0.7], [0.65, 0.75]])
        loss, stats = joint_virtual_open_prompt_loss(
            images,
            torch.tensor([0, 1]),
            children,
            unknowns,
            virtual,
            tau=0.1,
            lambda_id=1.0,
            lambda_virtual=1.0,
            lambda_coverage=0.1,
            lambda_diversity=0.1,
            lambda_separation=0.1,
            diversity_margin=0.2,
            separation_margin=0.5,
        )
        loss.backward()
        self.assertIsNotNone(children.grad)
        self.assertIsNotNone(unknowns.grad)
        self.assertIn("virtual_unknown_recall", stats)

    def test_dual_weihims_updates_images_and_prompts_without_ce(self):
        images = torch.nn.Parameter(torch.randn(8, 6))
        prompt_depth_1 = torch.nn.Parameter(torch.randn(2, 6))
        prompt_depth_2 = torch.nn.Parameter(torch.randn(4, 6))
        paths = torch.tensor([
            [0, 1, 3], [0, 1, 3],
            [0, 1, 4], [0, 1, 4],
            [0, 2, 5], [0, 2, 5],
            [0, 2, 6], [0, 2, 6],
        ])
        loss, stats = dual_weihims_positive_loss(
            images,
            paths,
            {1: prompt_depth_1, 2: prompt_depth_2},
            {1: torch.tensor([1, 2]), 2: torch.tensor([3, 4, 5, 6])},
            {
                1: torch.tensor([[0, 1, -1], [0, 2, -1]]),
                2: torch.tensor([
                    [0, 1, 3], [0, 1, 4], [0, 2, 5], [0, 2, 6]
                ]),
            },
            mining_margin=10.0,
        )
        loss.backward()
        self.assertIsNotNone(images.grad)
        self.assertIsNotNone(prompt_depth_1.grad)
        self.assertIsNotNone(prompt_depth_2.grad)
        self.assertEqual(stats["path_ce_loss"], 0.0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
