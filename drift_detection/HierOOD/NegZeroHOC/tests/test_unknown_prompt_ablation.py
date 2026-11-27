import random
import unittest

import torch
import torch.nn.functional as F

from negzerohoc.losses import unknown_regularization
from negzerohoc.prompt_models import HierPromptConfig, UnknownPromptLearner
from negzerohoc.training_data import EdgeExample, sample_leave_child_out_episode
from negzerohoc.unknown_scoring import grouped_unknown_logits


class GroupedUnknownScoringTest(unittest.TestCase):
    def test_single_prototype_matches_cosine_logits(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        known = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        unknown = torch.tensor([0.0, 1.0])

        actual = grouped_unknown_logits(
            images,
            known,
            unknown,
            logit_scale=3.0,
        )
        expected_features = torch.cat([known, unknown.unsqueeze(0)], dim=0)
        expected = 3.0 * (images @ expected_features.t())
        torch.testing.assert_close(actual, expected)

    def test_duplicate_prototypes_do_not_change_unknown_prior(self):
        images = torch.tensor([[0.8, 0.6]])
        known = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        unknown = torch.tensor([0.6, 0.8])

        single = grouped_unknown_logits(
            images,
            known,
            unknown,
            logit_scale=5.0,
        )
        repeated = grouped_unknown_logits(
            images,
            known,
            unknown.repeat(4, 1),
            logit_scale=5.0,
        )
        torch.testing.assert_close(single, repeated)


class UnknownEpisodeSamplingTest(unittest.TestCase):
    @staticmethod
    def child_examples(count: int):
        return {
            f"child-{index}": [
                EdgeExample(
                    image_index=index,
                    parent="parent",
                    child=f"child-{index}",
                    leaf=f"child-{index}",
                    parent_depth=1,
                )
            ]
            for index in range(count)
        }

    def test_hide_ratio_counts(self):
        child_examples = self.child_examples(8)
        hide25 = sample_leave_child_out_episode(
            "parent",
            child_examples,
            strategy="hide_ratio_25",
            max_examples=64,
            rng=random.Random(0),
        )
        hide50 = sample_leave_child_out_episode(
            "parent",
            child_examples,
            strategy="hide_ratio_50",
            max_examples=64,
            rng=random.Random(0),
        )
        self.assertEqual(len(hide25.hidden_children), 2)
        self.assertEqual(len(hide50.hidden_children), 4)


class UnknownPrototypeRegularizationTest(unittest.TestCase):
    def test_diversity_penalizes_collapsed_prototypes(self):
        parent = torch.tensor([1.0, 0.0])
        children = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        separated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        _, collapsed_stats = unknown_regularization(
            collapsed,
            parent,
            children,
            lambda_anchor=0.0,
            lambda_child_sep=0.0,
            lambda_prototype_diversity=1.0,
            prototype_diversity_margin=0.2,
        )
        _, separated_stats = unknown_regularization(
            separated,
            parent,
            children,
            lambda_anchor=0.0,
            lambda_child_sep=0.0,
            lambda_prototype_diversity=1.0,
            prototype_diversity_margin=0.2,
        )
        self.assertGreater(
            collapsed_stats["prototype_diversity_loss"],
            separated_stats["prototype_diversity_loss"],
        )


class UnknownPromptLearnerTest(unittest.TestCase):
    class DummyHierarchy:
        max_depth = 2
        id_node_list = ["root", "parent"]
        node_ancestors = {"root": [], "parent": [0]}
        node_description = {}

    class DummyTextEncoder:
        text_width = 4
        projection_dim = 3

        @staticmethod
        def encode_plain_texts(texts):
            return F.normalize(torch.ones(len(texts), 3), dim=-1)

        @staticmethod
        def encode_with_context(texts, context):
            del texts
            return F.normalize(context.mean(dim=1)[:, :3], dim=-1)

    def test_multiple_prototypes_have_distinct_trainable_contexts(self):
        cfg = HierPromptConfig(
            global_ctx_tokens=1,
            depth_ctx_tokens=1,
            parent_ctx_tokens=1,
            depth_embed_dim=2,
            parent_generator_hidden_dim=4,
            unknown_prompts=4,
            unknown_prototype_ctx_tokens=2,
        )
        learner = UnknownPromptLearner(
            "generic",
            self.DummyHierarchy(),
            self.DummyTextEncoder(),
            cfg,
        )
        features = learner.encode_unknown_prototypes(["parent"])
        self.assertEqual(tuple(features.shape), (1, 4, 3))
        self.assertFalse(torch.allclose(features[:, 0], features[:, 1]))

        features[..., 0].sum().backward()
        self.assertIsNotNone(learner.prototype_ctx.grad)
        self.assertGreater(float(learner.prototype_ctx.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
