import math
import unittest
from argparse import Namespace

import torch

from negzerohoc.idea3_inference import predict_features_idea3
from negzerohoc.losses import sparse_path_bottleneck_loss
from negzerohoc.semantic_index import LocalSemanticCandidates
from scripts.train_idea3_positive_prompts import (
    backward_sparse_path_bottleneck_streaming,
    build_sparse_path_decisions,
)


class DummyHierarchy:
    id_node_list = ["root", "A", "B", "A1", "A2", "B1", "B2"]
    parent2children = {
        "root": ["A", "B"],
        "A": ["A1", "A2"],
        "B": ["B1", "B2"],
    }
    node_ancestors = {
        "root": [],
        "A": [0],
        "B": [0],
        "A1": [0, 1],
        "A2": [0, 1],
        "B1": [0, 2],
        "B2": [0, 2],
    }


def unit_feature(first_coordinate: float) -> torch.Tensor:
    second = math.sqrt(max(0.0, 1.0 - first_coordinate * first_coordinate))
    return torch.tensor([first_coordinate, second], dtype=torch.float32)


def local_candidates(parent: str, children: list[str], scores: list[float]):
    return LocalSemanticCandidates(
        parent=parent,
        children=children,
        child_features=torch.stack([unit_feature(score) for score in scores]),
        unknown_feature=None,
        candidate_names=children,
        prompts={child: [] for child in children},
    )


class SparsePathLossTest(unittest.TestCase):
    def test_parent_stream_gradient_matches_full_loss(self):
        class DummyLearner(torch.nn.Module):
            edge_order = [
                ("root", "A"), ("root", "B"),
                ("A", "A1"), ("A", "A2"),
                ("B", "B1"), ("B", "B2"),
            ]

            def __init__(self):
                super().__init__()
                self.features = torch.nn.Parameter(torch.tensor([
                    [0.9, 0.1], [0.1, 0.9],
                    [0.8, 0.2], [0.2, 0.8],
                    [0.7, 0.3], [0.3, 0.7],
                ]))

            def encode_edges(self, edges):
                indices = [self.edge_order.index(edge) for edge in edges]
                return self.features[indices]

            def encode_children(self, parent, children):
                return self.encode_edges([(parent, child) for child in children])

        args = Namespace(
            tau=0.2,
            loss_bottleneck_weight=0.6,
            loss_bottleneck_temperature=0.4,
            loss_route_margin=0.05,
            loss_margin_weight=0.3,
        )
        hierarchy = DummyHierarchy()
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        leaves = ["A1", "B1"]

        full_learner = DummyLearner()
        full_adapter = torch.nn.Linear(2, 2, bias=False)
        stream_adapter = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            full_adapter.weight.copy_(torch.tensor([[1.0, 0.1], [0.2, 0.9]]))
            stream_adapter.load_state_dict(full_adapter.state_dict())
        decisions, targets, _ = build_sparse_path_decisions(
            hierarchy, full_learner, full_adapter(images), leaves, tau=args.tau
        )
        full_loss, _ = sparse_path_bottleneck_loss(
            decisions,
            targets,
            bottleneck_weight=args.loss_bottleneck_weight,
            bottleneck_temperature=args.loss_bottleneck_temperature,
            route_margin=args.loss_route_margin / args.tau,
            margin_weight=args.loss_margin_weight,
        )
        full_loss.backward()

        stream_learner = DummyLearner()
        stream_stats = backward_sparse_path_bottleneck_streaming(
            args,
            hierarchy,
            stream_learner,
            stream_adapter,
            images,
            leaves,
        )

        self.assertAlmostEqual(float(full_loss), stream_stats["loss"], places=6)
        self.assertTrue(torch.allclose(
            full_learner.features.grad,
            stream_learner.features.grad,
            atol=1e-6,
            rtol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            full_adapter.weight.grad,
            stream_adapter.weight.grad,
            atol=1e-6,
            rtol=1e-5,
        ))

    def test_active_path_uses_all_siblings_but_skips_inactive_subtrees(self):
        class DummyLearner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Parameter(torch.tensor([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.2],
                    [0.2, 0.8],
                ]))

            def encode_edges(self, edges):
                return self.features[:len(edges)]

        decisions, targets, stats = build_sparse_path_decisions(
            DummyHierarchy(),
            DummyLearner(),
            torch.tensor([[1.0, 0.0]]),
            ["A1"],
            tau=1.0,
        )

        self.assertEqual(stats["active_parents"], 2)
        self.assertEqual(stats["active_prompts"], 4)
        self.assertEqual(len(decisions[0]), 2)
        self.assertEqual([int(logits.numel()) for logits in decisions[0]], [2, 2])
        self.assertEqual(targets[0], [0, 0])

    def test_correct_path_has_lower_loss_and_gradients(self):
        good_logits = [
            torch.tensor([3.0, -1.0], requires_grad=True),
            torch.tensor([2.0, 0.0], requires_grad=True),
        ]
        bad_logits = [
            torch.tensor([3.0, -1.0], requires_grad=True),
            torch.tensor([-1.0, 2.0], requires_grad=True),
        ]
        good_loss, good_stats = sparse_path_bottleneck_loss(
            [good_logits], [[0, 0]], bottleneck_weight=0.5, margin_weight=0.25
        )
        bad_loss, bad_stats = sparse_path_bottleneck_loss(
            [bad_logits], [[0, 0]], bottleneck_weight=0.5, margin_weight=0.25
        )

        self.assertLess(float(good_loss), float(bad_loss))
        self.assertEqual(good_stats["path_acc"], 1.0)
        self.assertEqual(bad_stats["path_acc"], 0.0)
        good_loss.backward()
        self.assertIsNotNone(good_logits[0].grad)
        self.assertGreater(float(good_logits[0].grad.abs().sum()), 0.0)


class GlobalPathInferenceTest(unittest.TestCase):
    def test_global_map_can_recover_from_greedy_root_choice(self):
        hierarchy = DummyHierarchy()
        semantic_index = {
            "root": local_candidates("root", ["A", "B"], [0.2, 0.1]),
            "A": local_candidates("A", ["A1", "A2"], [0.01, 0.0]),
            "B": local_candidates("B", ["B1", "B2"], [0.9, -0.9]),
        }
        image = torch.tensor([[1.0, 0.0]])

        greedy = predict_features_idea3(
            image, hierarchy, semantic_index, mode="positive_child_only", tau=1.0
        )
        global_map = predict_features_idea3(
            image, hierarchy, semantic_index, mode="positive_global_path", tau=1.0
        )

        self.assertEqual(hierarchy.id_node_list[int(greedy["preds"][0])], "A1")
        self.assertEqual(hierarchy.id_node_list[int(global_map["preds"][0])], "B1")


if __name__ == "__main__":
    unittest.main()
