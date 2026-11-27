import math
import unittest

import torch

from negzerohoc.metric_terminal import (
    build_metric_terminal_specs,
    grouped_cosine_logmeanexp,
    metric_terminal_scores,
    normalized_softmin,
    predict_features_metric_terminal,
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


def unit_feature(cosine: float) -> torch.Tensor:
    return torch.tensor(
        [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))],
        dtype=torch.float32,
    )


class MetricTerminalTest(unittest.TestCase):
    def test_normalized_softmin_has_no_path_length_offset(self):
        short = normalized_softmin(torch.tensor([[0.4]]), 0.1)
        long = normalized_softmin(torch.tensor([[0.4, 0.4, 0.4]]), 0.1)
        self.assertTrue(torch.allclose(short, long, atol=1e-6))

    def test_positive_only_leaf_uses_terminal_and_route_consistency(self):
        hierarchy = DummyHierarchy()
        specs = build_metric_terminal_specs(hierarchy)
        edges = {
            ("root", "A"): unit_feature(0.9),
            ("root", "B"): unit_feature(0.2),
            ("A", "A1"): unit_feature(0.8),
            ("A", "A2"): unit_feature(0.1),
            ("B", "B1"): unit_feature(0.95),
            ("B", "B2"): unit_feature(-0.2),
        }
        output = predict_features_metric_terminal(
            torch.tensor([[1.0, 0.0]]),
            hierarchy,
            edges,
            specs,
            terminal_weight=0.25,
            bottleneck_temperature=0.05,
        )
        self.assertEqual(hierarchy.id_node_list[int(output["preds"][0])], "A1")
        self.assertEqual(output["diagnostics"]["candidate_type_counts"], {"leaf": 1})

    def test_parent_unknown_is_a_global_terminal_candidate(self):
        hierarchy = DummyHierarchy()
        specs = build_metric_terminal_specs(hierarchy, ["root", "A"])
        self.assertNotIn("root", [spec.unknown_parent for spec in specs])
        edges = {
            ("root", "A"): unit_feature(0.9),
            ("root", "B"): unit_feature(0.1),
            ("A", "A1"): unit_feature(0.2),
            ("A", "A2"): unit_feature(0.1),
            ("B", "B1"): unit_feature(0.4),
            ("B", "B2"): unit_feature(0.3),
        }
        output = predict_features_metric_terminal(
            torch.tensor([[1.0, 0.0]]),
            hierarchy,
            edges,
            specs,
            unknown_features_by_parent={"A": torch.stack([unit_feature(0.95)])},
            terminal_weight=0.5,
            bottleneck_temperature=0.05,
        )
        self.assertEqual(hierarchy.id_node_list[int(output["preds"][0])], "A")
        self.assertEqual(output["diagnostics"]["unknown_selection_rate"], 1.0)

    def test_unknown_logmeanexp_is_prototype_count_invariant(self):
        images = torch.tensor([[1.0, 0.0]])
        one = grouped_cosine_logmeanexp(images, unit_feature(0.7), 0.07)
        many = grouped_cosine_logmeanexp(
            images,
            torch.stack([unit_feature(0.7)] * 4),
            0.07,
        )
        self.assertTrue(torch.allclose(one, many, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
