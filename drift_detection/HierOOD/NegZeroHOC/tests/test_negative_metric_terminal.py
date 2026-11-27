import unittest

import torch

from negzerohoc.metric_terminal import build_metric_terminal_specs
from negzerohoc.negative_metric_terminal import (
    leave_one_child_out_terminal_recall,
    threshold_terminal_winner_indices,
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


class NegativeMetricTerminalTest(unittest.TestCase):
    def test_unknown_requires_margin_threshold(self):
        hierarchy = DummyHierarchy()
        specs = build_metric_terminal_specs(hierarchy, ["A", "B"])
        # Candidate order: A1, A2, B1, B2, unknown-A, unknown-B.
        scores = torch.tensor([[0.70, 0.20, 0.10, 0.00, 0.75, 0.30]])
        known = threshold_terminal_winner_indices(
            scores, specs, unknown_threshold=0.10
        )
        unknown = threshold_terminal_winner_indices(
            scores, specs, unknown_threshold=0.0
        )
        self.assertEqual(specs[int(known[0])].node, "A1")
        self.assertEqual(specs[int(unknown[0])].unknown_parent, "A")

    def test_loo_hides_child_subtree_only_for_decoder_validation(self):
        hierarchy = DummyHierarchy()
        specs = build_metric_terminal_specs(hierarchy, ["A", "B"])
        # Four validation samples, one per leaf. Each matching parent unknown
        # wins after that leaf is hidden.
        scores = torch.tensor([
            [0.95, 0.10, 0.05, 0.00, 0.80, 0.10],
            [0.10, 0.95, 0.05, 0.00, 0.80, 0.10],
            [0.05, 0.00, 0.95, 0.10, 0.10, 0.80],
            [0.05, 0.00, 0.10, 0.95, 0.10, 0.80],
        ])
        result = leave_one_child_out_terminal_recall(
            scores,
            specs,
            hierarchy,
            ["A1", "A2", "B1", "B2"],
            unknown_threshold=0.0,
        )
        self.assertEqual(result["fold_count"], 4)
        self.assertEqual(result["parent_count"], 2)
        self.assertAlmostEqual(result["fold_macro_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
