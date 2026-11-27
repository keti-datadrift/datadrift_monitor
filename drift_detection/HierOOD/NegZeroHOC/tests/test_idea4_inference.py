import math
import unittest

import torch

from negzerohoc.idea4_inference import (
    predict_features_terminal_global_path,
    terminal_global_path_scores,
)
from negzerohoc.semantic_index import LocalSemanticCandidates


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


def local_candidates(parent, children, child_scores, unknown_score=None):
    unknown_feature = None if unknown_score is None else unit_feature(unknown_score)
    candidate_names = list(children)
    if unknown_feature is not None:
        candidate_names.append(f"__unknown__:{parent}")
    return LocalSemanticCandidates(
        parent=parent,
        children=list(children),
        child_features=torch.stack([unit_feature(score) for score in child_scores]),
        unknown_feature=unknown_feature,
        candidate_names=candidate_names,
        prompts={name: [] for name in candidate_names},
    )


class TerminalGlobalPathTest(unittest.TestCase):
    def test_differentiable_scores_match_decoder_and_update_unknown_feature(self):
        hierarchy = DummyHierarchy()
        semantic_index = {
            "root": local_candidates("root", ["A", "B"], [0.8, -0.8]),
            "A": local_candidates("A", ["A1", "A2"], [0.2, 0.1], unknown_score=0.95),
            "B": local_candidates("B", ["B1", "B2"], [0.9, -0.9], unknown_score=-0.8),
        }
        semantic_index["A"].unknown_feature.requires_grad_(True)
        features = torch.tensor([[1.0, 0.0]])

        scores = terminal_global_path_scores(
            features,
            hierarchy,
            semantic_index,
            logit_scale=4.0,
        )
        prediction = predict_features_terminal_global_path(
            features,
            hierarchy,
            semantic_index,
            logit_scale=4.0,
        )
        winning_node = scores["candidate_nodes"][
            int(scores["score_matrix"].argmax(dim=1)[0])
        ]

        self.assertEqual(
            winning_node,
            hierarchy.id_node_list[int(prediction["preds"][0])],
        )
        scores["score_matrix"].sum().backward()
        self.assertIsNotNone(semantic_index["A"].unknown_feature.grad)
        self.assertGreater(
            float(semantic_index["A"].unknown_feature.grad.norm()),
            0.0,
        )

    def test_parent_unknown_competes_with_all_complete_leaf_paths(self):
        hierarchy = DummyHierarchy()
        semantic_index = {
            "root": local_candidates("root", ["A", "B"], [0.8, -0.8]),
            "A": local_candidates("A", ["A1", "A2"], [0.2, 0.1], unknown_score=0.95),
            "B": local_candidates("B", ["B1", "B2"], [0.9, -0.9], unknown_score=-0.8),
        }

        output = predict_features_terminal_global_path(
            torch.tensor([[1.0, 0.0]]),
            hierarchy,
            semantic_index,
            logit_scale=4.0,
        )

        self.assertEqual(hierarchy.id_node_list[int(output["preds"][0])], "A")
        self.assertEqual(output["diagnostics"]["candidate_type_counts"], {"unknown": 1})
        self.assertEqual(output["diagnostics"]["unknown_selection_rate"], 1.0)

    def test_leaf_wins_when_local_unknown_is_weak(self):
        hierarchy = DummyHierarchy()
        semantic_index = {
            "root": local_candidates("root", ["A", "B"], [0.8, -0.8]),
            "A": local_candidates("A", ["A1", "A2"], [0.95, -0.2], unknown_score=-0.9),
            "B": local_candidates("B", ["B1", "B2"], [0.1, -0.1], unknown_score=-0.8),
        }

        output = predict_features_terminal_global_path(
            torch.tensor([[1.0, 0.0]]),
            hierarchy,
            semantic_index,
            logit_scale=4.0,
        )

        self.assertEqual(hierarchy.id_node_list[int(output["preds"][0])], "A1")
        self.assertEqual(output["diagnostics"]["candidate_type_counts"], {"leaf": 1})

    def test_root_unknown_is_ignored_by_default(self):
        hierarchy = DummyHierarchy()
        semantic_index = {
            "root": local_candidates("root", ["A", "B"], [0.8, -0.8], unknown_score=1.0),
            "A": local_candidates("A", ["A1", "A2"], [0.95, -0.2], unknown_score=-0.9),
            "B": local_candidates("B", ["B1", "B2"], [0.1, -0.1], unknown_score=-0.8),
        }

        output = predict_features_terminal_global_path(
            torch.tensor([[1.0, 0.0]]),
            hierarchy,
            semantic_index,
            logit_scale=4.0,
            allow_root_unknown=False,
        )

        self.assertNotEqual(hierarchy.id_node_list[int(output["preds"][0])], "root")
        self.assertNotIn("root", output["diagnostics"]["stop_node_counts"])


if __name__ == "__main__":
    unittest.main()
