import unittest

import torch

from negzerohoc.oracle_parent import oracle_parent_diagnostics
from negzerohoc.semantic_index import LocalSemanticCandidates


class DummyHierarchy:
    id_node_list = ["root", "A", "A1", "A2", "B", "B1", "B2"]
    parent2children = {
        "root": ["A", "B"],
        "A": ["A1", "A2"],
        "B": ["B1", "B2"],
    }
    node_ancestors = {
        "root": [],
        "A": [0],
        "A1": [0, 1],
        "A2": [0, 1],
        "B": [0],
        "B1": [0, 4],
        "B2": [0, 4],
    }

    def gen_ds2node_map(self, classes):
        return torch.tensor([self.id_node_list.index("A") for _ in classes])


def local(parent, children, child_features, unknown=None):
    return LocalSemanticCandidates(
        parent=parent,
        children=children,
        child_features=torch.tensor(child_features, dtype=torch.float32),
        unknown_feature=None if unknown is None else torch.tensor(unknown, dtype=torch.float32),
        candidate_names=list(children),
        prompts={},
    )


class OracleParentTest(unittest.TestCase):
    def test_oracle_gate_is_separated_from_wrong_root_route(self):
        hierarchy = DummyHierarchy()
        semantic_index = {
            "root": local("root", ["A", "B"], [[1.0, 0.0], [0.0, 1.0]]),
            "A": local("A", ["A1", "A2"], [[1.0, 0.0], [-1.0, 0.0]], [[0.0, 1.0]]),
            "B": local("B", ["B1", "B2"], [[0.0, 1.0], [0.0, -1.0]], [[1.0, 0.0]]),
        }
        result = oracle_parent_diagnostics(
            torch.tensor([[0.0, 1.0]]),
            ["ood-a"],
            torch.tensor([0]),
            hierarchy,
            semantic_index,
            logit_scale=10.0,
        )
        self.assertEqual(result["oracle_unknown_recall"], 1.0)
        self.assertEqual(result["positive_route_reach_rate"], 0.0)
        self.assertEqual(result["joint_student_exact_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
