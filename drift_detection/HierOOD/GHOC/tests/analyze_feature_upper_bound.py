from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--id-split", required=True)
    args = parser.parse_args()

    hierarchy = Hierarchy(get_id_classes(args.id_split), args.hierarchy)
    feature_payload = torch.load(args.features, map_location="cpu", weights_only=False)
    density_payload = torch.load(args.density, map_location="cpu", weights_only=False)
    leaf_indices = [
        index
        for index, name in enumerate(hierarchy.id_node_list)
        if name not in hierarchy.parent2children
    ]
    leaf_index_tensor = torch.tensor(leaf_indices)
    features = torch.nn.functional.normalize(feature_payload["features"], dim=1)
    means = torch.nn.functional.normalize(
        density_payload["means"][leaf_indices],
        dim=1,
    )
    predictions = []
    for batch in features.split(2048):
        predictions.append(leaf_index_tensor[(batch @ means.T).argmax(dim=1)])
    predictions = torch.cat(predictions)
    targets = feature_payload["node_targets"]
    print(
        {
            "accuracy": float((predictions == targets).float().mean()),
            "balanced_accuracy": balanced_accuracy_score(
                targets.numpy(),
                predictions.numpy(),
            ),
        }
    )


if __name__ == "__main__":
    main()
