from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--id-split", required=True)
    args = parser.parse_args()

    hierarchy = Hierarchy(get_id_classes(args.id_split), args.hierarchy)
    result = torch.load(args.result, map_location="cpu", weights_only=False)
    experiment_dir = Path(args.experiment_dir)
    leaf_indices = {
        index
        for index, name in enumerate(hierarchy.id_node_list)
        if name not in hierarchy.parent2children
    }

    for split in ("val", "ood"):
        predictions = result["results"][split]["predictions"]
        artifact = torch.load(
            experiment_dir / f"features_{split}.pt",
            map_location="cpu",
            weights_only=False,
        )
        targets = artifact["node_targets"]
        depth_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        class_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            target_name = hierarchy.id_node_list[target]
            target_depth = len(hierarchy.node_ancestors[target_name])
            depth_counts[target_depth][1] += 1
            depth_counts[target_depth][0] += int(target == prediction)
            class_counts[target][1] += 1
            class_counts[target][0] += int(target == prediction)

        depth_class_recalls: dict[int, list[float]] = defaultdict(list)
        for target, (correct, count) in class_counts.items():
            target_name = hierarchy.id_node_list[target]
            target_depth = len(hierarchy.node_ancestors[target_name])
            depth_class_recalls[target_depth].append(correct / count)

        internal_rate = sum(
            int(prediction not in leaf_indices) for prediction in predictions.tolist()
        ) / len(predictions)
        prediction_depths = Counter(
            len(hierarchy.node_ancestors[hierarchy.id_node_list[prediction]])
            for prediction in predictions.tolist()
        )
        print(
            split,
            {
                "n": len(targets),
                "accuracy": float((predictions == targets).float().mean()),
                "pred_internal_rate": internal_rate,
                "target_depth_accuracy": {
                    depth: {
                        "correct": correct,
                        "count": count,
                        "accuracy": correct / count,
                    }
                    for depth, (correct, count) in sorted(depth_counts.items())
                },
                "target_depth_balanced_accuracy": {
                    depth: sum(recalls) / len(recalls)
                    for depth, recalls in sorted(depth_class_recalls.items())
                },
                "prediction_depth_counts": dict(sorted(prediction_depths.items())),
            },
        )


if __name__ == "__main__":
    main()
