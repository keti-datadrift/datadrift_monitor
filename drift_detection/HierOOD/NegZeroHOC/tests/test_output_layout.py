from __future__ import annotations

import unittest
from pathlib import Path

from negzerohoc.output_layout import (
    experiment_artifact_path,
    resolve_experiment_artifact,
    resolve_shared_feature_dir,
    shared_feature_dir,
)


class OutputLayoutTest(unittest.TestCase):
    def test_experiment_artifacts_are_grouped_by_experiment(self):
        path = experiment_artifact_path(
            "outputs",
            "joint-lora",
            "checkpoints",
            "best.pt",
        )
        self.assertEqual(
            path,
            Path("outputs/experiments/joint-lora/checkpoints/best.pt"),
        )

    def test_legacy_artifact_path_is_normalized(self):
        path = resolve_experiment_artifact(
            "outputs/results/legacy-name.result",
            output_root="outputs",
            experiment_name="joint-lora",
            kind="results",
            default_filename="default.result",
        )
        self.assertEqual(
            path,
            Path("outputs/experiments/joint-lora/results/legacy-name.result"),
        )

    def test_shared_features_are_separate_from_experiments(self):
        expected = Path("outputs/shared/features/fgvc/clip_b16")
        self.assertEqual(shared_feature_dir("outputs", "fgvc", "clip_b16"), expected)
        self.assertEqual(
            resolve_shared_feature_dir(
                "outputs/features/fgvc/clip_b16",
                output_root="outputs",
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
