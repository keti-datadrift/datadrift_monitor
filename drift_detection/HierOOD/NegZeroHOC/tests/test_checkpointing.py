from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from negzerohoc.checkpointing import (
    load_idea3_checkpoint,
    load_idea3_checkpoint_with_fallback,
    previous_checkpoint_path,
    save_idea3_checkpoint,
)


def save_test_checkpoint(
    path: Path, *, training_state=None, extra_payload=None
):
    return save_idea3_checkpoint(
        path,
        stage="test",
        dataset="test-dataset",
        clip_model="test-clip",
        hierarchy="test-hierarchy.json",
        id_split="test-split.csv",
        prompt_config={},
        positive_state_dict={"prompt": torch.tensor([1.0])},
        metrics={"score": 0.5},
        args={"epochs": 2},
        training_state=training_state,
        extra_payload=extra_payload,
    )


class AtomicCheckpointTest(unittest.TestCase):
    def test_round_trip_includes_resumable_training_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"
            training_state = {
                "epoch": 7,
                "optimizer_state_dict": {"state": {}, "param_groups": []},
            }

            save_test_checkpoint(path, training_state=training_state)
            payload = load_idea3_checkpoint(path)

            self.assertEqual(payload["training_state"]["epoch"], 7)
            self.assertFalse(path.with_name(".last.pt.tmp").exists())

    def test_round_trip_includes_atomic_extra_payload(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"
            proxies = torch.arange(6, dtype=torch.float32).reshape(2, 3)

            save_test_checkpoint(
                path,
                training_state={"epoch": 1},
                extra_payload={
                    "metric_proxies": proxies,
                    "metric_proxy_classes": ["a", "b"],
                },
            )
            payload = load_idea3_checkpoint(path)

            self.assertTrue(
                torch.equal(payload["metric_proxies"], proxies)
            )
            self.assertEqual(
                payload["metric_proxy_classes"], ["a", "b"]
            )

    def test_extra_payload_cannot_replace_reserved_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"

            with self.assertRaisesRegex(
                ValueError, "cannot replace checkpoint fields"
            ):
                save_test_checkpoint(
                    path, extra_payload={"stage": "replacement"}
                )

    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"
            path.write_bytes(b"previous-checkpoint")

            with mock.patch(
                "negzerohoc.checkpointing.torch.save",
                side_effect=RuntimeError("simulated power failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "power failure"):
                    save_test_checkpoint(path)

            self.assertEqual(path.read_bytes(), b"previous-checkpoint")
            self.assertFalse(path.with_name(".last.pt.tmp").exists())

    def test_resumable_checkpoint_keeps_one_previous_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"

            save_test_checkpoint(path, training_state={"epoch": 1})
            save_test_checkpoint(path, training_state={"epoch": 2})

            self.assertEqual(load_idea3_checkpoint(path)["training_state"]["epoch"], 2)
            previous_path = previous_checkpoint_path(path)
            self.assertEqual(
                load_idea3_checkpoint(previous_path)["training_state"]["epoch"],
                1,
            )

    def test_corrupt_primary_falls_back_to_previous_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "last.pt"

            save_test_checkpoint(path, training_state={"epoch": 1})
            save_test_checkpoint(path, training_state={"epoch": 2})
            path.write_bytes(b"interrupted-checkpoint")

            payload, loaded_path = load_idea3_checkpoint_with_fallback(path)

            self.assertEqual(payload["training_state"]["epoch"], 1)
            self.assertEqual(loaded_path, previous_checkpoint_path(path))


if __name__ == "__main__":
    unittest.main()
