from __future__ import annotations

import os
import random
import unittest

import numpy as np
import torch

from negzerohoc.runtime import configure_reproducibility, seed_data_loader_worker


class ReproducibilityTest(unittest.TestCase):
    def test_configure_reproducibility_repeats_all_rngs(self):
        configure_reproducibility(17)
        first = (
            random.random(),
            float(np.random.rand()),
            torch.rand(4),
        )

        configure_reproducibility(17)
        second = (
            random.random(),
            float(np.random.rand()),
            torch.rand(4),
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_data_loader_worker_seed_repeats_python_and_numpy(self):
        torch.manual_seed(23)
        seed_data_loader_worker(0)
        first = (random.random(), float(np.random.rand()))

        torch.manual_seed(23)
        seed_data_loader_worker(0)
        second = (random.random(), float(np.random.rand()))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
