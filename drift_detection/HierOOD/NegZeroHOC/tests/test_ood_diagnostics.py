import unittest

import torch

from negzerohoc.ood_diagnostics import binary_ood_metrics, max_cosine_scores


class OODDiagnosticsTest(unittest.TestCase):
    def test_max_cosine_scores_normalizes_inputs(self):
        images = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        texts = torch.tensor([[4.0, 0.0], [0.0, 5.0]])
        scores, predictions = max_cosine_scores(images, texts)
        self.assertTrue(torch.allclose(scores, torch.ones(2)))
        self.assertEqual(predictions.tolist(), [0, 1])

    def test_binary_ood_metrics_perfect_separation(self):
        metrics = binary_ood_metrics([-2.0, -1.0], [1.0, 2.0])
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["fpr95"], 0.0)
        self.assertEqual(metrics["best_balanced_acc_diagnostic_only"], 1.0)


if __name__ == "__main__":
    unittest.main()

