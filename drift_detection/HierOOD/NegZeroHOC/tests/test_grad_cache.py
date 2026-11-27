import copy
import unittest

import torch
from torch import nn

from negzerohoc.grad_cache import (
    ImageFeatureEncoder,
    capture_rng_state,
    grad_cache_forward_backward,
    restore_rng_state,
)


class ToyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, 3, bias=False)

    def get_image_features(self, pixel_values):
        return self.projection(pixel_values)


class GradCacheTest(unittest.TestCase):
    def test_image_feature_wrapper(self):
        model = ToyClip()
        images = torch.randn(4, 5)
        self.assertTrue(torch.equal(
            ImageFeatureEncoder(model)(images),
            model.get_image_features(pixel_values=images),
        ))

    def test_cached_gradient_matches_direct_logical_batch(self):
        torch.manual_seed(11)
        direct_model = ToyClip()
        cached_model = copy.deepcopy(direct_model)
        direct_prompt = nn.Parameter(torch.randn(4, 3))
        cached_prompt = nn.Parameter(direct_prompt.detach().clone())
        images = torch.randn(12, 5)

        def objective(features, prompt):
            similarities = features @ prompt.t()
            pair_term = (features @ features.t()).square().mean()
            return similarities.logsumexp(dim=1).mean() + 0.2 * pair_term

        direct_loss = objective(
            ImageFeatureEncoder(direct_model)(images),
            direct_prompt,
        )
        direct_loss.backward()

        def cached_objective(features):
            return objective(features, cached_prompt), {}

        grad_cache_forward_backward(
            images,
            ImageFeatureEncoder(cached_model),
            cached_objective,
            micro_batch_size=3,
        )

        self.assertTrue(torch.allclose(
            direct_model.projection.weight.grad,
            cached_model.projection.weight.grad,
            atol=1e-6,
            rtol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            direct_prompt.grad,
            cached_prompt.grad,
            atol=1e-6,
            rtol=1e-5,
        ))

    def test_rng_snapshot_replays_dropout_mask(self):
        torch.manual_seed(7)
        dropout = nn.Dropout(p=0.5)
        values = torch.ones(8, 8)
        snapshot = capture_rng_state()
        first = dropout(values)
        restore_rng_state(snapshot)
        second = dropout(values)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
