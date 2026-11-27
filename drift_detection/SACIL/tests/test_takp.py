from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from sacil.engine.trainer import SACILTrainer, inverse_class_sample_weights
from sacil.methods import takp_mixed_classification_loss
from sacil.models import TaKPIncrementalNet


def test_takp_public_bbn_mixing_and_forward_shapes() -> None:
    model = TaKPIncrementalNet(num_classes=10, mix_scale=2.0).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model.forward_detailed(images, alpha=0.5)
        logits, features = model(images, return_features=True)

    assert output.conventional_features.shape == (2, 512)
    assert output.rebalancing_features.shape == (2, 512)
    assert features.shape == (2, 1024)
    assert logits.shape == (2, 10)
    assert torch.allclose(
        features,
        torch.cat(
            (
                output.conventional_features,
                output.rebalancing_features,
            ),
            dim=1,
        ),
    )
    assert torch.allclose(logits, output.logits)


def test_takp_cifar_stem_keeps_cifar_resolution() -> None:
    model = TaKPIncrementalNet(
        num_classes=10, mix_scale=2.0, stem="cifar"
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        shared = model.backbone.shared_features(images)
        logits = model(images)

    assert shared.shape == (2, 512, 4, 4)
    assert logits.shape == (2, 10)


def test_takp_classifier_expansion_preserves_old_rows() -> None:
    model = TaKPIncrementalNet(num_classes=10)
    old_weight = model.classifier.weight.detach().clone()
    old_bias = model.classifier.bias.detach().clone()

    model.expand_classes(20)

    assert model.num_classes == 20
    assert torch.equal(model.classifier.weight[:10], old_weight)
    assert torch.equal(model.classifier.bias[:10], old_bias)


def test_inverse_sampler_matches_paper_class_probabilities() -> None:
    targets = [0, 0, 1, 1, 1, 1]
    weights = inverse_class_sample_weights(targets)
    class_zero_mass = weights[:2].sum()
    class_one_mass = weights[2:].sum()

    assert torch.allclose(
        class_zero_mass / class_one_mass,
        torch.tensor(2.0, dtype=torch.double),
    )


def test_takp_mixed_loss_updates_both_label_terms() -> None:
    logits = torch.randn(4, 6, requires_grad=True)
    conventional_targets = torch.tensor([0, 1, 2, 3])
    rebalancing_targets = torch.tensor([3, 2, 1, 0])

    loss = takp_mixed_classification_loss(
        logits,
        conventional_targets,
        rebalancing_targets,
        alpha=0.75,
    )
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_takp_initial_plain_ce_disables_incremental_rebalancing() -> None:
    trainer = SACILTrainer.__new__(SACILTrainer)
    trainer.model = TaKPIncrementalNet(
        num_classes=2, mix_scale=2.0, stem="cifar"
    )
    trainer.device = torch.device("cpu")
    trainer.config = {"debug": {"max_batches_per_epoch": 1}}
    loader = DataLoader(
        [
            {
                "image": torch.randn(3, 32, 32),
                "target": torch.tensor(index % 2),
            }
            for index in range(2)
        ],
        batch_size=2,
    )

    training_log = trainer._train_takp_initial_ce(
        loader,
        {
            "epochs": 1,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 2e-4,
            "nesterov": False,
            "scheduler": "cosine",
        },
    )

    assert training_log["initial_objective"] == "cross_entropy"
    assert training_log["dual_rebalancing"] is False
    assert training_log["lambda_kd"] == 0.0
    assert training_log["lambda_topology"] == 0.0
    assert training_log["epoch_logs"][0]["batches"] == 1
