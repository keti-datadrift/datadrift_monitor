from __future__ import annotations

import torch
from torch.nn import functional as F

from sacil.methods import (
    ClasswiseAutoencoderClassifier,
    RectifiedCosineLinear,
    casper_spectral_loss,
    controlled_transfer_loss,
    create_classification_loss,
    create_contrastive_loss,
    cross_space_clustering_loss,
    fgp_graph_preservation_loss,
    icarl_bce_loss,
    icarl_distillation_targets,
    parameter_l2_regularization,
    pod_flat_loss,
    pod_spatial_loss,
    podnet_nca_loss,
    prototype_cross_entropy,
    prototype_logits,
    reconstruction_confidence_weights,
)


def test_icarl_targets_keep_teacher_probabilities_for_old_classes():
    targets = torch.tensor([0, 3])
    old = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    combined = icarl_distillation_targets(
        targets,
        4,
        old_probabilities=old,
        known_classes=2,
    )
    assert torch.equal(combined[:, :2], old)
    assert torch.equal(combined[:, 2:], torch.tensor([[0.0, 0.0], [0.0, 1.0]]))


def test_icarl_bce_matches_explicit_combined_target():
    logits = torch.tensor(
        [[0.2, -0.4, 0.1], [-0.1, 0.7, -0.2]], requires_grad=True
    )
    old_logits = torch.tensor([[0.6, -0.2], [-0.3, 0.4]])
    targets = torch.tensor([2, 1])
    combined = F.one_hot(targets, 3).float()
    combined[:, :2] = torch.sigmoid(old_logits)
    expected = F.binary_cross_entropy_with_logits(logits, combined)
    actual = icarl_bce_loss(
        logits, targets, old_logits=old_logits, known_classes=2
    )
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_parameter_l2_regularization_matches_casper_formula():
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))
        model.bias.copy_(torch.tensor([3.0]))

    loss = parameter_l2_regularization(model, 1e-5)
    assert torch.allclose(loss, torch.tensor(14e-5))
    loss.backward()
    assert torch.allclose(model.weight.grad, 2e-5 * model.weight)
    assert torch.allclose(model.bias.grad, 2e-5 * model.bias)


def test_podnet_losses_are_zero_for_identical_representations():
    similarities = torch.tensor([[0.8, 0.1], [0.2, 0.9]])
    targets = torch.tensor([0, 1])
    assert torch.isfinite(podnet_nca_loss(similarities, targets))

    features = torch.randn(3, 5)
    feature_map = torch.randn(3, 4, 2, 2)
    assert pod_flat_loss(features, features).abs() < 1e-6
    assert pod_spatial_loss([feature_map], [feature_map]).abs() < 1e-6


def test_podnet_nca_penalizes_weaker_positive_similarity():
    targets = torch.tensor([0])
    strong = podnet_nca_loss(
        torch.tensor([[0.95, 0.1, 0.0]]), targets, margin=0.0
    )
    weak = podnet_nca_loss(
        torch.tensor([[0.2, 0.1, 0.0]]), targets, margin=0.0
    )
    assert strong < weak


def test_fgp_classifier_and_graph_loss():
    classifier = RectifiedCosineLinear(4, 3)
    features = torch.randn(5, 4, requires_grad=True)
    logits = classifier(features)
    assert logits.shape == (5, 3)

    weights = torch.randn(2, 4)
    loss = fgp_graph_preservation_loss(
        features,
        features.detach(),
        weights,
        weights,
    )
    assert loss.abs() < 1e-6
    (loss + logits.mean()).backward()
    assert torch.isfinite(features.grad).all()


def test_cscct_losses_have_gradients_and_safe_empty_transfer():
    current = torch.randn(4, 6, requires_grad=True)
    reference = torch.randn(4, 6)
    targets = torch.tensor([0, 0, 2, 3])
    csc = cross_space_clustering_loss(current, reference, targets)
    transfer = controlled_transfer_loss(
        current,
        reference,
        targets,
        known_classes=2,
        temperature=2.0,
    )
    (csc + transfer).backward()
    assert torch.isfinite(current.grad).all()

    empty = controlled_transfer_loss(
        current.detach(),
        reference,
        torch.tensor([2, 2, 3, 3]),
        known_classes=2,
    )
    assert empty == 0


def test_casper_spectral_loss_is_finite_and_differentiable():
    features = torch.randn(8, 5, requires_grad=True)
    loss = casper_spectral_loss(features, num_classes=2, k=3)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()


def test_create_native_classifier_and_losses():
    classifier = ClasswiseAutoencoderClassifier(
        6,
        3,
        hidden_layers=(5,),
        latent_features=4,
    )
    features = torch.randn(7, 6, requires_grad=True)
    output = classifier(features)
    assert output["logits"].shape == (7, 3)
    assert output["latents"].shape == (7, 3, 4)
    assert torch.allclose(
        output["logits"].sum(dim=1), torch.ones(7), atol=1e-5
    )

    targets = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    weights = reconstruction_confidence_weights(output["error_logits"])
    classification = create_classification_loss(output["logits"], targets)
    contrastive = create_contrastive_loss(
        output["latents"], targets, sample_weights=weights
    )
    loss = classification + contrastive
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()


def test_prototype_ce_uses_the_same_decision_as_normalized_nme():
    features = torch.tensor(
        [[1.0, 0.1], [0.2, 1.0], [-0.9, -0.1]],
        requires_grad=True,
    )
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    targets = torch.tensor([0, 1, 2])

    logits = prototype_logits(features, prototypes, temperature=0.1)
    normalized_features = F.normalize(features.detach(), dim=1)
    normalized_prototypes = F.normalize(prototypes, dim=1)
    nme = torch.cdist(normalized_features, normalized_prototypes).argmin(dim=1)
    assert torch.equal(logits.argmax(dim=1), nme)

    loss, returned_logits = prototype_cross_entropy(
        features, targets, prototypes, temperature=0.1
    )
    assert torch.equal(returned_logits, logits)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert prototypes.grad is None
