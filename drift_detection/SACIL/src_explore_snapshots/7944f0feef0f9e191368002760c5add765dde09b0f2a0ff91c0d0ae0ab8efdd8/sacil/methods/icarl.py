from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def pycil_icarl_kd_loss(
    current_old_logits: Tensor,
    reference_logits: Tensor,
    *,
    temperature: float = 2.0,
) -> Tensor:
    """The old-class softmax distillation used by PyCIL's iCaRL learner.

    This intentionally has no ``T**2`` multiplier: it mirrors PyCIL's
    ``models/icarl.py::_KD_loss`` rather than the original iCaRL BCE objective.
    """

    if current_old_logits.shape != reference_logits.shape:
        raise ValueError("current and reference old logits must have equal shape")
    if current_old_logits.ndim != 2 or current_old_logits.shape[1] == 0:
        raise ValueError("old logits must be a non-empty matrix")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prediction = F.log_softmax(current_old_logits / temperature, dim=1)
    target = F.softmax(reference_logits.detach() / temperature, dim=1)
    return -(target * prediction).sum() / current_old_logits.shape[0]


def icarl_distillation_targets(
    targets: Tensor,
    num_classes: int,
    *,
    old_probabilities: Tensor | None = None,
    known_classes: int = 0,
) -> Tensor:
    """Build the sigmoid targets used by the original iCaRL objective.

    New-class entries are one-hot labels.  When a frozen teacher is present,
    its sigmoid outputs replace the first ``known_classes`` entries.

    Reference:
        Rebuffi et al., iCaRL, CVPR 2017, official TensorFlow code.
    """

    if targets.ndim != 1:
        raise ValueError("targets must be a vector")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if known_classes < 0 or known_classes > num_classes:
        raise ValueError("known_classes is outside the classifier range")
    if targets.numel() and (
        int(targets.min()) < 0 or int(targets.max()) >= num_classes
    ):
        raise ValueError("target is outside the classifier range")

    result = F.one_hot(targets.long(), num_classes=num_classes).to(
        dtype=torch.float32
    )
    if known_classes == 0:
        if old_probabilities is not None and old_probabilities.numel() != 0:
            raise ValueError("old probabilities were provided without old classes")
        return result
    if old_probabilities is None:
        raise ValueError("old probabilities are required for old classes")
    if old_probabilities.shape != (targets.shape[0], known_classes):
        raise ValueError(
            "old probabilities must have shape [batch, known_classes]"
        )
    result[:, :known_classes] = old_probabilities.detach().to(result)
    return result


def icarl_bce_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    old_logits: Tensor | None = None,
    known_classes: int = 0,
) -> Tensor:
    """Original iCaRL classification/distillation loss.

    Unlike a CE plus KL reproduction, iCaRL uses one sigmoid binary
    cross-entropy over the combined hard-label and teacher target matrix.
    """

    if logits.ndim != 2:
        raise ValueError("logits must be a matrix")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets have different batch sizes")
    old_probabilities = None
    if old_logits is not None:
        if old_logits.shape != (logits.shape[0], known_classes):
            raise ValueError(
                "old logits must have shape [batch, known_classes]"
            )
        old_probabilities = torch.sigmoid(old_logits.detach())
    combined_targets = icarl_distillation_targets(
        targets,
        logits.shape[1],
        old_probabilities=old_probabilities,
        known_classes=known_classes,
    ).to(dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, combined_targets)


def parameter_l2_regularization(
    model: nn.Module, coefficient: float
) -> Tensor:
    """CaSpeR iCaRL's explicit ``wd_reg * sum(theta**2)`` penalty."""

    value = float(coefficient)
    if value < 0:
        raise ValueError("L2 regularization coefficient must be non-negative")
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("L2 regularization requires model parameters")
    penalty = parameters[0].new_zeros(())
    for parameter in parameters:
        penalty = penalty + parameter.float().square().sum()
    return value * penalty
