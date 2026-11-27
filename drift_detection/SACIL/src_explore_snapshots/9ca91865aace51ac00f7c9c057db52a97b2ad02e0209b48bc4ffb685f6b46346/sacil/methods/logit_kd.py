from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def old_logit_kl_loss(
    current_logits: Tensor,
    reference_logits: Tensor,
    *,
    temperature: float = 4.0,
) -> Tensor:
    """Temperature-scaled KL distillation over the teacher's classes."""
    if current_logits.ndim != 2 or reference_logits.ndim != 2:
        raise ValueError("logits must be rank-two tensors")
    if current_logits.shape[0] != reference_logits.shape[0]:
        raise ValueError("current and reference batch sizes differ")
    old_class_count = reference_logits.shape[1]
    if old_class_count <= 0 or current_logits.shape[1] < old_class_count:
        raise ValueError("current logits do not contain all old classes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    current_old = current_logits[:, :old_class_count]
    softened_current = F.log_softmax(
        current_old / float(temperature), dim=1
    )
    softened_reference = F.softmax(
        reference_logits.detach() / float(temperature), dim=1
    )
    return F.kl_div(
        softened_current,
        softened_reference,
        reduction="batchmean",
    ) * float(temperature) ** 2
