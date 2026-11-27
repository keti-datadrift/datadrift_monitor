from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def takp_mixed_classification_loss(
    mixed_logits: Tensor,
    conventional_targets: Tensor,
    rebalancing_targets: Tensor,
    *,
    alpha: float,
) -> Tensor:
    """BBN/TaKP classification loss from the released training code."""
    alpha = float(alpha)
    return (
        alpha * F.cross_entropy(mixed_logits, conventional_targets)
        + (1.0 - alpha)
        * F.cross_entropy(mixed_logits, rebalancing_targets)
    )
