from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def herding_prefix_extrapolated_means(
    exemplar_features: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    full_prefix: int = 20,
    reference_prefix: int = 10,
) -> tuple[Tensor, Tensor]:
    """Return ordinary NME and Richardson-extrapolated herding means.

    ``exemplar_features`` must preserve the stored herding priority order
    within every class.  Each row is the per-exemplar NME contribution: a
    normalized feature, or the average of normalized original/mirror
    features when horizontal-flip prototype construction is enabled.

    Under the first-order sequence model ``m_k = q + a/k``, eliminating the
    unknown leading error gives

        q = (K * m_K - J * m_J) / (K - J),

    where K is ``full_prefix`` and J is ``reference_prefix``.
    """

    if exemplar_features.ndim != 2:
        raise ValueError("exemplar_features must have shape [N, D]")
    labels = targets.detach().cpu().long().flatten()
    values = exemplar_features.detach().cpu().float()
    if values.shape[0] != labels.numel():
        raise ValueError("feature and target counts do not match")
    class_count = int(num_classes)
    full = int(full_prefix)
    reference = int(reference_prefix)
    if class_count <= 0:
        raise ValueError("num_classes must be positive")
    if not 0 < reference < full:
        raise ValueError(
            "reference_prefix must be positive and smaller than full_prefix"
        )

    ordinary: list[Tensor] = []
    extrapolated: list[Tensor] = []
    for class_id in range(class_count):
        class_values = values[labels == class_id]
        if class_values.shape[0] != full:
            raise ValueError(
                f"class {class_id} has {class_values.shape[0]} exemplars; "
                f"expected exactly {full} ordered exemplars"
            )
        mean_full = class_values.mean(dim=0)
        mean_reference = class_values[:reference].mean(dim=0)
        estimate = (
            float(full) * mean_full
            - float(reference) * mean_reference
        ) / float(full - reference)
        ordinary.append(F.normalize(mean_full, dim=0))
        extrapolated.append(F.normalize(estimate, dim=0))

    return torch.stack(ordinary), torch.stack(extrapolated)
