from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def pycil_finetune_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    known_classes: int,
) -> Tensor:
    """Fine-tune CE with the class scope used by PyCIL.

    PyCIL trains the base session over the complete base head.  In later
    sessions its Fine-tune learner receives new-class samples only and applies
    CE to the newly appended classifier slice.  Keeping this operation in a
    method module prevents it from being accidentally collapsed into Replay
    CE in the shared trainer.
    """

    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("logits must be a matrix and targets a vector")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets have different batch sizes")
    known = int(known_classes)
    if known < 0 or known >= logits.shape[1]:
        raise ValueError("known class count is outside the classifier")
    if known == 0:
        return F.cross_entropy(logits, targets)
    if targets.numel() and bool((targets < known).any()):
        raise ValueError("PyCIL Fine-tune incremental batches must be new-only")
    return F.cross_entropy(logits[:, known:], targets - known)
