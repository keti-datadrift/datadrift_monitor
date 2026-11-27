from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .icarl import pycil_icarl_kd_loss


@torch.no_grad()
def analytic_embedding_gradient_alignment(
    new_all_logits: Tensor,
    new_reference_old_logits: Tensor,
    new_targets: Tensor,
    classifier_weights: Tensor,
    *,
    temperature: float,
    threshold: float,
) -> tuple[Tensor, Tensor]:
    """Return detached CE/KD embedding-gradient cosine and keep mask."""

    known_classes = new_reference_old_logits.shape[1]
    new_count = new_all_logits.shape[0]
    weights = classifier_weights.detach()
    ce_delta = F.softmax(new_all_logits.detach(), dim=1)
    ce_delta[
        torch.arange(new_count, device=new_all_logits.device), new_targets
    ] -= 1.0
    g_ce = ce_delta @ weights
    student_old_prob = F.softmax(
        new_all_logits.detach()[:, :known_classes] / temperature, dim=1
    )
    teacher_old_prob = F.softmax(
        new_reference_old_logits.detach() / temperature, dim=1
    )
    g_kd = ((student_old_prob - teacher_old_prob) / temperature) @ weights[
        :known_classes
    ]
    alignment = F.cosine_similarity(g_ce, g_kd, dim=1, eps=1e-12)
    return alignment, (alignment >= float(threshold)).detach()


@dataclass
class SelectiveKDResult:
    loss: Tensor
    old_kd: Tensor
    new_kd: Tensor
    new_keep_ratio: Tensor
    alignment_mean: Tensor
    alignment_positive_ratio: Tensor
    old_count: int
    new_count: int

    def detached_metrics(self) -> dict[str, float | int]:
        return {
            "old_kd": float(self.old_kd.detach()),
            "new_kd": float(self.new_kd.detach()),
            "new_keep_ratio": float(self.new_keep_ratio.detach()),
            "alignment_mean": float(self.alignment_mean.detach()),
            "alignment_positive_ratio": float(
                self.alignment_positive_ratio.detach()
            ),
            "old_count": int(self.old_count),
            "new_count": int(self.new_count),
        }


def selective_pycil_icarl_kd_loss(
    all_logits: Tensor,
    reference_old_logits: Tensor,
    targets: Tensor,
    replay_mask: Tensor,
    classifier_weights: Tensor,
    *,
    temperature: float = 2.0,
    alignment_threshold: float = 0.0,
) -> SelectiveKDResult:
    """SRIL-style new-row routing with analytic embedding gradients.

    The routing decision is fully detached. Replay rows always retain the
    original PyCIL KD term. Retained per-row losses are summed and divided by
    the original batch size, so dropping new rows reduces total KD force.
    """

    if all_logits.ndim != 2 or all_logits.shape[0] == 0:
        raise ValueError("all_logits must be a non-empty matrix")
    batch_size, total_classes = all_logits.shape
    if reference_old_logits.ndim != 2 or reference_old_logits.shape[0] != batch_size:
        raise ValueError("reference old logits have the wrong batch size")
    known_classes = reference_old_logits.shape[1]
    if known_classes <= 0 or known_classes >= total_classes:
        raise ValueError("selective KD requires old and new classes")
    if targets.shape != (batch_size,):
        raise ValueError("targets must be a batch vector")
    if replay_mask.shape != (batch_size,) or replay_mask.dtype != torch.bool:
        raise ValueError("replay_mask must be a boolean batch vector")
    if classifier_weights.shape[0] != total_classes or classifier_weights.ndim != 2:
        raise ValueError("classifier weights do not match all logits")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(torch.tensor(float(alignment_threshold))):
        raise ValueError("alignment threshold must be finite")

    current_old_logits = all_logits[:, :known_classes]
    new_mask = ~replay_mask
    old_count = int(replay_mask.sum())
    new_count = int(new_mask.sum())
    zero = all_logits.sum() * 0.0
    old_kd = (
        pycil_icarl_kd_loss(
            current_old_logits[replay_mask],
            reference_old_logits[replay_mask],
            temperature=temperature,
        )
        if old_count
        else zero
    )

    if new_count == 0:
        loss = pycil_icarl_kd_loss(
            current_old_logits,
            reference_old_logits,
            temperature=temperature,
        )
        return SelectiveKDResult(
            loss=loss,
            old_kd=old_kd,
            new_kd=zero,
            new_keep_ratio=zero,
            alignment_mean=zero,
            alignment_positive_ratio=zero,
            old_count=old_count,
            new_count=0,
        )

    alignment, keep = analytic_embedding_gradient_alignment(
        all_logits[new_mask],
        reference_old_logits[new_mask],
        targets[new_mask],
        classifier_weights,
        temperature=temperature,
        threshold=alignment_threshold,
    )

    student_log = F.log_softmax(current_old_logits / temperature, dim=1)
    teacher_prob = F.softmax(
        reference_old_logits.detach() / temperature, dim=1
    )
    per_sample = -(teacher_prob * student_log).sum(dim=1)
    new_per_sample = per_sample[new_mask]
    keep_float = keep.to(new_per_sample)
    new_kd = (new_per_sample * keep_float).sum() / new_count

    if bool(keep.all()):
        # Exact numerical/gradient parity when routing retains every new row.
        loss = pycil_icarl_kd_loss(
            current_old_logits,
            reference_old_logits,
            temperature=temperature,
        )
    else:
        loss = (
            per_sample[replay_mask].sum()
            + (new_per_sample * keep_float).sum()
        ) / batch_size

    return SelectiveKDResult(
        loss=loss,
        old_kd=old_kd,
        new_kd=new_kd,
        new_keep_ratio=keep_float.mean(),
        alignment_mean=alignment.mean(),
        alignment_positive_ratio=(alignment >= 0.0).float().mean(),
        old_count=old_count,
        new_count=new_count,
    )
