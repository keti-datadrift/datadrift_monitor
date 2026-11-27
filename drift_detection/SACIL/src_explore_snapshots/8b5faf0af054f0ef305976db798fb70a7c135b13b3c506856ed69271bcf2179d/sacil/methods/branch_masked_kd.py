from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .icarl import pycil_icarl_kd_loss


@dataclass
class BranchMaskedKDResult:
    loss: Tensor
    old_kd: Tensor
    new_kd: Tensor
    teacher_retained_mass: Tensor
    student_retained_mass: Tensor
    masked_class_ratio: Tensor
    old_count: int
    new_count: int

    def detached_metrics(self) -> dict[str, float | int]:
        return {
            "old_kd": float(self.old_kd.detach()),
            "new_kd": float(self.new_kd.detach()),
            "teacher_retained_mass": float(
                self.teacher_retained_mass.detach()
            ),
            "student_retained_mass": float(
                self.student_retained_mass.detach()
            ),
            "masked_class_ratio": float(self.masked_class_ratio.detach()),
            "old_count": int(self.old_count),
            "new_count": int(self.new_count),
        }


def branch_masked_pycil_icarl_kd_loss(
    current_old_logits: Tensor,
    reference_logits: Tensor,
    replay_mask: Tensor,
    new_class_indices: Tensor,
    branch_class_mask: Tensor,
    *,
    temperature: float = 2.0,
    v_min: float = 0.25,
) -> BranchMaskedKDResult:
    """PyCIL iCaRL KD with branch masking on new samples only.

    Replay rows use the exact unmodified PyCIL distribution. For a new row,
    probabilities belonging to its fixed conflict branch are multiplied by
    ``v_min`` on both teacher and student sides and each distribution is then
    renormalized. No ``T**2`` multiplier is introduced.
    """

    if current_old_logits.shape != reference_logits.shape:
        raise ValueError("current and reference old logits must have equal shape")
    if current_old_logits.ndim != 2 or current_old_logits.shape[1] == 0:
        raise ValueError("old logits must be a non-empty matrix")
    batch_size, known_classes = current_old_logits.shape
    if replay_mask.shape != (batch_size,) or replay_mask.dtype != torch.bool:
        raise ValueError("replay_mask must be a boolean batch vector")
    new_mask = ~replay_mask
    new_count = int(new_mask.sum())
    old_count = int(replay_mask.sum())
    if new_class_indices.shape != (new_count,):
        raise ValueError("new_class_indices must contain one entry per new row")
    if branch_class_mask.ndim != 2 or branch_class_mask.shape[1] != known_classes:
        raise ValueError("branch_class_mask has the wrong old-class dimension")
    if branch_class_mask.dtype != torch.bool:
        raise ValueError("branch_class_mask must be boolean")
    if new_count and (
        int(new_class_indices.min()) < 0
        or int(new_class_indices.max()) >= branch_class_mask.shape[0]
    ):
        raise ValueError("new class index is outside branch mapping")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    value = float(v_min)
    if not 0.0 < value <= 1.0:
        raise ValueError("v_min must be in (0, 1]")

    student_log = F.log_softmax(current_old_logits / temperature, dim=1)
    teacher_prob = F.softmax(
        reference_logits.detach() / temperature, dim=1
    )
    per_sample = -(teacher_prob * student_log).sum(dim=1)
    zero = current_old_logits.sum() * 0.0
    old_kd = (
        pycil_icarl_kd_loss(
            current_old_logits[replay_mask],
            reference_logits[replay_mask],
            temperature=temperature,
        )
        if old_count
        else zero
    )

    selected_mask = branch_class_mask.to(current_old_logits.device)[
        new_class_indices.long()
    ]
    masked_ratio = (
        selected_mask.float().mean()
        if new_count
        else current_old_logits.new_zeros(())
    )

    if new_count == 0:
        loss = pycil_icarl_kd_loss(
            current_old_logits,
            reference_logits,
            temperature=temperature,
        )
        return BranchMaskedKDResult(
            loss=loss,
            old_kd=old_kd,
            new_kd=zero,
            teacher_retained_mass=zero,
            student_retained_mass=zero,
            masked_class_ratio=masked_ratio,
            old_count=old_count,
            new_count=0,
        )

    # This direct call is intentional: v_min=1 must be numerically and
    # gradient-identical to the existing PyCIL implementation.
    if value == 1.0:
        loss = pycil_icarl_kd_loss(
            current_old_logits,
            reference_logits,
            temperature=temperature,
        )
        new_kd = per_sample[new_mask].mean() if new_count else zero
        one = current_old_logits.new_ones(())
        return BranchMaskedKDResult(
            loss=loss,
            old_kd=old_kd,
            new_kd=new_kd,
            teacher_retained_mass=one if new_count else zero,
            student_retained_mass=one if new_count else zero,
            masked_class_ratio=masked_ratio,
            old_count=old_count,
            new_count=new_count,
        )

    if new_count:
        multiplier = torch.where(
            selected_mask,
            current_old_logits.new_full(selected_mask.shape, value),
            current_old_logits.new_ones(selected_mask.shape),
        )
        log_multiplier = multiplier.log()
        new_student_logits = (
            current_old_logits[new_mask] / temperature + log_multiplier
        )
        new_teacher_logits = (
            reference_logits.detach()[new_mask] / temperature + log_multiplier
        )
        new_student_log = F.log_softmax(new_student_logits, dim=1)
        new_teacher_prob = F.softmax(new_teacher_logits, dim=1)
        new_per_sample = -(new_teacher_prob * new_student_log).sum(dim=1)
        new_kd = new_per_sample.mean()
        student_prob = student_log[new_mask].exp()
        teacher_retained = (teacher_prob[new_mask] * multiplier).sum(dim=1)
        student_retained = (student_prob * multiplier).sum(dim=1)
        teacher_retained_mass = teacher_retained.mean()
        student_retained_mass = student_retained.mean()
        loss = (
            per_sample[replay_mask].sum() + new_per_sample.sum()
        ) / batch_size
    else:
        new_kd = zero
        teacher_retained_mass = zero
        student_retained_mass = zero
        loss = per_sample.sum() / batch_size

    return BranchMaskedKDResult(
        loss=loss,
        old_kd=old_kd,
        new_kd=new_kd,
        teacher_retained_mass=teacher_retained_mass,
        student_retained_mass=student_retained_mass,
        masked_class_ratio=masked_ratio,
        old_count=old_count,
        new_count=new_count,
    )


@dataclass
class BranchMaskedKDReference:
    session_id: int
    known_classes: int
    new_incremental_labels: tuple[int, ...]
    new_original_class_ids: tuple[int, ...]
    branch_node_ids: tuple[str, ...]
    branch_class_mask: Tensor
    teacher_tree_state: dict

    def __post_init__(self) -> None:
        new_count = len(self.new_incremental_labels)
        if self.session_id <= 0 or self.known_classes <= 0:
            raise ValueError("branch KD reference requires an incremental session")
        if not (
            len(self.new_original_class_ids)
            == len(self.branch_node_ids)
            == new_count
        ):
            raise ValueError("branch mapping metadata lengths differ")
        if self.branch_class_mask.shape != (new_count, self.known_classes):
            raise ValueError("branch mask shape does not match mapping")
        if self.branch_class_mask.dtype != torch.bool:
            raise ValueError("branch mask must be boolean")
        if new_count and not bool(self.branch_class_mask.any(dim=1).all()):
            raise ValueError("every new class must map to a non-empty branch")

    def state_dict(self) -> dict:
        return {
            "session_id": int(self.session_id),
            "known_classes": int(self.known_classes),
            "new_incremental_labels": list(self.new_incremental_labels),
            "new_original_class_ids": list(self.new_original_class_ids),
            "branch_node_ids": list(self.branch_node_ids),
            "branch_class_mask": self.branch_class_mask.detach().cpu().clone(),
            "teacher_tree_state": copy.deepcopy(self.teacher_tree_state),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "BranchMaskedKDReference":
        return cls(
            session_id=int(state["session_id"]),
            known_classes=int(state["known_classes"]),
            new_incremental_labels=tuple(
                int(value) for value in state["new_incremental_labels"]
            ),
            new_original_class_ids=tuple(
                int(value) for value in state["new_original_class_ids"]
            ),
            branch_node_ids=tuple(str(value) for value in state["branch_node_ids"]),
            branch_class_mask=state["branch_class_mask"].bool(),
            teacher_tree_state=copy.deepcopy(state["teacher_tree_state"]),
        )
