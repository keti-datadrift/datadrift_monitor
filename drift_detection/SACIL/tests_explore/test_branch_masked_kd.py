from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import (
    base_recipe_signature,
    resolve_branch_masked_kd_options,
    resolve_edge_topology_options,
)
from sacil.methods import (
    BranchMaskedKDReference,
    branch_masked_pycil_icarl_kd_loss,
    pycil_icarl_kd_loss,
)
from sacil.provenance import build_exploration_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_vmin_one_is_exact_pycil_loss_and_gradient() -> None:
    torch.manual_seed(3)
    teacher = torch.randn(6, 5)
    replay = torch.tensor([True, False, True, False, True, False])
    new_indices = torch.tensor([0, 1, 0])
    branch_mask = torch.tensor(
        [[True, True, False, False, False], [False, False, True, True, False]]
    )
    standard_logits = torch.randn(6, 5, requires_grad=True)
    candidate_logits = standard_logits.detach().clone().requires_grad_()

    standard = pycil_icarl_kd_loss(
        standard_logits, teacher, temperature=2.0
    )
    candidate = branch_masked_pycil_icarl_kd_loss(
        candidate_logits,
        teacher,
        replay,
        new_indices,
        branch_mask,
        temperature=2.0,
        v_min=1.0,
    ).loss
    standard.backward()
    candidate.backward()

    assert torch.equal(candidate, standard)
    assert torch.equal(candidate_logits.grad, standard_logits.grad)


def test_mixed_batch_keeps_old_rows_exact_and_masks_only_new_rows() -> None:
    torch.manual_seed(7)
    student = torch.randn(5, 4, requires_grad=True)
    teacher = torch.randn(5, 4)
    replay = torch.tensor([True, False, True, False, False])
    new_indices = torch.tensor([0, 1, 0])
    branch_mask = torch.tensor(
        [[True, True, False, False], [False, False, True, False]]
    )
    result = branch_masked_pycil_icarl_kd_loss(
        student,
        teacher,
        replay,
        new_indices,
        branch_mask,
        temperature=2.0,
        v_min=0.25,
    )

    exact_old = pycil_icarl_kd_loss(
        student[replay], teacher[replay], temperature=2.0
    )
    torch.testing.assert_close(result.old_kd, exact_old, rtol=0, atol=0)

    selected = branch_mask[new_indices]
    multiplier = torch.where(selected, 0.25, 1.0).to(student)
    manual_teacher = F.softmax(
        teacher[~replay] / 2.0 + multiplier.log(), dim=1
    )
    manual_student_log = F.log_softmax(
        student[~replay] / 2.0 + multiplier.log(), dim=1
    )
    manual_new = -(manual_teacher * manual_student_log).sum(dim=1).mean()
    torch.testing.assert_close(result.new_kd, manual_new)
    assert 0.0 < float(result.teacher_retained_mass) < 1.0
    assert 0.0 < float(result.student_retained_mass) < 1.0
    assert float(result.masked_class_ratio) == pytest.approx(5 / 12)
    result.loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_all_replay_batch_is_exact_pycil_and_handles_no_new_rows() -> None:
    torch.manual_seed(13)
    student = torch.randn(4, 3, requires_grad=True)
    teacher = torch.randn(4, 3)
    result = branch_masked_pycil_icarl_kd_loss(
        student,
        teacher,
        torch.ones(4, dtype=torch.bool),
        torch.empty(0, dtype=torch.long),
        torch.tensor([[True, False, False]]),
        temperature=2.0,
        v_min=0.25,
    )
    exact = pycil_icarl_kd_loss(student, teacher, temperature=2.0)
    torch.testing.assert_close(result.loss, exact, rtol=0, atol=0)
    assert result.old_count == 4 and result.new_count == 0
    assert float(result.teacher_retained_mass) == 0.0
    assert float(result.student_retained_mass) == 0.0


def test_branch_mapping_checkpoint_round_trip() -> None:
    reference = BranchMaskedKDReference(
        session_id=1,
        known_classes=4,
        new_incremental_labels=(4, 5),
        new_original_class_ids=(10, 20),
        branch_node_ids=("node:1", "node:2"),
        branch_class_mask=torch.tensor(
            [[True, True, False, False], [False, False, True, True]]
        ),
        teacher_tree_state={"root_id": "node:3", "nodes": []},
    )
    restored = BranchMaskedKDReference.from_state_dict(reference.state_dict())
    assert restored.branch_node_ids == reference.branch_node_ids
    torch.testing.assert_close(
        restored.branch_class_mask, reference.branch_class_mask
    )
    assert restored.teacher_tree_state == reference.teacher_tree_state


@pytest.mark.parametrize(
    ("filename", "v_min"),
    [
        ("icarl_htpl_branch_kd_v025.yaml", 0.25),
        ("icarl_htpl_branch_kd_v050.yaml", 0.5),
    ],
)
def test_i2_configs_preserve_icarl_base_signature(
    filename: str, v_min: float
) -> None:
    control = load_config_tree(
        PROJECT_ROOT
        / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    candidate = load_config_tree(
        PROJECT_ROOT / "configs/explore/cifar100" / filename
    )
    branch = resolve_branch_masked_kd_options("icarl", candidate["method"])
    edge = resolve_edge_topology_options("icarl", candidate["method"])
    assert branch == {
        "enabled": True,
        "v_min": v_min,
        "top_internal_branches": 1,
    }
    assert edge["enabled"] and edge["edge_weighting"] == "global"
    assert edge["representatives_per_class"] == 2
    assert edge["lambda_edge"] == 5.0
    assert candidate["output"]["directory"] == (
        "outputs/explore/branch_masked_kd"
    )
    assert base_recipe_signature(candidate) == base_recipe_signature(control)


def test_python_source_provenance_is_deterministic_and_diff_aware(
    tmp_path: Path,
) -> None:
    source = tmp_path / "explore"
    preserved = tmp_path / "preserved"
    source.mkdir()
    preserved.mkdir()
    (source / "same.py").write_text("x = 1\n", encoding="utf-8")
    (preserved / "same.py").write_text("x = 1\n", encoding="utf-8")
    (source / "modified.py").write_text("x = 2\n", encoding="utf-8")
    (preserved / "modified.py").write_text("x = 1\n", encoding="utf-8")
    (source / "added.py").write_text("x = 3\n", encoding="utf-8")
    (preserved / "deleted.py").write_text("x = 4\n", encoding="utf-8")

    first = build_exploration_provenance(source, preserved)
    second = build_exploration_provenance(source, preserved)
    assert first == second
    assert len(first["source_digest"]) == 64
    assert len(first["preserved_src_digest"]) == 64
    statuses = {item["path"]: item["status"] for item in first["changed_files"]}
    assert statuses == {
        "added.py": "added",
        "deleted.py": "deleted",
        "modified.py": "modified",
    }
