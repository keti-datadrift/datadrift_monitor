from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch.nn import functional as F

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank
from sacil.hierarchy import GriffinPeronaGreedy
from sacil.methods.sacil_v1 import (
    SACILV1PathLoss,
    compute_internal_node_radii,
    insert_planned_classes,
    plan_conflict_relaxation,
)
from sacil.engine.table1_trainer import base_recipe_signature
from sacil.engine.checkpoint import restore_rng_state


def _state():
    class_ids = (11, 3, 19, 7)
    prototypes = F.normalize(torch.eye(4), dim=1)
    affinity = torch.tensor(
        [
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.8],
            [0.1, 0.1, 0.8, 0.0],
        ]
    )
    tree = GriffinPeronaGreedy().build(class_ids, affinity)
    prototype_bank = PrototypeBank(class_ids, prototypes)
    anchors = HierarchicalAnchorBank.from_tree(prototype_bank, tree)
    features = prototypes.repeat_interleave(3, dim=0)
    targets = torch.tensor(class_ids).repeat_interleave(3)
    radii = compute_internal_node_radii(
        features, targets, anchors, tree, quantile=0.9
    )
    return class_ids, prototypes, tree, anchors, features, targets, radii


def test_conflict_plan_is_radius_gated_and_globally_bounded():
    _, prototypes, tree, anchors, _, _, radii = _state()
    incoming_ids = (91, 42)
    incoming = F.normalize(
        torch.stack((prototypes[0] + 0.02 * prototypes[1], prototypes[2])),
        dim=1,
    )
    plan = plan_conflict_relaxation(
        incoming,
        incoming_ids,
        anchors,
        tree,
        radii,
        radius_slack=0.1,
        max_conflict_leaf_ratio=0.5,
        relaxation_margin=0.1,
    )
    assert tuple(item.new_class_id for item in plan.assignments) == incoming_ids
    assert len(plan.relaxed_old_class_ids) <= plan.old_class_budget == 2
    for assignment in plan.assignments:
        if assignment.conflict_root_node_id is not None:
            assert assignment.overlap_score >= 0.0
            assert set(assignment.relaxed_old_class_ids) <= set(
                plan.relaxed_old_class_ids
            )


def test_local_margin_relaxes_only_ancestor_path_relations():
    class_ids, prototypes, tree, anchors, _, _, radii = _state()
    incoming = F.normalize((prototypes[0] + 0.01 * prototypes[1]).unsqueeze(0), dim=1)
    local = plan_conflict_relaxation(
        incoming,
        (91,),
        anchors,
        tree,
        radii,
        radius_slack=0.1,
        max_conflict_leaf_ratio=0.5,
        relaxation_margin=0.2,
    )
    strict = copy.deepcopy(local)
    strict.relaxed_margins = {}
    local_loss = SACILV1PathLoss(anchors, tree, local)
    strict_loss = SACILV1PathLoss(anchors, tree, strict)
    reference = prototypes.clone()
    current = F.normalize(reference + 0.1 * torch.roll(reference, 1, 1), dim=1)
    current.requires_grad_(True)
    local_value = local_loss(current, reference, torch.tensor(class_ids))
    strict_value = strict_loss(current, reference, torch.tensor(class_ids))
    assert local_value <= strict_value
    assert 0 < local_loss.diagnostics()["relaxed_relation_count"]
    local_value.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_online_insertion_preserves_protocol_order_and_old_nodes():
    _, prototypes, tree, anchors, _, _, radii = _state()
    old_nodes = {
        node_id: (node.left, node.right)
        for node_id, node in tree.nodes.items()
        if node.is_leaf
    }
    incoming_ids = (91, 42)
    plan = plan_conflict_relaxation(
        prototypes[:2],
        incoming_ids,
        anchors,
        tree,
        radii,
        radius_slack=0.1,
        max_conflict_leaf_ratio=0.5,
    )
    updated, insertions = insert_planned_classes(tree, plan)
    assert updated.class_order == (*tree.class_order, *incoming_ids)
    assert len(insertions) == len(incoming_ids)
    for node_id, children in old_nodes.items():
        assert (updated.nodes[node_id].left, updated.nodes[node_id].right) == children


def test_sacil_v1_ablation_fields_do_not_change_base_recipe():
    base = {
        "seed": 1,
        "data": {"name": "cifar100"},
        "model": {"backbone": "resnet32"},
        "memory": {"exemplars_per_class": 20},
        "evaluation": {"classifier": "nme"},
        "training": {"base": {"epochs": 2}, "incremental": {"epochs": 1}},
        "method": {
            "name": "icarl",
            "geometry_mode": "sacil",
            "lambda_geo": 1.0,
            "sacil_v1": {"relaxation_mode": "local_margin"},
        },
    }
    changed = copy.deepcopy(base)
    changed["method"]["lambda_geo"] = 64.0
    changed["method"]["sacil_v1"] = {
        "relaxation_mode": "none",
        "hierarchy_source": "random",
    }
    assert base_recipe_signature(base) == base_recipe_signature(changed)


def test_cuda_rng_state_is_transplanted_from_source_to_target(monkeypatch):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [torch.tensor([10], dtype=torch.uint8), torch.tensor([20], dtype=torch.uint8)],
    }
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: restored.append((value.clone(), device)),
    )
    restore_rng_state(
        state,
        source_cuda_device="cuda:1",
        target_cuda_device="cuda:0",
    )
    assert len(restored) == 1
    assert restored[0][1] == 0
    assert torch.equal(restored[0][0], state["cuda"][1])
