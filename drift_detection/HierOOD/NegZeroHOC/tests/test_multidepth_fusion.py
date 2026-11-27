from __future__ import annotations

from types import SimpleNamespace

import torch

from ProHOC.libs.utils.score_util import compprob
from negzerohoc.multidepth_fusion import (
    fuse_multidepth_probabilities,
    get_multidepth_classes,
    multidepth_route_conditionals,
    multidepth_targets,
    multidepth_unknown_probabilities,
    padded_multidepth_path,
)


def toy_hierarchy():
    hierarchy = SimpleNamespace(
        id_node_list=["root", "m1", "m2", "f1", "a", "b", "c"],
        parent2children={
            "root": ["m1", "m2"],
            "m1": ["f1", "b"],
            "m2": ["c"],
            "f1": ["a"],
        },
        node_ancestors={
            "root": [],
            "m1": [0],
            "m2": [0],
            "f1": [0, 1],
            "a": [0, 1, 3],
            "b": [0, 1],
            "c": [0, 2],
        },
        max_depth=3,
    )
    hierarchy.gen_ds2node_map = lambda classes: torch.tensor(
        [hierarchy.id_node_list.index(value) for value in classes]
    )
    return hierarchy


def test_padded_paths_repeat_shallow_leaf():
    hierarchy = toy_hierarchy()
    assert padded_multidepth_path(hierarchy, "a") == ["m1", "f1", "a"]
    assert padded_multidepth_path(hierarchy, "b") == ["m1", "b", "b"]


def test_multidepth_targets_follow_padded_paths():
    hierarchy = toy_hierarchy()
    classes = ["a", "b", "c"]
    multidepth = get_multidepth_classes(hierarchy, classes)
    targets = multidepth_targets(
        hierarchy, classes, torch.tensor([0, 1, 2]), multidepth
    )
    recovered = [
        [multidepth[d][int(targets[d][i])] for d in range(3)]
        for i in range(3)
    ]
    assert recovered == [
        ["m1", "f1", "a"],
        ["m1", "b", "b"],
        ["m2", "c", "c"],
    ]


def test_compprob_fusion_is_normalized():
    hierarchy = toy_hierarchy()
    classes = ["a", "b", "c"]
    multidepth = get_multidepth_classes(hierarchy, classes)
    probabilities = [
        torch.tensor([[0.7, 0.3]]),
        torch.tensor([[0.5, 0.3, 0.2]]),
        torch.tensor([[0.6, 0.25, 0.15]]),
    ]
    fused = fuse_multidepth_probabilities(
        probabilities, hierarchy, multidepth, compprob
    )
    assert torch.allclose(fused.sum(dim=1), torch.ones(1), atol=1e-5)
    assert float(fused[0, hierarchy.id_node_list.index("root")]) == 0.0


def test_global_depth_probabilities_become_local_routes():
    hierarchy = toy_hierarchy()
    classes = ["a", "b", "c"]
    multidepth = get_multidepth_classes(hierarchy, classes)
    probabilities = [
        torch.tensor([[0.8, 0.2]]),
        torch.tensor([[0.2, 0.3, 0.5]]),
        torch.tensor([[0.6, 0.25, 0.15]]),
    ]
    routes = multidepth_route_conditionals(
        probabilities, hierarchy, multidepth
    )
    assert torch.allclose(routes["root"].sum(dim=1), torch.ones(1))
    assert torch.allclose(routes["m1"].sum(dim=1), torch.ones(1))
    assert torch.allclose(routes["f1"], torch.ones(1, 1))


def test_multidepth_unknown_probabilities_cover_nonroot_parents():
    hierarchy = toy_hierarchy()
    multidepth = get_multidepth_classes(hierarchy, ["a", "b", "c"])
    probabilities = [
        torch.tensor([[0.8, 0.2]]),
        torch.tensor([[0.2, 0.3, 0.5]]),
        torch.tensor([[0.6, 0.25, 0.15]]),
    ]
    unknown = multidepth_unknown_probabilities(
        probabilities, hierarchy, multidepth, compprob
    )
    assert set(unknown) == {"m1", "m2", "f1"}
    assert unknown["m1"].shape == (1,)
    assert float(unknown["m1"][0]) > 0.0
    assert torch.equal(unknown["m2"], torch.zeros(1))
    assert torch.equal(unknown["f1"], torch.zeros(1))
