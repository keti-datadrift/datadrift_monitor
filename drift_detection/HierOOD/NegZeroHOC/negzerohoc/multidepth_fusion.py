from __future__ import annotations

import torch


def _node_path_without_root(hierarchy, node: str) -> list[str]:
    path = [
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.node_ancestors.get(node, [])
    ] + [node]
    return [value for value in path if value != "root"]


def padded_multidepth_path(hierarchy, leaf: str) -> list[str]:
    path = _node_path_without_root(hierarchy, leaf)
    if not path:
        raise ValueError(f"Leaf {leaf!r} has no non-root path")
    if len(path) > hierarchy.max_depth:
        raise ValueError(f"Leaf {leaf!r} exceeds hierarchy.max_depth")
    return path + [leaf] * (hierarchy.max_depth - len(path))


def get_multidepth_classes(hierarchy, leaf_classes: list[str]) -> list[list[str]]:
    by_depth = [set() for _ in range(hierarchy.max_depth)]
    for leaf in leaf_classes:
        for depth, node in enumerate(padded_multidepth_path(hierarchy, leaf)):
            by_depth[depth].add(node)
    return [sorted(nodes) for nodes in by_depth]


def multidepth_targets(
    hierarchy,
    dataset_classes: list[str],
    dataset_targets: torch.Tensor,
    multidepth_classes: list[list[str]],
) -> list[torch.Tensor]:
    dataset_targets = dataset_targets.detach().long().cpu()
    class_nodes = [
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.gen_ds2node_map(dataset_classes).tolist()
    ]
    maps = [
        {node: index for index, node in enumerate(nodes)}
        for nodes in multidepth_classes
    ]
    result = []
    for depth in range(hierarchy.max_depth):
        class_depth_targets = [
            maps[depth][padded_multidepth_path(hierarchy, leaf)[depth]]
            for leaf in class_nodes
        ]
        lookup = torch.tensor(class_depth_targets, dtype=torch.long)
        result.append(lookup[dataset_targets])
    return result


def children_maps_and_group_sizes(
    hierarchy,
    multidepth_classes: list[list[str]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    children_maps = []
    group_sizes = []
    for depth in range(len(multidepth_classes) - 1):
        parents = multidepth_classes[depth]
        children = multidepth_classes[depth + 1]
        parent_to_index = {node: index for index, node in enumerate(parents)}
        child_map = []
        for child in children:
            child_path = padded_multidepth_path(hierarchy, child)
            parent_node = child_path[min(depth, len(child_path) - 1)]
            if parent_node not in parent_to_index:
                raise RuntimeError(
                    f"Cannot map depth-{depth + 1} child {child!r} "
                    f"to depth-{depth} parent"
                )
            child_map.append(parent_to_index[parent_node])
        child_map_tensor = torch.tensor(child_map, dtype=torch.long)
        sizes = torch.bincount(
            child_map_tensor, minlength=len(parents)
        ).float()
        children_maps.append(child_map_tensor)
        group_sizes.append(sizes)
    return children_maps, group_sizes


def multidepth_route_conditionals(
    probabilities_by_depth: list[torch.Tensor],
    hierarchy,
    multidepth_classes: list[list[str]],
) -> dict[str, torch.Tensor]:
    """Convert global depth heads into normalized parent-local routes."""
    if len(probabilities_by_depth) != len(multidepth_classes):
        raise ValueError("One probability tensor is required per depth")
    class_maps = [
        {node: index for index, node in enumerate(nodes)}
        for nodes in multidepth_classes
    ]
    routes = {}
    for parent, children in hierarchy.parent2children.items():
        if parent == "root":
            child_depth = 0
        else:
            child_depth = len(hierarchy.node_ancestors.get(parent, []))
        if child_depth >= len(probabilities_by_depth):
            raise RuntimeError(
                f"Parent {parent!r} has no matching multi-depth child head"
            )
        missing = [
            child for child in children
            if child not in class_maps[child_depth]
        ]
        if missing:
            raise RuntimeError(
                f"Depth-{child_depth} head misses children of {parent!r}: "
                f"{missing[:3]}"
            )
        indices = torch.tensor(
            [class_maps[child_depth][child] for child in children],
            dtype=torch.long,
        )
        local = probabilities_by_depth[child_depth].float().cpu().index_select(
            1, indices
        )
        routes[parent] = local / local.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
    return routes


def multidepth_unknown_probabilities(
    probabilities_by_depth: list[torch.Tensor],
    hierarchy,
    multidepth_classes: list[list[str]],
    uncertainty_method,
) -> dict[str, torch.Tensor]:
    """Extract non-root parent unknown probabilities from ProHOC fusion."""
    if len(probabilities_by_depth) != len(multidepth_classes):
        raise ValueError("One probability tensor is required per depth")
    probabilities = [
        value.detach().float().cpu()
        for value in probabilities_by_depth
    ]
    sample_count = int(probabilities[0].shape[0])
    children_maps, group_sizes = children_maps_and_group_sizes(
        hierarchy, multidepth_classes
    )
    result = {}
    for depth, (child_map, sizes) in enumerate(
        zip(children_maps, group_sizes)
    ):
        _, p_unknown = uncertainty_method(
            probabilities[depth + 1],
            child_map,
            sizes,
            sample_count,
            len(multidepth_classes[depth]),
            device="cpu",
        )
        single_parent = sizes == 1
        if bool(single_parent.any()):
            p_unknown[:, single_parent] = 0.0
        for parent_index, parent in enumerate(
            multidepth_classes[depth]
        ):
            if parent == "root" or parent not in hierarchy.parent2children:
                continue
            result[parent] = p_unknown[:, parent_index].cpu()
    expected = {
        parent
        for parent in hierarchy.parent2children
        if parent != "root"
    }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(
            "Multi-depth unknown probabilities do not cover hierarchy: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return result


def fuse_multidepth_probabilities(
    probabilities_by_depth: list[torch.Tensor],
    hierarchy,
    multidepth_classes: list[list[str]],
    uncertainty_method,
) -> torch.Tensor:
    """ProHOC-style fusion into a normalized distribution over tree nodes."""
    if len(probabilities_by_depth) != len(multidepth_classes):
        raise ValueError("One probability tensor is required per depth")
    sample_count = int(probabilities_by_depth[0].shape[0])
    probabilities = [
        value.detach().float().cpu().clone()
        for value in probabilities_by_depth
    ]
    children_maps, group_sizes = children_maps_and_group_sizes(
        hierarchy, multidepth_classes
    )
    local_probabilities = [probabilities[0]]
    unknown_probabilities = []
    for depth, (child_map, sizes) in enumerate(
        zip(children_maps, group_sizes)
    ):
        child_probability = probabilities[depth + 1]
        result, p_unknown = uncertainty_method(
            child_probability,
            child_map,
            sizes,
            sample_count,
            len(multidepth_classes[depth]),
            device="cpu",
        )
        single_parent = sizes == 1
        if bool(single_parent.any()):
            result[:, single_parent[child_map]] = 1.0
            p_unknown[:, single_parent] = 0.0
        local_probabilities.append(result)
        unknown_probabilities.append(p_unknown)

    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    fused = torch.zeros(
        sample_count, len(hierarchy.id_node_list), dtype=torch.float32
    )
    finest_classes = multidepth_classes[-1]
    finest_paths = [
        padded_multidepth_path(hierarchy, leaf)
        for leaf in finest_classes
    ]
    depth_maps = [
        {node: index for index, node in enumerate(nodes)}
        for nodes in multidepth_classes
    ]

    for leaf, path in zip(finest_classes, finest_paths):
        reach = torch.ones(sample_count, dtype=torch.float32)
        for depth, node in enumerate(path):
            reach = reach * local_probabilities[depth][
                :, depth_maps[depth][node]
            ]
        fused[:, node_to_index[leaf]] += reach

    for depth, parents in enumerate(multidepth_classes[:-1]):
        for parent_index, parent in enumerate(parents):
            if parent not in hierarchy.parent2children:
                continue
            reach = torch.ones(sample_count, dtype=torch.float32)
            parent_path = padded_multidepth_path(hierarchy, parent)
            for prefix_depth in range(depth + 1):
                node = parent_path[prefix_depth]
                reach = reach * local_probabilities[prefix_depth][
                    :, depth_maps[prefix_depth][node]
                ]
            fused[:, node_to_index[parent]] += (
                reach * unknown_probabilities[depth][:, parent_index]
            )

    sums = fused.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
        raise RuntimeError(
            "Multi-depth fusion is not normalized: "
            f"range=({float(sums.min())}, {float(sums.max())})"
        )
    return fused
