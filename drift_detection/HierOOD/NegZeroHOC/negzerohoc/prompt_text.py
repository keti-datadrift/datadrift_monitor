from __future__ import annotations

from .prompts import (
    canonicalize_node_name,
    clean_node_name,
    display_node_name,
    infer_node_role,
    node_path_names,
)


TEXT_TEMPLATES_VERSION = "idea3_path_aware_v1"
UNKNOWN_TEXT_VARIANTS = {
    "generic",
    "parent_conditioned",
    "unseen_child",
    "literal_not",
}


def node_depth(hierarchy, node: str) -> int:
    return len(hierarchy.node_ancestors.get(node, []))


def _path_display(hierarchy, dataset_name: str, node: str, include_self: bool = True) -> list[str]:
    return node_path_names(hierarchy, node, include_self=include_self, dataset_name=dataset_name)


def _path_canonical(hierarchy, dataset_name: str, node: str) -> str:
    path = _path_display(hierarchy, dataset_name, node, include_self=True)
    role = infer_node_role(dataset_name, node, node_depth(hierarchy, node))
    return canonicalize_node_name(dataset_name, node, path, role)


def _fgvc_edge_text(hierarchy, parent: str, child: str) -> str:
    child_role = infer_node_role("fgvc-aircraft", child)
    child_name = _path_canonical(hierarchy, "fgvc-aircraft", child)
    parent_name = _path_canonical(hierarchy, "fgvc-aircraft", parent)

    if child_role == "manufacturer":
        return f"a photo of an aircraft manufactured by {child_name}"
    if child_role == "family":
        return f"a photo of a {child_name} aircraft family under manufacturer {parent_name}"
    if child_role == "model":
        return f"a photo of a {child_name} aircraft model under the {parent_name} family"
    return f"a photo of {child_name} aircraft under {parent_name}"


def _generic_edge_text(dataset_name: str, hierarchy, parent: str, child: str) -> str:
    child_name = display_node_name(hierarchy, dataset_name, child)
    parent_name = display_node_name(hierarchy, dataset_name, parent)
    child_name = clean_node_name(child_name)
    parent_name = clean_node_name(parent_name)

    if parent == "root":
        return f"a photo of {child_name}, a visual category"
    return f"a photo of {child_name}, a visual category under {parent_name}"


def build_edge_text(dataset_name: str, hierarchy, parent: str, child: str) -> str:
    dataset_key = dataset_name.lower()
    if dataset_key == "fgvc-aircraft":
        return _fgvc_edge_text(hierarchy, parent, child)
    return _generic_edge_text(dataset_name, hierarchy, parent, child)


def build_parent_text(dataset_name: str, hierarchy, parent: str) -> str:
    dataset_key = dataset_name.lower()
    if parent == "root":
        if dataset_key == "fgvc-aircraft":
            return "a photo of an aircraft"
        return "a photo of a visual category"

    parent_name = _path_canonical(hierarchy, dataset_name, parent)
    parent_role = infer_node_role(dataset_name, parent, node_depth(hierarchy, parent))

    if dataset_key == "fgvc-aircraft":
        if parent_role == "manufacturer":
            return f"a photo of aircraft manufactured by {parent_name}"
        if parent_role == "family":
            return f"a photo of aircraft in the {parent_name} family"
        if parent_role == "model":
            return f"a photo of aircraft model {parent_name}"

    path = _path_display(hierarchy, dataset_name, parent, include_self=True)
    clean_path = [clean_node_name(x) for x in path if x != "root"]
    if len(clean_path) >= 2:
        return f"a photo of {clean_path[-1]}, a visual category under {clean_path[-2]}"
    return f"a photo of {parent_name}, a visual category"


def build_parent_unknown_text(
    dataset_name: str,
    hierarchy,
    parent: str,
    variant: str = "parent_conditioned",
) -> str:
    if variant not in UNKNOWN_TEXT_VARIANTS:
        raise ValueError(
            f"Unsupported unknown text variant {variant!r}; "
            f"expected one of {sorted(UNKNOWN_TEXT_VARIANTS)}"
        )
    dataset_key = dataset_name.lower()
    if variant == "generic":
        if dataset_key == "fgvc-aircraft":
            return "a photo of an unknown aircraft"
        return "a photo of an unknown visual category"

    if parent == "root":
        if dataset_key == "fgvc-aircraft":
            if variant == "literal_not":
                return "a photo of an aircraft that is not from a known manufacturer"
            if variant == "unseen_child":
                return "a photo of an aircraft from an unseen manufacturer"
            return "a photo of an aircraft from an unknown manufacturer"
        if variant == "literal_not":
            return "a photo of a visual category that is not known"
        if variant == "unseen_child":
            return "a photo of an unseen visual category"
        return "a photo of an unknown visual category"

    parent_name = _path_canonical(hierarchy, dataset_name, parent)
    parent_role = infer_node_role(dataset_name, parent, node_depth(hierarchy, parent))

    if dataset_key == "fgvc-aircraft":
        if parent_role == "manufacturer":
            if variant == "literal_not":
                return (
                    f"a photo of an aircraft manufactured by {parent_name} "
                    "that is not from a known aircraft family"
                )
            if variant == "unseen_child":
                return (
                    f"a photo of an unseen aircraft family manufactured by "
                    f"{parent_name}"
                )
            return f"a photo of an unknown aircraft family manufactured by {parent_name}"
        if parent_role == "family":
            if variant == "literal_not":
                return (
                    f"a photo of an aircraft in the {parent_name} family "
                    "that is not a known aircraft model"
                )
            if variant == "unseen_child":
                return (
                    f"a photo of an unseen aircraft model under the "
                    f"{parent_name} family"
                )
            return f"a photo of an unknown aircraft model under the {parent_name} family"
        if variant == "literal_not":
            return (
                f"a photo of an aircraft under {parent_name} "
                "that is not a known child category"
            )
        if variant == "unseen_child":
            return f"a photo of an unseen aircraft category under {parent_name}"
        return f"a photo of another aircraft category under {parent_name}"

    if variant == "literal_not":
        return (
            f"a photo of a visual category under {parent_name} "
            "that is not a known child category"
        )
    if variant == "unseen_child":
        return f"a photo of an unseen visual category under {parent_name}"
    return f"a photo of another visual category under {parent_name}"


def build_child_negative_text(
    dataset_name: str,
    hierarchy,
    parent: str,
    child: str,
) -> str:
    dataset_key = dataset_name.lower()
    parent_name = _path_canonical(hierarchy, dataset_name, parent)
    child_name = _path_canonical(hierarchy, dataset_name, child)
    child_role = infer_node_role(dataset_name, child, node_depth(hierarchy, child))

    if dataset_key == "fgvc-aircraft":
        if child_role == "family":
            return (
                f"an aircraft manufactured by {parent_name} that is not from "
                f"the {child_name} family"
            )
        if child_role == "model":
            return (
                f"an aircraft in the {parent_name} family that is not model "
                f"{child_name}"
            )
        return f"an aircraft under {parent_name} that is not {child_name}"

    return f"a visual category under {parent_name} that is not {child_name}"
