from __future__ import annotations

from typing import Iterable


FGVC_ROLE_BY_PREFIX = {
    "m-": "manufacturer",
    "f-": "family",
    "v-": "model",
}


def clean_node_name(node_name: str) -> str:
    for prefix in FGVC_ROLE_BY_PREFIX:
        if node_name.startswith(prefix):
            node_name = node_name[len(prefix):]
            break
    return node_name.replace("_", " ").strip()


def clean_description(description: str) -> str:
    description = description.strip()
    if ":" in description:
        prefix, value = description.split(":", 1)
        if prefix.lower() in {
            "name",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        }:
            description = value.strip()
    return description.replace("_", " ").strip()


def display_node_name(hierarchy, dataset_name: str, node_name: str) -> str:
    if dataset_name.lower() == "fgvc-aircraft":
        return node_name
    if node_name == "root":
        return "root"
    description = getattr(hierarchy, "node_description", {}).get(node_name)
    if description:
        return clean_description(description)
    return clean_node_name(node_name)


def infer_node_role(dataset_name: str, node_name: str, depth: int | None = None) -> str:
    dataset_name = dataset_name.lower()
    if dataset_name == "fgvc-aircraft":
        for prefix, role in FGVC_ROLE_BY_PREFIX.items():
            if node_name.startswith(prefix):
                return role
        if node_name == "root":
            return "root"
    if depth is not None:
        return f"depth_{depth}"
    return "node"


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def _fgvc_canonical(node_name: str, path_names: list[str], node_role: str) -> str:
    if node_name == "root":
        return "aircraft"

    clean_path = [clean_node_name(x) for x in path_names if x != "root"]
    raw = clean_node_name(node_name)

    manufacturer = next((x for x in clean_path if x.startswith("m-")), None)
    family = next((x for x in clean_path if x.startswith("f-")), None)

    # clean_path has already been stripped, so recover by role from original path.
    for original, clean in zip([x for x in path_names if x != "root"], clean_path):
        role = infer_node_role("fgvc-aircraft", original)
        if role == "manufacturer":
            manufacturer = clean
        elif role == "family":
            family = clean

    if node_role == "manufacturer":
        return raw

    if node_role == "family":
        if manufacturer and not _contains(raw, manufacturer):
            return f"{manufacturer} {raw}"
        return raw

    if node_role == "model":
        if manufacturer and family:
            family_tail = family
            if _contains(family_tail, manufacturer):
                family_tail = family_tail[len(manufacturer):].strip()
            if _contains(raw, family_tail) or _contains(raw, manufacturer):
                if _contains(raw, manufacturer):
                    return raw
                return f"{manufacturer} {raw}"
            return f"{manufacturer} {family_tail}-{raw}"
        if manufacturer and not _contains(raw, manufacturer):
            return f"{manufacturer} {raw}"
        return raw

    return raw


def canonicalize_node_name(
    dataset_name: str,
    node_name: str,
    path_names: Iterable[str] | None = None,
    node_role: str | None = None,
) -> str:
    path_names = list(path_names or [])
    node_role = node_role or infer_node_role(dataset_name, node_name)

    if dataset_name.lower() == "fgvc-aircraft":
        return _fgvc_canonical(node_name, path_names, node_role)

    if path_names:
        clean = clean_node_name(path_names[-1])
        return clean if clean != "root" else "root"

    clean = clean_node_name(node_name)
    return clean if clean != "root" else "root"


def build_positive_prompts(
    dataset_name: str,
    node_name: str,
    parent_name: str | None,
    path_names: Iterable[str],
    node_role: str | None = None,
) -> list[str]:
    node_role = node_role or infer_node_role(dataset_name, node_name)
    full_name = canonicalize_node_name(dataset_name, node_name, path_names, node_role)

    if dataset_name.lower() == "fgvc-aircraft":
        if node_role == "manufacturer":
            return [
                f"a photo of a {full_name} aircraft",
                f"an image of an airplane manufactured by {full_name}",
                f"a photo of a commercial aircraft made by {full_name}",
                f"a photo of an aircraft from manufacturer {full_name}",
            ]
        if node_role == "family":
            return [
                f"a photo of a {full_name} aircraft",
                f"a photo of an aircraft from the {full_name} family",
                f"a photo of a {full_name} passenger airplane",
                f"a photo of a commercial aircraft in the {full_name} family",
            ]
        if node_role == "model":
            return [
                f"a photo of a {full_name} aircraft",
                f"a photo of a {full_name} passenger airplane",
                f"a photo of an aircraft model {full_name}",
                f"a photo of a commercial aircraft variant {full_name}",
            ]

    return [
        f"a photo of {full_name}",
        f"a photo of {full_name}, a class in a visual hierarchy",
    ]


def build_unknown_prompts(
    dataset_name: str,
    parent_name: str,
    parent_path_names: Iterable[str],
    unknown_role: str | None = None,
) -> list[str]:
    parent_path_names = list(parent_path_names)
    parent_role = infer_node_role(dataset_name, parent_name)
    parent_full = canonicalize_node_name(dataset_name, parent_name, parent_path_names, parent_role)

    if dataset_name.lower() == "fgvc-aircraft":
        if parent_name == "root":
            return [
                "a photo of an aircraft from an unknown manufacturer",
                "a photo of a commercial aircraft made by another manufacturer",
                "a photo of an airplane whose manufacturer is not among the known categories",
            ]
        if parent_role == "manufacturer":
            return [
                f"a photo of an unknown aircraft family manufactured by {parent_full}",
                f"a photo of another {parent_full} aircraft family",
                f"a photo of a {parent_full} aircraft outside the known families",
                f"a photo of a {parent_full} airplane not belonging to the listed aircraft families",
            ]
        if parent_role == "family":
            return [
                f"a photo of an unknown aircraft model from the {parent_full} family",
                f"a photo of another model variant of the {parent_full} aircraft",
                f"a photo of a {parent_full} aircraft outside the known model variants",
                f"a photo of a {parent_full} airplane not belonging to the listed models",
            ]

    if parent_name == "root":
        return [
            "a photo of an unknown visual category",
            "a photo of an object or organism outside the known categories",
            "a photo that does not belong to the listed visual hierarchy categories",
        ]

    return [
        f"a photo of another kind of {parent_full}",
        f"a photo of {parent_full} outside the listed child categories",
    ]


def node_path_names(
    hierarchy,
    node_name: str,
    include_self: bool = True,
    dataset_name: str | None = None,
) -> list[str]:
    ancestor_indices = hierarchy.node_ancestors.get(node_name, [])
    names = [hierarchy.id_node_list[i] for i in ancestor_indices]
    if include_self:
        names.append(node_name)
    if dataset_name is not None:
        names = [display_node_name(hierarchy, dataset_name, name) for name in names]
    return names


def node_depth(hierarchy, node_name: str) -> int:
    return len(hierarchy.node_ancestors.get(node_name, []))
