from __future__ import annotations

import argparse
from argparse import Namespace
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.prompts import (
    build_positive_prompts,
    build_unknown_prompts,
    canonicalize_node_name,
    display_node_name,
    infer_node_role,
    node_path_names,
)


def load_config(path):
    cfg = load_yaml_config(path)
    dataset_cfg = cfg.get("dataset", {})
    inspect_cfg = cfg.get("prompt_inspection", {})
    inference_cfg = cfg.get("inference", {})
    return Namespace(
        config=str(path),
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        nodes=inspect_cfg.get("nodes", ["m-Boeing", "f-Boeing_737", "v-737-300"]),
        parents=inspect_cfg.get("parents", ["root", "m-Boeing", "f-Boeing_737"]),
        allow_root_unknown=inference_cfg.get("allow_root_unknown", False),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    config_arg = parser.parse_args()
    return load_config(config_arg.config)


def format_node_for_display(hierarchy, dataset_name: str, node_name: str) -> str:
    display = display_node_name(hierarchy, dataset_name, node_name)
    if display == node_name:
        return node_name
    return f"{display} ({node_name})"


def main():
    args = parse_args()
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)

    print("== Positive prompts ==")
    for node in args.nodes:
        if node not in hierarchy.id_node_list:
            print(f"[skip] {node}: not in retained ID hierarchy")
            continue
        path = node_path_names(hierarchy, node, include_self=True, dataset_name=args.dataset)
        role = infer_node_role(args.dataset, node)
        canonical = canonicalize_node_name(args.dataset, node, path, role)
        parent = None
        if node in hierarchy.child2parent:
            parent = hierarchy.child2parent[node]
        print(f"\nnode_id: {node}")
        print(f"display: {display_node_name(hierarchy, args.dataset, node)}")
        print(f"role: {role}")
        print(f"path: {' > '.join(path)}")
        print(f"canonical: {canonical}")
        for prompt in build_positive_prompts(args.dataset, node, parent, path, role):
            print(f"- {prompt}")

    print("\n== Unknown prompts ==")
    for parent in args.parents:
        if parent not in hierarchy.id_node_list:
            print(f"[skip] {parent}: not in retained ID hierarchy")
            continue
        if parent not in hierarchy.parent2children:
            print(f"[skip] {parent}: not an internal parent")
            continue
        if parent == "root" and not args.allow_root_unknown:
            print(f"[skip] {parent}: root unknown disabled by inference.allow_root_unknown")
            continue
        path = node_path_names(hierarchy, parent, include_self=True, dataset_name=args.dataset)
        canonical = canonicalize_node_name(args.dataset, parent, path, infer_node_role(args.dataset, parent))
        children = [
            format_node_for_display(hierarchy, args.dataset, child)
            for child in hierarchy.parent2children[parent]
        ]
        print(f"\nparent_id: {parent}")
        print(f"display: {display_node_name(hierarchy, args.dataset, parent)}")
        print(f"path: {' > '.join(path)}")
        print(f"canonical: {canonical}")
        print(f"children: {', '.join(children)}")
        for prompt in build_unknown_prompts(args.dataset, parent, path):
            print(f"- {prompt}")


if __name__ == "__main__":
    main()
