from __future__ import annotations

import argparse
from argparse import Namespace
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.prompt_text import build_edge_text, build_parent_text, build_parent_unknown_text, node_depth
from negzerohoc.prompts import display_node_name


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
        nodes=inspect_cfg.get("nodes", []),
        parents=inspect_cfg.get("parents", []),
        allow_root_unknown=bool(inference_cfg.get("allow_root_unknown", False)),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def fmt(hierarchy, dataset: str, node: str) -> str:
    display = display_node_name(hierarchy, dataset, node)
    return node if display == node else f"{display} ({node})"


def main():
    args = parse_args()
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)

    print("== Idea 3 positive edge texts ==")
    for node in args.nodes:
        if node not in hierarchy.id_node_list:
            print(f"[skip] {node}: not in retained ID hierarchy")
            continue
        if node == "root" or node not in hierarchy.child2parent:
            print(f"[skip] {node}: no retained parent edge")
            continue
        parent = hierarchy.child2parent[node]
        print(f"\nparent: {fmt(hierarchy, args.dataset, parent)}")
        print(f"child: {fmt(hierarchy, args.dataset, node)}")
        print(f"parent_depth: {node_depth(hierarchy, parent)}")
        print(f"- {build_edge_text(args.dataset, hierarchy, parent, node)}")

    print("\n== Idea 3 parent texts and unknown texts ==")
    for parent in args.parents:
        if parent not in hierarchy.id_node_list:
            print(f"[skip] {parent}: not in retained ID hierarchy")
            continue
        if parent not in hierarchy.parent2children:
            print(f"[skip] {parent}: not an internal parent")
            continue
        print(f"\nparent: {fmt(hierarchy, args.dataset, parent)}")
        print(f"depth: {node_depth(hierarchy, parent)}")
        print(f"children: {', '.join(fmt(hierarchy, args.dataset, c) for c in hierarchy.parent2children[parent])}")
        print(f"parent_text: {build_parent_text(args.dataset, hierarchy, parent)}")
        if parent == "root" and not args.allow_root_unknown:
            print("unknown_text: [disabled for root]")
        else:
            print(f"unknown_text: {build_parent_unknown_text(args.dataset, hierarchy, parent)}")


if __name__ == "__main__":
    main()
