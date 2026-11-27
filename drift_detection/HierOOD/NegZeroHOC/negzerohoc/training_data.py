from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

import torch

from .prompt_text import node_depth


UNKNOWN_LABEL = "__unknown__"


@dataclass(frozen=True)
class EdgeExample:
    image_index: int
    parent: str
    child: str
    leaf: str
    parent_depth: int


@dataclass(frozen=True)
class UnknownEpisode:
    parent: str
    known_children: list[str]
    hidden_children: list[str]
    examples: list[EdgeExample]
    labels: list[str]


@dataclass(frozen=True)
class ParentKnownEpisode:
    parent: str
    children: list[str]
    examples: list[EdgeExample]
    labels: list[str]


def node_path(hierarchy, node: str) -> list[str]:
    ancestors = [hierarchy.id_node_list[idx] for idx in hierarchy.node_ancestors.get(node, [])]
    return ancestors + [node]


def build_leaf_paths(hierarchy) -> dict[str, list[str]]:
    leaves = [node for node in hierarchy.id_node_list if node not in hierarchy.parent2children]
    return {leaf: node_path(hierarchy, leaf) for leaf in leaves}


def _target_to_retained_node(hierarchy, class_name: str) -> str:
    node = class_name
    while node not in hierarchy.id_node_list:
        node = hierarchy.child2parent[node]
    return node


def build_positive_edge_examples(hierarchy, feature_payload: dict) -> list[EdgeExample]:
    classes = list(feature_payload["classes"])
    targets = feature_payload["targets"].long()
    examples = []

    for image_index, target in enumerate(targets.tolist()):
        leaf = _target_to_retained_node(hierarchy, classes[target])
        path = node_path(hierarchy, leaf)
        for parent, child in zip(path[:-1], path[1:]):
            if parent not in hierarchy.parent2children:
                continue
            if child not in hierarchy.parent2children[parent]:
                continue
            examples.append(
                EdgeExample(
                    image_index=image_index,
                    parent=parent,
                    child=child,
                    leaf=leaf,
                    parent_depth=node_depth(hierarchy, parent),
                )
            )
    return examples


def group_examples_by_parent(examples: list[EdgeExample]) -> dict[str, list[EdgeExample]]:
    grouped = defaultdict(list)
    for example in examples:
        grouped[example.parent].append(example)
    return dict(grouped)


def group_examples_by_parent_child(examples: list[EdgeExample]) -> dict[str, dict[str, list[EdgeExample]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for example in examples:
        grouped[example.parent][example.child].append(example)
    return {parent: dict(child_map) for parent, child_map in grouped.items()}


def sample_examples(examples: list[EdgeExample], max_examples: int, rng: random.Random) -> list[EdgeExample]:
    if len(examples) <= max_examples:
        return list(examples)
    return rng.sample(examples, max_examples)


def sample_leave_child_out_episode(
    parent: str,
    child_examples: dict[str, list[EdgeExample]],
    strategy: str,
    max_examples: int,
    rng: random.Random,
) -> UnknownEpisode | None:
    eligible_children = [child for child, exs in child_examples.items() if exs]
    if parent == "root" or len(eligible_children) < 2:
        return None

    shuffled = list(eligible_children)
    rng.shuffle(shuffled)
    if strategy == "hide_ratio_25":
        n_hide = max(1, int(round(len(shuffled) * 0.25)))
    elif strategy == "hide_ratio_50":
        n_hide = max(1, int(round(len(shuffled) * 0.50)))
    elif strategy == "hide_one_child":
        n_hide = 1
    else:
        raise ValueError(f"Unsupported hide strategy: {strategy}")

    n_hide = min(n_hide, len(shuffled) - 1)
    hidden = sorted(shuffled[:n_hide])
    known = sorted(child for child in eligible_children if child not in set(hidden))
    if not known or not hidden:
        return None

    per_child = max(1, max_examples // len(eligible_children))
    examples = []
    labels = []
    for child in known:
        chosen = sample_examples(child_examples[child], per_child, rng)
        examples.extend(chosen)
        labels.extend([child] * len(chosen))
    for child in hidden:
        chosen = sample_examples(child_examples[child], per_child, rng)
        examples.extend(chosen)
        labels.extend([UNKNOWN_LABEL] * len(chosen))

    if not examples:
        return None

    order = list(range(len(examples)))
    rng.shuffle(order)
    return UnknownEpisode(
        parent=parent,
        known_children=known,
        hidden_children=hidden,
        examples=[examples[i] for i in order],
        labels=[labels[i] for i in order],
    )


def sample_parent_known_episode(
    parent: str,
    child_examples: dict[str, list[EdgeExample]],
    max_examples: int,
    rng: random.Random,
) -> ParentKnownEpisode | None:
    children = sorted(child for child, examples in child_examples.items() if examples)
    if parent == "root" or not children:
        return None

    per_child = max(1, max_examples // len(children))
    examples = []
    labels = []
    for child in children:
        chosen = sample_examples(child_examples[child], per_child, rng)
        examples.extend(chosen)
        labels.extend([child] * len(chosen))
    if not examples:
        return None

    order = list(range(len(examples)))
    rng.shuffle(order)
    return ParentKnownEpisode(
        parent=parent,
        children=children,
        examples=[examples[index] for index in order],
        labels=[labels[index] for index in order],
    )


def gather_image_features(features: torch.Tensor, examples: list[EdgeExample], device: str | torch.device) -> torch.Tensor:
    indices = torch.tensor([example.image_index for example in examples], dtype=torch.long)
    return features.index_select(0, indices).to(device)
