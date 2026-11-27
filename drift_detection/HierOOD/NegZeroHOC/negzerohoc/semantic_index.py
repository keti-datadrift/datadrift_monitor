from __future__ import annotations

from dataclasses import dataclass

import torch

from .prompts import (
    build_positive_prompts,
    build_unknown_prompts,
    infer_node_role,
    node_path_names,
)


@dataclass
class LocalSemanticCandidates:
    parent: str
    children: list[str]
    child_features: torch.Tensor
    unknown_feature: torch.Tensor | None
    candidate_names: list[str]
    prompts: dict[str, list[str]]


def build_semantic_index(
    dataset_name: str,
    hierarchy,
    clip_backend,
    mode: str,
    allow_root_unknown: bool = False,
) -> dict[str, LocalSemanticCandidates]:
    if mode not in {"child_only", "manual_unknown"}:
        raise ValueError(f"Unsupported mode for this implementation pass: {mode}")

    index = {}
    for parent, children in hierarchy.parent2children.items():
        child_features = []
        prompts_by_candidate = {}
        for child in children:
            path = node_path_names(hierarchy, child, include_self=True, dataset_name=dataset_name)
            role = infer_node_role(dataset_name, child)
            prompts = build_positive_prompts(dataset_name, child, parent, path, role)
            prompts_by_candidate[child] = prompts
            child_features.append(clip_backend.encode_prompt_ensemble(prompts).cpu())

        child_tensor = torch.stack(child_features, dim=0)
        unknown_feature = None
        candidate_names = list(children)

        if mode == "manual_unknown" and (parent != "root" or allow_root_unknown):
            parent_path = node_path_names(hierarchy, parent, include_self=True, dataset_name=dataset_name)
            prompts = build_unknown_prompts(dataset_name, parent, parent_path)
            unknown_name = f"__unknown__:{parent}"
            prompts_by_candidate[unknown_name] = prompts
            unknown_feature = clip_backend.encode_prompt_ensemble(prompts).cpu()
            candidate_names.append(unknown_name)

        index[parent] = LocalSemanticCandidates(
            parent=parent,
            children=list(children),
            child_features=child_tensor,
            unknown_feature=unknown_feature,
            candidate_names=candidate_names,
            prompts=prompts_by_candidate,
        )

    return index
