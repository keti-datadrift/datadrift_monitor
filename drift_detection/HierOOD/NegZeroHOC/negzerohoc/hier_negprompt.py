from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .semantic_index import LocalSemanticCandidates
from .unknown_scoring import grouped_unknown_logits


def hierarchical_negprompt_loss(
    image_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    target_indices: torch.Tensor,
    parent_feature: torch.Tensor,
    *,
    tau: float,
    lambda_nis: float,
    lambda_npd: float,
    lambda_nnd: float,
    lambda_stop: float,
    lambda_parent: float,
) -> tuple[torch.Tensor, dict]:
    """NegPrompt losses for one hierarchy parent without held-out children."""
    if float(tau) <= 0.0:
        raise ValueError("tau must be positive")
    if negative_features.ndim != 3:
        raise ValueError("negative_features must have shape [children, prompts, dim]")
    if positive_features.ndim != 2:
        raise ValueError("positive_features must have shape [children, dim]")
    if negative_features.shape[0] != positive_features.shape[0]:
        raise ValueError("Positive and negative child counts must match")

    images = F.normalize(image_features.float(), dim=-1)
    positives = F.normalize(positive_features.float(), dim=-1)
    negatives = F.normalize(negative_features.float(), dim=-1)
    negative_flat = negatives.flatten(0, 1)
    targets = target_indices.long().to(images.device)

    negative_logits = images @ negative_flat.t() / float(tau)
    negative_log_probs = F.log_softmax(negative_logits, dim=1)
    nis_loss = -negative_log_probs.mean()
    nis_excess = nis_loss - math.log(negative_logits.shape[1])

    stop_logits = grouped_unknown_logits(
        images,
        positives,
        negative_flat,
        logit_scale=1.0 / float(tau),
    )
    stop_loss = F.cross_entropy(stop_logits, targets)

    positive_per_negative = positives.unsqueeze(1).expand_as(negatives)
    npd_loss = -(negatives * positive_per_negative).sum(dim=-1).mean()

    num_prompts = int(negatives.shape[1])
    if num_prompts > 1:
        similarities = negatives @ negatives.transpose(1, 2)
        off_diagonal = ~torch.eye(
            num_prompts,
            dtype=torch.bool,
            device=negatives.device,
        )
        nnd_loss = similarities[:, off_diagonal].mean()
    else:
        nnd_loss = negatives.new_zeros(())

    parent = F.normalize(parent_feature.float(), dim=-1)
    parent_loss = -(negatives @ parent).mean()
    total = (
        float(lambda_nis) * nis_loss
        + float(lambda_npd) * npd_loss
        + float(lambda_nnd) * nnd_loss
        + float(lambda_stop) * stop_loss
        + float(lambda_parent) * parent_loss
    )

    predictions = stop_logits.argmax(dim=1)
    unknown_index = positive_features.shape[0]
    return total, {
        "loss": float(total.detach().cpu()),
        "nis_loss": float(nis_loss.detach().cpu()),
        "nis_excess": float(nis_excess.detach().cpu()),
        "npd_loss": float(npd_loss.detach().cpu()),
        "nnd_loss": float(nnd_loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
        "parent_loss": float(parent_loss.detach().cpu()),
        "known_acc": float((predictions == targets).float().mean().detach().cpu()),
        "unknown_selection_rate": float(
            (predictions == unknown_index).float().mean().detach().cpu()
        ),
    }


def build_differentiable_hier_negprompt_semantic_index(
    hierarchy,
    positive_index,
    negative_learner,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Encode every unknown bank with gradients for decoder-aligned training."""
    index = {}
    negative_features_by_parent = {}
    for parent, local in positive_index.items():
        negative_features = None
        candidate_names = list(local.children)
        prompts = dict(local.prompts)
        if parent != "root":
            child_negatives = negative_learner.encode_negative_prototypes(
                parent,
                list(local.children),
            )
            negative_features_by_parent[parent] = child_negatives
            negative_features = child_negatives.flatten(0, 1)
            unknown_name = f"__unknown__:{parent}"
            candidate_names.append(unknown_name)
            prompts[unknown_name] = [
                negative_learner.negative_text(parent, child)
                for child in local.children
                for _ in range(negative_learner.num_negative_prompts)
            ]
        index[parent] = LocalSemanticCandidates(
            parent=parent,
            children=list(local.children),
            child_features=local.child_features,
            unknown_feature=negative_features,
            candidate_names=candidate_names,
            prompts=prompts,
        )
    return index, negative_features_by_parent


@torch.no_grad()
def build_hier_negprompt_semantic_index(hierarchy, positive_index, negative_learner):
    was_training = negative_learner.training
    negative_learner.eval()
    index = {}
    for parent, local in positive_index.items():
        negative_features = None
        candidate_names = list(local.children)
        prompts = dict(local.prompts)
        if parent != "root":
            child_negatives = negative_learner.encode_negative_prototypes(
                parent,
                list(local.children),
            )
            negative_features = child_negatives.flatten(0, 1).detach().cpu()
            unknown_name = f"__unknown__:{parent}"
            candidate_names.append(unknown_name)
            prompts[unknown_name] = [
                negative_learner.negative_text(parent, child)
                for child in local.children
                for _ in range(negative_learner.num_negative_prompts)
            ]
        index[parent] = LocalSemanticCandidates(
            parent=parent,
            children=list(local.children),
            child_features=local.child_features,
            unknown_feature=negative_features,
            candidate_names=candidate_names,
            prompts=prompts,
        )
    if was_training:
        negative_learner.train()
    return index
