from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .unknown_scoring import grouped_unknown_logits


def spherical_sibling_mixup(
    image_features: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    num_samples: int,
    mix_min: float,
    mix_max: float,
    rng: random.Random,
) -> torch.Tensor:
    """Create detached points between examples from different sibling classes."""
    if image_features.ndim != 2:
        raise ValueError("image_features must have shape [images, dim]")
    if int(image_features.shape[0]) != int(target_indices.numel()):
        raise ValueError("image_features and target_indices must have matching lengths")
    if not 0.0 <= float(mix_min) <= float(mix_max) <= 1.0:
        raise ValueError("mix_min and mix_max must satisfy 0 <= min <= max <= 1")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")

    groups: dict[int, list[int]] = {}
    for index, target in enumerate(target_indices.detach().cpu().tolist()):
        groups.setdefault(int(target), []).append(index)
    labels = sorted(groups)
    if len(labels) < 2:
        raise ValueError("Sibling mixup requires examples from at least two children")

    normalized = F.normalize(image_features.detach().float(), dim=-1)
    mixed = []
    for _ in range(int(num_samples)):
        first_label, second_label = rng.sample(labels, 2)
        first = normalized[rng.choice(groups[first_label])]
        second = normalized[rng.choice(groups[second_label])]
        weight = rng.uniform(float(mix_min), float(mix_max))
        dot = (first * second).sum().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(dot)
        sin_angle = torch.sin(angle)
        if float(sin_angle.abs().cpu()) < 1e-5:
            point = (1.0 - weight) * first + weight * second
        else:
            point = (
                torch.sin((1.0 - weight) * angle) / sin_angle * first
                + torch.sin(weight * angle) / sin_angle * second
            )
        mixed.append(F.normalize(point, dim=0))
    return torch.stack(mixed).detach()


def refine_virtual_open_features(
    virtual_features: torch.Tensor,
    child_features: torch.Tensor,
    parent_feature: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    child_temperature: float,
    parent_weight: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict]:
    """Move virtual points away from child evidence while retaining locality."""
    if float(child_temperature) <= 0.0:
        raise ValueError("child_temperature must be positive")
    if int(steps) < 0 or float(step_size) < 0.0:
        raise ValueError("steps and step_size must be non-negative")

    initial = F.normalize(virtual_features.detach().float(), dim=-1)
    children = F.normalize(child_features.detach().float(), dim=-1)
    parent = F.normalize(parent_feature.detach().float(), dim=-1)

    def measurements(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        child_logits = points @ children.t()
        child_energy = float(child_temperature) * torch.logsumexp(
            child_logits / float(child_temperature), dim=1
        )
        parent_similarity = points @ parent
        return child_energy, parent_similarity

    initial_child_energy, initial_parent_similarity = measurements(initial)
    points = initial
    for _ in range(int(steps)):
        points = points.detach().requires_grad_(True)
        normalized = F.normalize(points, dim=-1)
        child_energy, parent_similarity = measurements(normalized)
        anchor_similarity = (normalized * initial).sum(dim=1)
        objective = (
            child_energy
            - float(parent_weight) * parent_similarity
            + float(anchor_weight) * (1.0 - anchor_similarity)
        ).sum()
        gradient = torch.autograd.grad(objective, points, create_graph=False)[0]
        points = F.normalize(
            (points - float(step_size) * gradient).detach(), dim=-1
        )

    refined = F.normalize(points.detach(), dim=-1)
    final_child_energy, final_parent_similarity = measurements(refined)
    return refined, {
        "virtual_initial_child_energy": float(initial_child_energy.mean().cpu()),
        "virtual_refined_child_energy": float(final_child_energy.mean().cpu()),
        "virtual_initial_parent_similarity": float(
            initial_parent_similarity.mean().cpu()
        ),
        "virtual_refined_parent_similarity": float(
            final_parent_similarity.mean().cpu()
        ),
        "virtual_anchor_similarity": float((refined * initial).sum(dim=1).mean().cpu()),
    }


def _prototype_diversity(
    unknown_features: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if int(unknown_features.shape[0]) <= 1:
        return unknown_features.new_zeros(())
    similarities = unknown_features @ unknown_features.t()
    off_diagonal = ~torch.eye(
        unknown_features.shape[0],
        dtype=torch.bool,
        device=unknown_features.device,
    )
    return F.relu(similarities[off_diagonal] - float(margin)).square().mean()


def virtual_open_negprompt_loss(
    id_features: torch.Tensor,
    child_features: torch.Tensor,
    unknown_features: torch.Tensor,
    virtual_features: torch.Tensor,
    *,
    tau: float,
    lambda_virtual: float,
    lambda_id_teacher: float,
    lambda_coverage: float,
    lambda_diversity: float,
    diversity_margin: float,
) -> tuple[torch.Tensor, dict]:
    """Decoder-aligned virtual-open loss with gradients only to unknown prompts."""
    if float(tau) <= 0.0:
        raise ValueError("tau must be positive")
    images = F.normalize(id_features.detach().float(), dim=-1)
    children = F.normalize(child_features.detach().float(), dim=-1)
    unknowns = F.normalize(unknown_features.float(), dim=-1)
    virtual = F.normalize(virtual_features.detach().float(), dim=-1)
    logit_scale = 1.0 / float(tau)

    virtual_logits = grouped_unknown_logits(
        virtual,
        children,
        unknowns,
        logit_scale=logit_scale,
        aggregation="logmeanexp",
    )
    unknown_index = int(children.shape[0])
    virtual_targets = torch.full(
        (virtual.shape[0],), unknown_index, dtype=torch.long, device=virtual.device
    )
    virtual_loss = F.cross_entropy(virtual_logits, virtual_targets)

    child_logits = logit_scale * (images @ children.t())
    teacher_probabilities = F.softmax(child_logits, dim=1).detach()
    id_logits = grouped_unknown_logits(
        images,
        children,
        unknowns,
        logit_scale=logit_scale,
        aggregation="logmeanexp",
    )
    student_log_probabilities = F.log_softmax(id_logits, dim=1)
    id_teacher_loss = -(
        teacher_probabilities * student_log_probabilities[:, :unknown_index]
    ).sum(dim=1).mean()
    teacher_entropy = -(
        teacher_probabilities * teacher_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1).mean()

    similarities = virtual @ unknowns.t()
    virtual_coverage = similarities.max(dim=1).values.mean()
    prototype_coverage = similarities.max(dim=0).values.mean()
    coverage_loss = 1.0 - 0.5 * (virtual_coverage + prototype_coverage)
    diversity_loss = _prototype_diversity(unknowns, diversity_margin)

    open_objective = (
        float(lambda_virtual) * virtual_loss
        + float(lambda_coverage) * coverage_loss
        + float(lambda_diversity) * diversity_loss
    )
    total = open_objective + float(lambda_id_teacher) * id_teacher_loss
    virtual_predictions = virtual_logits.argmax(dim=1)
    id_predictions = id_logits.argmax(dim=1)
    return total, {
        "loss": float(total.detach().cpu()),
        "open_objective": float(open_objective.detach().cpu()),
        "virtual_loss": float(virtual_loss.detach().cpu()),
        "virtual_unknown_recall": float(
            (virtual_predictions == unknown_index).float().mean().detach().cpu()
        ),
        "id_teacher_loss": float(id_teacher_loss.detach().cpu()),
        "id_teacher_excess": float(
            (id_teacher_loss - teacher_entropy).detach().cpu()
        ),
        "id_unknown_selection_rate": float(
            (id_predictions == unknown_index).float().mean().detach().cpu()
        ),
        "coverage_loss": float(coverage_loss.detach().cpu()),
        "virtual_coverage": float(virtual_coverage.detach().cpu()),
        "prototype_coverage": float(prototype_coverage.detach().cpu()),
        "diversity_loss": float(diversity_loss.detach().cpu()),
        "virtual_unknown_margin": float(
            (
                virtual_logits[:, unknown_index]
                - virtual_logits[:, :unknown_index].max(dim=1).values
            ).mean().detach().cpu()
        ),
        "id_child_unknown_margin": float(
            (
                child_logits.max(dim=1).values - id_logits[:, unknown_index]
            ).mean().detach().cpu()
        ),
    }


def joint_virtual_open_prompt_loss(
    id_features: torch.Tensor,
    targets: torch.Tensor,
    child_features: torch.Tensor,
    unknown_features: torch.Tensor,
    virtual_features: torch.Tensor,
    *,
    tau: float,
    lambda_id: float,
    lambda_virtual: float,
    lambda_coverage: float,
    lambda_diversity: float,
    lambda_separation: float,
    diversity_margin: float,
    separation_margin: float,
) -> tuple[torch.Tensor, dict]:
    """Jointly fit positive and unknown prompts to a frozen image space."""
    if float(tau) <= 0.0:
        raise ValueError("tau must be positive")
    images = F.normalize(id_features.detach().float(), dim=-1)
    children = F.normalize(child_features.float(), dim=-1)
    unknowns = F.normalize(unknown_features.float(), dim=-1)
    virtual = F.normalize(virtual_features.detach().float(), dim=-1)
    targets = targets.long().to(images.device)
    logit_scale = 1.0 / float(tau)
    unknown_index = int(children.shape[0])

    id_logits = grouped_unknown_logits(
        images,
        children,
        unknowns,
        logit_scale=logit_scale,
        aggregation="logmeanexp",
    )
    id_loss = F.cross_entropy(id_logits, targets)

    virtual_logits = grouped_unknown_logits(
        virtual,
        children,
        unknowns,
        logit_scale=logit_scale,
        aggregation="logmeanexp",
    )
    virtual_targets = torch.full(
        (virtual.shape[0],), unknown_index, dtype=torch.long, device=virtual.device
    )
    virtual_loss = F.cross_entropy(virtual_logits, virtual_targets)

    similarities = virtual @ unknowns.t()
    virtual_coverage = similarities.max(dim=1).values.mean()
    prototype_coverage = similarities.max(dim=0).values.mean()
    coverage_loss = 1.0 - 0.5 * (virtual_coverage + prototype_coverage)
    diversity_loss = _prototype_diversity(unknowns, diversity_margin)
    child_unknown_cosines = children @ unknowns.t()
    separation_loss = F.relu(
        child_unknown_cosines - float(separation_margin)
    ).square().mean()

    open_objective = (
        float(lambda_virtual) * virtual_loss
        + float(lambda_coverage) * coverage_loss
        + float(lambda_diversity) * diversity_loss
        + float(lambda_separation) * separation_loss
    )
    total = float(lambda_id) * id_loss + open_objective
    id_predictions = id_logits.argmax(dim=1)
    virtual_predictions = virtual_logits.argmax(dim=1)
    return total, {
        "loss": float(total.detach().cpu()),
        "id_loss": float(id_loss.detach().cpu()),
        "id_acc": float((id_predictions == targets).float().mean().detach().cpu()),
        "id_unknown_selection_rate": float(
            (id_predictions == unknown_index).float().mean().detach().cpu()
        ),
        "open_objective": float(open_objective.detach().cpu()),
        "virtual_loss": float(virtual_loss.detach().cpu()),
        "virtual_unknown_recall": float(
            (virtual_predictions == unknown_index).float().mean().detach().cpu()
        ),
        "coverage_loss": float(coverage_loss.detach().cpu()),
        "diversity_loss": float(diversity_loss.detach().cpu()),
        "separation_loss": float(separation_loss.detach().cpu()),
        "positive_negative_cosine": float(child_unknown_cosines.mean().detach().cpu()),
        "virtual_unknown_margin": float(
            (
                virtual_logits[:, unknown_index]
                - virtual_logits[:, :unknown_index].max(dim=1).values
            ).mean().detach().cpu()
        ),
        "id_child_unknown_margin": float(
            (
                id_logits[:, :unknown_index].max(dim=1).values
                - id_logits[:, unknown_index]
            ).mean().detach().cpu()
        ),
    }
