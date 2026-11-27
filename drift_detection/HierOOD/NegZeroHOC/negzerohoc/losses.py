from __future__ import annotations

import torch
import torch.nn.functional as F


def positive_ce_loss(
    image_features: torch.Tensor,
    child_features: torch.Tensor,
    target_indices: torch.Tensor,
    tau: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    image_features = F.normalize(image_features.float(), dim=-1)
    child_features = F.normalize(child_features.float(), dim=-1)
    logits = image_features @ child_features.t() / float(tau)
    loss = F.cross_entropy(logits, target_indices.long().to(logits.device))
    acc = (logits.argmax(dim=1) == target_indices.to(logits.device)).float().mean()
    return loss, {"acc": float(acc.detach().cpu()), "loss": float(loss.detach().cpu())}


def sparse_path_bottleneck_loss(
    decision_logits: list[list[torch.Tensor]],
    target_indices: list[list[int]],
    bottleneck_weight: float = 0.5,
    bottleneck_temperature: float = 0.5,
    route_margin: float = 0.0,
    margin_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Optimize exhaustive sibling decisions on each sample's true path.

    ``decision_logits[i][d]`` contains every child logit for the parent at
    depth ``d`` on sample ``i``'s ground-truth path. No hierarchy-wide prompt
    set or sampled negative set is required: all siblings at each active
    parent are exact negatives.

    The normalized smooth maximum emphasizes the weakest decision on a path.
    This matches the all-edges-correct requirement of hierarchical routing
    while keeping the objective decomposable over samples for exact gradient
    accumulation with small microbatches.
    """
    if len(decision_logits) != len(target_indices):
        raise ValueError("decision_logits and target_indices must have the same number of samples")
    if not 0.0 <= float(bottleneck_weight) <= 1.0:
        raise ValueError("bottleneck_weight must be in [0, 1]")
    if float(bottleneck_temperature) <= 0.0:
        raise ValueError("bottleneck_temperature must be positive")

    sample_losses = []
    local_losses_all = []
    margin_losses_all = []
    local_correct = 0
    local_total = 0
    path_correct = 0

    for sample_logits, sample_targets in zip(decision_logits, target_indices):
        if len(sample_logits) != len(sample_targets):
            raise ValueError("Each sample must have one target for every path decision")
        if not sample_logits:
            continue

        local_losses = []
        local_margin_losses = []
        sample_is_correct = True
        for logits, target in zip(sample_logits, sample_targets):
            if logits.ndim != 1:
                raise ValueError("Each local decision tensor must be one-dimensional")
            if not 0 <= int(target) < int(logits.numel()):
                raise ValueError(f"Target {target} is outside a decision with {logits.numel()} children")

            target_tensor = torch.tensor([int(target)], dtype=torch.long, device=logits.device)
            local_loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
            local_losses.append(local_loss)
            local_losses_all.append(local_loss)

            prediction = int(torch.argmax(logits.detach()).item())
            is_correct = prediction == int(target)
            local_correct += int(is_correct)
            local_total += 1
            sample_is_correct = sample_is_correct and is_correct

            if logits.numel() > 1 and (float(margin_weight) > 0.0 or float(route_margin) > 0.0):
                negative_mask = torch.ones_like(logits, dtype=torch.bool)
                negative_mask[int(target)] = False
                negative_lse = torch.logsumexp(logits[negative_mask], dim=0)
                margin_loss = F.softplus(negative_lse - logits[int(target)] + float(route_margin))
                local_margin_losses.append(margin_loss)
                margin_losses_all.append(margin_loss)

        path_correct += int(sample_is_correct)
        local_tensor = torch.stack(local_losses)
        mean_local = local_tensor.mean()
        temperature = float(bottleneck_temperature)
        smooth_worst = temperature * (
            torch.logsumexp(local_tensor / temperature, dim=0)
            - torch.log(torch.tensor(float(local_tensor.numel()), device=local_tensor.device))
        )
        sample_loss = (
            (1.0 - float(bottleneck_weight)) * mean_local
            + float(bottleneck_weight) * smooth_worst
        )
        if local_margin_losses and float(margin_weight) > 0.0:
            sample_loss = sample_loss + float(margin_weight) * torch.stack(local_margin_losses).mean()
        sample_losses.append(sample_loss)

    if not sample_losses:
        raise ValueError("Sparse path bottleneck loss received no valid path decisions")

    loss = torch.stack(sample_losses).mean()
    mean_local_loss = torch.stack(local_losses_all).mean()
    mean_margin_loss = (
        torch.stack(margin_losses_all).mean()
        if margin_losses_all
        else loss.detach().new_zeros(())
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mean_local_loss": float(mean_local_loss.detach().cpu()),
        "mean_margin_loss": float(mean_margin_loss.detach().cpu()),
        "local_acc": local_correct / max(1, local_total),
        "path_acc": path_correct / max(1, len(sample_losses)),
        "num_samples": len(sample_losses),
        "num_decisions": local_total,
    }


def ms_loss_level(
    sim_mat: torch.Tensor,
    labels: torch.Tensor,
    neg_weights: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sim_mat.device
    batch_size = sim_mat.size(0)
    labels = labels.view(-1)
    loss_list = []
    set_size_list = []

    for i in range(batch_size):
        label_i = labels[i]
        mask_pos = labels == label_i
        mask_neg = labels != label_i
        mask_pos[i] = False

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]
        current_neg_w = None if neg_weights is None else neg_weights[i][mask_neg]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        hardest_pos_sim = torch.min(pos_sim)
        hardest_neg_sim = torch.max(neg_sim)

        neg_keep = (neg_sim + mining_margin) > hardest_pos_sim
        pos_keep = (pos_sim - mining_margin) < hardest_neg_sim
        neg_sim = neg_sim[neg_keep]
        pos_sim = pos_sim[pos_keep]
        if current_neg_w is not None:
            current_neg_w = current_neg_w[neg_keep]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        pos_term = torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))) / alpha
        if current_neg_w is None:
            neg_exp = torch.exp(beta * (neg_sim - lam))
        else:
            neg_exp = current_neg_w * torch.exp(beta * (neg_sim - lam))
        neg_term = torch.log(1.0 + torch.sum(neg_exp)) / beta

        loss_list.append(pos_term + neg_term)
        set_size_list.append(torch.tensor(pos_sim.numel() + neg_sim.numel(), dtype=sim_mat.dtype, device=device))

    return torch.stack(loss_list), torch.stack(set_size_list)


def multi_similarity_loss(
    features: torch.Tensor,
    path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    sim_mat = torch.clamp(features @ features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    _, labels = torch.unique(path_labels, dim=0, return_inverse=True)
    losses, _ = ms_loss_level(
        sim_mat,
        labels,
        alpha=alpha,
        beta=beta,
        lam=lam,
        mining_margin=mining_margin,
    )
    return losses.mean()


def get_slice_distance_weights(
    slice_labels: torch.Tensor,
    scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    matches = (slice_labels.unsqueeze(1) == slice_labels.unsqueeze(0)).float()
    continuous_matches = torch.cumprod(matches, dim=2)
    shared_depth = continuous_matches.sum(dim=2)
    tree_dist = (float(slice_labels.size(1)) - shared_depth).pow(dist_pow)
    return torch.pow(float(scale), tree_dist - 1.0)


def hierarchical_ms_min_loss(
    features: torch.Tensor,
    path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    use_distance_weights: bool = False,
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    if minimum_mode not in {"batch", "sample"}:
        raise ValueError(f"Unsupported minimum_mode: {minimum_mode}")

    features = F.normalize(features.float(), dim=-1)
    sim_mat = torch.clamp(features @ features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    num_levels = int(path_labels.size(1))
    device = features.device

    level_losses = []
    for level in range(num_levels):
        current_path = path_labels[:, : level + 1]
        _, labels_level = torch.unique(current_path, dim=0, return_inverse=True)
        neg_weights = None
        if use_distance_weights:
            neg_weights = get_slice_distance_weights(
                current_path,
                scale=dist_scale,
                dist_pow=dist_pow,
            )
        losses_l, _ = ms_loss_level(
            sim_mat,
            labels_level,
            neg_weights=neg_weights,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
        level_losses.append(losses_l)

    total_loss = torch.tensor(0.0, dtype=features.dtype, device=device)
    cur_min = None
    for rev_level in range(num_levels - 1, -1, -1):
        base_loss_l = level_losses[rev_level]
        if cur_min is None:
            cur_loss = base_loss_l
        else:
            cur_loss = torch.min(base_loss_l, cur_min)
        cur_min = torch.min(cur_loss) if minimum_mode == "batch" else cur_loss
        k_l = torch.exp(torch.tensor(1.0 / float(rev_level + 1), dtype=features.dtype, device=device))
        total_loss = total_loss + (k_l * cur_loss.mean()) / float(num_levels)
    return total_loss


def _cross_modal_ms_loss_level(
    sim_mat: torch.Tensor,
    image_labels: torch.Tensor,
    prompt_labels: torch.Tensor,
    neg_weights: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sim_mat.device
    num_images = sim_mat.size(0)
    loss_list = []
    set_size_list = []

    for i in range(num_images):
        mask_pos = prompt_labels == image_labels[i]
        mask_neg = ~mask_pos

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]
        current_neg_w = None if neg_weights is None else neg_weights[i][mask_neg]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        hardest_pos_sim = torch.min(pos_sim)
        hardest_neg_sim = torch.max(neg_sim)

        neg_keep = (neg_sim + mining_margin) > hardest_pos_sim
        pos_keep = (pos_sim - mining_margin) < hardest_neg_sim
        neg_sim = neg_sim[neg_keep]
        pos_sim = pos_sim[pos_keep]
        if current_neg_w is not None:
            current_neg_w = current_neg_w[neg_keep]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        pos_term = torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))) / alpha
        if current_neg_w is None:
            neg_exp = torch.exp(beta * (neg_sim - lam))
        else:
            neg_exp = current_neg_w * torch.exp(beta * (neg_sim - lam))
        neg_term = torch.log(1.0 + torch.sum(neg_exp)) / beta

        loss_list.append(pos_term + neg_term)
        set_size_list.append(torch.tensor(pos_sim.numel() + neg_sim.numel(), dtype=sim_mat.dtype, device=device))

    return torch.stack(loss_list), torch.stack(set_size_list)


def _cross_modal_distance_weights(
    image_slice_labels: torch.Tensor,
    prompt_slice_labels: torch.Tensor,
    scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    matches = (image_slice_labels.unsqueeze(1) == prompt_slice_labels.unsqueeze(0)).float()
    continuous_matches = torch.cumprod(matches, dim=2)
    shared_depth = continuous_matches.sum(dim=2)
    tree_dist = (float(image_slice_labels.size(1)) - shared_depth).pow(dist_pow)
    return torch.pow(float(scale), tree_dist - 1.0)


def cross_modal_multi_similarity_loss(
    image_features: torch.Tensor,
    prompt_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1)
    prompt_features = F.normalize(prompt_features.float(), dim=-1)
    sim_mat = torch.clamp(image_features @ prompt_features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    _, all_labels = torch.unique(
        torch.cat([image_path_labels, prompt_path_labels], dim=0),
        dim=0,
        return_inverse=True,
    )
    image_labels = all_labels[: image_path_labels.size(0)]
    prompt_labels = all_labels[image_path_labels.size(0):]
    losses, _ = _cross_modal_ms_loss_level(
        sim_mat,
        image_labels,
        prompt_labels,
        alpha=alpha,
        beta=beta,
        lam=lam,
        mining_margin=mining_margin,
    )
    return losses.mean()


def cross_modal_hierarchical_ms_min_loss(
    image_features: torch.Tensor,
    prompt_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_path_labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    use_distance_weights: bool = False,
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> torch.Tensor:
    if minimum_mode not in {"batch", "sample"}:
        raise ValueError(f"Unsupported minimum_mode: {minimum_mode}")

    image_features = F.normalize(image_features.float(), dim=-1)
    prompt_features = F.normalize(prompt_features.float(), dim=-1)
    sim_mat = torch.clamp(image_features @ prompt_features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    num_levels = int(image_path_labels.size(1))
    device = image_features.device

    level_losses = []
    for level in range(num_levels):
        image_path = image_path_labels[:, : level + 1]
        prompt_path = prompt_path_labels[:, : level + 1]
        _, all_labels = torch.unique(
            torch.cat([image_path, prompt_path], dim=0),
            dim=0,
            return_inverse=True,
        )
        image_labels = all_labels[: image_path.size(0)]
        prompt_labels = all_labels[image_path.size(0):]
        neg_weights = None
        if use_distance_weights:
            neg_weights = _cross_modal_distance_weights(
                image_path,
                prompt_path,
                scale=dist_scale,
                dist_pow=dist_pow,
            )

        losses_l, _ = _cross_modal_ms_loss_level(
            sim_mat,
            image_labels,
            prompt_labels,
            neg_weights=neg_weights,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
        level_losses.append(losses_l)

    total_loss = torch.tensor(0.0, dtype=image_features.dtype, device=device)
    cur_min = None
    for rev_level in range(num_levels - 1, -1, -1):
        base_loss_l = level_losses[rev_level]
        if cur_min is None:
            cur_loss = base_loss_l
        else:
            cur_loss = torch.min(base_loss_l, cur_min)
        cur_min = torch.min(cur_loss) if minimum_mode == "batch" else cur_loss
        k_l = torch.exp(torch.tensor(1.0 / float(rev_level + 1), dtype=image_features.dtype, device=device))
        total_loss = total_loss + (k_l * cur_loss.mean()) / float(num_levels)
    return total_loss


def prompt_metric_loss(
    image_features: torch.Tensor,
    prompt_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_path_labels: torch.Tensor,
    loss_name: str,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    image_path_labels = image_path_labels.to(image_features.device)
    prompt_path_labels = prompt_path_labels.to(image_features.device)
    loss_name = loss_name.lower()

    if loss_name == "ms":
        loss = cross_modal_multi_similarity_loss(
            image_features,
            prompt_features,
            image_path_labels,
            prompt_path_labels,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
    elif loss_name in {"hims_min", "himsmin"}:
        loss = cross_modal_hierarchical_ms_min_loss(
            image_features,
            prompt_features,
            image_path_labels,
            prompt_path_labels,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
            minimum_mode=minimum_mode,
            use_distance_weights=False,
        )
    elif loss_name in {"weihims", "hims_min_wei"}:
        loss = cross_modal_hierarchical_ms_min_loss(
            image_features,
            prompt_features,
            image_path_labels,
            prompt_path_labels,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
            minimum_mode=minimum_mode,
            use_distance_weights=True,
            dist_scale=dist_scale,
            dist_pow=dist_pow,
        )
    else:
        raise ValueError(f"Unsupported prompt metric loss: {loss_name}")

    return loss, {
        "pair_mode": "cross_modal_image_prompt_only",
        "metric_loss": float(loss.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


def depthwise_prompt_metric_loss(
    image_features: torch.Tensor,
    prompt_features_by_depth: dict[int, torch.Tensor],
    image_node_labels_by_depth: torch.Tensor,
    prompt_node_labels_by_depth: dict[int, torch.Tensor],
    prompt_path_labels_by_depth: dict[int, torch.Tensor],
    loss_name: str,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    if minimum_mode not in {"batch", "sample"}:
        raise ValueError(f"Unsupported minimum_mode: {minimum_mode}")

    loss_name = loss_name.lower()
    if loss_name not in {"ms", "hims_min", "himsmin", "weihims", "hims_min_wei"}:
        raise ValueError(f"Unsupported depthwise prompt metric loss: {loss_name}")

    image_features = F.normalize(image_features.float(), dim=-1)
    image_node_labels_by_depth = image_node_labels_by_depth.to(image_features.device)
    use_distance_weights = loss_name in {"weihims", "hims_min_wei"}
    num_images = int(image_features.shape[0])
    device = image_features.device

    level_losses = {}
    valid_counts = {}
    for depth in sorted(prompt_features_by_depth):
        if depth <= 0:
            continue
        prompt_features = prompt_features_by_depth[depth]
        if prompt_features.numel() == 0:
            continue

        image_labels = image_node_labels_by_depth[:, depth]
        valid = image_labels >= 0
        if not bool(valid.any()):
            continue

        prompt_labels = prompt_node_labels_by_depth[depth].to(device)
        prompt_paths = prompt_path_labels_by_depth[depth].to(device)
        valid_image_features = image_features[valid]
        valid_image_labels = image_labels[valid]
        prompt_features = F.normalize(prompt_features.float(), dim=-1)
        sim_mat = torch.clamp(valid_image_features @ prompt_features.t(), min=-1.0 + 1e-6, max=1.0 - 1e-6)

        neg_weights = None
        if use_distance_weights:
            image_paths = image_node_labels_by_depth[valid, : depth + 1]
            prompt_paths = prompt_paths[:, : depth + 1]
            neg_weights = _cross_modal_distance_weights(
                image_paths,
                prompt_paths,
                scale=dist_scale,
                dist_pow=dist_pow,
            )

        losses_l, _ = _cross_modal_ms_loss_level(
            sim_mat,
            valid_image_labels,
            prompt_labels,
            neg_weights=neg_weights,
            alpha=alpha,
            beta=beta,
            lam=lam,
            mining_margin=mining_margin,
        )
        full_losses = torch.full((num_images,), float("inf"), dtype=losses_l.dtype, device=device)
        full_losses[valid] = losses_l
        level_losses[depth] = full_losses
        valid_counts[depth] = int(valid.sum().detach().cpu())

    if not level_losses:
        zero = image_features.sum() * 0.0
        return zero, {
            "pair_mode": "depthwise_global_image_prompt",
            "metric_loss": 0.0,
            "loss": 0.0,
            "valid_counts": valid_counts,
        }

    if loss_name == "ms":
        loss_terms = []
        for full_losses in level_losses.values():
            finite = torch.isfinite(full_losses)
            if bool(finite.any()):
                loss_terms.append(full_losses[finite].mean())
        loss = torch.stack(loss_terms).mean() if loss_terms else image_features.sum() * 0.0
    else:
        cur_min = torch.full((num_images,), float("inf"), dtype=image_features.dtype, device=device)
        loss_terms = []
        max_depth = max(level_losses)
        for depth in range(max_depth, 0, -1):
            if depth not in level_losses:
                continue
            base_loss = level_losses[depth]
            finite = torch.isfinite(base_loss)
            if not bool(finite.any()):
                continue
            if minimum_mode == "batch":
                depth_min = torch.min(base_loss[finite])
                cur_min[finite] = torch.minimum(base_loss[finite], depth_min.expand_as(base_loss[finite]))
            else:
                cur_min[finite] = torch.minimum(base_loss[finite], cur_min[finite])
            k_l = torch.exp(torch.tensor(1.0 / float(depth + 1), dtype=image_features.dtype, device=device))
            loss_terms.append(k_l * cur_min[finite].mean())
        loss = torch.stack(loss_terms).mean() if loss_terms else image_features.sum() * 0.0

    return loss, {
        "pair_mode": "depthwise_global_image_prompt",
        "metric_loss": float(loss.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "valid_counts": valid_counts,
    }


def dual_weihims_positive_loss(
    image_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_features_by_depth: dict[int, torch.Tensor],
    prompt_node_labels_by_depth: dict[int, torch.Tensor],
    prompt_path_labels_by_depth: dict[int, torch.Tensor],
    *,
    image_weight: float = 1.0,
    alignment_weight: float = 1.0,
    alpha: float = 2.0,
    beta: float = 50.0,
    lam: float = 0.5,
    mining_margin: float = 0.1,
    minimum_mode: str = "sample",
    dist_scale: float = 2.0,
    dist_pow: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Image-image WeiHiMS plus image-prompt WeiHiMS, without CE.

    The image term shapes the Vision-LoRA representation according to the
    label tree.  The alignment term places learned positive node prompts in
    that same hierarchy-aware space.  No path/classification CE is evaluated
    or added here.
    """
    if float(image_weight) < 0.0 or float(alignment_weight) < 0.0:
        raise ValueError("Dual WeiHiMS weights must be non-negative")
    if float(image_weight) == 0.0 and float(alignment_weight) == 0.0:
        raise ValueError("At least one Dual WeiHiMS weight must be positive")

    image_loss = hierarchical_ms_min_loss(
        image_features,
        image_path_labels,
        alpha=alpha,
        beta=beta,
        lam=lam,
        mining_margin=mining_margin,
        minimum_mode=minimum_mode,
        use_distance_weights=True,
        dist_scale=dist_scale,
        dist_pow=dist_pow,
    )
    alignment_loss, alignment_stats = depthwise_prompt_metric_loss(
        image_features,
        prompt_features_by_depth,
        image_path_labels,
        prompt_node_labels_by_depth,
        prompt_path_labels_by_depth,
        loss_name="weihims",
        alpha=alpha,
        beta=beta,
        lam=lam,
        mining_margin=mining_margin,
        minimum_mode=minimum_mode,
        dist_scale=dist_scale,
        dist_pow=dist_pow,
    )
    total = float(image_weight) * image_loss + float(alignment_weight) * alignment_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "image_weihims_loss": float(image_loss.detach().cpu()),
        "image_prompt_weihims_loss": float(alignment_loss.detach().cpu()),
        "path_ce_loss": 0.0,
        "pair_mode": "image_image_plus_depthwise_image_prompt",
        "alignment_valid_counts": alignment_stats["valid_counts"],
    }


def unknown_ce_loss(
    image_features: torch.Tensor,
    candidate_features: torch.Tensor,
    target_indices: torch.Tensor,
    tau: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    return positive_ce_loss(image_features, candidate_features, target_indices, tau=tau)


def unknown_regularization(
    unknown_feature: torch.Tensor,
    parent_feature: torch.Tensor,
    child_features: torch.Tensor,
    lambda_anchor: float = 0.1,
    lambda_child_sep: float = 0.1,
    lambda_prototype_diversity: float = 0.0,
    prototype_diversity_margin: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    if unknown_feature.ndim == 1:
        unknown_feature = unknown_feature.unsqueeze(0)
    if unknown_feature.ndim != 2:
        raise ValueError(
            "unknown_feature must have shape [dim] or [num_prototypes, dim]"
        )
    unknown_feature = F.normalize(unknown_feature.float(), dim=-1)
    parent_feature = F.normalize(parent_feature.float(), dim=-1)
    child_features = F.normalize(child_features.float(), dim=-1)

    anchor_loss = -torch.mean(unknown_feature @ parent_feature)
    child_sep_loss = torch.mean(child_features @ unknown_feature.t())
    if unknown_feature.shape[0] > 1:
        similarities = unknown_feature @ unknown_feature.t()
        off_diagonal = ~torch.eye(
            unknown_feature.shape[0],
            dtype=torch.bool,
            device=unknown_feature.device,
        )
        diversity_loss = F.relu(
            similarities[off_diagonal] - float(prototype_diversity_margin)
        ).mean()
    else:
        diversity_loss = unknown_feature.new_zeros(())
    total = (
        float(lambda_anchor) * anchor_loss
        + float(lambda_child_sep) * child_sep_loss
        + float(lambda_prototype_diversity) * diversity_loss
    )
    return total, {
        "anchor_loss": float(anchor_loss.detach().cpu()),
        "child_sep_loss": float(child_sep_loss.detach().cpu()),
        "prototype_diversity_loss": float(diversity_loss.detach().cpu()),
        "regularizer": float(total.detach().cpu()),
    }
