import torch
import torch.nn.functional as F

def get_tree_distance_matrix(hi_labels, num_hi):
    B, L = hi_labels.size()
    matches = (hi_labels.unsqueeze(1) == hi_labels.unsqueeze(0)).float()
    continuous_matches = torch.cumprod(matches, dim=2)
    shared_depth = continuous_matches.sum(dim=2)
    tree_dist = float(L) - shared_depth
    return tree_dist

def weighted_ms_loss_mined(sim_mat, tree_dist, alpha=2.0, beta=50.0, lam=0.5, dist_scale=1.5, mining_margin=0.1):
    device = sim_mat.device
    batch_size = sim_mat.size(0)
    loss_list = []
    
    diag_mask = torch.eye(batch_size, device=device).bool()

    for i in range(batch_size):
        mask_pos = (tree_dist[i] == 0) & (~diag_mask[i])
        pos_sim = sim_mat[i][mask_pos]

        mask_neg = tree_dist[i] > 0
        neg_sim = sim_mat[i][mask_neg]
        neg_dist = tree_dist[i][mask_neg]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            continue

        hardest_pos_sim = torch.min(pos_sim)
        hardest_neg_sim = torch.max(neg_sim)

        mining_idx_neg = (neg_sim + mining_margin) > hardest_pos_sim
        neg_sim_mined = neg_sim[mining_idx_neg]
        neg_dist_mined = neg_dist[mining_idx_neg]

        mining_idx_pos = (pos_sim - mining_margin) < hardest_neg_sim
        pos_sim_mined = pos_sim[mining_idx_pos]

        if pos_sim_mined.numel() == 0 or neg_sim_mined.numel() == 0:
            continue

        pos_term = (
            torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim_mined - lam)))) / alpha
        )

        neg_weights = torch.pow(dist_scale, neg_dist_mined)
        
        neg_term_weighted = torch.sum(
            neg_weights * torch.exp(beta * (neg_sim_mined - lam))
        )
        neg_term = torch.log(1.0 + neg_term_weighted) / beta

        loss_list.append(pos_term + neg_term)

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(loss_list).mean()

def ms_wei_loss(
    features,
    hi_labels,
    batch_size,
    num_hi,
    alpha=2.0,
    beta=50.0,
    lam=0.5,
    dist_scale=2.0,
    mining_margin=0.1,
):
    sim_mat = torch.matmul(features, features.t())
    sim_mat = torch.clamp(sim_mat, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    tree_dist_mat = get_tree_distance_matrix(hi_labels, num_hi)

    total_loss = weighted_ms_loss_mined(
        sim_mat,
        tree_dist_mat,
        alpha=alpha,
        beta=beta,
        lam=lam,
        dist_scale=dist_scale,
        mining_margin=mining_margin
    )

    return total_loss