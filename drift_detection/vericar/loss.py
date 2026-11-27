import torch
import torch.nn.functional as F

def ms_loss_level(sim_mat, labels, alpha=2.0, beta=50.0, lam=0.5, mining_margin=0.1):
    device = sim_mat.device
    batch_size = sim_mat.size(0)
    labels = labels.view(-1)

    loss_list = []
    set_size_list = []

    for i in range(batch_size):
        label_i = labels[i]
        
        mask_pos = labels == label_i
        mask_pos[i] = False
        mask_neg = labels != label_i

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]

        if len(pos_sim) == 0 or len(neg_sim) == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        hardest_pos_sim = torch.min(pos_sim)
        hardest_neg_sim = torch.max(neg_sim)

        mining_idx_neg = (neg_sim + mining_margin) > hardest_pos_sim
        neg_sim = neg_sim[mining_idx_neg]

        mining_idx_pos = (pos_sim - mining_margin) < hardest_neg_sim
        pos_sim = pos_sim[mining_idx_pos]
        
        if len(pos_sim) == 0 or len(neg_sim) == 0:
            loss_list.append(torch.tensor(0.0, device=device))
            set_size_list.append(torch.tensor(1.0, device=device))
            continue

        if len(pos_sim) > 0:
            pos_term = (
                torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))) / alpha
            )
        else:
            pos_term = torch.tensor(0.0, device=device)

        if len(neg_sim) > 0:
            neg_term = (
                torch.log(1.0 + torch.sum(torch.exp(beta * (neg_sim - lam)))) / beta
            )
        else:
            neg_term = torch.tensor(0.0, device=device)

        loss_list.append(pos_term + neg_term)
        set_size = torch.tensor(len(pos_sim) + len(neg_sim), dtype=pos_sim.dtype, device=device)
        set_size_list.append(set_size)

    return torch.stack(loss_list), torch.stack(set_size_list)


def HiMS_min_loss(
    features, hi_labels, batch_size, num_hi, alpha=2.0, beta=50.0, lam=0.5, mining_margin=0.1
):
    sim_mat = torch.matmul(features, features.t())
    
    sim_mat = torch.clamp(sim_mat, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    B = batch_size
    L = num_hi
    device = features.device

    level_losses = []
    level_set_sizes = []

    for level in range(L):
        current_hierarchy_path = hi_labels[:, 0 : level + 1]

        _, labels_level = torch.unique(
            current_hierarchy_path, dim=0, return_inverse=True
        )

        loss_l, set_size_l = ms_loss_level(sim_mat, labels_level, alpha=alpha, beta=beta, lam=lam, mining_margin=mining_margin)
        level_losses.append(loss_l)
        level_set_sizes.append(set_size_l)

    total_loss = 0.0
    L_float = float(L)

    cur_min = None

    for rev_level in range(L - 1, -1, -1):
        l = rev_level
        base_loss_l = level_losses[l]
        set_size_base_l = level_set_sizes[l]

        if cur_min is None:
            cur_loss = base_loss_l
        else:
            cur_loss = torch.min(base_loss_l, cur_min)

        cur_min = torch.min(cur_loss)

        k_l = torch.exp(torch.tensor(1.0 / (l + 1.0), device=device))
        level_loss = cur_loss # / set_size_base_l
        total_loss = total_loss + (k_l * level_loss.mean()) / L_float

    return total_loss