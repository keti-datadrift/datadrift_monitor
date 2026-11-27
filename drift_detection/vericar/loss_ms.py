import torch
import torch.nn.functional as F

def MS_loss(features, hi_labels, alpha=2.0, beta=50.0, lam=0.5):
    batch_size = features.size(0)
    
    sim_mat = torch.matmul(features, features.t())
    sim_mat = torch.clamp(sim_mat, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    label_comparison = hi_labels.unsqueeze(1) == hi_labels.unsqueeze(0)
    mask_pos = label_comparison.all(dim=2)
    mask_neg = ~mask_pos

    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    mask_pos[diag_mask] = False
    mask_neg[diag_mask] = False

    loss = []

    for i in range(batch_size):
        pos_sim_i = sim_mat[i][mask_pos[i]]
        neg_sim_i = sim_mat[i][mask_neg[i]]

        if pos_sim_i.numel() == 0 or neg_sim_i.numel() == 0:
            continue

        hardest_pos_sim = torch.min(pos_sim_i)
        hardest_neg_sim = torch.max(neg_sim_i)
        
        neg_sim_i = neg_sim_i[neg_sim_i + 0.1 > hardest_pos_sim] 
        pos_sim_i = pos_sim_i[pos_sim_i - 0.1 < hardest_neg_sim]
        
        if pos_sim_i.numel() == 0 or neg_sim_i.numel() == 0:
            continue

        pos_term = (1.0 / alpha) * torch.log(
            1.0 + torch.sum(torch.exp(-alpha * (pos_sim_i - lam)))
        )
        
        neg_term = (1.0 / beta) * torch.log(
            1.0 + torch.sum(torch.exp(beta * (neg_sim_i - lam)))
        )

        loss.append(pos_term + neg_term)

    if len(loss) == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    return torch.stack(loss).mean()