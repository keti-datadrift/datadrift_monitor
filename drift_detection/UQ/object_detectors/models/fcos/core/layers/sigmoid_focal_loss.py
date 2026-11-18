import torch
from torch import nn


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0] if isinstance(gamma, (list, tuple)) else gamma
    alpha = alpha[0] if isinstance(alpha, (list, tuple)) else alpha
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits).clamp(min=1e-6, max=1.0 - 1e-6)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return (
        -(t == class_range).float() * term1 * alpha
        - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)
    )


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        return sigmoid_focal_loss_cpu(logits, targets, self.gamma, self.alpha).sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
