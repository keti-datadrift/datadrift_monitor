"""Stock PyCIL ``SimpleLinear`` classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PyCILSimpleLinear(nn.Module):
    """Exact classifier primitive from PyCIL ``convs/linears.py``."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        return {"logits": F.linear(inputs, self.weight, self.bias)}
