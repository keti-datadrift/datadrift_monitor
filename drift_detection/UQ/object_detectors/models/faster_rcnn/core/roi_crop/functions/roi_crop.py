import torch.nn.functional as F


class RoICropFunction(object):
    """Compatibility wrapper using torch.grid_sample instead of the old extension."""

    def __call__(self, input1, input2):
        return self.forward(input1, input2)

    def forward(self, input1, input2):
        return F.grid_sample(input1, input2, align_corners=False)
