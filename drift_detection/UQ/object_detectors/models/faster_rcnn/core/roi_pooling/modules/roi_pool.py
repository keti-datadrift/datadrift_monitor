from torch.nn.modules.module import Module
from torchvision.ops import roi_pool


class _RoIPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_pool(
            features,
            rois,
            output_size=(self.pooled_height, self.pooled_width),
            spatial_scale=self.spatial_scale,
        )
