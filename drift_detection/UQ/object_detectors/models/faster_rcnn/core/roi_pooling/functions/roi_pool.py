from torchvision.ops import roi_pool


class RoIPoolFunction(object):
    """Compatibility callable for the original extension-backed API."""

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def __call__(self, features, rois):
        return self.forward(features, rois)

    def forward(self, features, rois):
        return roi_pool(
            features,
            rois,
            output_size=(self.pooled_height, self.pooled_width),
            spatial_scale=self.spatial_scale,
        )
