from torchvision.ops import roi_align


class RoIAlignFunction(object):
    """Compatibility callable for the original extension-backed API."""

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def __call__(self, features, rois):
        return self.forward(features, rois)

    def forward(self, features, rois):
        return roi_align(
            features,
            rois,
            output_size=(self.aligned_height, self.aligned_width),
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1,
            aligned=False,
        )
