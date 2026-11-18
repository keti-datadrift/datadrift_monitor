from torch.nn.functional import avg_pool2d, max_pool2d
from torch.nn.modules.module import Module
from torchvision.ops import roi_align


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_align(
            features,
            rois,
            output_size=(self.aligned_height, self.aligned_width),
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1,
            aligned=False,
        )


class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = roi_align(
            features,
            rois,
            output_size=(self.aligned_height + 1, self.aligned_width + 1),
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1,
            aligned=False,
        )
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = roi_align(
            features,
            rois,
            output_size=(self.aligned_height + 1, self.aligned_width + 1),
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1,
            aligned=False,
        )
        return max_pool2d(x, kernel_size=2, stride=1)
