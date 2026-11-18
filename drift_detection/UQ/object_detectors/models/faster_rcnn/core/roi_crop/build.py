"""Legacy extension build script intentionally disabled.

Runtime ROI Crop compatibility is provided by torch.nn.functional.grid_sample.
"""


def build_extension():
    raise RuntimeError("Legacy ROI Crop extension build is disabled; use torch.nn.functional.grid_sample.")


if __name__ == "__main__":
    build_extension()
