"""Legacy extension build script intentionally disabled.

Runtime ROI Pooling is provided by torchvision.ops through modules/roi_pool.py.
"""


def build_extension():
    raise RuntimeError("Legacy ROI Pooling extension build is disabled; use torchvision.ops.roi_pool.")


if __name__ == "__main__":
    build_extension()
