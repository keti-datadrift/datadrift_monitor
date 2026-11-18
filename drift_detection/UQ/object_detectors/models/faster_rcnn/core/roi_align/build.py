"""Legacy extension build script intentionally disabled.

Runtime ROI Align is provided by torchvision.ops through modules/roi_align.py.
"""


def build_extension():
    raise RuntimeError("Legacy ROI Align extension build is disabled; use torchvision.ops.roi_align.")


if __name__ == "__main__":
    build_extension()
