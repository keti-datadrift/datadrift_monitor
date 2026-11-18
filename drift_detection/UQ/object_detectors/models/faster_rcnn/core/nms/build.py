"""Legacy extension build script intentionally disabled.

Runtime NMS is provided by torchvision.ops through nms_wrapper.py.
"""


def build_extension():
    raise RuntimeError("Legacy NMS extension build is disabled; use torchvision.ops.nms.")


if __name__ == "__main__":
    build_extension()
