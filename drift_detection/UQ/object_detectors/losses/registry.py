from losses.yolov10 import build_yolov10_loss
from losses.yolov5 import build_yolov5_loss


def build_loss(model_type, model, config=None):
    key = str(model_type or "").strip().lower().replace("-", "_")
    if key in {"yolo", "yolov5"}:
        return build_yolov5_loss(model, config)
    if key in {"yolov10", "yolo_v10"}:
        return build_yolov10_loss(model, config)
    raise ValueError(
        f"Unsupported external object-detector loss adapter: {model_type}. "
        "FCOS and Faster R-CNN use their native model forward loss dict."
    )


__all__ = ["build_loss"]
