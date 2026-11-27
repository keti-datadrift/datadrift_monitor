import importlib


_MODEL_ALIASES = {
    "yolo": "yolov5",
    "yolov5": "yolov5",
    "yolov10": "yolov10",
    "yolo_v10": "yolov10",
    "faster_rcnn": "faster_rcnn",
    "faster-rcnn": "faster_rcnn",
    "frcnn": "faster_rcnn",
    "fcos": "fcos",
}


def normalize_train_model_type(model_type: str) -> str:
    key = str(model_type or "yolov5").strip().lower()
    model = _MODEL_ALIASES.get(key)
    if model is None:
        raise ValueError(f"Unsupported object detector model type for train: {model_type}")
    return model


def resolve_train_runner(model_type: str = "yolov5"):
    model = normalize_train_model_type(model_type)
    module = importlib.import_module(f"commands.train.{model}")
    return getattr(module, "run_train")


__all__ = ["normalize_train_model_type", "resolve_train_runner"]
