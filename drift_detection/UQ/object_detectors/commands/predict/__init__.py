"""Predict runners organized by object detector model.

Model-specific entry points live under:
    commands.predict.yolov5
    commands.predict.faster_rcnn
    commands.predict.fcos
"""

from .registry import normalize_predict_model_type, resolve_predict_runner

__all__ = ["normalize_predict_model_type", "resolve_predict_runner"]

