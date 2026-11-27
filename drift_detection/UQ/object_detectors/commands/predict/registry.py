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

_UNCERTAINTY_MODULES = {
    "gt": ("gt", "run_tp_csv"),
    "score": ("score", "run_score_csv"),
    "class_probability": ("class_probability", "run_class_probability_csv"),
    "entropy": ("entropy", "run_entropy_csv"),
    "energy": ("energy", "run_energy_csv"),
    "deterministic": ("deterministic", "run_deterministic_uncertainties_csv"),
    "mc_dropout": ("mc_dropout", "run_mc_dropout_csv"),
    "ensemble": ("ensemble", "run_ensemble_csv"),
    "meta_detect": ("meta_detect", "run_meta_detect_csv"),
    "layer_grad": ("layer_grad", "run_layer_grad_csv"),
    "null_detect": ("null_detect", "run_null_detect_csv"),
}

_SUPPORTED = {
    "yolov5": set(_UNCERTAINTY_MODULES),
    "faster_rcnn": {
        "gt",
        "score",
        "class_probability",
        "entropy",
        "energy",
        "deterministic",
        "mc_dropout",
        "ensemble",
        "meta_detect",
        "layer_grad",
        "null_detect",
    },
    "fcos": {
        "gt",
        "score",
        "class_probability",
        "entropy",
        "energy",
        "deterministic",
        "mc_dropout",
        "ensemble",
        "meta_detect",
        "layer_grad",
        "null_detect",
    },
    "yolov10": {
        "gt",
        "score",
        "class_probability",
        "entropy",
        "energy",
        "deterministic",
        "mc_dropout",
        "ensemble",
        "meta_detect",
        "layer_grad",
        "null_detect",
    },
}


def normalize_predict_model_type(model_type: str) -> str:
    key = str(model_type or "yolov5").strip().lower()
    model = _MODEL_ALIASES.get(key)
    if model is None:
        raise ValueError(f"Unsupported object detector model type for predict: {model_type}")
    return model


def resolve_predict_runner(uncertainty: str, model_type: str = "yolov5"):
    model = normalize_predict_model_type(model_type)
    u = str(uncertainty).strip().lower()
    if u not in _SUPPORTED[model]:
        raise NotImplementedError(f"{model} does not support uncertainty='{uncertainty}' yet.")

    module_name, function_name = _UNCERTAINTY_MODULES[u]
    module = importlib.import_module(f"commands.predict.{model}.{module_name}")
    return getattr(module, function_name)


__all__ = ["normalize_predict_model_type", "resolve_predict_runner"]
