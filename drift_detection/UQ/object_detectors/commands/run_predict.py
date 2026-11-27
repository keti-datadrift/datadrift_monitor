import torch

from commands.predict.registry import resolve_predict_runner


def _active_uncertainty(config):
    raw = config.get("output", {}).get("uncertainty", "gt")
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError("run_predict expects exactly one output.uncertainty value.")
        raw = raw[0]
    value = str(raw).strip().lower()
    return value or "gt"


def run_predict(config, run_dir):
    uncertainty = _active_uncertainty(config)
    model_type = str(config.get("model", {}).get("type", "yolov5")).strip().lower()
    device = str(config.get("model", {}).get("device", "cuda")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] device={device}")
    runner = resolve_predict_runner(uncertainty=uncertainty, model_type=model_type)
    runner(config, run_dir)


__all__ = ["run_predict"]
