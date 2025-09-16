from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

def get_model(cfg, nc, weights=None, verbose=False):
    """Return YOLO Detection Model."""
    model = DetectionModel(cfg, nc=nc, verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)
    return model
