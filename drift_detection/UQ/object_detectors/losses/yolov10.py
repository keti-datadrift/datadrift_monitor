from models.yolov10.core import V10DetectLoss


def build_yolov10_loss(model, _config=None):
    return V10DetectLoss(model)


__all__ = ["build_yolov10_loss"]
