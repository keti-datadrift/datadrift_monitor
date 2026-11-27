from models.yolov5.core.utils.loss import ComputeLoss


class YoloV5Loss:
    def __init__(self, model):
        self.compute_loss = ComputeLoss(model)

    def __call__(self, predictions, targets):
        return self.compute_loss(predictions, targets)


def build_yolov5_loss(model, _config=None):
    return YoloV5Loss(model)


__all__ = ["YoloV5Loss", "build_yolov5_loss"]
