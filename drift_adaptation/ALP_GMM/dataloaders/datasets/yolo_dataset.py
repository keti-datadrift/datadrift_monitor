from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_yolo_dataset

def build_dataset(args, img_path, mode="train", batch=None, data=None, model=None):
    """
    Build YOLO Dataset (from DetectionTrainer.build_dataset)
    """
    gs = max(int(de_parallel(model).stride.max() if model else 0), 32)
    return build_yolo_dataset(args, img_path, batch, data, mode=mode, rect=(mode == "val"), stride=gs)
