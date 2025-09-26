import os
import re
import numpy as np
import torch
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ultralytics.utils.plotting import plot_images, output_to_target
from ultralytics.utils import ops

def get_unique_dir(base_dir, exp_name):
    """Create unique directory for experiments."""
    base_path = Path(base_dir)
    target_dir = base_path / exp_name
    idx = 2
    while target_dir.exists():
        target_dir = base_path / f"{exp_name}{idx}"
        idx += 1
    return target_dir

def load_previous_scalars(log_dir):
    """Load scalar data from previous TensorBoard logs."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        scalars[tag] = event_acc.Scalars(tag)
    return scalars

def copy_scalars_to_writer(writer, scalars):
    """Copy scalar data to new TensorBoard writer."""
    for tag, events in scalars.items():
        for e in events:
            writer.add_scalar(tag, e.value, e.step)

def plot_samples(batch, epoch, mode, idx, model, args, img_dir):
    """Plot images with ground truth and predictions (if validation mode)."""
    fname_gt = img_dir / f"{mode}_epoch{epoch+1}_batch{idx}.jpg"
    # print(batch["img"].shape, batch["batch_idx"], batch["cls"].squeeze(-1), batch["bboxes"], batch["im_file"])
    plot_images(
        images=batch["img"],
        batch_idx=batch["batch_idx"],
        cls=batch["cls"].squeeze(-1),
        bboxes=batch["bboxes"],
        paths=batch["im_file"],
        fname=fname_gt,
        names=model.names
    )

    if not mode == "train":
        fname_pred = img_dir / f"{mode}_epoch{epoch+1}_batch{idx}_pred.jpg"
        model.eval()
        with torch.no_grad():
            imgs = batch["img"]
            raw_preds = model(imgs)

            preds_nms = ops.non_max_suppression(
                raw_preds,
                conf_thres=getattr(args, "conf", None) or 0.25,
                iou_thres=getattr(args, "iou", None) or 0.45,
                max_det=getattr(args, "max_det", None) or 300
            )

        plot_images(
            imgs,
            *output_to_target(preds_nms, max_det=getattr(args, "max_det", 300)),
            paths=batch["im_file"],
            fname=fname_pred,
            names=model.names
        )

def log_results_csv(csv_file, epoch, train_loss, val_loss):
    """Append training and validation loss to CSV file."""
    if not csv_file.exists():
        with open(csv_file, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
    with open(csv_file, "a") as f:
        f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")

def match_predictions(pred_classes, true_classes, iou, iouv, use_scipy=False):
    """Matches predictions to ground truth using IoU thresholds for multiple levels."""
    # N predictions x T thresholds
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)

    # Match classes
    correct_class = true_classes[:, None] == pred_classes  # (M, N)
    iou = iou * correct_class  # Zero out non-matching classes
    iou = iou.cpu().numpy()

    for i, threshold in enumerate(iouv.cpu().tolist()):
        if use_scipy:
            # Hungarian algorithm for optimal matching
            import scipy.optimize

            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            # Greedy matching
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                # Sort by IoU descending
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                # Remove duplicate detections and labels
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

def calculate_brightness_tensor(img_tensor):
    """
    img_tensor: (B, C, H, W) normalized [0,1]
    return: (B,) brightness values
    """
    # Convert to grayscale approximation: Y = 0.299R + 0.587G + 0.114B
    if img_tensor.shape[1] == 3:
        r, g, b = img_tensor[:, 0], img_tensor[:, 1], img_tensor[:, 2]
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        return brightness.mean(dim=[1, 2])
    else:
        # Single channel image
        return img_tensor.mean(dim=[1, 2, 3])

def proportional_choice(v, eps=0.0):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.random.choice(len(probas), p=probas)
    
def norm_path(p: str) -> str:
    """Normalize a file path (case + separators)."""
    return os.path.normcase(os.path.normpath(p))

def extract_image_paths_from_batch(batch):
    """
    Try to extract image paths from a YOLO batch dict.
    """
    for k in ("im_file", "im_files", "path", "paths"):
        if k in batch:
            v = batch[k]
            if isinstance(v, (list, tuple)):
                return list(map(str, v))
            elif isinstance(v, str):
                return [v]
    return []

def get_versioned_run_dir(base_dir: str, exp_name: str) -> Path:
    """
    Create/return next version folder under base_dir/exp_name as v1, v2, ...
    Example: runs/EXP_A/v1, runs/EXP_A/v2, ...
    """
    root = Path(base_dir) / exp_name
    root.mkdir(parents=True, exist_ok=True)

    existing = []
    for d in root.iterdir():
        if d.is_dir():
            m = re.fullmatch(r"v(\d+)", d.name)
            if m:
                existing.append(int(m.group(1)))
    next_ver = (max(existing) + 1) if existing else 1
    return root / f"v{next_ver}"