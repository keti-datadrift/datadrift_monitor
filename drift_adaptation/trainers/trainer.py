import csv
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloaders.yolo_dataloader import get_dataloader
from models.yolo_model import get_model
from ultralytics.utils.metrics import box_iou, ap_per_class
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.loss import v8DetectionLoss
from losses.yolo_custom_loss import SampleWiseDetectionLoss

from utils.trainer_utils import (
    load_previous_scalars,
    copy_scalars_to_writer,
    plot_samples,
    log_results_csv,
    match_predictions,
    norm_path,
    extract_image_paths_from_batch,
    get_versioned_run_dir
)


class YOLOTrainer:
    def __init__(self, args, data, custom_cfg=None):
        self.args = args
        self.data = data
        self.custom_cfg = custom_cfg or {}

        # Resume checkpoint path
        self.resume_path = self.custom_cfg.get("resume", None)

        # Custom config
        self.exp_name = self.custom_cfg["exp_name"]
        self.plot_period = self.custom_cfg["plot_period"]

        # Device setup
        self.device = select_device(self.args.device)

        # Experiment directories
        base_dir = getattr(args, "save_dir", "./runs")
        self.save_dir = get_versioned_run_dir(base_dir, self.exp_name)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.img_dir = self.save_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        # Save configs
        with open(self.save_dir / "args.yaml", "w") as f:
            yaml.safe_dump(vars(args), f)
        with open(self.save_dir / "custom.yaml", "w") as f:
            yaml.safe_dump(custom_cfg, f)

        # Checkpoint and logs
        self.last_ckpt = self.weights_dir / "last.pt"
        self.best_ckpt = self.weights_dir / "best.pt"
        self.csv_file = self.save_dir / "results.csv"

        # Components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        
        # pretrain csv path
        self.pretrain_csv_path = self.custom_cfg.get("pretrain_csv", None)
        self.pretrain_brisque_map = {}

    def _load_brisque_map(self):
        p = Path(self.pretrain_csv_path or "")
        if not p.is_file():
            LOGGER.warning(f"pretrain_csv not found: {p}")
            return
        try:
            df = pd.read_csv(p)
            if {"image_path", "brisque"} <= set(df.columns):
                self.pretrain_brisque_map = {norm_path(ip): float(b) for ip, b in zip(df["image_path"], df["brisque"]) if isinstance(ip, str)}
                LOGGER.info(f"Loaded BRISQUE map: {len(self.pretrain_brisque_map)} entries")
            else:
                LOGGER.warning("pretrain_csv missing required columns.")
        except Exception as e:
            LOGGER.warning(f"Failed to read pretrain_csv: {e}")
    
    def setup_model(self):
        LOGGER.info(f"Loading YOLO model: {self.args.model}")
        self.model = get_model(self.args.model, nc=self.data["nc"], weights=None)
        self.model.args = self.args
        self.model.to(self.device)
        self.criterion = v8DetectionLoss(self.model)
        self._load_brisque_map()
    
    def setup_dataloaders(self):
        self.train_loader = get_dataloader(self.data["train"], self.args.batch, self.args, self.data, self.model, mode="train")
        self.val_loader = get_dataloader(self.data["val"], self.args.batch, self.args, self.data, self.model, mode="val")

    def setup_optimizer(self):
        if self.args.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr0, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr0, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

    def setup_scheduler(self):
        lf = lambda x: max(1 - x / self.args.epochs, 0)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

    def preprocess_batch(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        batch["img"] = batch["img"].float() / 255.0
        return batch
  
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        loss_items_sum = None
        loss_names = ["box", "cls", "dfl"]
        batch_size = self.args.batch

        log_steps = self.custom_cfg["log_every_n_steps"]
        nb = len(self.val_loader)
        
        stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)
        
        history_records = []
        sample_loss_fn = SampleWiseDetectionLoss(self.model)

        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=nb, desc=f"[Val] Epoch {epoch+1}/{self.args.epochs}")
            for i, batch in pbar:
                step = epoch * nb + i
                batch = self.preprocess_batch(batch)
                
                loss, loss_items = self.model(batch)
                total_loss += (loss.item() / batch_size)
                if loss_items_sum is None:
                    loss_items_sum = loss_items.clone()
                else:
                    loss_items_sum += loss_items

                imgs = batch["img"]
                preds = self.model(imgs)
                preds_logits = preds if isinstance(preds, torch.Tensor) else preds[0]
                preds_nms = ops.non_max_suppression(preds_logits)
                
                if epoch == self.args.epochs - 1:
                    if isinstance(preds, (tuple, list)) and len(preds) == 2:
                        _, sample_losses = sample_loss_fn(preds, batch)
                    else:
                        raise ValueError("Unexpected preds structure for sample_loss_fn")
                    
                    im_paths = extract_image_paths_from_batch(batch)
                    if im_paths and len(im_paths) == len(sample_losses):
                        for pth, s_loss in zip(im_paths, sample_losses):
                            bq = self.pretrain_brisque_map.get(norm_path(pth), None)
                            if bq is not None:
                                history_records.append({"brisque": float(bq), "loss": float(s_loss)})
                            else:
                                LOGGER.debug(f"No BRISQUE for path: {pth}")
                    else:
                        LOGGER.debug("No image paths in batch or length mismatch; skip history record for this batch.")
                
                # Metrics
                for si, pred in enumerate(preds_nms):
                    gt_cls = batch["cls"][batch["batch_idx"] == si].squeeze(-1)
                    gt_boxes = batch["bboxes"][batch["batch_idx"] == si]

                    if len(gt_boxes):
                        gt_boxes = ops.xywh2xyxy(gt_boxes)
                        img_shape = imgs.shape[2:]
                        gt_boxes[:, [0, 2]] *= img_shape[1]
                        gt_boxes[:, [1, 3]] *= img_shape[0]
                    if len(pred):
                        predn = pred.clone()
                        iou = box_iou(gt_boxes, predn[:, :4]) if len(gt_boxes) else torch.zeros((0, len(pred)), device=self.device)
                        correct = match_predictions(predn[:, 5], gt_cls, iou, iouv)
                    else:
                        correct = torch.zeros((0, len(iouv)), dtype=torch.bool, device=self.device)

                    stats["tp"].append(correct)
                    stats["conf"].append(pred[:, 4] if len(pred) else torch.empty(0, device=self.device))
                    stats["pred_cls"].append(pred[:, 5] if len(pred) else torch.empty(0, device=self.device))
                    stats["target_cls"].append(gt_cls.to(self.device))
                
                # Display
                postfix = {"total": f"{(loss.item() / batch_size):.4f}"}
                for name, val in zip(loss_names, loss_items):
                    postfix[name] = f"{val.item():.4f}"
                pbar.set_postfix(postfix)

                # Log step
                if log_steps > 0 and (i + 1) % log_steps == 0:
                    self.writer.add_scalar("Val_step/total", loss.item() / batch_size, step)
                    for name, val in zip(loss_names, loss_items):
                        self.writer.add_scalar(f"Val_step/{name}", val.item(), step)

                # Visualization
                if self.plot_period > 0 and (epoch + 1) % self.plot_period == 0 and i < 3:
                    plot_samples(batch, epoch, mode="val", idx=i, model=self.model, args=self.args, img_dir=self.img_dir)

        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
        if len(stats["tp"]):
            _, _, p, r, _, ap, *_ = ap_per_class(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"])
            precision = float(p.mean())
            recall = float(r.mean())
            mAP50 = float(ap[:, 0].mean())
            mAP50_95 = float(ap.mean())
        else:
            precision = recall = mAP50 = mAP50_95 = 0.0
            
        self.writer.add_scalar("metrics/precision", precision, epoch)
        self.writer.add_scalar("metrics/recall", recall, epoch)
        self.writer.add_scalar("metrics/mAP50", mAP50, epoch)
        self.writer.add_scalar("metrics/mAP50-95", mAP50_95, epoch)
        
        avg_loss = total_loss / nb
        avg_loss_items = (loss_items_sum / nb).tolist() if loss_items_sum is not None else []

        self.writer.add_scalar("Val_epoch/total", avg_loss, epoch)
        for name, val in zip(loss_names, avg_loss_items):
            self.writer.add_scalar(f"Val_epoch/{name}", val, epoch)
        
        LOGGER.info(
            f"Loss: {avg_loss:.4f} (Box: {avg_loss_items[0]:.4f}, Cls: {avg_loss_items[1]:.4f}, DFL: {avg_loss_items[2]:.4f}) | "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, mAP@0.5: {mAP50:.4f}, mAP@0.5:0.95: {mAP50_95:.4f}"
        )
        
        if epoch == self.args.epochs - 1 and len(history_records) > 0:
            history_path = self.save_dir / "history0.csv"
            with open(history_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["brisque", "loss"])
                writer.writeheader()
                writer.writerows(history_records)
            LOGGER.info(f"history0.csv saved: {history_path}, {len(history_records)} samples")
        
        return avg_loss

    def train(self):
        self.setup_model()
        self.setup_dataloaders()
        self.setup_optimizer()
        self.setup_scheduler()

        # Resume variables
        best_val_loss = float("inf")
        no_improve_count = 0
        start_epoch = 0
        prev_scalars = {}

        if self.resume_path and Path(self.resume_path).exists():
            LOGGER.info(f"Resuming from checkpoint: {self.resume_path}")
            ckpt = torch.load(self.resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            no_improve_count = ckpt.get("no_improve_count", 0)
            start_epoch = ckpt.get("epoch", -1) + 1

            # Load previous logs
            old_log_dir = Path(self.resume_path).parent.parent / "logs"
            if old_log_dir.exists():
                prev_scalars = load_previous_scalars(str(old_log_dir))
                LOGGER.info(f"Loaded {len(prev_scalars)} tags from previous logs.")

            LOGGER.info(f"Resumed at epoch {start_epoch}, best_val_loss {best_val_loss:.4f}")

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))
        if prev_scalars:
            copy_scalars_to_writer(self.writer, prev_scalars)
            LOGGER.info(f"Copied previous logs to new writer.")

        LOGGER.info(f"Starting training from epoch {start_epoch} to {self.args.epochs}...")

        # Custom configs
        log_steps = self.custom_cfg.get("log_every_n_steps", 0)
        save_period = self.custom_cfg.get("save_period", 0)
        patience = self.custom_cfg.get("patience", 0)
        min_improve = self.custom_cfg.get("min_improve", 0.0)
        loss_names = ["box", "cls", "dfl"]

        # Training loop
        for epoch in range(start_epoch, self.args.epochs):
            self.model.train()
            total_loss = 0
            loss_items_sum = None
            nb = len(self.train_loader)
            batch_size = self.args.batch

            pbar = tqdm(enumerate(self.train_loader), total=nb, desc=f"[Train] Epoch {epoch+1}/{self.args.epochs}")
            for i, batch in pbar:
                step = epoch * nb + i
                batch = self.preprocess_batch(batch)
                loss, loss_items = self.model(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += (loss.item() / batch_size)
                if loss_items_sum is None:
                    loss_items_sum = loss_items.clone()
                else:
                    loss_items_sum += loss_items

                # Display
                postfix = {"total": f"{(loss.item() / batch_size):.4f}"}
                for name, val in zip(loss_names, loss_items):
                    postfix[name] = f"{val.item():.4f}"
                pbar.set_postfix(postfix)

                # Log steps
                if log_steps > 0 and (i + 1) % log_steps == 0:
                    self.writer.add_scalar("Train_step/total", loss.item() / batch_size, step)
                    for name, val in zip(loss_names, loss_items):
                        self.writer.add_scalar(f"Train_step/{name}", val.item(), step)

                # Visualization
                if self.plot_period > 0 and (epoch + 1) % self.plot_period == 0 and i < 3:
                    plot_samples(batch, epoch, mode="train", idx=i, model=self.model, args=self.args, img_dir=self.img_dir)

            avg_loss = total_loss / nb
            avg_loss_items = (loss_items_sum / nb).tolist() if loss_items_sum is not None else []

            self.writer.add_scalar("Train_epoch/total", avg_loss, epoch)
            for name, val in zip(loss_names, avg_loss_items):
                self.writer.add_scalar(f"Train_epoch/{name}", val, epoch)

            val_loss = self.validate(epoch)

            # Save checkpoint
            if save_period > 0 and (epoch + 1) % save_period == 0:
                epoch_ckpt = self.weights_dir / f"epoch{epoch+1}.pt"
                torch.save({
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "no_improve_count": no_improve_count,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict()
                }, epoch_ckpt)

            # Best checkpoint
            if (best_val_loss - val_loss) > min_improve:
                best_val_loss = val_loss
                no_improve_count = 0
                torch.save({
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "no_improve_count": no_improve_count,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict()
                }, self.best_ckpt)
            else:
                no_improve_count += 1

            # Last checkpoint
            torch.save({
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "no_improve_count": no_improve_count,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict()
            }, self.last_ckpt)

            # Log results
            log_results_csv(self.csv_file, epoch, avg_loss, val_loss)
            self.scheduler.step()

            # Early stopping
            if patience > 0 and no_improve_count >= patience:
                LOGGER.info(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                break

        self.writer.close()
