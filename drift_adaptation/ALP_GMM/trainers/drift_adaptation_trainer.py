import copy
import yaml
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture as GMM
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ultralytics.utils import LOGGER, ops
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import box_iou, ap_per_class

from dataloaders.datasets.yolo_custom_dataset import YoloCustomDataset, yolo_collate_fn
from losses.yolo_custom_loss import SampleWiseDetectionLoss
from models.yolo_model import get_model
from utils.trainer_utils import (
    plot_samples,
    match_predictions,
    proportional_choice,
    norm_path,
    extract_image_paths_from_batch,
    get_versioned_run_dir
)


class DriftAdaptTrainer:
    def __init__(self, args, custom_cfg, main_cfg):
        self.args = args
        self.custom_cfg = custom_cfg
        self.device = select_device(args.device)
        self.nc = custom_cfg["nc"]
        self.workers = custom_cfg["workers"]
        
        # Paths
        self.segments_csv = custom_cfg["segments_csv"]
        self.history0_csv = custom_cfg["history0_csv"]
        self.pretrained_path = custom_cfg["pretrained"]
        self.pretrain_csv = custom_cfg["pretrain_csv"]
        
        # Dirs
        base_dir = custom_cfg["save_dir"]
        exp_name = custom_cfg["exp_name"]
        self.save_dir = get_versioned_run_dir(base_dir, exp_name)   # e.g., runs/EXP_A/v3
        self.img_dir = self.save_dir / "images"
        self.weights_dir = self.save_dir / "weights"
        self.logs_dir = self.save_dir / "logs"
        for d in [self.img_dir, self.weights_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Save configs
        full_cfg = {
            "main": main_cfg,
            "trainer_args": vars(args),
            "trainer_custom": custom_cfg
        }
        with open(self.save_dir / "config.yaml", "w") as f:
            yaml.safe_dump(full_cfg, f)

        # Custom config
        self.log_every_n_steps = custom_cfg["log_every_n_steps"]
        self.plot_period = custom_cfg["plot_period"]
        self.save_period = custom_cfg["save_period"]
        self.temperature = custom_cfg["temperature"]
        self.subset_size = custom_cfg["subset_size"]
        self.use_pretrain_data = custom_cfg["use_pretrain_data"]
        self.inner_loops = custom_cfg["inner_loops"]
        self.potential_ks = custom_cfg["potential_ks"]
        self.gmm_fitness_fun = custom_cfg["gmm_fitness_fun"]
        self.sampling_method = custom_cfg["sampling_method"]
        
        # Checkpoint
        self.best_val_loss = float("inf")
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))
        
        # Components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.sample_loss_fn = None
        self.gmm = None
        self.brisque_map = {}
       
    def _load_brisque_map(self):
        """
        Build a dict: normalized(image_path) -> brisque (float) from pretrain.csv and segments.csv (if available).
        """
        m = {}

        def ingest(csv_path: str):
            if not csv_path:
                return
            p = Path(csv_path)
            if not p.is_file():
                LOGGER.warning(f"CSV not found: {p}")
                return
            try:
                df = pd.read_csv(p)
            except Exception as e:
                LOGGER.warning(f"Failed to read CSV: {p} ({e})")
                return
            if not {"image_path", "brisque"} <= set(df.columns):
                LOGGER.warning(f"CSV missing columns (image_path, brisque): {p}")
                return
            for ip, b in zip(df["image_path"], df["brisque"]):
                if isinstance(ip, str):
                    try:
                        m[norm_path(ip)] = float(b)
                    except Exception:
                        pass

        ingest(self.pretrain_csv)
        ingest(self.segments_csv)

        self.brisque_map = m
        LOGGER.info(f"Loaded BRISQUE lookup: {len(self.brisque_map)} entries")
    
    def setup_model(self):
        self.model = get_model(self.args.model, nc=self.nc, weights=None)
        self.model.args = self.args
        self.model.to(self.device)

        if self.pretrained_path and Path(self.pretrained_path).exists():
            LOGGER.info(f"Loading pretrained weights from {self.pretrained_path}")
            ckpt = torch.load(self.pretrained_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
        
        self.criterion = v8DetectionLoss(self.model)
        self.sample_loss_fn = SampleWiseDetectionLoss(self.model)
        self._load_brisque_map()
    
    def setup_optimizer(self):
        if self.args.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr0, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr0, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
    
    def preprocess_batch(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        return batch
    
    def build_loader_from_df(self, df):
        dataset = YoloCustomDataset(df, img_size=self.args.imgsz)
        return DataLoader(dataset, batch_size=self.args.batch, shuffle=True, num_workers=self.workers, collate_fn=yolo_collate_fn)
    
    def save_checkpoint(self, epoch, val_loss):
        # Save checkpoint
        if self.save_period > 0 and (epoch + 1) % self.save_period == 0:
            epoch_ckpt = self.weights_dir / f"epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }, epoch_ckpt)

        # Best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_ckpt = self.weights_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }, best_ckpt)

        # Last checkpoint
        last_ckpt = self.weights_dir / "last.pt"
        torch.save({
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, last_ckpt)
    
    def drift_adaptation(self, method):

        if method == "gmm":
            return self._gmm_adaptation()
        elif method == "naive1":
            return self._naive_adaptation(version=1)
        elif method == "naive2":
            return self._naive_adaptation(version=2)
        elif method == "no_adapt":
            return self._no_adaptation()
        else:
            raise ValueError(f"Unsupported drift adaptation method: {method}")
    
    def _gmm_adaptation(self):
        pretrain_all = pd.read_csv(self.pretrain_csv)
        pretrain_train_df = pretrain_all[pretrain_all["split"] == "train"].copy()
        pretrain_val_df = pretrain_all[pretrain_all["split"] == "val"].copy()
        LOGGER.info(f"Loaded pretrain dataset: train {len(pretrain_train_df)} | val {len(pretrain_val_df)} samples")
        
        df_segments = pd.read_csv(self.segments_csv)
        history_prev = pd.read_csv(self.history0_csv)
        LOGGER.info(f"Loaded segments metadata: {len(df_segments)} samples")
        LOGGER.info(f"Loaded initial history0: {len(history_prev)} samples")
        segment_ids = sorted(df_segments["segment_id"].unique())

        self.setup_model()
        self.setup_optimizer()

        # Directory to save histories
        history_dir = self.save_dir / "histories"
        history_dir.mkdir(parents=True, exist_ok=True)

        # Buffer for previous data segments
        prev_segments_df = pd.DataFrame()
        
        potential_ks = self.potential_ks
        gmm_fitness_fun = self.gmm_fitness_fun
        for epoch_idx, seg_id in enumerate(segment_ids, start=1):
            LOGGER.info(f"\n=== [ALP-GMM] Segment {seg_id} ===")

            # Current segment subsets
            seg_train = df_segments[(df_segments.segment_id == seg_id) & (df_segments.split == "train")]
            seg_probe = df_segments[(df_segments.segment_id == seg_id) & (df_segments.split == "probe")]
            seg_test = df_segments[(df_segments.segment_id == seg_id) & (df_segments.split == "test")]

            # --------------------------
            # 1. Backup state & temporary update on validation set
            # --------------------------
            backup_state = copy.deepcopy(self.model.state_dict())
            backup_opt_state = copy.deepcopy(self.optimizer.state_dict())

            LOGGER.info("Performing temporary update on validation set...")
            self.train_one_epoch(seg_probe, log=False, epoch=epoch_idx)
            
            # --------------------------
            # 2. Compute ALP for pretrain validation set
            # --------------------------
            LOGGER.info("Computing ALP...")
            history_temp = self.evaluate_and_get_history(pretrain_val_df, log=False, epoch=epoch_idx, log_prefix="Val")
            df_temp = pd.DataFrame(history_temp, columns=["brisque", "loss"])

            alp_records = []
            for _, row in df_temp.iterrows():
                p = row["brisque"]
                curr_loss = row["loss"]
                idx = (history_prev["brisque"] - p).abs().idxmin()
                prev_loss = history_prev.iloc[idx]["loss"]
                alp = abs(prev_loss - curr_loss)
                alp_records.append((p, alp))

            alp_array = np.array(alp_records)

            # Restore original state
            self.model.load_state_dict(backup_state)
            self.optimizer.load_state_dict(backup_opt_state)
            
            # --------------------------
            # 3. Fit GMM on (p, alp)
            # --------------------------
            LOGGER.info(f"Fitting best GMM on (p, alp) with candidates {potential_ks} using {gmm_fitness_fun.upper()}...")
            
            valid_gmms = []
            scores = []

            for k in potential_ks:
                if len(alp_array) < k:
                    continue
                gmm_candidate = GMM(n_components=k)
                gmm_candidate.fit(alp_array)
                if gmm_fitness_fun == "bic":
                    score = gmm_candidate.bic(alp_array)
                elif gmm_fitness_fun == "aic":
                    score = gmm_candidate.aic(alp_array)
                elif gmm_fitness_fun == "aicc":
                    n = len(alp_array)
                    d = alp_array.shape[1]
                    params_per_gmm = ((d * d - d) / 2) + 2 * d + 1
                    k_params = k * params_per_gmm - 1
                    penalty = (2 * k_params * (k_params + 1)) / (n - k_params - 1)
                    score = gmm_candidate.aic(alp_array) + penalty
                else:
                    raise NotImplementedError(f"Unsupported fitness metric: {gmm_fitness_fun}")

                valid_gmms.append(gmm_candidate)
                scores.append(score)

            if not valid_gmms:
                raise RuntimeError("No valid GMM could be fitted. Check data size vs potential_ks.")

            best_idx = int(np.argmin(scores))
            self.gmm = valid_gmms[best_idx]
            LOGGER.info(f"Selected GMM with {self.gmm.n_components} components (Score: {scores[best_idx]:.2f})")

            # --------------------------
            # 4. Sample subset S from previous + current train
            # --------------------------
            for inner_idx in range(self.inner_loops):
                candidate_dfs = [prev_segments_df, seg_train]
                if epoch_idx == 1 and self.use_pretrain_data:
                    candidate_dfs.append(pretrain_train_df)
                candidate_df = pd.concat(candidate_dfs, ignore_index=True)

                candidate_points = candidate_df["brisque"].values.reshape(-1, 1)          
                pred_alp_means, _ = self.predict_alp_given_p(candidate_points)

                subset_size = min(self.subset_size, len(candidate_df))
                if self.sampling_method == "prob":
                    exp_scores = np.exp((pred_alp_means - np.max(pred_alp_means)) / self.temperature)
                    probs = exp_scores / exp_scores.sum()
                    sampled_idx = np.random.choice(len(candidate_df), size=subset_size, replace=False, p=probs)             
                elif self.sampling_method == "descending":
                    sampled_idx = np.argsort(-pred_alp_means.flatten())[:subset_size]  # sort descending
                else:
                    raise NotImplementedError(f"Unsupported sampling method: {self.sampling_method}")
                sampled_df = candidate_df.iloc[sampled_idx]
                # --------------------------
                # 5. Train main model on sampled subset
                # --------------------------
                global_epoch = (epoch_idx - 1) * self.inner_loops + (inner_idx + 1)
                if inner_idx == 0:
                    LOGGER.info(f"Training main model on subset size {subset_size} (pool size: {len(candidate_df)})...")
                self.train_one_epoch(sampled_df, log=True, epoch=global_epoch, loop_idx=inner_idx + 1)

            # --------------------------
            # 6. Validate on current segment test → Save history
            # --------------------------
            LOGGER.info(f"Evaluating on current test set...")
            history_current = self.evaluate_and_get_history(seg_test, log=True, epoch=epoch_idx, log_prefix="Adaptation")
            pd.DataFrame(history_current, columns=["brisque", "loss"]).to_csv(history_dir / f"Adaptation{epoch_idx}.csv", index=False)

            LOGGER.info("Evaluating on pretrain validation set...")
            pretrain_history = self.evaluate_and_get_history(pretrain_val_df, log=True, epoch=epoch_idx, log_prefix="Retention")
            pretrain_history_df = pd.DataFrame(pretrain_history, columns=["brisque", "loss"])
            pretrain_history_df.to_csv(history_dir / f"Retention{epoch_idx}.csv", index=False)
            
            val_loss = float(np.mean([loss for _, loss in history_current]))
            self.save_checkpoint(epoch_idx, val_loss)
            
            # Update buffers
            history_prev = pretrain_history_df.copy()
            prev_segments_df = candidate_df.copy()
        
        self.writer.close()

    def _naive_adaptation(self, version):
        """
        Naive adaptation strategies:
        - version 1: Subset from current segment train only
        - version 2: Subset from current + all previous segment trains
        """
        pretrain_all = pd.read_csv(self.pretrain_csv)
        pretrain_train_df = pretrain_all[pretrain_all["split"] == "train"].copy()
        pretrain_val_df = pretrain_all[pretrain_all["split"] == "val"].copy()
        LOGGER.info(f"Loaded pretrain dataset: train {len(pretrain_train_df)} | val {len(pretrain_val_df)} samples")
        
        df_segments = pd.read_csv(self.custom_cfg["segments_csv"])
        LOGGER.info(f"Loaded segments metadata: {len(df_segments)} samples")
        segment_ids = sorted(df_segments["segment_id"].unique())

        self.setup_model()
        self.setup_optimizer()

        history_dir = self.save_dir / "histories"
        history_dir.mkdir(parents=True, exist_ok=True)

        prev_train_segments = []
        if self.use_pretrain_data:
            prev_train_segments.append(pretrain_train_df)
        for epoch_idx, seg_id in enumerate(segment_ids, start=1):
            LOGGER.info(f"\n=== [Naive Adaptation v{version}] Segment {seg_id} ===")
            
            seg_train = df_segments[(df_segments.segment_id == seg_id) & (df_segments.split == "train")]
            seg_test  = df_segments[(df_segments.segment_id == seg_id) & (df_segments.split == "test")]

            # ------------------------
            # Subset Selection Logic
            # ------------------------
            for inner_idx in range(self.inner_loops):
                if version == 1:
                    candidate_df = seg_train
                elif version == 2:
                    candidate_dfs = prev_train_segments + [seg_train]
                    candidate_df = pd.concat(candidate_dfs, ignore_index=True)
                else:
                    raise ValueError(f"Unsupported naive adaptation version: {version}")
                
                subset_size = min(self.subset_size, len(candidate_df))
                sampled_subset = candidate_df.sample(n=subset_size)
                if inner_idx == 0:
                    LOGGER.info(f"Training model on segment {seg_id} with subset size {subset_size} (pool size: {len(candidate_df)})...")
                
                # ------------------------
                # Train on selected subset
                # ------------------------
                global_epoch = (epoch_idx - 1) * self.inner_loops + (inner_idx + 1)
                self.train_one_epoch(sampled_subset, log=True, epoch=global_epoch, loop_idx=inner_idx + 1)

            # ------------------------
            # Evaluate on test set
            # ------------------------
            LOGGER.info(f"Evaluating on segment {seg_id} test set...")
            history_current = self.evaluate_and_get_history(seg_test, log=True, epoch=epoch_idx, log_prefix="Adaptation")
            pd.DataFrame(history_current, columns=["brisque","loss"]).to_csv(history_dir / f"Adaptation{epoch_idx}.csv", index=False)

            LOGGER.info("Evaluating on pretrain validation set...")
            pretrain_history = self.evaluate_and_get_history(pretrain_val_df, log=True, epoch=epoch_idx, log_prefix="Retention")
            pd.DataFrame(pretrain_history, columns=["brisque", "loss"]).to_csv(history_dir / f"Retention{epoch_idx}.csv", index=False)
            
            val_loss = np.mean([l for _,l in history_current])
            self.save_checkpoint(epoch_idx, val_loss)

            prev_train_segments.append(seg_train)
        
        self.writer.close()
    
    def _no_adaptation(self):
        """
        No adaptation: do not update the model at all.
        Iterate segments and evaluate ONLY on seg_test; save per-segment histories.
        """
        df_segments = pd.read_csv(self.segments_csv)
        segment_ids = sorted(df_segments["segment_id"].unique())

        self.setup_model()

        history_dir = self.save_dir / "histories"
        history_dir.mkdir(parents=True, exist_ok=True)

        for epoch_idx, seg_id in enumerate(segment_ids, start=1):
            LOGGER.info(f"\n=== [No-Adapt] Segment {seg_id} ===")
            seg_test = df_segments[(df_segments.segment_id == seg_id) & (df_segments.split == "test")]

            LOGGER.info(f"Evaluating on segment {seg_id} test set (no adaptation)...")
            history_current = self.evaluate_and_get_history(seg_test, log=True, epoch=epoch_idx, log_prefix="Adaptation")
            pd.DataFrame(history_current, columns=["brisque", "loss"]).to_csv(history_dir / f"Adaptation{epoch_idx}.csv", index=False)

        self.writer.close()

    def train_one_epoch(self, df_subset, log, epoch, loop_idx=None):
        loader = self.build_loader_from_df(df_subset)
        self.model.train()
        
        if loop_idx is not None:
            desc = f"[Train {loop_idx}/{self.inner_loops}] Updating model"
        else:
            desc = f"[Train] Updating model"
        
        total_loss, loss_items_sum = 0, None
        nb = len(loader)
        step_base = epoch * nb
        for i, batch in enumerate(tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)):
            step = step_base + i
            batch = self.preprocess_batch(batch)
            imgs = batch["img"]
            preds = self.model(imgs)
            loss, loss_items = self.criterion(preds, batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            if loss_items_sum is None:
                loss_items_sum = loss_items.clone()
            else:
                loss_items_sum += loss_items
            
            if self.log_every_n_steps > 0 and (i + 1) % self.log_every_n_steps == 0 and log:
                self.writer.add_scalar("Train_step/total", loss.item(), step)
                for name, val in zip(["box", "cls", "dfl"], loss_items):
                    self.writer.add_scalar(f"Train_step/{name}", val.item(), step)
            
            if self.plot_period > 0 and (epoch + 1) % self.plot_period == 0 and i < 3 and log:
                plot_samples(batch, epoch - 1, mode="train", idx=i, model=self.model, args=self.args, img_dir=self.img_dir)

        avg_loss = total_loss / nb
        avg_loss_items = (loss_items_sum / nb).tolist()
        
        if log:
            self.writer.add_scalar("Train_epoch/total", avg_loss, epoch)
            for name, val in zip(["box", "cls", "dfl"], avg_loss_items):
                self.writer.add_scalar(f"Train_epoch/{name}", val, epoch)
        
    def evaluate_and_get_history(self, df_subset, log, epoch, log_prefix):
        loader = self.build_loader_from_df(df_subset)
        self.model.eval()
        records = []
        
        total_loss, loss_items_sum = 0, None
        nb = len(loader)
        stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)
        step_base = epoch * nb
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc="[Eval] Collecting history", leave=False, dynamic_ncols=True)):
                step = step_base + i
                batch = self.preprocess_batch(batch)
                imgs = batch["img"]
                preds = self.model(imgs)
                loss, loss_items = self.criterion(preds, batch)

                total_loss += loss.item()
                if loss_items_sum is None:
                    loss_items_sum = loss_items.clone()
                else:
                    loss_items_sum += loss_items
                
                if self.log_every_n_steps > 0 and (i + 1) % self.log_every_n_steps == 0 and log:
                    self.writer.add_scalar(f"{log_prefix}_step/total", loss.item(), step)
                    for name, val in zip(["box", "cls", "dfl"], loss_items):
                        self.writer.add_scalar(f"{log_prefix}_step/{name}", val.item(), step)
                
                _, sample_losses = self.sample_loss_fn(preds, batch)
                im_paths = extract_image_paths_from_batch(batch)
                if im_paths and len(im_paths) == len(sample_losses):
                    for pth, s_loss in zip(im_paths, sample_losses.cpu().numpy().tolist()):
                        bq = self.brisque_map.get(norm_path(pth), None)
                        if bq is not None:
                            records.append((float(bq), float(s_loss)))
                        else:
                            LOGGER.debug(f"No BRISQUE for path: {pth}")
                else:
                    LOGGER.debug("Image paths missing or length mismatch; skipping batch history.")

                if self.plot_period > 0 and (epoch + 1) % self.plot_period == 0 and i < 3 and log:
                    plot_samples(batch, epoch - 1, mode=log_prefix.lower(), idx=i, model=self.model, args=self.args, img_dir=self.img_dir)
                
                if log:
                    preds_nms = ops.non_max_suppression(preds)
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
        
        avg_loss = total_loss / nb
        avg_loss_items = (loss_items_sum / nb).tolist()
        
        if log:
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
            if len(stats["tp"]):
                _, _, p, r, _, ap, *_ = ap_per_class(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"])
                precision, recall = float(p.mean()), float(r.mean())
                mAP50, mAP50_95 = float(ap[:, 0].mean()), float(ap.mean())
            else:
                precision = recall = mAP50 = mAP50_95 = 0.0

            self.writer.add_scalar(f"{log_prefix}_epoch/total", avg_loss, epoch)
            for name, val in zip(["box", "cls", "dfl"], avg_loss_items):
                self.writer.add_scalar(f"{log_prefix}_epoch/{name}", val, epoch)
            self.writer.add_scalar(f"{log_prefix}_metrics/precision", precision, epoch)
            self.writer.add_scalar(f"{log_prefix}_metrics/recall", recall, epoch)
            self.writer.add_scalar(f"{log_prefix}_metrics/mAP50", mAP50, epoch)
            self.writer.add_scalar(f"{log_prefix}_metrics/mAP50-95", mAP50_95, epoch)

        return records

    def predict_alp_given_p(self, p):
        assert self.gmm is not None, "GMM must be fitted first."
        assert p.shape[1] == 1, "Input must have shape (N, 1)"

        means = self.gmm.means_
        covariances = self.gmm.covariances_

        # Select Gaussian with highest ALP mean
        alp_means_per_gauss = means[:, 1]
        k = proportional_choice(alp_means_per_gauss, eps=0.0)

        mean_k = means[k]
        cov_k  = covariances[k]

        mu_p = mean_k[0]
        mu_alp = mean_k[1]
        var_p = cov_k[0, 0]
        cov_alp_p = cov_k[1, 0]
        var_alp = cov_k[1, 1]

        # Conditional mean and variance for ALP | p
        diff = p.flatten() - mu_p  # shape (N,)
        cond_means = mu_alp + (cov_alp_p / var_p) * diff
        cond_vars = var_alp - (cov_alp_p ** 2) / var_p
        cond_vars = np.clip(cond_vars, 1e-6, None)

        return cond_means.astype(np.float32), np.sqrt(cond_vars).astype(np.float32)
