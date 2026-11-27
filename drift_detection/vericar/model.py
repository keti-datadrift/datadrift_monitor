import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig
import torch.nn.functional as F
import pytorch_lightning as pl
import tqdm

from loss import HiMS_min_loss
from loss_ms import MS_loss
from loss_ms_wei import ms_wei_loss
from loss_HiMS_wei import HiMS_min_wei_loss


class VehiInfoRet(pl.LightningModule):
    def __init__(
        self,
        pretrained=True,
        model_name="openai/clip-vit-base-patch32",
        exemplar_k=15,
        knn=1,
        alpha=2.0,
        beta=50.0,
        lam=0.5,
        loss="HiMS",
        lr=1e-5,
        weight_decay=1e-4,
        dist_scale=2.0,
        dist_pow=1.0,
    ):
        super(VehiInfoRet, self).__init__()

        self.exemplar_k = exemplar_k
        self.knn = knn
        self.exemplar_features = None
        self.exemplar_labels = None
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.dist_scale = dist_scale
        self.dist_pow = dist_pow

        if pretrained:
            self.feature_extractor = CLIPVisionModel.from_pretrained(model_name)
        else:
            model_config = CLIPVisionConfig(model_name)
            self.feature_extractor = CLIPVisionModel(model_config)

        self.feature_extractor.train()
        self.fc = nn.Linear(768, 128, bias=False)

    def forward(self, images):
        outputs = self.feature_extractor(images)
        features = outputs.pooler_output
        low_features = self.fc(features)
        return F.normalize(low_features, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        images, all_labels, hi_labels = batch

        batch_size = images.size(0)
        num_hi = hi_labels.size(1)

        features = self(images)
        if self.loss == "HiMS":
            loss = HiMS_min_loss(
                features, hi_labels, batch_size, num_hi, self.alpha, self.beta, self.lam
            )
        elif self.loss == "MS":
            loss = MS_loss(features, hi_labels, self.alpha, self.beta, self.lam)
        elif self.loss == "MSwei":
            loss = ms_wei_loss(
                features,
                hi_labels,
                batch_size,
                num_hi,
                self.alpha,
                self.beta,
                self.lam,
                self.dist_scale,
            )
        elif self.loss == "HiMSwei":
            loss = HiMS_min_wei_loss(
                features,
                hi_labels,
                batch_size,
                num_hi,
                self.alpha,
                self.beta,
                self.lam,
                self.dist_scale,
                self.dist_pow,
            )

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_start(self):
        print("\nBuilding exemplar set for retrieval validation...")
        self.feature_extractor.eval()
        train_loader = self.trainer.datamodule.train_dataloader()

        features_by_class = {}

        with torch.no_grad():
            for batch in tqdm.tqdm(train_loader, desc="Building exemplar set"):
                images, all_labels, _ = batch
                images = images.to(self.device)
                features = self(images)

                for feature, label in zip(features, all_labels):
                    label_item = label.item()
                    if label_item not in features_by_class:
                        features_by_class[label_item] = []
                    features_by_class[label_item].append(feature)

        exemplar_features = []
        exemplar_labels = []

        for class_idx, class_features_list in features_by_class.items():
            if len(class_features_list) == 0:
                continue

            class_features = torch.stack(class_features_list)
            if self.exemplar_k >= len(class_features):
                exemplar_features.append(class_features)
                exemplar_labels.extend([class_idx] * len(class_features))
            else:
                mean_feature = torch.mean(class_features, dim=0)
                distances = torch.cdist(
                    mean_feature.unsqueeze(0), class_features
                ).squeeze(0)
                _, top_k_indices = torch.topk(distances, self.exemplar_k, largest=False)
                exemplar_features.append(class_features[top_k_indices])
                exemplar_labels.extend([class_idx] * self.exemplar_k)

        if exemplar_features:
            self.exemplar_features = torch.cat(exemplar_features, dim=0)
            self.exemplar_labels = torch.tensor(
                exemplar_labels, device=self.device, dtype=torch.long
            )
            print(f"Exemplar set built with {len(self.exemplar_features)} samples.")
        else:
            print("Could not build exemplar set.")

    def calc_retrieval_metrics(self, query_features, query_labels):
        if self.exemplar_features is None or self.exemplar_labels is None:
            return 0.0, 0.0

        dists = torch.cdist(query_features, self.exemplar_features)

        sorted_indices = torch.argsort(dists, dim=1, descending=False)

        ranked_labels = self.exemplar_labels[sorted_indices]

        matches = ranked_labels == query_labels.unsqueeze(1)

        prec1 = matches[:, 0].float().mean()

        num_positives = matches.sum(dim=1)  # [B]

        has_positives = num_positives > 0

        aps = []
        cumulative_matches = matches.cumsum(dim=1).float()

        ranks = torch.arange(1, matches.size(1) + 1, device=self.device).float()

        prec_at_k = cumulative_matches / ranks

        relevant_precisions = prec_at_k * matches.float()

        sum_precisions = relevant_precisions.sum(dim=1)

        ap = torch.zeros_like(sum_precisions)
        ap[has_positives] = (
            sum_precisions[has_positives] / num_positives[has_positives].float()
        )

        map_r = ap.mean()

        return prec1, map_r

    def validation_step(self, batch, batch_idx):
        images, all_labels, hi_labels = batch
        batch_size = images.size(0)
        num_hi = hi_labels.size(1)

        features = self(images)

        if self.loss == "HiMS":
            loss = HiMS_min_loss(
                features, hi_labels, batch_size, num_hi, self.alpha, self.beta, self.lam
            )
        elif self.loss == "MS":
            loss = MS_loss(features, hi_labels, self.alpha, self.beta, self.lam)
        elif self.loss == "MSwei":
            loss = ms_wei_loss(
                features,
                hi_labels,
                batch_size,
                num_hi,
                self.alpha,
                self.beta,
                self.lam,
                self.dist_scale,
            )
        elif self.loss == "HiMSwei":
            loss = HiMS_min_wei_loss(
                features,
                hi_labels,
                batch_size,
                num_hi,
                self.alpha,
                self.beta,
                self.lam,
                self.dist_scale,
                self.dist_pow,
            )

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        prec1, map_r = self.calc_retrieval_metrics(features, all_labels)

        self.log("val_prec1", prec1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_map_r", map_r, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, all_labels, hi_labels = batch
        batch_size = images.size(0)
        num_hi = hi_labels.size(1)

        features = self(images)

        if self.loss == "HiMS":
            loss = HiMS_min_loss(
                features, hi_labels, batch_size, num_hi, self.alpha, self.beta, self.lam
            )
        elif self.loss == "MS":
            loss = MS_loss(features, hi_labels, self.alpha, self.beta, self.lam)
        elif self.loss == "MSwei":
            loss = ms_wei_loss(
                features,
                hi_labels,
                batch_size,
                num_hi,
                self.alpha,
                self.beta,
                self.lam,
                self.dist_scale,
            )
        elif self.loss == "HiMSwei":
            loss = HiMS_min_wei_loss(
                features,
                hi_labels,
                batch_size,
                num_hi,
                self.alpha,
                self.beta,
                self.lam,
                self.dist_scale,
                self.dist_pow,
            )
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        prec1, map_r = self.calc_retrieval_metrics(features, all_labels)

        self.log("test_prec1", prec1, on_step=False, on_epoch=True)
        self.log("test_map_r", map_r, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        if self.exemplar_features is not None:
            checkpoint["exemplar_features"] = self.exemplar_features
            checkpoint["exemplar_labels"] = self.exemplar_labels

    def on_load_checkpoint(self, checkpoint):
        if "exemplar_features" in checkpoint:
            self.exemplar_features = checkpoint["exemplar_features"]
            self.exemplar_labels = checkpoint["exemplar_labels"]
