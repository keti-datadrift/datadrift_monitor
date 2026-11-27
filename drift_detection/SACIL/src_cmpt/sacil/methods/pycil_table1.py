"""PyCIL lifecycle adapters for the SACIL Table-1 baselines.

This module is imported only after :func:`sacil.pycil.activate_pycil` puts the
pinned PyCIL checkout on ``sys.path``.  The reference checkouts remain
read-only; all trainable method code lives in this package.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import BaseNet, CosineIncrementalNet, IncrementalNet

from sacil.methods.afc import (
    afc_nca_loss,
    afc_pod_loss,
    scheduled_afc_factor,
)
from sacil.methods.casper import casper_spectral_loss
from sacil.methods.create import (
    ClasswiseAutoencoderClassifier,
    create_classification_loss,
    create_contrastive_loss,
    reconstruction_confidence_weights,
)
from sacil.methods.cscct import (
    controlled_transfer_loss,
    cross_space_clustering_loss,
)
from sacil.methods.fgp import (
    RectifiedCosineLinear,
    fgp_graph_preservation_loss,
    scheduled_fgp_weight,
)
from sacil.methods.icarl import icarl_bce_loss
from sacil.methods.logit_kd import old_logit_kl_loss
from sacil.methods.podnet import (
    pod_flat_loss,
    pod_spatial_loss,
    podnet_nca_loss,
)


TABLE1_METHODS = {
    "joint",
    "finetune",
    "replay",
    "icarl",
    "podnet",
    "afc",
    "create",
    "fgp",
    "cscct",
    "casper",
}


class _FGPIncrementalNet(BaseNet):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, False)

    def update_fc(self, num_classes: int) -> None:
        expanded = RectifiedCosineLinear(
            self.feature_dim, num_classes, bias=True, learnable_scale=True
        )
        if self.fc is not None:
            old_classes = self.fc.out_features
            with torch.no_grad():
                expanded.weight[:old_classes].copy_(self.fc.weight)
                expanded.bias[:old_classes].copy_(self.fc.bias)
                expanded.scale.copy_(self.fc.scale)
        self.fc = expanded

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        backbone = self.convnet(inputs)
        return {**backbone, "logits": self.fc(backbone["features"])}


class _CREATEIncrementalNet(BaseNet):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, False)
        self.hidden_layers = tuple(
            int(value) for value in args.get("create_hidden_layers", [])
        )
        self.latent_features = int(args.get("create_latent_features", 32))
        self.reconstruction_scale = float(
            args.get("create_reconstruction_scale", 0.1)
        )

    def update_fc(self, num_classes: int) -> None:
        expanded = ClasswiseAutoencoderClassifier(
            self.feature_dim,
            num_classes,
            hidden_layers=self.hidden_layers,
            latent_features=self.latent_features,
            reconstruction_scale=self.reconstruction_scale,
        )
        if self.fc is not None:
            old_classes = self.fc.num_classes
            for class_id in range(old_classes):
                expanded.class_autoencoders[class_id] = copy.deepcopy(
                    self.fc.class_autoencoders[class_id]
                )
        self.fc = expanded

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        backbone = self.convnet(inputs)
        output = self.fc(backbone["features"])
        output["fmaps"] = backbone["fmaps"]
        return output


class _ScaleShiftConv2d(nn.Conv2d):
    """CSCCT scale-and-shift convolution.

    The convolution kernel copied from the previous task is frozen.  Training
    only updates a channel-pair scale and, when present, an additive bias.
    This is the ``branch_1=ss`` layer used by the official CSCCT release.
    """

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
        )
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None and source.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.mtl_weight = nn.Parameter(
            torch.ones(
                self.out_channels,
                self.in_channels // self.groups,
                1,
                1,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        if self.bias is None:
            self.register_parameter("mtl_bias", None)
        else:
            self.mtl_bias = nn.Parameter(
                torch.zeros_like(self.bias, memory_format=torch.preserve_format)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        scaled_weight = self.weight * self.mtl_weight.expand_as(self.weight)
        bias = (
            None
            if self.bias is None
            else self.bias + self.mtl_bias
        )
        return self._conv_forward(inputs, scaled_weight, bias)


def _convert_to_scale_shift(module: nn.Module) -> None:
    """Replace every ordinary convolution in ``module`` in-place."""

    for name, child in list(module.named_children()):
        if isinstance(child, _ScaleShiftConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, _ScaleShiftConv2d(child))
        else:
            _convert_to_scale_shift(child)


class _CSCCTIncrementalNet(CosineIncrementalNet):
    """CSCCT dual-branch, layer-wise fused feature extractor."""

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, pretrained=False, nb_proxy=1)
        self.second_convnet: nn.Module | None = None
        self.branch1_style = str(
            args.get("cscct_branch1", "scale_shift")
        ).lower()
        if self.branch1_style not in {"scale_shift", "free", "fixed"}:
            raise ValueError(
                "cscct_branch1 must be scale_shift, free, or fixed"
            )
        level_count = 3 if args["convnet_type"].lower() == "resnet32" else 4
        self.fusion = nn.ParameterList(
            nn.Parameter(torch.tensor([0.5])) for _ in range(level_count)
        )
        self._task_index = -1

    def update_fc(self, num_classes: int) -> None:
        self._task_index += 1
        if self._task_index > 0 and self.second_convnet is None:
            self.second_convnet = copy.deepcopy(self.convnet)
            if self.branch1_style == "scale_shift":
                _convert_to_scale_shift(self.convnet)
            elif self.branch1_style == "fixed":
                for parameter in self.convnet.parameters():
                    parameter.requires_grad = False
        super().update_fc(num_classes, self._task_index)

    def extract_vector(self, inputs: Tensor) -> Tensor:
        return self._fused_backbone(inputs)["features"]

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        backbone = self._fused_backbone(inputs)
        output = self.fc(backbone["features"])
        output.update(backbone)
        return output

    def _mix(self, level: int, first: Tensor, second: Tensor) -> Tensor:
        alpha = self.fusion[level]
        return alpha * first + (1.0 - alpha) * second

    def _fused_backbone(self, inputs: Tensor) -> dict[str, Tensor]:
        if self.second_convnet is None:
            return self.convnet(inputs)
        first = self.convnet
        second = self.second_convnet
        if hasattr(first, "stage_1"):
            first_stem = F.relu(
                first.bn_1(first.conv_1_3x3(inputs)), inplace=False
            )
            second_stem = F.relu(
                second.bn_1(second.conv_1_3x3(inputs)), inplace=False
            )
            map1 = self._mix(
                0,
                first.stage_1(first_stem),
                second.stage_1(second_stem),
            )
            map2 = self._mix(
                1, first.stage_2(map1), second.stage_2(map1)
            )
            map3 = self._mix(
                2, first.stage_3(map2), second.stage_3(map2)
            )
            pooled = first.avgpool(map3)
            return {
                "fmaps": [map1, map2, map3],
                "features": pooled.flatten(start_dim=1),
            }

        first_map = first.layer1(first.conv1(inputs))
        second_map = second.layer1(second.conv1(inputs))
        map1 = self._mix(0, first_map, second_map)
        map2 = self._mix(
            1, first.layer2(map1), second.layer2(map1)
        )
        map3 = self._mix(
            2, first.layer3(map2), second.layer3(map2)
        )
        map4 = self._mix(
            3, first.layer4(map3), second.layer4(map3)
        )
        return {
            "fmaps": [map1, map2, map3, map4],
            "features": first.avgpool(map4).flatten(start_dim=1),
        }


def _freeze_old_proxy_weights(network: nn.Module) -> None:
    classifier = network.fc
    if hasattr(classifier, "fc1"):
        for parameter in classifier.fc1.parameters():
            parameter.requires_grad = False


class Table1Baseline(BaseLearner):
    """Common PyCIL task/memory harness with method-faithful objectives."""

    def __init__(self, args: dict[str, Any], method: str) -> None:
        super().__init__(args)
        method = method.lower()
        if method not in TABLE1_METHODS:
            raise ValueError(f"unsupported Table-1 method: {method}")
        self.method = method
        self._network_args = copy.deepcopy(args)
        if method in {"podnet", "afc"}:
            self._network = CosineIncrementalNet(
                args,
                pretrained=False,
                nb_proxy=int(args.get("proxy_per_class", 10)),
            )
        elif method == "create":
            self._network = _CREATEIncrementalNet(args)
        elif method == "fgp":
            self._network = _FGPIncrementalNet(args)
        elif method == "cscct":
            self._network = _CSCCTIncrementalNet(args)
        else:
            self._network = IncrementalNet(args, False)

        self.batch_size = int(args.get("batch_size", 128))
        self.num_workers = int(args.get("num_workers", 0))
        self.pin_memory = bool(args.get("pin_memory", True))
        self.init_epochs = int(args.get("init_epochs", 200))
        self.epochs = int(args.get("epochs", self._default_epochs()))
        self.init_lr = float(args.get("init_lr", 0.1))
        self.lr = float(args.get("lr", self._default_lr()))
        self.init_weight_decay = float(
            args.get("init_weight_decay", 5e-4)
        )
        self.weight_decay = float(args.get("weight_decay", 5e-4))
        self.momentum = float(args.get("momentum", 0.9))
        self.scheduler = str(
            args.get("scheduler", self._default_scheduler())
        ).lower()
        self.init_scheduler = str(
            args.get("init_scheduler", self.scheduler)
        ).lower()
        self.milestones = [
            int(value) for value in args.get("milestones", [80, 120])
        ]
        self.init_milestones = [
            int(value)
            for value in args.get("init_milestones", [60, 120, 170])
        ]
        self.lr_decay = float(args.get("lr_decay", 0.1))
        self.eval_interval = int(args.get("eval_interval", 5))
        self.disable_tqdm = bool(args.get("disable_tqdm", False))
        self.max_batches_per_epoch = args.get("max_batches_per_epoch")
        if self.max_batches_per_epoch is not None:
            self.max_batches_per_epoch = int(self.max_batches_per_epoch)

        self.kd_temperature = float(args.get("kd_temperature", 2.0))
        default_kd_weight = 0.25 if method == "cscct" else 1.0
        self.kd_weight = float(args.get("kd_weight", default_kd_weight))
        self.pod_flat_weight = float(args.get("pod_flat_weight", 1.0))
        self.pod_spatial_weight = float(
            args.get("pod_spatial_weight", 5.0)
        )
        self.afc_weight = float(args.get("afc_weight", 4.0))
        self.fgp_weight = float(args.get("fgp_weight", 0.1))
        self.csc_weight = float(args.get("csc_weight", 3.0))
        self.ct_weight = float(args.get("ct_weight", 1.5))
        self.ct_temperature = float(args.get("ct_temperature", 2.0))
        self.cscct_fusion_lr = float(args.get("cscct_fusion_lr", 1e-8))
        self.casper_weight = float(args.get("casper_weight", 0.01))
        self.casper_k = int(args.get("casper_k", 10))
        self.casper_classes = args.get("casper_classes")
        self.create_contrastive_weight = float(
            args.get("create_contrastive_weight", 1.0)
        )
        self.create_confidence_alpha = float(
            args.get("create_confidence_alpha", 2.0)
        )

        default_ft = 20 if method in {"podnet", "afc", "create"} else 0
        self.finetune_epochs = int(args.get("finetune_epochs", default_ft))
        default_ft_lr = 0.05 if method == "afc" else 0.005
        self.finetune_lr = float(args.get("finetune_lr", default_ft_lr))
        self.store_exemplars = bool(args.get("store_exemplars", True))
        self.classifier_scale = float(args.get("classifier_scale", 3.0))
        self.learnable_classifier_scale = bool(
            args.get("learnable_classifier_scale", False)
        )
        seed = args.get("seed", 1)
        self.random_seed = int(seed[0] if isinstance(seed, list) else seed)
        self._afc_importance: list[Tensor] | None = None
        self._casper_loader: DataLoader | None = None
        self._cscct_balanced_loader: DataLoader | None = None

    def _default_epochs(self) -> int:
        if self.method in {"podnet", "afc"}:
            return 160
        if self.method == "create":
            return 120
        return 170

    def _default_lr(self) -> float:
        if self.method == "create":
            return 0.001
        return 0.1

    def _default_scheduler(self) -> str:
        if self.method in {"podnet", "afc", "create"}:
            return "cosine"
        return "multistep"

    @property
    def _uses_replay(self) -> bool:
        return self.method not in {"joint", "finetune"}

    def after_task(self) -> None:
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: %d", self.exemplar_size)

    def incremental_train(self, data_manager) -> None:
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.method == "joint" and self._cur_task > 0:
            self._network = IncrementalNet(self._network_args, False)
        if self.method in {"podnet", "afc"}:
            self._network.update_fc(self._total_classes, self._cur_task)
            self._configure_cosine_scale()
            if self._cur_task > 0:
                self._imprint_new_proxies(data_manager)
                _freeze_old_proxy_weights(self._network)
        else:
            self._network.update_fc(self._total_classes)
            if self.method == "cscct" and self._cur_task > 0:
                self._imprint_new_proxies(data_manager)
                _freeze_old_proxy_weights(self._network)

        memory = self._get_memory() if self._uses_replay else None
        train_classes = (
            np.arange(0, self._total_classes)
            if self.method == "joint"
            else np.arange(self._known_classes, self._total_classes)
        )
        train_dataset = data_manager.get_dataset(
            train_classes,
            source="train",
            mode="train",
            appendent=memory,
        )
        self.train_loader = self._loader(train_dataset, shuffle=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = self._loader(test_dataset, shuffle=False)
        self._casper_loader = self._build_casper_loader(data_manager)
        self._cscct_balanced_loader = self._build_cscct_balanced_loader(
            data_manager
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(
                self._network, self._multiple_gpus
            )
        self._train(self.train_loader, self.test_loader)
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module

        if (
            self._cur_task > 0
            and self.finetune_epochs > 0
            and self.method in {"podnet", "afc", "create"}
        ):
            self._balanced_finetune(data_manager)
        if self.method == "afc":
            self._afc_importance = self._estimate_afc_importance(
                self.train_loader
            )
        if self.store_exemplars:
            self.build_rehearsal_memory(
                data_manager, self.samples_per_class
            )

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _configure_cosine_scale(self) -> None:
        scale = self._network.fc.sigma
        with torch.no_grad():
            scale.fill_(self.classifier_scale)
        scale.requires_grad = self.learnable_classifier_scale

    @torch.inference_mode()
    def _imprint_new_proxies(self, data_manager) -> None:
        """Initialize PODNet/AFC new proxies from incoming-class features."""

        network = self._network
        network.to(self._device)
        network.eval()
        proxy_count = int(network.nb_proxy)
        old_weights = network.fc.fc1.weight
        average_norm = old_weights.norm(dim=1).mean().cpu()
        imprinted = []
        for class_id in range(self._known_classes, self._total_classes):
            dataset = data_manager.get_dataset(
                np.arange(class_id, class_id + 1),
                source="train",
                mode="test",
            )
            loader = self._loader(dataset, shuffle=False)
            features = []
            for _, inputs, _ in loader:
                features.append(
                    network.extract_vector(inputs.to(self._device)).cpu()
                )
            normalized = F.normalize(torch.cat(features, dim=0), dim=1)
            if proxy_count == 1:
                centers = F.normalize(
                    normalized.mean(dim=0, keepdim=True), dim=1
                )
            else:
                if normalized.shape[0] < proxy_count:
                    raise ValueError(
                        "not enough samples to imprint all class proxies"
                    )
                cluster = KMeans(
                    n_clusters=proxy_count,
                    n_init=10,
                    random_state=self.random_seed + class_id,
                )
                cluster.fit(normalized.numpy())
                centers = torch.from_numpy(cluster.cluster_centers_).to(
                    dtype=normalized.dtype
                )
            imprinted.append(centers * average_norm)
        values = torch.cat(imprinted, dim=0).to(
            device=network.fc.fc2.weight.device,
            dtype=network.fc.fc2.weight.dtype,
        )
        network.fc.fc2.weight.copy_(values)

    def _build_casper_loader(self, data_manager) -> DataLoader | None:
        if self.method != "casper" or self._cur_task == 0:
            return None
        memory = self._get_memory()
        if memory is None:
            raise RuntimeError("CaSpeR requires exemplar replay memory")
        dataset = data_manager.get_dataset(
            [], source="train", mode="train", appendent=memory
        )
        return self._loader(dataset, shuffle=True)

    def _build_cscct_balanced_loader(
        self, data_manager
    ) -> DataLoader | None:
        if self.method != "cscct" or self._cur_task == 0:
            return None
        memory = self._get_memory()
        if memory is None:
            raise RuntimeError("CSCCT requires exemplar replay memory")
        new_data = []
        new_targets = []
        samples = self.samples_per_class
        generator = np.random.default_rng(
            self.random_seed + self._cur_task
        )
        for class_id in range(self._known_classes, self._total_classes):
            data, targets, _ = data_manager.get_dataset(
                np.arange(class_id, class_id + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            count = min(samples, len(targets))
            selected = generator.choice(len(targets), count, replace=False)
            new_data.append(np.asarray(data)[selected])
            new_targets.append(np.asarray(targets)[selected])
        balanced_data = np.concatenate(
            (np.asarray(memory[0]), *new_data), axis=0
        )
        balanced_targets = np.concatenate(
            (np.asarray(memory[1]), *new_targets), axis=0
        )
        dataset = data_manager.get_dataset(
            [],
            source="train",
            mode="train",
            appendent=(balanced_data, balanced_targets),
        )
        return self._loader(dataset, shuffle=True)

    def _phase(self) -> tuple[int, float, str, list[int], float]:
        if self._cur_task == 0:
            return (
                self.init_epochs,
                self.init_lr,
                self.init_scheduler,
                self.init_milestones,
                self.init_weight_decay,
            )
        return (
            self.epochs,
            self.lr,
            self.scheduler,
            self.milestones,
            self.weight_decay,
        )

    def _make_scheduler(
        self,
        optimizer: optim.Optimizer,
        name: str,
        epochs: int,
        milestones: Sequence[int],
    ):
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-8
            )
        if name == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=list(milestones), gamma=self.lr_decay
            )
        raise ValueError(f"unsupported scheduler: {name}")

    def _train(self, train_loader, test_loader) -> None:
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        epochs, lr, schedule, milestones, decay = self._phase()
        fusion_parameters: list[nn.Parameter] = []
        if self.method == "cscct":
            cscct_network = self._network
            fusion_parameters = list(cscct_network.fusion.parameters())
            fusion_ids = {id(parameter) for parameter in fusion_parameters}
            parameters = [
                parameter
                for parameter in self._network.parameters()
                if parameter.requires_grad and id(parameter) not in fusion_ids
            ]
        else:
            parameters = [
                parameter
                for parameter in self._network.parameters()
                if parameter.requires_grad
            ]
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=self.momentum,
            weight_decay=decay,
        )
        scheduler = self._make_scheduler(
            optimizer, schedule, epochs, milestones
        )
        fusion_optimizer = (
            optim.SGD(
                fusion_parameters,
                lr=self.cscct_fusion_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            if fusion_parameters
            else None
        )
        casper_iterator = (
            iter(self._casper_loader)
            if self._casper_loader is not None
            else None
        )

        progress = tqdm(range(epochs), disable=self.disable_tqdm)
        for epoch in progress:
            self._network.train()
            totals: dict[str, float] = {}
            sample_count = 0
            correct = 0
            for batch_index, (_, inputs, targets) in enumerate(train_loader):
                if (
                    self.max_batches_per_epoch is not None
                    and batch_index >= self.max_batches_per_epoch
                ):
                    break
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True).long()
                replay_inputs = None
                if casper_iterator is not None:
                    try:
                        _, replay_inputs, _ = next(casper_iterator)
                    except StopIteration:
                        casper_iterator = iter(self._casper_loader)
                        _, replay_inputs, _ = next(casper_iterator)
                    replay_inputs = replay_inputs.to(
                        self._device, non_blocking=True
                    )

                output = self._network(inputs)
                reference = None
                if self._old_network is not None:
                    with torch.no_grad():
                        reference = self._old_network(inputs)
                components = self._loss_components(
                    inputs,
                    targets,
                    output,
                    reference,
                    replay_inputs=replay_inputs,
                )
                loss = sum(components.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                count = targets.numel()
                sample_count += count
                prediction_logits = output["logits"]
                correct += int(
                    prediction_logits.argmax(dim=1).eq(targets).sum()
                )
                for name, value in components.items():
                    totals[name] = totals.get(name, 0.0) + float(
                        value.detach()
                    ) * count
            scheduler.step()
            if (
                fusion_optimizer is not None
                and self._cscct_balanced_loader is not None
            ):
                self._update_cscct_fusion(fusion_optimizer)
            if sample_count == 0:
                raise RuntimeError("Table-1 learner processed no samples")
            pieces = " ".join(
                f"{name}={value / sample_count:.4f}"
                for name, value in sorted(totals.items())
            )
            info = (
                f"{self.method} task={self._cur_task} "
                f"epoch={epoch + 1}/{epochs} {pieces} "
                f"train={100.0 * correct / sample_count:.2f}"
            )
            if self.eval_interval > 0 and epoch % self.eval_interval == 0:
                accuracy = self._compute_accuracy(
                    self._network, test_loader
                )
                info += f" test={accuracy:.2f}"
            progress.set_description(info)
            logging.info(info)

    def _update_cscct_fusion(
        self, optimizer: optim.Optimizer
    ) -> None:
        network = self._network
        network.eval()
        for _, inputs, targets in self._cscct_balanced_loader:
            inputs = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True).long()
            loss = F.cross_entropy(network(inputs)["logits"], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _loss_components(
        self,
        inputs: Tensor,
        targets: Tensor,
        output: dict[str, Tensor],
        reference: dict[str, Tensor] | None,
        *,
        replay_inputs: Tensor | None,
    ) -> dict[str, Tensor]:
        logits = output["logits"]
        zero = logits.sum() * 0.0
        if self.method in {"joint", "finetune", "replay"}:
            return {"ce": F.cross_entropy(logits, targets)}
        if self.method == "icarl":
            old_logits = None if reference is None else reference["logits"]
            return {
                "icarl": icarl_bce_loss(
                    logits,
                    targets,
                    old_logits=old_logits,
                    known_classes=self._known_classes,
                )
            }
        if self.method == "podnet":
            losses = {
                "nca": podnet_nca_loss(
                    logits,
                    targets,
                    scale=1.0,
                )
            }
            if reference is not None:
                factor = math.sqrt(
                    self._total_classes
                    / (self._total_classes - self._known_classes)
                )
                losses["pod_flat"] = (
                    self.pod_flat_weight
                    * factor
                    * pod_flat_loss(
                        output["features"], reference["features"]
                    )
                )
                losses["pod_spatial"] = (
                    self.pod_spatial_weight
                    * factor
                    * pod_spatial_loss(
                        output["fmaps"], reference["fmaps"]
                    )
                )
            return losses
        if self.method == "afc":
            network = (
                self._network.module
                if isinstance(self._network, nn.DataParallel)
                else self._network
            )
            losses = {
                "nca": afc_nca_loss(
                    logits, targets, 1.0
                )
            }
            if reference is not None:
                if self._afc_importance is None:
                    raise RuntimeError("AFC has no previous channel importance")
                losses["afc"] = scheduled_afc_factor(
                    self._total_classes,
                    self._total_classes - self._known_classes,
                    self.afc_weight,
                ) * afc_pod_loss(
                    reference["fmaps"],
                    output["fmaps"],
                    self._afc_importance,
                )
            return losses
        if self.method == "create":
            losses = {
                "create_cls": create_classification_loss(logits, targets)
            }
            weights = reconstruction_confidence_weights(
                output["error_logits"],
                alpha=self.create_confidence_alpha,
            )
            losses["create_contrast"] = (
                self.create_contrastive_weight
                * create_contrastive_loss(
                    output["latents"], targets, sample_weights=weights
                )
            )
            if reference is not None:
                losses["create_kd"] = self.kd_weight * old_logit_kl_loss(
                    output["error_logits"][:, : self._known_classes],
                    reference["error_logits"],
                    temperature=self.kd_temperature,
                )
            return losses
        if self.method == "fgp":
            one_hot = F.one_hot(
                targets, num_classes=self._total_classes
            ).to(logits)
            losses = {
                "fgp_bce": F.binary_cross_entropy_with_logits(
                    logits, one_hot
                )
            }
            if reference is not None:
                network = (
                    self._network.module
                    if isinstance(self._network, nn.DataParallel)
                    else self._network
                )
                losses["fgp_graph"] = scheduled_fgp_weight(
                    self._known_classes,
                    self._total_classes,
                    base_weight=self.fgp_weight,
                ) * fgp_graph_preservation_loss(
                    output["features"],
                    reference["features"],
                    network.fc.weight[: self._known_classes],
                    self._old_network.fc.weight,
                )
            return losses

        if self.method == "casper":
            losses = {
                "icarl": icarl_bce_loss(
                    logits,
                    targets,
                    old_logits=(
                        None if reference is None else reference["logits"]
                    ),
                    known_classes=self._known_classes,
                )
            }
        else:
            losses = {"ce": F.cross_entropy(logits, targets)}
        if reference is not None:
            if self.method != "casper":
                losses["kd"] = self.kd_weight * old_logit_kl_loss(
                    logits[:, : self._known_classes],
                    reference["logits"],
                    temperature=self.kd_temperature,
                )
        if self.method == "cscct" and reference is not None:
            losses["csc"] = self.csc_weight * cross_space_clustering_loss(
                output["features"], reference["features"], targets
            )
            losses["ct"] = self.ct_weight * controlled_transfer_loss(
                output["features"],
                reference["features"],
                targets,
                known_classes=self._known_classes,
                temperature=self.ct_temperature,
            )
        if self.method == "casper" and replay_inputs is not None:
            replay_output = self._network(replay_inputs)
            num_classes = (
                int(self.casper_classes)
                if self.casper_classes is not None
                else self._total_classes - self._known_classes
            )
            losses["casper"] = self.casper_weight * casper_spectral_loss(
                replay_output["features"],
                num_classes=num_classes,
                k=self.casper_k,
            )
        return losses

    def _balanced_finetune(self, data_manager) -> None:
        per_class = self.samples_per_class
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)
        dataset = data_manager.get_dataset(
            [], source="train", mode="train", appendent=self._get_memory()
        )
        loader = self._loader(dataset, shuffle=True)

        network = self._network
        for parameter in network.parameters():
            parameter.requires_grad = False
        for parameter in network.fc.parameters():
            parameter.requires_grad = True
        if self.method == "podnet" and self._cur_task > 0:
            _freeze_old_proxy_weights(network)
        optimizer = optim.SGD(
            [p for p in network.fc.parameters() if p.requires_grad],
            lr=self.finetune_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.finetune_epochs
        )
        network.to(self._device)
        for _ in range(self.finetune_epochs):
            network.train()
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device).long()
                output = network(inputs)
                if self.method == "create":
                    loss = create_classification_loss(
                        output["logits"], targets
                    )
                elif self.method == "podnet":
                    loss = podnet_nca_loss(
                        output["logits"],
                        targets,
                        scale=1.0,
                    )
                else:
                    loss = afc_nca_loss(
                        output["logits"],
                        targets,
                        1.0,
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        for parameter in network.parameters():
            parameter.requires_grad = True
        if self.method in {"podnet", "afc"} and self._cur_task > 0:
            _freeze_old_proxy_weights(network)

        added = per_class * (
            self._total_classes - self._known_classes
        )
        if added:
            self._data_memory = self._data_memory[:-added]
            self._targets_memory = self._targets_memory[:-added]

    def _estimate_afc_importance(
        self, loader: DataLoader
    ) -> list[Tensor]:
        network = self._network
        network.to(self._device)
        network.eval()
        accumulators: list[Tensor] | None = None
        batches = 0
        for batch_index, (_, inputs, targets) in enumerate(loader):
            if (
                self.max_batches_per_epoch is not None
                and batch_index >= self.max_batches_per_epoch
            ):
                break
            inputs = inputs.to(self._device)
            targets = targets.to(self._device).long()
            output = network(inputs)
            feature_maps = output["fmaps"]
            gradients = torch.autograd.grad(
                F.cross_entropy(output["logits"], targets),
                feature_maps,
                retain_graph=False,
                allow_unused=False,
            )
            values = [
                gradient.detach().square().sum(dim=(0, 2, 3))
                / gradient.shape[0]
                for gradient in gradients
            ]
            if accumulators is None:
                accumulators = [value.clone() for value in values]
            else:
                for accumulator, value in zip(accumulators, values):
                    accumulator.add_(value)
            batches += 1
        if accumulators is None or batches == 0:
            raise RuntimeError("AFC importance pass processed no samples")
        return [
            (value / batches)
            / (value.mean() / batches).clamp_min(torch.finfo(value.dtype).eps)
            for value in accumulators
        ]


class Table1FineTune(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "finetune")


class Table1Joint(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "joint")


class Table1Replay(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "replay")


class Table1ICaRL(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "icarl")


class Table1PODNet(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "podnet")


class Table1AFC(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "afc")


class Table1CREATE(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "create")


class Table1FGP(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "fgp")


class Table1CSCCT(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "cscct")


class Table1CaSpeR(Table1Baseline):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__(args, "casper")


_FACTORY = {
    "table1_joint": Table1Joint,
    "table1_finetune": Table1FineTune,
    "table1_replay": Table1Replay,
    "table1_icarl": Table1ICaRL,
    "table1_podnet": Table1PODNet,
    "table1_afc": Table1AFC,
    "table1_create": Table1CREATE,
    "table1_fgp": Table1FGP,
    "table1_cscct": Table1CSCCT,
    "table1_casper": Table1CaSpeR,
}


def get_table1_model(model_name: str, args: dict[str, Any]) -> BaseLearner:
    name = str(model_name).lower()
    if name not in _FACTORY:
        raise ValueError(f"unknown Table-1 model: {model_name}")
    return _FACTORY[name](args)


def table1_model_names() -> frozenset[str]:
    return frozenset(_FACTORY)
