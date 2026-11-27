import argparse
import random
import subprocess
import sys
import torch
import torch.distributed as dist
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

import os

from libs import train_util
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import gen_datasets, get_id_classes
from libs.utils.hierarchy_utils import get_multidepth_classes, get_multidepth_target_transform
import torchvision.models as models
from torchvision.models import ResNet50_Weights

parser = argparse.ArgumentParser()
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--epochs", type=int, default=90)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.90)
parser.add_argument("--lr_decay", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--datadir", type=str, required=True)
parser.add_argument("--hierarchy", type=str, required=True)
parser.add_argument("--traindir", type=str, required=True)
parser.add_argument("--seed", type=int, default=123456)
parser.add_argument("--loginterval", type=int, default=250)
parser.add_argument("--id_split", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument("--dist_backend", type=str, default=None)
parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:29500?use_libuv=0")


def setup_distributed(args, local_rank=0, rank=0, world_size=None):
    world_size = int(os.environ.get("WORLD_SIZE", world_size or "1"))
    distributed = args.multi_gpu and world_size > 1

    if args.multi_gpu and not distributed and torch.cuda.device_count() > 1:
        raise RuntimeError(
            "--multi_gpu now uses DistributedDataParallel. "
            "Launch this script normally and let it spawn workers, e.g. "
            "`python main_multidepth.py ... --multi_gpu`."
        )

    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        rank = int(os.environ.get("RANK", rank))
        backend = args.dist_backend or ("gloo" if os.name == "nt" else "nccl")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend=backend,
            init_method=args.dist_url,
            rank=rank,
            world_size=world_size,
        )
        return distributed, rank, world_size, local_rank, device, backend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device, None


class TargetTransform:
    def __init__(self, target_transform):
        self.target_transform = target_transform

    def __call__(self, target):
        return self.target_transform[target]


def main(args, local_rank=0, rank=0, world_size=None):

    distributed, rank, world_size, local_rank, device, backend = setup_distributed(
        args, local_rank=local_rank, rank=rank, world_size=world_size)
    is_main_process = rank == 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    id_classes = get_id_classes(args.id_split)

    hierarchy = Hierarchy(id_classes,
                          args.hierarchy,
                          )

    ood_classes = hierarchy.ood_train_classes

    if is_main_process:
        print("==> Preparing data..")
    train_ds, val_ds, _ = gen_datasets(args.datadir,
                                       id_classes,
                                       ood_classes,
                                       )

    height = args.height

    assert height < hierarchy._max_depth

    multi_classes = get_multidepth_classes(hierarchy, train_ds.classes)

    target_transform = get_multidepth_target_transform(train_ds.classes,
                                                       multi_classes,
                                                       args.height,
                                                       hierarchy)

    num_id_classes = len(multi_classes[-(height + 1)])

    if is_main_process:
        print(f"Height: {height}")
        print(f"Nr classes: {num_id_classes}")
    
    target_transform_fn = TargetTransform(target_transform)
    train_ds.target_transform = target_transform_fn
    val_ds.target_transform = target_transform_fn
    
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if distributed else None

    trainloader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        sampler=train_sampler, pin_memory=torch.cuda.is_available())
    valloader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        sampler=val_sampler, pin_memory=torch.cuda.is_available())

    checkpoint_fn = os.path.join(args.traindir, "checkpoint.pt")
    if is_main_process:
        print(f"checkpoint filename: {checkpoint_fn}")

    tensorboard_dir = os.path.join(args.traindir, "tensorboard")
    summary_writer = SummaryWriter(tensorboard_dir) if is_main_process and SummaryWriter is not None else None
    if is_main_process and SummaryWriter is None:
        print("TensorBoard is not installed; continuing without TensorBoard logging.")

    weights = None if "imagenet" in args.datadir else ResNet50_Weights.IMAGENET1K_V1

    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_id_classes)
    
    criterion = nn.CrossEntropyLoss()

    if is_main_process:
        print(f"Device: {device}")
        print(f"Model: ResNet-50, output classes: {num_id_classes}, weights: {weights}")
        if distributed:
            print(f"Using {world_size} processes with DistributedDataParallel ({backend})")
    net = net.to(device)
    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}
        net = DistributedDataParallel(net, **ddp_kwargs)

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    
    top1_acc, top5_acc = train_util.train(
        net,
        trainloader,
        valloader,
        criterion,
        optimizer,
        args.epochs,
        args.batch_size,
        num_id_classes,
        checkpoint=checkpoint_fn,
        log_every_n=args.loginterval,
        summary_writer=summary_writer,
        device=device,
        distributed=distributed,
        rank=rank,
        train_sampler=train_sampler,
        is_main_process=is_main_process,
    )

    if is_main_process:
        print(f"Top 1 Accuracy: {top1_acc}")
        print(f"Top 5 Accuracy: {top5_acc}")

    if distributed:
        dist.destroy_process_group()


def launch_ddp_subprocesses():
    world_size = int(os.environ.get("NPROC_PER_NODE", "2"))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        device_ids = [device.strip() for device in visible_devices.split(",") if device.strip()]
    else:
        device_ids = [str(i) for i in range(world_size)]

    if len(device_ids) < world_size:
        raise RuntimeError(
            f"NPROC_PER_NODE={world_size}, but CUDA_VISIBLE_DEVICES only exposes "
            f"{len(device_ids)} device(s): {','.join(device_ids)}"
        )

    base_env = os.environ.copy()
    base_env.setdefault("USE_LIBUV", "0")
    base_env.setdefault("MASTER_ADDR", "127.0.0.1")
    base_env.setdefault("MASTER_PORT", "29500")

    processes = []
    for rank in range(world_size):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_ids[rank]
        env["WORLD_SIZE"] = str(world_size)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = "0"
        cmd = [sys.executable] + sys.argv
        processes.append(subprocess.Popen(cmd, env=env))

    exit_codes = []
    try:
        for process in processes:
            exit_codes.append(process.wait())
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise

    failed = [code for code in exit_codes if code != 0]
    if failed:
        raise SystemExit(f"DDP worker failed with exit codes: {exit_codes}")


if __name__=="__main__":
    args = parser.parse_args()
    if args.multi_gpu and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        launch_ddp_subprocesses()
    else:
        main(args)
