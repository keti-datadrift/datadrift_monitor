from __future__ import annotations

import argparse
from argparse import Namespace
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.clip_backend import ClipBackend, safe_model_name
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.feature_io import ensure_dir, save_feature_file, save_json
from negzerohoc.output_layout import shared_feature_dir
from negzerohoc.runtime import configured_device

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class DatasetWithPaths(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        path, _ = self.dataset.samples[index]
        return image, target, path

    @property
    def classes(self):
        return self.dataset.classes


def collate_pil(batch):
    images, targets, paths = zip(*batch)
    return list(images), torch.tensor(targets, dtype=torch.long), list(paths)


def load_config(path):
    cfg = load_yaml_config(path)
    dataset_cfg = cfg.get("dataset", {})
    runtime_cfg = cfg.get("runtime", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    feature_cfg = cfg.get("feature_extraction", {})
    experiment_cfg = cfg.get("experiment", {})

    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing required config key: dataset.datadir in {path}")

    return Namespace(
        config=str(path),
        experiment_name=experiment_cfg.get("name", "fgvc-aircraft-clip-features"),
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        datadir=datadir,
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch32"),
        outdir=experiment_cfg.get("output_root", "outputs"),
        batch_size=dataloader_cfg.get("batch_size", 128),
        num_workers=dataloader_cfg.get("num_workers", 4),
        device=configured_device(runtime_cfg),
        local_files_only=clip_cfg.get("local_files_only", True),
        skip_train=feature_cfg.get("skip_train", True),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    config_arg = parser.parse_args()
    return load_config(config_arg.config)


@torch.no_grad()
def encode_split(backend, split_name, dataset, loader, clip_model):
    feats = []
    targets = []
    paths = []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"Encoding {split_name}", unit="batch")

    for images, batch_targets, batch_paths in iterator:
        batch_feats = backend.encode_images(images).cpu()
        feats.append(batch_feats)
        targets.append(batch_targets.cpu())
        paths.extend(batch_paths)

    return {
        "features": torch.cat(feats, dim=0),
        "targets": torch.cat(targets, dim=0),
        "classes": list(dataset.classes),
        "paths": paths,
        "split": split_name,
        "clip_model": clip_model,
    }


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()
    from negzerohoc.prohoc_compat.utils.dataset_util import SubsetImageFolder, get_id_classes

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    id_classes = get_id_classes(args.id_split)
    ood_classes = hierarchy.ood_train_classes

    datadir = Path(args.datadir)
    train_ds = DatasetWithPaths(SubsetImageFolder(datadir / "train", id_classes, transform=None))
    val_ds = DatasetWithPaths(SubsetImageFolder(datadir / "val", id_classes, transform=None))
    ood_ds = DatasetWithPaths(SubsetImageFolder(datadir / "val", ood_classes, transform=None))

    backend = ClipBackend(args.clip_model, device=args.device, local_files_only=args.local_files_only)

    model_key = f"clip_{safe_model_name(args.clip_model)}"
    feature_dir = ensure_dir(shared_feature_dir(args.outdir, args.dataset, model_key))
    split_items = [("val", val_ds), ("ood", ood_ds)]
    if not args.skip_train:
        split_items.insert(0, ("train", train_ds))

    for split_name, dataset in split_items:
        print(f"Encoding {split_name}: {len(dataset)} images")
        payload = encode_split(
            backend,
            split_name,
            dataset,
            make_loader(dataset, args),
            args.clip_model,
        )
        save_path = feature_dir / f"{split_name}-features.pt"
        save_feature_file(save_path, payload)
        print(f"saved: {save_path}")

    save_json(
        feature_dir / "meta.json",
        {
            "dataset": args.dataset,
            "clip_model": args.clip_model,
            "model_key": model_key,
            "hierarchy": str(args.hierarchy),
            "id_split": str(args.id_split),
            "datadir": str(args.datadir),
            "config": str(args.config),
            "num_id_classes": len(id_classes),
            "num_ood_classes": len(ood_classes),
        },
    )


if __name__ == "__main__":
    main()
