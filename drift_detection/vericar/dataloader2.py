import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
import json
import glob
import tqdm
from pathlib import Path
from torchvision.transforms import RandomHorizontalFlip


class VehicleDataset(Dataset):
    def __init__(
        self,
        metadata,
        data_root,
        view_to_idx,
        type_to_idx,
        make_to_idx,
        model_to_idx,
        year_to_idx,
        all_labels_to_idx,
        flip_p=0.0,
    ):
        self.metadata = metadata
        self.data_root = Path(data_root)

        self.view_to_idx = view_to_idx
        self.type_to_idx = type_to_idx
        self.make_to_idx = make_to_idx
        self.model_to_idx = model_to_idx
        self.year_to_idx = year_to_idx
        self.all_labels_to_idx = all_labels_to_idx

        self.flipper = RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        pt_path = self.data_root / item["image_path"]
        
        image = torch.load(pt_path)
        flipped_img = self.flipper(image)
        
        view_str = item["view"]
        type_str = item["type"]
        make_str = item["make"]
        model_str = item["model"]
        year_str = item["year"]
        all_str = item["all_labels"]

        view_label = self.view_to_idx[view_str]
        type_label = self.type_to_idx[type_str]
        make_label = self.make_to_idx[make_str]
        model_label = self.model_to_idx[model_str]
        year_label = self.year_to_idx[year_str]
        all_label = self.all_labels_to_idx[all_str]

        hi_labels = torch.tensor(
            [view_label, type_label, make_label, model_label, year_label], dtype=torch.long
        )

        return flipped_img, all_label, hi_labels


class VehicleDataModule(pl.LightningDataModule):
    def __init__(
        self, data_root, batch_size=64, num_workers=4, flip_p=0.5,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flip_p = flip_p

    def setup(self, stage=None):
        cache_path = self.data_root / "metadata_cache_split.pkl"

        if cache_path.exists():
            print(f"Loading metadata from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            self.train_meta = cache_data["train_meta"]
            self.val_meta = cache_data["val_meta"]
            self.test_meta = cache_data["test_meta"]
            
            self.view_to_idx = cache_data["view_to_idx"]
            self.type_to_idx = cache_data["type_to_idx"]
            self.make_to_idx = cache_data["make_to_idx"]
            self.model_to_idx = cache_data["model_to_idx"]
            self.year_to_idx = cache_data["year_to_idx"]
            self.all_labels_to_idx = cache_data["all_labels_to_idx"]
        else:
            print(f"Cache not found. Scanning train/val/test folders in {self.data_root}...")
            
            self.train_meta = []
            self.val_meta = []
            self.test_meta = []
            
            view_labels = set()
            type_labels = set()
            make_labels = set()
            model_labels = set()
            year_labels = set()
            all_labels = set()

            splits = ["train", "val", "test"]
            meta_dict = {"train": self.train_meta, "val": self.val_meta, "test": self.test_meta}

            for split in splits:
                split_label_dir = self.data_root / split / "labels"
                json_files = glob.glob(str(split_label_dir / "**/*.json"), recursive=True)
                
                print(f"Processing {split} split: found {len(json_files)} files.")

                for json_path in tqdm.tqdm(json_files, desc=f"Scanning {split}"):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        attrs = data["car"]["attributes"]
                        car_view = attrs["view"]
                        car_type = attrs["type"]
                        make = attrs["brand"]
                        model = attrs["model"]
                        year = attrs["year"]

                        view_labels.add(car_view)
                        type_labels.add(car_type)
                        make_labels.add(make)
                        model_labels.add(model)
                        year_labels.add(year)
                        all_labels.add(f"{car_view}-{car_type}-{make}-{model}-{year}")

                        rel_path = Path(json_path).relative_to(split_label_dir)
                        image_rel_path = Path(split) / "images" / rel_path.with_suffix(".pt")

                        meta_dict[split].append(
                            {
                                "image_path": str(image_rel_path),
                                "all_labels": f"{car_view}-{car_type}-{make}-{model}-{year}",
                                "view": car_view,
                                "type": car_type,
                                "make": make,
                                "model": model,
                                "year": year,
                            }
                        )
                    except (KeyError, json.JSONDecodeError) as e:
                        print(f"Warning: Skipping file {json_path} due to error: {e}")

            self.view_to_idx = {l: i for i, l in enumerate(sorted(view_labels))}
            self.type_to_idx = {l: i for i, l in enumerate(sorted(type_labels))}
            self.make_to_idx = {l: i for i, l in enumerate(sorted(make_labels))}
            self.model_to_idx = {l: i for i, l in enumerate(sorted(model_labels))}
            self.year_to_idx = {l: i for i, l in enumerate(sorted(year_labels))}
            self.all_labels_to_idx = {l: i for i, l in enumerate(sorted(all_labels))}

            print(f"Saving metadata to cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "train_meta": self.train_meta,
                    "val_meta": self.val_meta,
                    "test_meta": self.test_meta,
                    "view_to_idx": self.view_to_idx,
                    "type_to_idx": self.type_to_idx,
                    "make_to_idx": self.make_to_idx,
                    "model_to_idx": self.model_to_idx,
                    "year_to_idx": self.year_to_idx,
                    "all_labels_to_idx": self.all_labels_to_idx,
                }, f)

        if stage == "fit" or stage is None:
            self.train_dataset = VehicleDataset(
                self.train_meta, self.data_root, self.view_to_idx, self.type_to_idx, self.make_to_idx,
                self.model_to_idx, self.year_to_idx, self.all_labels_to_idx, flip_p=self.flip_p
            )
            self.val_dataset = VehicleDataset(
                self.val_meta, self.data_root, self.view_to_idx, self.type_to_idx, self.make_to_idx,
                self.model_to_idx, self.year_to_idx, self.all_labels_to_idx, flip_p=0.0
            )

        if stage == "test" or stage is None:
            self.test_dataset = VehicleDataset(
                self.test_meta, self.data_root, self.view_to_idx, self.type_to_idx, self.make_to_idx,
                self.model_to_idx, self.year_to_idx, self.all_labels_to_idx, flip_p=0.0
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)
