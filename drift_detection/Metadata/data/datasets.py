from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SplitInfo:
    id_classes: tuple[int, ...]
    ood_classes: tuple[int, ...]


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, input_kind: str):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.input_kind = input_kind

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _split_id_train_val(
    x: np.ndarray,
    y: np.ndarray,
    id_classes: tuple[int, ...],
    val_ratio: float,
    seed: int,
):
    id_mask = np.isin(y, id_classes)
    x_id = x[id_mask]
    y_id = y[id_mask]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y_id))
    val_size = int(round(len(indices) * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x_id[train_idx], y_id[train_idx], x_id[val_idx], y_id[val_idx]


def _split_test(x: np.ndarray, y: np.ndarray, id_classes: tuple[int, ...]):
    id_mask = np.isin(y, id_classes)
    return x[id_mask], y[id_mask], x[~id_mask], y[~id_mask]


def _load_fashionmnist(root: Path):
    base = root / "FashionMNIST"
    train = pd.read_csv(base / "fashion-mnist_train.csv")
    test = pd.read_csv(base / "fashion-mnist_test.csv")
    x_train = train.drop(columns=["label"]).to_numpy(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_train = train["label"].to_numpy(np.int64)
    x_test = test.drop(columns=["label"]).to_numpy(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_test = test["label"].to_numpy(np.int64)
    split = SplitInfo(id_classes=tuple(range(7)), ood_classes=(7, 8, 9))
    return x_train, y_train, x_test, y_test, split, "image"


def _load_usps(root: Path):
    with h5py.File(root / "usps.h5", "r") as f:
        x_train = f["train"]["data"][:].astype(np.float32).reshape(-1, 1, 16, 16)
        y_train = f["train"]["target"][:].astype(np.int64)
        x_test = f["test"]["data"][:].astype(np.float32).reshape(-1, 1, 16, 16)
        y_test = f["test"]["target"][:].astype(np.int64)
    split = SplitInfo(id_classes=tuple(range(7)), ood_classes=(7, 8, 9))
    return x_train, y_train, x_test, y_test, split, "image"


def _load_reuters10k(root: Path):
    train = np.load(root / "reuters-10k" / "train.npy", allow_pickle=True).item()
    test = np.load(root / "reuters-10k" / "test.npy", allow_pickle=True).item()
    x_train = train["data"].astype(np.float32)
    y_train = train["label"].astype(np.int64)
    x_test = test["data"].astype(np.float32)
    y_test = test["label"].astype(np.int64)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    classes, counts = np.unique(y_train, return_counts=True)
    id_classes = tuple(int(c) for c in classes[np.argsort(counts)[::-1][:2]])
    ood_classes = tuple(int(c) for c in classes if int(c) not in id_classes)
    split = SplitInfo(id_classes=id_classes, ood_classes=ood_classes)
    return x_train, y_train, x_test, y_test, split, "tabular"


def _read_reuters_raw_split(base: Path, split_name: str, label_by_path: dict[str, str]):
    texts = []
    labels = []
    split_dir = base / split_name
    for path in sorted(split_dir.iterdir(), key=lambda p: int(p.name)):
        rel = f"{split_name}/{path.name}"
        if rel not in label_by_path:
            continue
        texts.append(path.read_text(encoding="latin-1", errors="ignore"))
        labels.append(label_by_path[rel])
    return texts, np.asarray(labels, dtype=object)


def _reuters_primary_labels(base: Path):
    label_by_path = {}
    for line in (base / "cats.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        label_by_path[parts[0]] = parts[1]
    return label_by_path


def _encode_reuters_labels(train_labels, test_labels, num_classes: int):
    classes, counts = np.unique(train_labels, return_counts=True)
    ordered = classes[np.argsort(counts)[::-1]]
    selected = ordered[:num_classes]
    label_to_id = {label: idx for idx, label in enumerate(selected)}

    train_mask = np.isin(train_labels, selected)
    test_mask = np.isin(test_labels, selected)
    y_train = np.asarray([label_to_id[label] for label in train_labels[train_mask]], dtype=np.int64)
    y_test = np.asarray([label_to_id[label] for label in test_labels[test_mask]], dtype=np.int64)
    return selected, train_mask, test_mask, y_train, y_test


def _balanced_id_classes(train_labels: np.ndarray, selected: np.ndarray, id_class_count: int):
    counts = np.asarray([np.sum(train_labels == label) for label in selected], dtype=np.int64)
    target = int(counts.sum() // 2)
    dp = {(0, 0): None}

    for idx, count in enumerate(counts.tolist()):
        for key in list(dp.keys())[::-1]:
            k, total = key
            next_key = (k + 1, total + count)
            if k + 1 <= id_class_count and next_key not in dp:
                dp[next_key] = (k, total, idx)

    candidates = [total for k, total in dp if k == id_class_count]
    if not candidates:
        raise ValueError(f"Cannot choose {id_class_count} ID classes from {len(selected)} classes.")
    best_total = min(candidates, key=lambda total: abs(total - target))

    id_indices = []
    state = (id_class_count, best_total)
    while state != (0, 0):
        prev_k, prev_total, idx = dp[state]
        id_indices.append(idx)
        state = (prev_k, prev_total)
    return tuple(sorted(int(idx) for idx in id_indices))


def _tokenize_clip_texts(texts: list[str], config: dict):
    try:
        from transformers import CLIPTokenizer
    except ImportError as exc:
        raise ImportError("CLIP text encoder requires transformers.") from exc

    model_name = config["model"].get("clip_model_name", "openai/clip-vit-base-patch32")
    max_length = int(config["model"].get("max_length", 77))
    local_files_only = bool(config["model"].get("local_files_only", False))
    tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return np.stack([encoded["input_ids"], encoded["attention_mask"]], axis=1).astype(np.int64)


def _load_reuters_raw(root: Path, config: dict):
    base = root / "reuters"
    num_classes = int(config["dataset"].get("num_classes", 46))
    id_class_count = int(config["dataset"].get("id_class_count", 2))
    split_strategy = config["dataset"].get("split_strategy", "top_frequency")
    label_by_path = _reuters_primary_labels(base)
    train_texts, train_labels = _read_reuters_raw_split(base, "training", label_by_path)
    test_texts, test_labels = _read_reuters_raw_split(base, "test", label_by_path)

    selected, train_mask, test_mask, y_train, y_test = _encode_reuters_labels(
        train_labels, test_labels, num_classes
    )
    train_texts = [text for text, keep in zip(train_texts, train_mask) if keep]
    test_texts = [text for text, keep in zip(test_texts, test_mask) if keep]

    id_class_count = min(id_class_count, num_classes)
    if split_strategy == "balanced_sample_count":
        id_classes = _balanced_id_classes(train_labels[train_mask], selected, id_class_count)
    elif split_strategy == "top_frequency":
        id_classes = tuple(range(id_class_count))
    else:
        raise ValueError(f"Unknown Reuters split_strategy: {split_strategy}")
    ood_classes = tuple(range(len(selected))[len(id_classes):])
    ood_classes = tuple(idx for idx in range(len(selected)) if idx not in id_classes)
    split = SplitInfo(id_classes=id_classes, ood_classes=ood_classes)
    model_type = config["model"].get("type", "tfidf_mlp").lower()

    if model_type == "clip_text":
        x_train = _tokenize_clip_texts(train_texts, config)
        x_test = _tokenize_clip_texts(test_texts, config)
        return x_train, y_train, x_test, y_test, split, "clip_text"

    max_features = int(config["model"].get("max_features", 20000))
    id_train_texts = [
        text for text, label in zip(train_texts, y_train)
        if label in split.id_classes
    ]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=True,
        stop_words="english",
        dtype=np.float32,
    )
    vectorizer.fit(id_train_texts)
    x_train = vectorizer.transform(train_texts).toarray().astype(np.float32)
    x_test = vectorizer.transform(test_texts).toarray().astype(np.float32)
    return x_train, y_train, x_test, y_test, split, "tabular"


def build_datasets(config: dict):
    dataset_name = config["dataset"]["name"].lower()
    root = Path(config["dataset"].get("root", "dataset"))
    val_ratio = float(config["dataset"].get("val_ratio", 0.2))
    seed = int(config.get("seed", 42))

    if dataset_name == "fashionmnist":
        x_train, y_train, x_test, y_test, split, input_kind = _load_fashionmnist(root)
    elif dataset_name == "usps":
        x_train, y_train, x_test, y_test, split, input_kind = _load_usps(root)
    elif dataset_name in {"reuters10k", "reuters-10k"}:
        x_train, y_train, x_test, y_test, split, input_kind = _load_reuters10k(root)
    elif dataset_name == "reuters":
        x_train, y_train, x_test, y_test, split, input_kind = _load_reuters_raw(root, config)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']['name']}")

    tr_x, tr_y, val_x, val_y = _split_id_train_val(
        x_train, y_train, split.id_classes, val_ratio, seed
    )
    id_x, id_y, ood_x, ood_y = _split_test(x_test, y_test, split.id_classes)

    return {
        "train": ArrayDataset(tr_x, tr_y, input_kind),
        "val": ArrayDataset(val_x, val_y, input_kind),
        "id_test": ArrayDataset(id_x, id_y, input_kind),
        "ood_test": ArrayDataset(ood_x, ood_y, input_kind),
        "split_info": split,
        "input_shape": tuple(tr_x.shape[1:]),
        "input_dim": int(np.prod(tr_x.shape[1:])),
        "input_kind": input_kind,
    }


def build_dataloaders(config: dict):
    datasets = build_datasets(config)
    batch_size = int(config["training"].get("batch_size", 128))
    num_workers = int(config["training"].get("num_workers", 0))
    loaders = {
        key: DataLoader(
            datasets[key],
            batch_size=batch_size,
            shuffle=(key == "train"),
            num_workers=num_workers,
        )
        for key in ("train", "val", "id_test", "ood_test")
    }
    return datasets, loaders
