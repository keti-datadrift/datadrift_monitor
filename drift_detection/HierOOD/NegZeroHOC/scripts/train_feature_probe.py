from __future__ import annotations

import argparse
from argparse import Namespace
from collections import Counter, defaultdict
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, load_feature_file, save_json
from negzerohoc.output_layout import resolve_experiment_artifact, resolve_shared_feature_dir
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from negzerohoc.training_data import build_positive_edge_examples, gather_image_features, group_examples_by_parent, node_path, sample_examples


class FeatureClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, model_type: str = "linear", hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        if model_type == "linear":
            self.net = nn.Linear(input_dim, output_dim)
        elif model_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unsupported linear_probe.model: {model_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)

    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    features_cfg = cfg.get("features", {})
    inference_cfg = cfg.get("inference", {})
    probe_cfg = cfg.get("linear_probe", {})

    experiment_name = experiment_cfg.get("name", "idea3")
    output_root = experiment_cfg.get("output_root", "outputs")
    probe_name = probe_cfg.get("name", f"{experiment_name}-clip-feature-probe")

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=output_root,
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        features_dir=str(resolve_shared_feature_dir(
            features_cfg.get("dir") or inference_cfg.get("features_dir"),
            output_root=output_root,
        )) if features_cfg.get("dir") or inference_cfg.get("features_dir") else None,
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        probe=probe_cfg.get("probe", "both"),
        model=probe_cfg.get("model", "linear"),
        hidden_dim=int(probe_cfg.get("hidden_dim", 1024)),
        dropout=float(probe_cfg.get("dropout", 0.0)),
        normalize_features=bool(probe_cfg.get("normalize_features", True)),
        epochs=int(probe_cfg.get("epochs", 100)),
        batch_size=int(probe_cfg.get("batch_size", 1024)),
        parents_per_step=int(probe_cfg.get("parents_per_step", 4)),
        lr=float(probe_cfg.get("lr", 1e-3)),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-4)),
        eval_every=int(probe_cfg.get("eval_every", 1)),
        eval_batch_size=int(probe_cfg.get("eval_batch_size", 4096)),
        checkpoint=str(resolve_experiment_artifact(
            probe_cfg.get("checkpoint"),
            output_root=output_root,
            experiment_name=experiment_name,
            kind="checkpoints",
            default_filename=f"{probe_name}.pt",
        )),
        result_path=str(resolve_experiment_artifact(
            probe_cfg.get("result_path"),
            output_root=output_root,
            experiment_name=experiment_name,
            kind="results",
            default_filename=f"{probe_name}.result",
        )),
        diagnostics_path=str(resolve_experiment_artifact(
            probe_cfg.get("diagnostics_path"),
            output_root=output_root,
            experiment_name=experiment_name,
            kind="diagnostics",
            default_filename=f"{probe_name}.json",
        )),
    )


def maybe_normalize(features: torch.Tensor, enabled: bool) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1) if enabled else features.float()


def metric_summary(metrics: dict) -> dict:
    out = {}
    for key in ["acc", "balanced_acc", "avg_hdist", "balanced_hdist"]:
        value = metrics.get(key)
        if value is not None:
            out[key] = float(value)
    return out


@torch.no_grad()
def evaluate_leaf_classification(args, model: nn.Module, payload: dict, device: str) -> dict:
    model.eval()
    features = maybe_normalize(payload["features"], args.normalize_features)
    targets = payload["targets"].long()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    steps = 0

    for start in range(0, features.shape[0], args.eval_batch_size):
        batch_x = features[start:start + args.eval_batch_size].to(device)
        batch_y = targets[start:start + args.eval_batch_size].to(device)
        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        total_loss += float(loss.detach().cpu())
        total_correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
        total_seen += int(batch_y.numel())
        steps += 1

    return {
        "loss": total_loss / max(1, steps),
        "acc": total_correct / max(1, total_seen),
    }


def train_leaf_probe(args, train_payload: dict, val_payload: dict | None, device: str) -> tuple[nn.Module, list[dict]]:
    features = maybe_normalize(train_payload["features"], args.normalize_features)
    targets = train_payload["targets"].long()
    input_dim = int(features.shape[1])
    num_classes = len(train_payload["classes"])
    model = FeatureClassifier(input_dim, num_classes, args.model, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(features.shape[0], generator=generator)
        iterator = range(0, features.shape[0], args.batch_size)
        if tqdm:
            iterator = tqdm(iterator, desc=f"leaf probe epoch {epoch}/{args.epochs}", leave=False)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        steps = 0

        for start in iterator:
            indices = order[start:start + args.batch_size]
            batch_x = features.index_select(0, indices).to(device)
            batch_y = targets.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
            total_seen += int(batch_y.numel())
            steps += 1

        train_loss = total_loss / max(1, steps)
        train_acc = total_correct / max(1, total_seen)
        stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "steps": steps,
        }
        if val_payload is not None and args.eval_every > 0 and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs):
            val_stats = evaluate_leaf_classification(args, model, val_payload, device)
            stats["val_loss"] = val_stats["loss"]
            stats["val_acc"] = val_stats["acc"]
        history.append(stats)
        message = f"leaf epoch {epoch}: train_loss={stats['train_loss']:.6f}, train_acc={stats['train_acc']:.6f}"
        if "val_acc" in stats:
            message += f", val_loss={stats['val_loss']:.6f}, val_acc={stats['val_acc']:.6f}"
        print(message)

    return model, history


@torch.no_grad()
def predict_leaf_probe(args, model: nn.Module, payload: dict, train_classes: list[str], hierarchy, device: str) -> tuple[torch.Tensor, dict]:
    model.eval()
    features = maybe_normalize(payload["features"], args.normalize_features)
    preds = []
    raw_correct = 0
    raw_total = 0
    same_class_space = list(payload["classes"]) == list(train_classes)

    for start in range(0, features.shape[0], args.eval_batch_size):
        batch = features[start:start + args.eval_batch_size].to(device)
        logits = model(batch)
        pred_class = logits.argmax(dim=1).cpu()
        preds.append(pred_class)
        if same_class_space:
            target = payload["targets"][start:start + args.eval_batch_size].long()
            raw_correct += int((pred_class == target).sum())
            raw_total += int(target.numel())

    pred_class_indices = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    ds2node = hierarchy.gen_ds2node_map(train_classes)
    pred_nodes = ds2node[pred_class_indices.long()]
    diagnostics = {
        "raw_class_acc": raw_correct / raw_total if raw_total else None,
        "pred_class_counts": dict(Counter(int(x) for x in pred_class_indices.tolist()).most_common(20)),
    }
    return pred_nodes.long(), diagnostics


class LocalProbeBank(nn.Module):
    def __init__(self, input_dim: int, parent_children: dict[str, list[str]], model_type: str, hidden_dim: int, dropout: float):
        super().__init__()
        self.parents = sorted(parent_children)
        self.parent_to_key = {parent: f"p{idx}" for idx, parent in enumerate(self.parents)}
        self.parent_children = {parent: list(parent_children[parent]) for parent in self.parents}
        self.models = nn.ModuleDict({
            self.parent_to_key[parent]: FeatureClassifier(input_dim, len(children), model_type, hidden_dim, dropout)
            for parent, children in self.parent_children.items()
        })

    def forward_parent(self, parent: str, features: torch.Tensor) -> torch.Tensor:
        return self.models[self.parent_to_key[parent]](features)


def train_local_probe(args, train_payload: dict, hierarchy, device: str) -> tuple[LocalProbeBank, list[dict]]:
    features = maybe_normalize(train_payload["features"], args.normalize_features)
    examples = build_positive_edge_examples(hierarchy, train_payload)
    grouped = group_examples_by_parent(examples)
    parent_children = {
        parent: list(hierarchy.parent2children[parent])
        for parent, items in grouped.items()
        if parent in hierarchy.parent2children and len(hierarchy.parent2children[parent]) >= 2 and items
    }
    model = LocalProbeBank(int(features.shape[1]), parent_children, args.model, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    parents = sorted(parent_children)
    n_per_parent = max(1, args.batch_size // max(1, args.parents_per_step))
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = list(parents)
        rng.shuffle(shuffled)
        chunks = [shuffled[i:i + args.parents_per_step] for i in range(0, len(shuffled), args.parents_per_step)]
        iterator = tqdm(chunks, desc=f"local probe epoch {epoch}/{args.epochs}", leave=False) if tqdm else chunks
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0

        for parent_chunk in iterator:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            accs = []
            for parent in parent_chunk:
                parent_examples = sample_examples(grouped[parent], n_per_parent, rng)
                if not parent_examples:
                    continue
                children = parent_children[parent]
                child_to_idx = {child: idx for idx, child in enumerate(children)}
                targets = torch.tensor([child_to_idx[ex.child] for ex in parent_examples], dtype=torch.long, device=device)
                batch_features = gather_image_features(features, parent_examples, device)
                logits = model.forward_parent(parent, batch_features)
                loss = F.cross_entropy(logits, targets)
                losses.append(loss)
                accs.append(float((logits.argmax(dim=1) == targets).float().mean().detach().cpu()))

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_acc += sum(accs) / max(1, len(accs))
            steps += 1

        stats = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, steps),
            "local_acc": epoch_acc / max(1, steps),
            "steps": steps,
        }
        history.append(stats)
        print(f"local epoch {epoch}: loss={stats['loss']:.6f}, local_acc={stats['local_acc']:.6f}")

    return model, history


@torch.no_grad()
def predict_local_probe(args, model: LocalProbeBank, payload: dict, hierarchy, device: str) -> tuple[torch.Tensor, dict]:
    model.eval()
    features = maybe_normalize(payload["features"], args.normalize_features)
    node_to_idx = {node: idx for idx, node in enumerate(hierarchy.id_node_list)}
    pred_nodes = []
    stop_depth_counts = Counter()
    stop_node_counts = Counter()

    for start in range(0, features.shape[0], args.eval_batch_size):
        batch = features[start:start + args.eval_batch_size].to(device)
        current = ["root"] * int(batch.shape[0])
        active = torch.ones(batch.shape[0], dtype=torch.bool, device=device)

        while bool(active.any()):
            next_active = torch.zeros_like(active)
            for parent in sorted(set(current[i] for i in torch.nonzero(active, as_tuple=False).flatten().tolist())):
                if parent not in model.parent_to_key:
                    continue
                row_indices = [i for i, node in enumerate(current) if node == parent and bool(active[i])]
                if not row_indices:
                    continue
                idx = torch.tensor(row_indices, dtype=torch.long, device=device)
                logits = model.forward_parent(parent, batch.index_select(0, idx))
                pred_child_idx = logits.argmax(dim=1).cpu().tolist()
                children = model.parent_children[parent]
                for local_i, child_idx in zip(row_indices, pred_child_idx):
                    child = children[child_idx]
                    current[local_i] = child
                    if child in model.parent_to_key:
                        next_active[local_i] = True
            active = next_active

        pred_nodes.extend(current)

    for node in pred_nodes:
        stop_depth_counts[len(hierarchy.node_ancestors.get(node, []))] += 1
        stop_node_counts[node] += 1
    preds = torch.tensor([node_to_idx[node] for node in pred_nodes], dtype=torch.long)
    diagnostics = {
        "stop_depth_counts": dict(sorted(stop_depth_counts.items())),
        "stop_node_counts": dict(stop_node_counts.most_common(20)),
    }
    return preds, diagnostics


@torch.no_grad()
def oracle_local_diagnostics(args, model: LocalProbeBank, payload: dict, hierarchy, device: str) -> dict:
    model.eval()
    features = maybe_normalize(payload["features"], args.normalize_features)
    examples = build_positive_edge_examples(hierarchy, payload)
    by_depth = defaultdict(lambda: {"correct": 0, "total": 0})
    by_parent = defaultdict(lambda: {"correct": 0, "total": 0})

    for parent, parent_examples in group_examples_by_parent(examples).items():
        if parent not in model.parent_to_key:
            continue
        children = model.parent_children[parent]
        child_to_idx = {child: idx for idx, child in enumerate(children)}
        for start in range(0, len(parent_examples), args.eval_batch_size):
            batch_examples = parent_examples[start:start + args.eval_batch_size]
            batch_features = gather_image_features(features, batch_examples, device)
            logits = model.forward_parent(parent, batch_features)
            pred = logits.argmax(dim=1).cpu().tolist()
            targets = [child_to_idx[ex.child] for ex in batch_examples]
            depth = len(hierarchy.node_ancestors.get(parent, []))
            for pred_idx, target_idx in zip(pred, targets):
                correct = int(pred_idx == target_idx)
                by_depth[depth]["correct"] += correct
                by_depth[depth]["total"] += 1
                by_parent[parent]["correct"] += correct
                by_parent[parent]["total"] += 1

    def finish(stats):
        total = stats["total"]
        return {
            "acc": stats["correct"] / total if total else None,
            "correct": int(stats["correct"]),
            "total": int(total),
        }

    worst = {
        parent: {
            "depth": len(hierarchy.node_ancestors.get(parent, [])),
            **finish(stats),
        }
        for parent, stats in sorted(
            by_parent.items(),
            key=lambda item: (item[1]["correct"] / max(1, item[1]["total"]), -item[1]["total"]),
        )[:20]
    }
    return {
        "by_depth": {str(depth): finish(stats) for depth, stats in sorted(by_depth.items())},
        "worst_parents": worst,
    }


def evaluate_probe(args, hierarchy, features_dir: Path, leaf_model=None, local_model=None, train_classes=None) -> dict:
    dists_mats = make_distance_mats(hierarchy)
    results = {
        "args": vars(args),
        "probe": args.probe,
    }
    for probe_name, model in [("leaf", leaf_model), ("local", local_model)]:
        if model is None:
            continue
        results[probe_name] = {}
        for split in ["val", "ood"]:
            payload = load_feature_file(features_dir / f"{split}-features.pt")
            if probe_name == "leaf":
                preds, diagnostics = predict_leaf_probe(args, model, payload, train_classes, hierarchy, args.device)
            else:
                preds, diagnostics = predict_local_probe(args, model, payload, hierarchy, args.device)
                diagnostics["oracle_local"] = oracle_local_diagnostics(args, model, payload, hierarchy, args.device)
            node_labels, metrics = evaluate_split(hierarchy, payload, preds.cpu(), dists_mats=dists_mats)
            results[probe_name][split] = {
                "preds": preds.cpu(),
                "targets": node_labels.cpu(),
                "metrics": metrics,
                "diagnostics": diagnostics,
            }
        results[probe_name]["mixed"] = mixed_summary(
            results[probe_name]["val"]["metrics"],
            results[probe_name]["ood"]["metrics"],
        )
    return results


def json_safe_result(results: dict) -> dict:
    summary = {"probe": results.get("probe"), "args": results.get("args", {})}
    for probe_name in ["leaf", "local"]:
        if probe_name not in results:
            continue
        summary[probe_name] = {}
        for split in ["val", "ood"]:
            summary[probe_name][split] = {
                "metrics": metric_summary(results[probe_name][split]["metrics"]),
                "diagnostics": results[probe_name][split]["diagnostics"],
            }
        summary[probe_name]["mixed"] = {
            key: float(value) for key, value in results[probe_name]["mixed"].items()
        }
    return summary


def print_eval_summary(diagnostics: dict) -> None:
    for probe_name in ["leaf", "local"]:
        if probe_name not in diagnostics:
            continue
        print(f"== {probe_name} probe evaluation ==")
        for split in ["val", "ood"]:
            metrics = diagnostics[probe_name][split]["metrics"]
            acc = metrics.get("acc")
            bacc = metrics.get("balanced_acc")
            bh = metrics.get("balanced_hdist")
            print(
                f"{split}: "
                f"acc={acc:.6f}, "
                f"balanced_acc={bacc:.6f}, "
                f"balanced_hdist={bh:.6f}"
            )
        mixed = diagnostics[probe_name]["mixed"]
        print(
            "mixed: "
            f"balanced_acc={mixed['mixed_balanced_acc']:.6f}, "
            f"balanced_hdist={mixed['mixed_balanced_hdist']:.6f}"
        )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def main() -> None:
    args = parse_args()
    if not args.features_dir:
        raise ValueError("Missing features.dir or inference.features_dir in config")
    if args.probe not in {"leaf", "local", "both"}:
        raise ValueError("linear_probe.probe must be one of: leaf, local, both")

    configure_reproducibility(args.seed)
    device = available_device(args.device)
    args.device = device

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    features_dir = Path(args.features_dir)
    train_payload = load_feature_file(features_dir / "train-features.pt")
    val_payload = load_feature_file(features_dir / "val-features.pt") if args.probe in {"leaf", "both"} else None

    leaf_model = None
    local_model = None
    train_classes = list(train_payload["classes"])

    if args.probe in {"leaf", "both"}:
        leaf_model, leaf_history = train_leaf_probe(args, train_payload, val_payload, device)
    else:
        leaf_history = None

    if args.probe in {"local", "both"}:
        local_model, local_history = train_local_probe(args, train_payload, hierarchy, device)
    else:
        local_history = None

    results = evaluate_probe(
        args,
        hierarchy,
        features_dir,
        leaf_model=leaf_model,
        local_model=local_model,
        train_classes=train_classes,
    )
    diagnostics = json_safe_result(results)
    diagnostics["train_history"] = {
        "leaf": leaf_history,
        "local": local_history,
    }
    print_eval_summary(diagnostics)

    ensure_dir(Path(args.result_path).parent)
    ensure_dir(Path(args.checkpoint).parent)
    torch.save(results, args.result_path)
    torch.save(
        {
            "args": vars(args),
            "train_classes": train_classes,
            "leaf_state_dict": leaf_model.state_dict() if leaf_model is not None else None,
            "local_state_dict": local_model.state_dict() if local_model is not None else None,
            "local_parent_children": local_model.parent_children if local_model is not None else None,
            "diagnostics": diagnostics,
        },
        args.checkpoint,
    )
    save_json(args.diagnostics_path, diagnostics)
    print(f"saved result: {args.result_path}")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved diagnostics: {args.diagnostics_path}")


if __name__ == "__main__":
    main()
