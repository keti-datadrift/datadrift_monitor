from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_RUNS = {
    "finetune": "finetune_25spc_2sessions",
    "replay": "replay_25spc_2sessions",
    "icarl": "icarl_25spc_2sessions",
    "podnet": "podnet_25spc_2sessions",
    "afc": "afc_25spc_2sessions",
    "create": "create_25spc_2sessions",
    "fgp": "fgp_25spc_2sessions",
    "cscct": "cscct_25spc_2sessions",
    "casper": "casper_25spc_2sessions",
    "joint": "joint_25spc_2sessions",
    "sacil": "sacil_25spc_2sessions",
}

EXPECTED_INCREMENTAL_COMPONENTS = {
    "finetune": {"classification"},
    "replay": {"classification"},
    "icarl": {"classification", "kd"},
    "podnet": {"classification", "pod_flat", "pod_spatial"},
    "afc": {"classification", "afc"},
    "create": {"classification", "contrastive", "kd"},
    "fgp": {"classification", "graph"},
    "cscct": {"classification", "kd", "csc", "ct"},
    "casper": {"classification", "regularization", "spectral"},
    "joint": {"classification"},
    "sacil": {"classification", "geometry"},
}


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError(f"non-finite value at {path}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _assert_tensor_tree_finite(value: Any, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise AssertionError(f"non-finite tensor at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_tensor_tree_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_tensor_tree_finite(item, f"{path}[{index}]")


def verify_run(root: Path, method: str, run_name: str, seed: int) -> dict[str, Any]:
    run_dir = root / run_name / f"seed_{seed}"
    session_path = run_dir / "sessions.jsonl"
    metrics_path = run_dir / "metrics.json"
    if not session_path.is_file() or not metrics_path.is_file():
        raise AssertionError(f"missing result files for {method}: {run_dir}")
    records = [
        json.loads(line)
        for line in session_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 2:
        raise AssertionError(f"{method}: expected two sessions, got {len(records)}")
    if [record["session_id"] for record in records] != [0, 1]:
        raise AssertionError(f"{method}: invalid session ids")
    if [record["seen_class_count"] for record in records] != [50, 55]:
        raise AssertionError(f"{method}: invalid seen-class counts")
    if [record["memory_size"] for record in records] != [1000, 1100]:
        raise AssertionError(f"{method}: invalid memory sizes")
    expected_samples = 1375 if method == "joint" else (125 if method == "finetune" else 1125)
    if records[0]["training"]["samples"] != 1250:
        raise AssertionError(f"{method}: base subset is not 50x25")
    if records[1]["training"]["samples"] != expected_samples:
        raise AssertionError(f"{method}: invalid incremental training size")
    components = set(records[1]["training"]["epoch_logs"][-1])
    missing_components = EXPECTED_INCREMENTAL_COMPONENTS[method] - components
    if missing_components:
        raise AssertionError(
            f"{method}: missing incremental losses {sorted(missing_components)}"
        )
    _assert_finite(records, f"{method}.sessions")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _assert_finite(metrics, f"{method}.metrics")

    checkpoint_summaries = []
    for session_id, expected_classes in enumerate((50, 55)):
        checkpoint_path = run_dir / "checkpoints" / f"session_{session_id:02d}.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint["framework"] != "sacil-unified":
            raise AssertionError(f"{method}: invalid framework metadata")
        if checkpoint["pycil_used"] or checkpoint["reference_code_executed"]:
            raise AssertionError(f"{method}: external reference was executed")
        if checkpoint["method"] != method or checkpoint["session_id"] != session_id:
            raise AssertionError(f"{method}: checkpoint identity mismatch")
        memory = checkpoint["memory"]
        if memory["exemplars_per_class"] != 20:
            raise AssertionError(f"{method}: memory limit is not 20")
        classes = memory["indices"]
        if len(classes) != expected_classes:
            raise AssertionError(f"{method}: checkpoint memory class count mismatch")
        flattened: list[int] = []
        for class_id, indices in classes.items():
            if len(indices) != 20 or len(set(indices)) != 20:
                raise AssertionError(
                    f"{method}: class {class_id} does not contain 20 unique exemplars"
                )
            flattened.extend(int(index) for index in indices)
        if len(set(flattened)) != expected_classes * 20:
            raise AssertionError(f"{method}: duplicated exemplar across classes")
        _assert_tensor_tree_finite(checkpoint["model"], f"{method}.model")
        _assert_tensor_tree_finite(
            checkpoint.get("class_means"), f"{method}.class_means"
        )
        checkpoint_summaries.append(
            {
                "session_id": session_id,
                "classes": expected_classes,
                "memory_size": len(flattened),
            }
        )
    return {
        "method": method,
        "status": "passed",
        "final_accuracy": metrics["summary"]["final_accuracy"],
        "checkpoints": checkpoint_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/validation/reference_parity"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    results = [
        verify_run(root, method, run_name, args.seed)
        for method, run_name in EXPECTED_RUNS.items()
    ]
    report = {
        "status": "passed",
        "protocol": {
            "dataset": "CIFAR-100",
            "train_samples_per_class": 25,
            "sessions": 2,
            "base_classes": 50,
            "increment_classes": 5,
            "exemplars_per_class": 20,
        },
        "methods": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
