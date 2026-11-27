from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_cmpt"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.cmpt.drift_grouping import (  # noqa: E402
    analyze_drift_grouping_support,
    summarize_drift_grouping_sessions,
)
from sacil.cmpt.evaluator import (  # noqa: E402
    CMPTCheckpointEvaluator,
    CMPTExperimentSettings,
    _load_model,
    _paired_support_features,
)
from sacil.config import load_config_tree  # noqa: E402
from sacil.memory import ExemplarMemory  # noqa: E402
from sacil.provenance import build_exploration_provenance  # noqa: E402
from sacil.utils import dump_json, git_commit  # noqa: E402


DEFAULT_CONFIGS = (
    "configs/cmpt/common_recipe/evaluate_icarl_component_ablation.yaml",
    "configs/cmpt/common_recipe/evaluate_lucir_natural_component_ablation.yaml",
    "configs/cmpt/common_recipe/evaluate_fgp_icl_component_ablation.yaml",
    "configs/cmpt/common_recipe/evaluate_podnet_component_ablation.yaml",
    "configs/cmpt/common_recipe/evaluate_afc_component_ablation.yaml",
    "configs/cmpt/common_recipe/evaluate_cscct_component_ablation.yaml",
    "configs/cmpt/common_recipe/evaluate_casper_il_component_ablation.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether visually close old classes share drift directions"
        )
    )
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="CMPT evaluation configs; defaults to seven CIFAR-100 learners",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/cmpt/drift_grouping/cifar100/seed_1/results.json"
        ),
    )
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--split-repeats", type=int, default=50)
    parser.add_argument("--class-crossfit-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def analyze_trajectory(
    config_path: Path,
    *,
    device: str,
    permutations: int,
    split_repeats: int,
    class_crossfit_folds: int,
    seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(load_config_tree(config_path))
    config["device"] = str(device)
    settings = CMPTExperimentSettings.from_config(config, PROJECT_ROOT)
    evaluator = CMPTCheckpointEvaluator(
        settings,
        PROJECT_ROOT,
        source_root=SOURCE_ROOT,
    )
    trainer = evaluator._trainer()
    sessions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position in range(1, len(evaluator.checkpoints)):
        previous_checkpoint = evaluator.checkpoints[position - 1]
        current_checkpoint = evaluator.checkpoints[position]
        session_id = int(current_checkpoint["session_id"])
        old_class_count = trainer.protocol.session(session_id).start
        previous_model = _load_model(
            trainer, previous_checkpoint, session_id - 1
        )
        current_model = _load_model(trainer, current_checkpoint, session_id)
        trainer.model = current_model
        trainer.memory = ExemplarMemory.from_state_dict(
            previous_checkpoint["memory"]
        )
        support = _paired_support_features(
            trainer,
            previous_model,
            current_model,
            session_id - 1,
            horizontal_flip=settings.support_horizontal_flip,
        )
        result = analyze_drift_grouping_support(
            support.old_fit_features,
            support.current_fit_features,
            support.fit_targets,
            support.old_exemplar_features,
            support.current_exemplar_features,
            support.targets,
            num_classes=old_class_count,
            ridge=settings.affine_ridge,
            permutations=permutations,
            split_repeats=split_repeats,
            class_crossfit_folds=class_crossfit_folds,
            seed=seed + 100 * position,
        )
        result.update(
            {
                "session_id": session_id,
                "previous_checkpoint": str(
                    evaluator.checkpoint_paths[position - 1]
                ),
                "current_checkpoint": str(
                    evaluator.checkpoint_paths[position]
                ),
            }
        )
        sessions.append(result)
        raw = result["raw_drift"]["spearman_rho"]
        residual = result["global_residual_drift"]["spearman_rho"]
        cross_fitted = result[
            "cross_fitted_global_residual_drift"
        ]["spearman_rho"]
        print(
            f"{settings.learner} S{session_id}: "
            f"raw-rho={raw:+.3f}, residual-rho={residual:+.3f}, "
            f"crossfit-residual-rho={cross_fitted:+.3f}"
        )
        del previous_model, current_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "learner": settings.learner,
        "config": str(config_path.resolve()),
        "checkpoint_directory": str(settings.checkpoint_directory),
        "trajectory": evaluator.audit.to_dict(),
        "sessions": sessions,
        "summary": summarize_drift_grouping_sessions(sessions),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    if args.permutations <= 0:
        raise ValueError("--permutations must be positive")
    if args.split_repeats <= 0:
        raise ValueError("--split-repeats must be positive")
    if args.class_crossfit_folds < 2:
        raise ValueError("--class-crossfit-folds must be at least 2")
    output = (
        args.output
        if args.output.is_absolute()
        else PROJECT_ROOT / args.output
    ).resolve()
    if output.exists() and not args.force:
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(f"complete result already exists: {output}")
            return 0
        raise FileExistsError(f"partial result exists: {output}")

    configs = list(args.configs) or [Path(value) for value in DEFAULT_CONFIGS]
    configs = [
        path if path.is_absolute() else PROJECT_ROOT / path for path in configs
    ]
    started = time.perf_counter()
    trajectories: list[dict[str, Any]] = []
    for index, config_path in enumerate(configs):
        trajectory = analyze_trajectory(
            config_path.resolve(),
            device=args.device,
            permutations=args.permutations,
            split_repeats=args.split_repeats,
            class_crossfit_folds=args.class_crossfit_folds,
            seed=args.seed + 10000 * index,
        )
        trajectories.append(trajectory)
        partial = {
            "schema_version": 1,
            "status": "running",
            "dataset": "cifar100",
            "seed": int(args.seed),
            "device": str(args.device),
            "permutations": int(args.permutations),
            "split_repeats": int(args.split_repeats),
            "class_crossfit_folds": int(args.class_crossfit_folds),
            "test_data_used": False,
            "checkpoint_weights_modified": False,
            "trajectories": trajectories,
            "elapsed_seconds": time.perf_counter() - started,
            "git_commit": git_commit(PROJECT_ROOT),
            "source_provenance": build_exploration_provenance(
                SOURCE_ROOT, PROJECT_ROOT / "src_explore"
            ),
        }
        dump_json(partial, output)

    payload = dict(partial)
    payload["status"] = "complete"
    payload["elapsed_seconds"] = time.perf_counter() - started
    dump_json(payload, output)
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
