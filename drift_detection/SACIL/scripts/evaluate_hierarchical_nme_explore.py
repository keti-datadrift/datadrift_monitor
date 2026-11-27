from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_explore"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.engine import UnifiedTable1Trainer  # noqa: E402
from sacil.engine.checkpoint import load_checkpoint  # noqa: E402
from sacil.engine.evaluator import (  # noqa: E402
    compute_nme_class_means,
    evaluate_nme,
)
from sacil.features import collect_features  # noqa: E402
from sacil.memory import ExemplarMemory  # noqa: E402
from sacil.methods.hierarchical_nme import (  # noqa: E402
    hierarchical_shrink_nme_means,
)
from sacil.provenance import build_exploration_provenance  # noqa: E402
from sacil.utils import dump_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc co-moving hierarchical-NME checkpoint rescore"
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shrinkage", type=float, default=0.35)
    parser.add_argument("--taxonomy-temperature", type=float, default=0.2)
    parser.add_argument(
        "--full-data-oracle",
        action="store_true",
        help=(
            "also compute a diagnostic mean from all historical training "
            "images; this violates CIL access and is never a reported result"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict] = []
    for checkpoint_path in args.checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        config = copy.deepcopy(checkpoint["config"])
        config["device"] = args.device
        config["output"]["directory"] = str(
            (PROJECT_ROOT / "outputs" / "explore" / "nme_rescore_tmp").resolve()
        )
        config["output"]["run_name"] = "hierarchical_nme_rescore"
        config["exploration_provenance"] = build_exploration_provenance(
            SOURCE_ROOT, PROJECT_ROOT / "src"
        )
        session_id = int(checkpoint["session_id"])
        trainer = UnifiedTable1Trainer(
            config, PROJECT_ROOT, max_sessions=session_id + 1
        )
        seen = trainer.protocol.session(session_id).stop
        model = trainer._new_model(seen).to(trainer.device)
        expected_type = str(checkpoint.get("model_type", ""))
        if expected_type and type(model).__name__ != expected_type:
            raise TypeError(
                f"checkpoint model mismatch: {expected_type} != "
                f"{type(model).__name__}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        trainer.model = model
        trainer.memory = ExemplarMemory.from_state_dict(checkpoint["memory"])

        memory_loader = trainer._memory_loader(session_id, augment=False)
        means = compute_nme_class_means(
            model,
            memory_loader,
            trainer.device,
            seen,
            horizontal_flip=bool(
                config.get("evaluation", {}).get("horizontal_flip", True)
            ),
        ).cpu()
        memory = collect_features(model, memory_loader, trainer.device)
        shrunken, diagnostics, tree = hierarchical_shrink_nme_means(
            means,
            memory.features,
            memory.targets,
            trainer.protocol.seen_classes(session_id),
            taxonomy_temperature=args.taxonomy_temperature,
            max_shrinkage=args.max_shrinkage,
        )
        test_dataset = trainer.data.cumulative_test_dataset(session_id)
        test_loader = trainer._loader(
            test_dataset, shuffle=False, session_id=session_id + 11000
        )
        old = trainer.protocol.session(session_id).start
        baseline = evaluate_nme(
            model, test_loader, trainer.device, old, means
        ).to_dict()
        hierarchical = evaluate_nme(
            model, test_loader, trainer.device, old, shrunken
        ).to_dict()
        classifier = getattr(model, "classifier", None)
        classifier_weight = getattr(classifier, "weight", None)
        proxy = None
        if isinstance(classifier_weight, torch.Tensor):
            proxy = evaluate_nme(
                model,
                test_loader,
                trainer.device,
                old,
                torch.nn.functional.normalize(
                    classifier_weight.detach().cpu().float(), dim=1
                ),
            ).to_dict()
        full_data_oracle = None
        old_full_data_hybrid_oracle = None
        if args.full_data_oracle:
            oracle_dataset = trainer.data.train_eval_dataset_for_classes(
                trainer.protocol.seen_classes(session_id)
            )
            oracle_loader = trainer._loader(
                oracle_dataset,
                shuffle=False,
                session_id=session_id + 12000,
            )
            oracle_means = compute_nme_class_means(
                model,
                oracle_loader,
                trainer.device,
                seen,
                horizontal_flip=bool(
                    config.get("evaluation", {}).get(
                        "horizontal_flip", True
                    )
                ),
            )
            full_data_oracle = evaluate_nme(
                model,
                test_loader,
                trainer.device,
                old,
                oracle_means,
            ).to_dict()
            hybrid_means = means.clone()
            if old > 0:
                hybrid_means[:old] = oracle_means[:old].cpu()
            old_full_data_hybrid_oracle = evaluate_nme(
                model,
                test_loader,
                trainer.device,
                old,
                hybrid_means,
            ).to_dict()
        alpha = torch.tensor(diagnostics.shrinkage)
        records.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "session_id": session_id,
                "baseline": baseline,
                "hierarchical": hierarchical,
                "learned_proxy_diagnostic": proxy,
                "full_historical_data_oracle": full_data_oracle,
                "old_full_data_new_memory_hybrid_oracle": (
                    old_full_data_hybrid_oracle
                ),
                "delta_accuracy": hierarchical["accuracy"]
                - baseline["accuracy"],
                "shrinkage_summary": {
                    "mean": float(alpha.mean().item()),
                    "min": float(alpha.min().item()),
                    "max": float(alpha.max().item()),
                    "capped_ratio": float(
                        (alpha >= args.max_shrinkage - 1.0e-8)
                        .float()
                        .mean()
                        .item()
                    ),
                },
                "shrinkage_diagnostics": diagnostics.to_dict(),
                "tree": tree,
            }
        )

    payload = {
        "method": "co-moving empirical-Bayes hierarchical NME shrinkage",
        "max_shrinkage": args.max_shrinkage,
        "taxonomy_temperature": args.taxonomy_temperature,
        "test_labels_used_for_selection": False,
        "full_historical_data_oracle_enabled": args.full_data_oracle,
        "full_historical_data_oracle_is_cil_valid": False,
        "exploration_provenance": build_exploration_provenance(
            SOURCE_ROOT, PROJECT_ROOT / "src"
        ),
        "records": records,
    }
    output = args.output.resolve()
    dump_json(payload, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
