from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_explore"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.config import load_config_tree  # noqa: E402
from sacil.engine import UnifiedTable1Trainer  # noqa: E402
from sacil.provenance import build_exploration_provenance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated SACIL exploration code from src_explore without "
            "mutating or importing the preserved src tree"
        )
    )
    parser.add_argument("config", type=Path, help="method YAML config")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="reuse a compatible unified session-0 checkpoint and start at S1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config_tree(args.config)
    config["exploration_provenance"] = build_exploration_provenance(
        SOURCE_ROOT, PROJECT_ROOT / "src"
    )
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir.resolve())
    if args.run_name is not None:
        config["output"]["run_name"] = args.run_name
    trainer = UnifiedTable1Trainer(
        config,
        PROJECT_ROOT,
        max_sessions=args.max_sessions,
        base_checkpoint=args.base_checkpoint,
    )
    if args.dry_run:
        base_checkpoint = trainer.validate_base_checkpoint()
        result = {
            "status": "validated",
            "framework": "sacil-unified-explore",
            "source_root": str(SOURCE_ROOT),
            "preserved_source_root": str(PROJECT_ROOT / "src"),
            "exploration_provenance": config["exploration_provenance"],
            "method": trainer.method,
            "geometry_mode": trainer.geometry_mode,
            "geometry_options": trainer.geometry_options,
            "edge_topology_options": trainer.edge_topology_options,
            "branch_masked_kd_options": trainer.branch_masked_kd_options,
            "selective_kd_options": trainer.selective_kd_options,
            "icarl_kd_weight": trainer.icarl_kd_weight,
            "geodesic_distillation_options": (
                trainer.geodesic_distillation_options
            ),
            "feature_cosine_distillation_options": (
                trainer.feature_cosine_distillation_options
            ),
            "bgs_options": trainer.bgs_options,
            "replay_batching_options": trainer.replay_batching_options,
            "prototype_consolidation_options": (
                trainer.prototype_consolidation_options
            ),
            "casper_options": trainer.casper_options,
            "base_checkpoint": base_checkpoint,
            "method_contract": trainer.method_contract.as_dict(),
            "protocol_id": trainer.protocol.protocol_id,
            "sessions": trainer.max_sessions,
            "device": str(trainer.device),
            "run_dir": str(trainer.run_dir),
        }
    else:
        result = trainer.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
