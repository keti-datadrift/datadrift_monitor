from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src_v1"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.config import load_config_tree  # noqa: E402
from sacil.engine import UnifiedTable1Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated SACIL-v1 experiments from src_v1"
    )
    parser.add_argument("config", type=Path, help="SACIL-v1 YAML config")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config_tree(args.config)
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
        result = {
            "status": "validated",
            "framework": "sacil-unified-v1",
            "source_root": str(SOURCE_ROOT),
            "method": trainer.method,
            "geometry_mode": trainer.geometry_mode,
            "sacil_v1_options": trainer.sacil_v1_options,
            "base_checkpoint": trainer.validate_base_checkpoint(),
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
