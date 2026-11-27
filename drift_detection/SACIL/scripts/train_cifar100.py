from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.config import load_config  # noqa: E402
from sacil.engine import SACILTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Replay/HAP/SACIL-v0 on CIFAR-100 CIL"
    )
    parser.add_argument("config", type=Path, help="YAML experiment config")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and dataset without training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir.resolve())
    if args.run_name is not None:
        config["output"]["run_name"] = args.run_name
    trainer = SACILTrainer(
        config,
        PROJECT_ROOT,
        resume=args.resume,
        max_sessions=args.max_sessions,
    )
    if args.dry_run:
        summary = {
            "status": "validated",
            "protocol_id": trainer.protocol.protocol_id,
            "num_sessions": trainer.protocol.num_sessions,
            "method": trainer.method_name,
            "device": str(trainer.device),
            "run_dir": str(trainer.run_dir),
        }
    else:
        summary = trainer.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

