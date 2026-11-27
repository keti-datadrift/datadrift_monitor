from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "unified" / "table1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "mds" / "results" / "table1_cifar100_b50_inc5_unified.md"
)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    run_name: str
    evaluator: str = "NME"
    backbone: str = "ResNet-32"


METHODS = (
    MethodSpec("Joint (upper bound)", "joint_nme_c100_b50_inc5_r32"),
    MethodSpec(
        "Fine-tune", "finetune_reference_hparams_nme_c100_b50_inc5_r32"
    ),
    MethodSpec(
        "Replay-CE", "replay_reference_hparams_nme_c100_b50_inc5_r32"
    ),
    MethodSpec("iCaRL", "icarl_pycil_semantics_nme_c100_b50_inc5_r32"),
    MethodSpec("iCaRL + CaSpeR-IL", "icarl_casper_nme_c100_b50_inc5_r32"),
    MethodSpec("PODNet", "podnet_nme_c100_b50_inc5_r32"),
    MethodSpec("AFC", "afc_nme_c100_b50_inc5_r32"),
    MethodSpec(
        "CREATE",
        "create_native_c100_b50_inc5_r32",
        evaluator="Native reconstruction error",
    ),
    MethodSpec("FGP-ICL", "fgp_nme_c100_b50_inc5_r32"),
    MethodSpec("iCaRL + CSCCT", "cscct_nme_c100_b50_inc5_r32"),
    MethodSpec("SACIL", "sacil_nme_c100_b50_inc5_r32"),
)


def _mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    mean = 100.0 * statistics.fmean(values)
    std = 100.0 * (statistics.pstdev(values) if len(values) > 1 else 0.0)
    return f"{mean:.3f} ± {std:.3f}"


def _load_complete_run(
    root: Path, spec: MethodSpec, seed: int, expected_sessions: int
) -> dict | None:
    path = root / spec.run_name / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["summary"]["num_sessions_completed"]) != expected_sessions:
        return None
    payload["_path"] = path
    return payload


def build_report(
    root: Path, *, seeds: tuple[int, ...], expected_sessions: int
) -> tuple[str, list[str]]:
    rows: list[str] = []
    details: list[str] = []
    incomplete: list[str] = []
    for spec in METHODS:
        runs = [
            run
            for seed in seeds
            if (
                run := _load_complete_run(root, spec, seed, expected_sessions)
            )
            is not None
        ]
        if len(runs) != len(seeds):
            incomplete.append(
                f"{spec.label}: {len(runs)}/{len(seeds)} seeds complete"
            )
        rows.append(
            "| "
            + " | ".join(
                (
                    spec.label,
                    spec.backbone,
                    spec.evaluator,
                    str(len(runs)),
                    _mean_std(
                        [
                            run["summary"]["average_incremental_accuracy"]
                            for run in runs
                        ]
                    ),
                    _mean_std(
                        [run["summary"]["final_accuracy"] for run in runs]
                    ),
                    _mean_std(
                        [
                            run["summary"]["average_forgetting"]
                            for run in runs
                        ]
                    ),
                )
            )
            + " |"
        )
        if runs:
            details.extend((f"### {spec.label}", ""))
            for run in runs:
                seed = Path(run["_path"]).parent.name.removeprefix("seed_")
                curve = ", ".join(
                    f"{100.0 * item['accuracy']:.2f}"
                    for item in run["sessions"]
                )
                details.append(f"- seed {seed}: [{curve}]")
            details.append("")

    lines = [
        "# CIFAR-100 B50-Inc5 unified Table 1",
        "",
        f"- generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "- engine: one in-repo SACIL unified runner",
        "- upstream code policy: reference-only; never executed",
        "- class order: AFC/PODNet public order 1",
        "- backbone: CIFAR ResNet-32 for every method",
        "- evaluator: NME, except CREATE's native reconstruction-error rule",
        "- memory: 20 exemplars per class with shared herding",
        "- AIA: arithmetic mean of all session accuracies, including session 0",
        f"- expected sessions: {expected_sessions}",
        "",
        "| Method | Backbone | Evaluator | Seeds | AIA | Final accuracy | Forgetting ↓ |",
        "|---|---|---|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    if incomplete:
        lines.extend(
            ("## Incomplete", "", *[f"- {item}" for item in incomplete], "")
        )
    lines.extend(("## Session curves (%)", "", *details))
    return "\n".join(lines).rstrip() + "\n", incomplete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize unified Table-1 runs")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--expected-sessions", type=int, default=11)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, incomplete = build_report(
        args.root.resolve(),
        seeds=tuple(args.seeds),
        expected_sessions=int(args.expected_sessions),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)
    if args.strict and incomplete:
        for item in incomplete:
            print(item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
