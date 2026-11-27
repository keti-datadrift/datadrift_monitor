from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = (
    PROJECT_ROOT / "outputs" / "pycil" / "table1" / "runner_logs_resnet32"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "mds"
    / "results"
    / "table1_cifar100_b50_inc5_resnet32.md"
)
CONFIG_ROOT = PROJECT_ROOT / "configs" / "pycil" / "table1" / "cifar100"


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    label: str
    config: str
    evaluator: str


METHODS = (
    MethodSpec(
        "joint", "Joint (upper bound)", "joint_nme_b50_inc5_resnet32.json", "NME"
    ),
    MethodSpec(
        "finetune", "Fine-tune", "finetune_nme_b50_inc5_resnet32.json", "NME"
    ),
    MethodSpec(
        "replay", "Replay-CE", "replay_nme_b50_inc5_resnet32.json", "NME"
    ),
    MethodSpec("icarl", "iCaRL", "icarl_nme_b50_inc5_resnet32.json", "NME"),
    MethodSpec(
        "podnet", "PODNet", "podnet_nme_b50_inc5_resnet32.json", "NME"
    ),
    MethodSpec("afc", "AFC", "afc_nme_b50_inc5_resnet32.json", "NME"),
    MethodSpec(
        "create",
        "CREATE",
        "create_native_b50_inc5_resnet32.json",
        "CNN",
    ),
    MethodSpec("fgp", "FGP-ICL", "fgp_nme_b50_inc5_resnet32.json", "NME"),
    MethodSpec(
        "cscct",
        "iCaRL + CSCCT",
        "icarl_cscct_nme_b50_inc5_resnet32.json",
        "NME",
    ),
    MethodSpec(
        "casper",
        "iCaRL + CaSpeR",
        "icarl_casper_nme_b50_inc5_resnet32.json",
        "NME",
    ),
    MethodSpec(
        "sacil",
        "Proto-SACIL",
        "proto_sacil_nme_b50_inc5_resnet32.json",
        "NME",
    ),
)


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class RunMetrics:
    curve: tuple[float, ...]
    forgetting: float

    @property
    def average(self) -> float:
        return statistics.fmean(self.curve)

    @property
    def final(self) -> float:
        return self.curve[-1]


def _numbers(payload: str) -> tuple[float, ...]:
    normalized = re.sub(r"np\.float(?:32|64)\(", "(", payload)
    return tuple(float(value) for value in re.findall(FLOAT_PATTERN, normalized))


def parse_runs(
    text: str, *, evaluator: str, expected_tasks: int
) -> tuple[RunMetrics, ...]:
    curve_pattern = re.compile(
        rf"{re.escape(evaluator)} top1 curve:\s*\[([^\]]*)\]"
    )
    curves = [
        values
        for match in curve_pattern.finditer(text)
        if len(values := _numbers(match.group(1))) == expected_tasks
    ]
    forgetting_pattern = re.compile(
        rf"=> Forgetting \({re.escape(evaluator)}\):\s*({FLOAT_PATTERN})"
    )
    forgetting = [
        float(match.group(1))
        for match in forgetting_pattern.finditer(text)
    ]
    count = min(len(curves), len(forgetting))
    return tuple(
        RunMetrics(curve=curves[index], forgetting=forgetting[index])
        for index in range(count)
    )


def _mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {std:.3f}"


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _seed_count(config_path: Path) -> int:
    config = _load_config(config_path)
    seeds = config.get("seed", [])
    return len(seeds) if isinstance(seeds, list) else 1


def _native_log_path(spec: MethodSpec) -> Path:
    config = _load_config(CONFIG_ROOT / spec.config)
    seeds = config.get("seed", [1])
    first_seed = seeds[0] if isinstance(seeds, list) else seeds
    init_cls = int(config["init_cls"])
    increment = int(config["increment"])
    log_init_cls = 0 if init_cls == increment else init_cls
    return (
        PROJECT_ROOT
        / "logs"
        / str(config["model_name"])
        / str(config["dataset"])
        / str(log_init_cls)
        / str(increment)
        / (
            f"{config['prefix']}_{first_seed}_"
            f"{config['convnet_type']}.log"
        )
    )


def _best_available_runs(
    spec: MethodSpec,
    log_root: Path,
    *,
    expected_tasks: int,
) -> tuple[tuple[RunMetrics, ...], Path | None]:
    candidates = (
        log_root / f"{spec.slug}.stdout.log",
        _native_log_path(spec),
    )
    best_runs: tuple[RunMetrics, ...] = ()
    best_path: Path | None = None
    for path in candidates:
        if not path.is_file():
            continue
        runs = parse_runs(
            path.read_text(encoding="utf-8", errors="replace"),
            evaluator=spec.evaluator,
            expected_tasks=expected_tasks,
        )
        if best_path is None or len(runs) > len(best_runs):
            best_runs = runs
            best_path = path
    return best_runs, best_path


def build_report(
    log_root: Path, *, expected_tasks: int
) -> tuple[str, list[str]]:
    rows: list[str] = []
    details: list[str] = []
    incomplete: list[str] = []
    for spec in METHODS:
        expected_seeds = _seed_count(CONFIG_ROOT / spec.config)
        runs, source_path = _best_available_runs(
            spec,
            log_root,
            expected_tasks=expected_tasks,
        )
        if len(runs) != expected_seeds:
            incomplete.append(
                f"{spec.slug}: {len(runs)}/{expected_seeds} completed seeds"
            )
        averages = [run.average for run in runs]
        finals = [run.final for run in runs]
        forgetting = [run.forgetting for run in runs]
        rows.append(
            "| "
            + " | ".join(
                (
                    spec.label,
                    spec.evaluator,
                    str(len(runs)),
                    _mean_std(averages),
                    _mean_std(finals),
                    _mean_std(forgetting),
                )
            )
            + " |"
        )
        if runs:
            details.extend(
                (
                    f"### {spec.label}",
                    "",
                    f"- config: `{spec.config}`",
                    f"- main evaluator: `{spec.evaluator}`",
                    f"- source log: `{source_path}`",
                )
            )
            for index, run in enumerate(runs, start=1):
                curve = ", ".join(f"{value:.2f}" for value in run.curve)
                details.append(
                    f"- seed run {index}: [{curve}], "
                    f"Avg={run.average:.3f}, Final={run.final:.3f}, "
                    f"Forgetting={run.forgetting:.3f}"
                )
            details.append("")

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# CIFAR-100 B50-Inc5 ResNet-32 Table 1 결과",
        "",
        f"- generated: `{timestamp}`",
        "- class order: AFC/PODNet order 1",
        "- sessions: base 50 classes + 10 increments × 5 classes (총 11 sessions)",
        "- memory: 20 exemplars/class, maximum 2,000 exemplars",
        "- seeds: 1, 2, 3",
        "- 표준편차: population standard deviation",
        "- CREATE만 논문 고유 reconstruction classifier(CNN column), "
        "나머지는 NME를 주 지표로 사용",
        "- Proto-SACIL은 learnable FC/old-logit KD 없이 prototype CE로 "
        "학습하고 NME로 평가",
        "",
        "| Method | Evaluator | Seeds | Average Accuracy ↑ | "
        "Final Accuracy ↑ | Forgetting ↓ |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    if incomplete:
        lines.extend(
            [
                "## 미완료 항목",
                "",
                *[f"- {item}" for item in incomplete],
                "",
            ]
        )
    lines.extend(["## Seed별 curve", "", *details])
    return "\n".join(lines).rstrip() + "\n", incomplete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed SACIL Table-1 PyCIL logs."
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-tasks", type=int, default=11)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any configured seed is missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_tasks <= 0:
        raise ValueError("expected-tasks must be positive")
    report, incomplete = build_report(
        args.log_root.expanduser().resolve(),
        expected_tasks=args.expected_tasks,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")
    if args.strict and incomplete:
        raise SystemExit("Incomplete runs: " + "; ".join(incomplete))


if __name__ == "__main__":
    main()
