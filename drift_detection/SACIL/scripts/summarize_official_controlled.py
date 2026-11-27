from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "mds"
    / "results"
    / "table1_official_and_controlled.md"
)
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class MethodSpec:
    group: str
    label: str
    config: Path
    evaluator: str


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


OFFICIAL_ROOT = Path("configs/pycil/official/cifar100")
CONTROLLED_ROOT = Path("configs/pycil/controlled/cifar100")
METHODS = (
    MethodSpec(
        "official",
        "Fine-tune",
        OFFICIAL_ROOT / "finetune_b50_inc5_resnet32.json",
        "CNN",
    ),
    MethodSpec(
        "official",
        "Replay-CE",
        OFFICIAL_ROOT / "replay_b50_inc5_resnet32.json",
        "CNN",
    ),
    MethodSpec(
        "official",
        "iCaRL",
        OFFICIAL_ROOT / "icarl_b50_inc5_resnet32.json",
        "NME",
    ),
    MethodSpec(
        "official",
        "PODNet",
        OFFICIAL_ROOT / "podnet_b50_inc5_resnet32.json",
        "NME",
    ),
    MethodSpec(
        "controlled",
        "Prototype-CE control",
        CONTROLLED_ROOT / "prototype_control_nme_b50_inc5_resnet32.json",
        "NME",
    ),
    MethodSpec(
        "controlled",
        "Global-HAP",
        CONTROLLED_ROOT / "global_hap_nme_b50_inc5_resnet32.json",
        "NME",
    ),
    MethodSpec(
        "controlled",
        "Flat-LRHAP",
        CONTROLLED_ROOT / "flat_lrhap_nme_b50_inc5_resnet32.json",
        "NME",
    ),
    MethodSpec(
        "controlled",
        "SACIL",
        CONTROLLED_ROOT / "sacil_nme_b50_inc5_resnet32.json",
        "NME",
    ),
)


def _load_config(path: Path) -> dict:
    with (PROJECT_ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        rf"Forgetting \({re.escape(evaluator)}\):\s*({FLOAT_PATTERN})"
    )
    forgetting = [
        float(match.group(1))
        for match in forgetting_pattern.finditer(text)
    ]
    count = min(len(curves), len(forgetting))
    return tuple(
        RunMetrics(curves[index], forgetting[index]) for index in range(count)
    )


def native_log_paths(spec: MethodSpec) -> tuple[Path, ...]:
    config = _load_config(spec.config)
    init_cls = int(config["init_cls"])
    increment = int(config["increment"])
    log_init_cls = 0 if init_cls == increment else init_cls
    seeds = config["seed"] if isinstance(config["seed"], list) else [config["seed"]]
    base = (
        PROJECT_ROOT
        / "logs"
        / str(config["model_name"])
        / str(config["dataset"])
        / str(log_init_cls)
        / str(increment)
    )
    return tuple(
        base
        / f"{config['prefix']}_{seed}_{config['convnet_type']}.log"
        for seed in seeds
    )


def collect_runs(
    spec: MethodSpec, *, expected_tasks: int
) -> tuple[tuple[RunMetrics, ...], tuple[Path, ...]]:
    runs: list[RunMetrics] = []
    sources: list[Path] = []
    for path in dict.fromkeys(native_log_paths(spec)):
        if not path.is_file():
            continue
        parsed = parse_runs(
            path.read_text(encoding="utf-8", errors="replace"),
            evaluator=spec.evaluator,
            expected_tasks=expected_tasks,
        )
        if parsed:
            runs.extend(parsed)
            sources.append(path)
    expected = len(_load_config(spec.config)["seed"])
    return tuple(runs[:expected]), tuple(sources)


def _mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {std:.3f}"


def build_report(*, expected_tasks: int) -> tuple[str, list[str]]:
    grouped_rows: dict[str, list[str]] = {"official": [], "controlled": []}
    details: list[str] = []
    incomplete: list[str] = []
    for spec in METHODS:
        config = _load_config(spec.config)
        expected = len(config["seed"])
        runs, sources = collect_runs(spec, expected_tasks=expected_tasks)
        if len(runs) != expected:
            incomplete.append(f"{spec.label}: {len(runs)}/{expected} seeds")
        grouped_rows[spec.group].append(
            "| "
            + " | ".join(
                (
                    spec.label,
                    spec.evaluator,
                    str(len(runs)),
                    _mean_std([run.average for run in runs]),
                    _mean_std([run.final for run in runs]),
                    _mean_std([run.forgetting for run in runs]),
                )
            )
            + " |"
        )
        if runs:
            details.extend(
                [
                    f"### {spec.label}",
                    "",
                    f"- config: `{spec.config.as_posix()}`",
                    f"- implementation: `{config['implementation_source']}`",
                    "- logs: " + ", ".join(f"`{path}`" for path in sources),
                ]
            )
            for index, run in enumerate(runs, start=1):
                curve = ", ".join(f"{value:.2f}" for value in run.curve)
                details.append(
                    f"- run {index}: [{curve}], Avg={run.average:.3f}, "
                    f"Final={run.final:.3f}, Forgetting={run.forgetting:.3f}"
                )
            details.append("")

    header = [
        "| Method | Evaluator | Seeds | Average accuracy | Final accuracy | Forgetting ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines = [
        "# CIFAR-100 B50-Inc5: official baselines and controlled SACIL",
        "",
        f"- generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "- protocol: ResNet-32, AFC/PODNet order 1, 20 exemplars/class",
        "- official table: untouched upstream PyCIL learners with method-native evaluator",
        "- controlled table: one SACIL PyCIL learner; only geometry mode changes",
        "- standalone reimplementation outputs are excluded",
        "",
        "## Official PyCIL baselines",
        "",
        *header,
        *grouped_rows["official"],
        "",
        "## Controlled geometry comparison",
        "",
        *header,
        *grouped_rows["controlled"],
        "",
    ]
    if incomplete:
        lines.extend(
            ["## Incomplete", "", *[f"- {item}" for item in incomplete], ""]
        )
    lines.extend(["## Run details", "", *details])
    return "\n".join(lines).rstrip() + "\n", incomplete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize official PyCIL and controlled SACIL logs"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-tasks", type=int, default=11)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, incomplete = build_report(expected_tasks=args.expected_tasks)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)
    return 1 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
