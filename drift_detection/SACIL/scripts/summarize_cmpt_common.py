from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/lucir/seed_1/cmpt_rigid_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/fgp_icl_common_v2_lr2/seed_1/cmpt_rigid_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/casper_icarl_common_v2_lr03/seed_1/cmpt_rigid_matched_nme.json",
)
AFFINE_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/lucir_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/fgp_icl_common_v2_lr2_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/casper_icarl_common_v2_lr03_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
)
CIFAR100_BASELINE_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/icarl/seed_1/cmpt_rigid_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/replay/seed_1/cmpt_rigid_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/podnet/seed_1/cmpt_rigid_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/afc/seed_1/cmpt_rigid_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/cscct/seed_1/cmpt_rigid_matched_nme.json",
)
CIFAR100_BASELINE_AFFINE_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/icarl_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/replay_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/podnet_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/afc_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/cscct_affine_ridge1e-2/seed_1/cmpt_affine_matched_nme.json",
)
IMAGENET100_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/lucir/seed_1/cmpt_rigid.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/fgp_icl/seed_1/cmpt_rigid.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/casper_il/seed_1/cmpt_rigid.json",
)
IMAGENET100_AFFINE_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/lucir/seed_1/cmpt_affine_ridge.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/fgp_icl/seed_1/cmpt_affine_ridge.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/casper_il/seed_1/cmpt_affine_ridge.json",
)
IMAGENET100_BASELINE_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/icarl/seed_1/cmpt_rigid.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/replay/seed_1/cmpt_rigid.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/podnet/seed_1/cmpt_rigid.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/afc/seed_1/cmpt_rigid.json",
)
IMAGENET100_BASELINE_AFFINE_RESULTS = (
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/icarl/seed_1/cmpt_affine_ridge.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/replay/seed_1/cmpt_affine_ridge.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/podnet/seed_1/cmpt_affine_ridge.json",
    PROJECT_ROOT
    / "outputs/cmpt/imagenet100_b50_inc5/evaluation/afc/seed_1/cmpt_affine_ridge.json",
)
LEGACY_RESULT = (
    PROJECT_ROOT
    / "outputs/cmpt/common_recipe/evaluation/lucir_like_legacy/seed_1/cmpt_rigid_matched_nme.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired NME/CMPT shared-protocol results"
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="include the margin-off LUCIR-like bridge result",
    )
    parser.add_argument(
        "--affine",
        action="store_true",
        help="summarize the ridge-regularized affine CMPT evaluations",
    )
    parser.add_argument(
        "--imagenet100",
        action="store_true",
        help="summarize ImageNet-100 B50-Inc5 results instead of CIFAR-100",
    )
    parser.add_argument(
        "--all-learners",
        action="store_true",
        help=(
            "include iCaRL, Replay, PODNet, AFC, and CIFAR-100 CSCCT "
            "paired evaluations"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optionally write the Markdown table to this path",
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"CMPT result is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"CMPT result is not complete: {path}")
    return payload


def _mean_residual(payload: dict) -> float:
    values = [
        float(record["transport_diagnostics"]["fit_residual"])
        for record in payload["records"]
        if record["transport_diagnostics"].get("fit_residual") is not None
    ]
    return sum(values) / len(values)


def _markdown(
    payloads: list[dict],
    *,
    affine: bool = False,
    dataset: str = "CIFAR-100",
) -> str:
    transport = "Affine CMPT" if affine else "CMPT"
    lines = [
        f"# {transport} {dataset} shared-protocol results (seed 1)",
        "",
        "| Learner | Native classifier | Native AIA | NME AIA | CMPT AIA | CMPT−NME | Native Final | NME Final | CMPT Final | CMPT−NME | Mean fit residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        summary = payload["summary"]
        if "native_aia_percent" not in summary:
            raise ValueError(
                f"{payload['learner']} lacks native-classifier metrics; "
                "rerun its existing evaluate_cmpt.py command once to "
                "upgrade the result"
            )
        lines.append(
            "| {learner} | {native_classifier} | {native_aia:.3f} | "
            "{baseline_aia:.3f} | {cmpt_aia:.3f} | {aia_delta:+.3f} | "
            "{native_final:.3f} | {baseline_final:.3f} | "
            "{cmpt_final:.3f} | {final_delta:+.3f} | {residual:.5f} |".format(
                learner=payload["learner"],
                native_classifier=payload["native_classifier"]["classifier"],
                native_aia=float(summary["native_aia_percent"]),
                baseline_aia=float(summary["baseline_aia_percent"]),
                cmpt_aia=float(summary["cmpt_aia_percent"]),
                aia_delta=float(summary["aia_delta_percent_points"]),
                native_final=float(summary["native_final_percent"]),
                baseline_final=float(summary["baseline_final_percent"]),
                cmpt_final=float(summary["cmpt_final_percent"]),
                final_delta=float(summary["final_delta_percent_points"]),
                residual=_mean_residual(payload),
            )
        )
    lines.extend(
        [
            "",
            "All deltas are paired comparisons from the same frozen session checkpoints.",
            "Query horizontal-flip TTA is disabled.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.imagenet100:
        paths = list(
            IMAGENET100_AFFINE_RESULTS
            if args.affine
            else IMAGENET100_RESULTS
        )
        if args.all_learners:
            paths.extend(
                IMAGENET100_BASELINE_AFFINE_RESULTS
                if args.affine
                else IMAGENET100_BASELINE_RESULTS
            )
    else:
        paths = list(AFFINE_RESULTS if args.affine else DEFAULT_RESULTS)
        if args.all_learners:
            paths.extend(
                CIFAR100_BASELINE_AFFINE_RESULTS
                if args.affine
                else CIFAR100_BASELINE_RESULTS
            )
    if args.include_legacy:
        if args.affine:
            raise ValueError("--include-legacy is unavailable with --affine")
        paths.append(LEGACY_RESULT)
    markdown = _markdown(
        [_load(path) for path in paths],
        affine=bool(args.affine),
        dataset="ImageNet-100 B50-Inc5" if args.imagenet100 else "CIFAR-100",
    )
    print(markdown)
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
