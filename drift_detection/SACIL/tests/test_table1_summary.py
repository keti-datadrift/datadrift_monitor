from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pycil_table1.py"
SPEC = importlib.util.spec_from_file_location("summarize_pycil_table1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_runs_uses_only_complete_curves_and_logging_forgetting():
    text = """
2026 [trainer.py] => NME top1 curve: [90.0]
2026 [trainer.py] => NME top1 curve: [90.0, np.float64(70.0)]
Forgetting (NME): 10.0
2026 [trainer.py] => Forgetting (NME): 10.0
2026 [trainer.py] => NME top1 curve: [80.0, 60.0]
2026 [trainer.py] => Forgetting (NME): 20.0
"""
    runs = MODULE.parse_runs(text, evaluator="NME", expected_tasks=2)
    assert len(runs) == 2
    assert runs[0].curve == (90.0, 70.0)
    assert runs[0].average == 80.0
    assert runs[0].forgetting == 10.0
    assert runs[1].final == 60.0


def test_native_log_path_matches_pycil_logging_contract():
    spec = next(item for item in MODULE.METHODS if item.slug == "sacil")
    path = MODULE._native_log_path(spec)
    assert path == (
        MODULE.PROJECT_ROOT
        / "logs"
        / "sacil"
        / "cifar100"
        / "50"
        / "5"
        / "table1_proto_sacil_nme_c100_b50_inc5_r32_1_resnet32.log"
    )


def test_best_available_runs_falls_back_to_native_log(
    tmp_path, monkeypatch
):
    native_log = tmp_path / "native.log"
    native_log.write_text(
        "\n".join(
            (
                "=> NME top1 curve: [90.0, 70.0]",
                "=> Forgetting (NME): 10.0",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "_native_log_path", lambda spec: native_log)
    spec = next(item for item in MODULE.METHODS if item.slug == "sacil")

    runs, source = MODULE._best_available_runs(
        spec,
        tmp_path / "missing-runner-logs",
        expected_tasks=2,
    )

    assert len(runs) == 1
    assert runs[0].curve == (90.0, 70.0)
    assert source == native_log
