from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_table1.py"
SPEC = importlib.util.spec_from_file_location("summarize_table1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_unified_summary_reads_only_complete_seed_runs(tmp_path) -> None:
    spec = MODULE.METHODS[0]
    run_dir = tmp_path / spec.run_name / "seed_1"
    run_dir.mkdir(parents=True)
    payload = {
        "summary": {
            "average_incremental_accuracy": 0.75,
            "final_accuracy": 0.65,
            "average_forgetting": 0.1,
            "num_sessions_completed": 2,
        },
        "sessions": [{"accuracy": 0.85}, {"accuracy": 0.65}],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    report, incomplete = MODULE.build_report(
        tmp_path, seeds=(1,), expected_sessions=2
    )

    assert "75.000 ± 0.000" in report
    assert "[85.00, 65.00]" in report
    assert spec.label not in "\n".join(incomplete)
