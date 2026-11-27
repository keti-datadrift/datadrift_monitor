from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_official_controlled.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_official_controlled", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_runs_keeps_only_complete_curves():
    text = """
NME top1 curve: [90.0]
NME top1 curve: [90.0, np.float64(70.0)]
Forgetting (NME): 10.0
NME top1 curve: [80.0, 60.0]
Forgetting (NME): 20.0
"""
    runs = MODULE.parse_runs(text, evaluator="NME", expected_tasks=2)
    assert len(runs) == 2
    assert runs[0].curve == (90.0, 70.0)
    assert runs[0].average == 80.0
    assert runs[1].final == 60.0


def test_native_paths_cover_every_config_seed():
    spec = next(item for item in MODULE.METHODS if item.label == "Fine-tune")
    paths = MODULE.native_log_paths(spec)
    assert len(paths) == 3
    assert paths[0].name.endswith("_1_resnet32.log")
    assert paths[-1].name.endswith("_3_resnet32.log")
