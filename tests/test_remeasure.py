"""Post-tune re-measure (plan C9): driver marker, summary fields, run-node wiring.

GPU-less seams are unit-tested here (marker parsing, summary fields, the
cli-side precise lookup, and _remeasure_best against a stub driver). The real
driver's re-measure mode is exercised end-to-end by the full-run smoke tests,
which need a GPU.
"""

import json
import sys
from pathlib import Path

import pytest

from nodes.run import _remeasure_best
from utils.results import (
    get_results_summary,
    parse_precise_best_time_from_output,
)


def test_parse_marker():
    assert parse_precise_best_time_from_output("noise\nCACAO_PRECISE_BEST_US=123.456\n") == 123.456
    assert parse_precise_best_time_from_output("no marker here") is None


def test_summary_precise_fields(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "Results": [
            {"Status": "Ok", "Configuration": [{"Name": "TILE", "Value": 16}],
             "ComputationResults": [{"Duration": 10.0}]},
        ],
    }))
    summary = get_results_summary(results, reference_time_us=100.0)
    assert summary["precise_best_time_us"] is None
    assert summary["speedup_precise"] is None
    assert summary["speedup"] == 10.0  # single-shot semantics unchanged

    summary = get_results_summary(results, 100.0, precise_best_time_us=8.0)
    assert summary["precise_best_time_us"] == 8.0
    assert summary["speedup_precise"] == 12.5
    assert summary["best_time_us"] == 10.0  # never rewritten by the precise value


def test_framework_template_has_remeasure_mode():
    from utils.framework import assemble_framework_cpp

    meta = {
        "grid": {"x": "1"},
        "reference": {"type": "cpu_c", "file": "ref_cpu.c", "function": "f"},
        "tuning": {"duration_s": 1.0},
        "validation": {"tolerance": 1e-4},
    }
    fw = assemble_framework_cpp(meta, {}, cuda_include="/usr/local/cuda/include")
    assert 'profileResults = (argc > 8) ? argv[8] : ""' in fw
    assert "preciseRuns = (argc > 9) ? std::stoi(argv[9]) : 0" in fw
    # marker STRING also appears in a comment — require the real marker PRINT
    # (a previous half-wired version passed by matching the comment alone)
    assert fw.count("CACAO_PRECISE_BEST_US=") >= 2
    assert "<< std::fixed << std::setprecision(3)" in fw
    assert "for (int i = 0; i < preciseRuns; i++)" in fw
    assert "GetKernelDuration() / 1000.0" in fw


def _stub_driver(tmp_path: Path, body: str) -> tuple:
    """A fake `driver`: python script that prints `body` plus its argv tail
    (so tests can assert what _remeasure_best appended)."""
    stub = tmp_path / "driver.py"
    stub.write_text(
        f"import sys\nprint({body!r})\nprint('CMDARG=' + sys.argv[-2])\n"
    )
    return [sys.executable, str(stub)], str(stub)


def _results_json(tmp_path: Path) -> Path:
    it = tmp_path / "iter1"
    it.mkdir(exist_ok=True)
    (it / "results.json").write_text(json.dumps({
        "Results": [{"Status": "Ok", "Configuration": [],
                     "ComputationResults": [{"Duration": 10.0}]}],
    }))
    return it


@pytest.mark.asyncio
async def test_remeasure_best_parses_marker(tmp_path):
    cmd, _ = _stub_driver(tmp_path, "CACAO_PRECISE_BEST_US=42.5")
    it = _results_json(tmp_path)
    precise, log_text = await _remeasure_best(cmd, it)
    assert precise == 42.5
    assert "RE-MEASURE" in log_text
    # KTT LoadResults appends ".json" itself — the profileResults argv must be
    # the BASE name, else the driver looks for results.json.json (a real bug:
    # silent no-marker on every iteration).
    assert f"CMDARG={it}/results" in log_text
    assert f"CMDARG={it}/results.json" not in log_text


@pytest.mark.asyncio
async def test_remeasure_best_is_never_fatal(tmp_path):
    cmd, _ = _stub_driver(tmp_path, "driver finished without a marker")
    it = _results_json(tmp_path)
    precise, log_text = await _remeasure_best(cmd, it)
    assert precise is None  # single-shot numbers remain the fallback
    assert "driver finished" in log_text


@pytest.mark.asyncio
async def test_remeasure_skipped_without_success(tmp_path):
    it = tmp_path / "iter1"
    it.mkdir()
    (it / "results.json").write_text(json.dumps({"Results": []}))
    cmd, _ = _stub_driver(tmp_path, "CACAO_PRECISE_BEST_US=1.0")
    precise, log_text = await _remeasure_best(cmd, it)
    assert precise is None and log_text == ""


def test_cli_find_precise_result(tmp_path):
    from cli import _find_precise_result

    branch = tmp_path / "branches" / "b1"
    (branch / "iter1").mkdir(parents=True)
    (branch / "iter2").mkdir()
    states = {
        "iter1": {"results_summary": {"best_time_us": 12.0,
                                       "precise_best_time_us": 11.5,
                                       "speedup_precise": 8.7}},
        "iter2": {"results_summary": {"best_time_us": 9.0}},
    }
    for name, snap in states.items():
        (branch / name / "state.json").write_text(json.dumps(snap))

    # winning iteration (min best_time) has no precise data -> None, no guess
    assert _find_precise_result(tmp_path, {"best_time_us": 9.0,
                                            "branch_path": str(branch)}) == (None, None)
    # the 12.0 iteration does -> surfaced
    assert _find_precise_result(tmp_path, {"best_time_us": 12.0,
                                            "branch_path": str(branch)}) == (11.5, 8.7)
    assert _find_precise_result(tmp_path, {"best_time_us": None,
                                            "branch_path": str(branch)}) == (None, None)
