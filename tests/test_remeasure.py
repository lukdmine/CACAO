"""Post-tune re-measure: driver marker, summary fields, run-node wiring, ranking.

The GPU-less seams are tested here. The real driver's remeasure mode runs in
test_driver_modes_integration.py.
"""

import asyncio
import json
import sys
from pathlib import Path

from nodes.run import _remeasure_best, precise_runs
from tests.helpers.run_harness import commands, order_log, run_two_case_node
from utils.build import driver_command
from utils.results import get_results_summary, parse_precise_best_time_from_output


def test_parse_marker():
    assert parse_precise_best_time_from_output("noise\nCACAO_PRECISE_BEST_US=123.456\n") == 123.456
    assert parse_precise_best_time_from_output("no marker here") is None


def _one_result(path: Path, ok: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    results = (
        [{"Status": "Ok", "Configuration": [{"Name": "TILE", "Value": 16}],
          "ComputationResults": [{"Duration": 10.0}]}] if ok else []
    )
    path.write_text(json.dumps({"Results": results}), encoding="utf-8")
    return path


def test_summary_precise_fields(tmp_path):
    results = _one_result(tmp_path / "results.json")
    summary = get_results_summary(results, reference_time_us=100.0)
    assert summary["precise_best_time_us"] is None
    assert summary["speedup_precise"] is None
    assert summary["speedup"] == 10.0

    summary = get_results_summary(results, 100.0, precise_best_time_us=8.0)
    assert summary["precise_best_time_us"] == 8.0
    assert summary["speedup_precise"] == 12.5
    assert summary["best_time_us"] == 10.0  # never rewritten by the precise value


def test_driver_command_puts_the_replay_count_in_argv_11():
    cmd = driver_command(
        "./driver", platform=0, device=0, duration=1.0, tolerance=1e-4,
        output_base="results", kernel_file="k.cu", ref_file="r.cu",
        profile_results="/it/results", mode="remeasure", precise_runs=25,
    )
    assert cmd[8:] == ["/it/results", "remeasure", "", "25"]


def test_framework_template_has_remeasure_mode():
    from utils.framework import assemble_framework_cpp

    meta = {
        "grid": {"x": "1"},
        "reference": {"type": "cpu_c", "file": "ref_cpu.c", "function": "f"},
        "tuning": {"duration_s": 1.0},
        "validation": {"tolerance": 1e-4},
    }
    fw = assemble_framework_cpp(meta, {}, cuda_include="/usr/local/cuda/include")
    # The searcher mode keeps argv[9] and the MCMC seed argv[10]; the count is argv[11].
    assert 'mode = (argc > 9) ? argv[9] : "random"' in fw
    assert "preciseRuns = (argc > 11) ? std::stoi(argv[11]) : 25" in fw
    assert 'if (mode == "remeasure")' in fw
    assert 'std::cout << "CACAO_PRECISE_BEST_US="' in fw
    assert "for (int i = 0; i < preciseRuns; i++)" in fw
    assert "GetKernelDuration() / 1000.0" in fw


def _stub_cmd(tmp_path: Path, body: str) -> list:
    stub = tmp_path / "driver.py"
    stub.write_text(f"print({body!r})\n", encoding="utf-8")
    return [sys.executable, str(stub)]


def test_remeasure_best_parses_marker(tmp_path):
    it = _one_result(tmp_path / "iter1" / "results.json").parent
    precise, text = asyncio.run(
        _remeasure_best(_stub_cmd(tmp_path, "CACAO_PRECISE_BEST_US=42.5"), it)
    )
    assert precise == 42.5
    assert "RE-MEASURE" in text


def test_remeasure_best_is_never_fatal(tmp_path):
    it = _one_result(tmp_path / "iter1" / "results.json").parent
    precise, text = asyncio.run(
        _remeasure_best(_stub_cmd(tmp_path, "driver finished without a marker"), it)
    )
    assert precise is None
    assert "driver finished" in text

    precise, text = asyncio.run(_remeasure_best([str(tmp_path / "no-such-driver")], it))
    assert precise is None and "re-measure failed" in text


def test_remeasure_skipped_without_success(tmp_path):
    it = _one_result(tmp_path / "iter1" / "results.json", ok=False).parent
    precise, text = asyncio.run(
        _remeasure_best(_stub_cmd(tmp_path, "CACAO_PRECISE_BEST_US=1.0"), it)
    )
    assert precise is None and text == ""


def test_precise_runs_env_overrides_config(monkeypatch):
    import config

    monkeypatch.delenv("CACAO_PRECISE_RUNS", raising=False)
    assert precise_runs() == config.PRECISE_RUNS
    monkeypatch.setenv("CACAO_PRECISE_RUNS", "0")
    assert precise_runs() == 0
    monkeypatch.setenv("CACAO_PRECISE_RUNS", "seven")
    assert precise_runs() == config.PRECISE_RUNS


def test_run_node_remeasures_every_case(tmp_path, monkeypatch):
    """Each case is re-measured from its own directory, after its own tune, and a
    multi-case summary carries the geometric mean of the precise speedups."""
    monkeypatch.delenv("CACAO_PRECISE_RUNS", raising=False)
    state = run_two_case_node(tmp_path, precise_us=50.0)
    assert [o for o in order_log() if not o.startswith("compile")] == [
        "tune:case_big", "remeasure:case_big", "tune:case_small", "remeasure:case_small",
    ]
    remeasures = [c for c in commands() if "remeasure" in c]
    assert len(remeasures) == 2
    for cmd in remeasures:
        case_dir = Path(cmd[6]).parent
        assert cmd[8:] == [str(case_dir / "results"), "remeasure", "", "25"]
    summary = state.results_summary
    assert summary["precise_best_time_us"] == 50.0
    assert summary["cases"]["small"]["precise_best_time_us"] == 50.0
    # No reference time in the harness, so there is no precise speedup to average.
    assert summary["speedup_precise"] is None


def test_run_node_skips_the_remeasure_when_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CACAO_PRECISE_RUNS", "0")
    run_two_case_node(tmp_path, precise_us=50.0)
    assert not [o for o in order_log() if o.startswith("remeasure")]


def test_aggregate_precise_speedup_needs_every_case():
    from utils.results import aggregate_case_summaries

    a = {"has_success": True, "speedup": 2.0, "speedup_precise": 4.0}
    b = {"has_success": True, "speedup": 2.0, "speedup_precise": 9.0}
    assert aggregate_case_summaries({"a": a, "b": b}, "a")["speedup_precise"] == 6.0
    b = dict(b, speedup_precise=None)
    assert aggregate_case_summaries({"a": a, "b": b}, "a")["speedup_precise"] is None


def test_outranks_uses_precise_scores_only_when_both_sides_have_them():
    from utils.branch_bests import outranks

    noisy_win = {"speedup": 2.1, "best_time_us": 95.0, "speedup_precise": 1.9}
    steady = {"speedup": 2.0, "best_time_us": 100.0, "speedup_precise": 2.0}
    assert outranks(steady, noisy_win)
    assert not outranks(noisy_win, steady)
    # One side never re-measured: single-shot against single-shot.
    old = {"speedup": 2.0, "best_time_us": 100.0}
    assert outranks(noisy_win, old)
    assert not outranks(steady, dict(steady))


def test_find_precise_result_reads_the_winning_iteration(tmp_path):
    from nodes.merge import find_precise_result

    branch = tmp_path / "branches" / "b1"
    states = {
        "iter1": {"results_summary": {"best_time_us": 12.0, "precise_best_time_us": 11.5,
                                      "speedup_precise": 8.7}},
        "iter2": {"results_summary": {"best_time_us": 9.0}},
    }
    for name, snap in states.items():
        (branch / name).mkdir(parents=True)
        (branch / name / "state.json").write_text(json.dumps(snap), encoding="utf-8")

    # The winning iteration has no precise data: None, never another iteration's.
    assert find_precise_result({"best_time_us": 9.0, "branch_path": str(branch)}) == (None, None)
    assert find_precise_result({"best_time_us": 12.0, "branch_path": str(branch)}) == (11.5, 8.7)
    assert find_precise_result({"best_time_us": None, "branch_path": str(branch)}) == (None, None)
