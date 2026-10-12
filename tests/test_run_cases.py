"""The run node's case loop: ordering, fail-fast, and the summary it produces.

The tuner subprocess is stubbed — this tests the loop, not KTT.
"""

from pathlib import Path

from tests.helpers.run_harness import order_log, run_two_case_node


def test_compile_failure_names_the_case(tmp_path):
    """A build failure in case 2 must say which case, or the diagnosis is unactionable."""
    from utils.build import BuildResult

    def fake_compile(iter_dir, **kw):
        ok = "small" not in Path(iter_dir).name
        return BuildResult(ok, Path(iter_dir) / "driver" if ok else None,
                           "" if ok else "error: no matching function", [])

    state = run_two_case_node(tmp_path, compile_framework=fake_compile)
    assert state.status == "proposing"
    assert "[COMPILE ERROR]" in state.run_output
    assert "small" in state.run_output


def test_all_cases_compile_before_any_tunes(tmp_path):
    """Pass 1 is the whole point: a build-level failure must not cost tuning budget."""
    run_two_case_node(tmp_path)
    assert order_log() == [
        "compile:case_big", "compile:case_small", "tune:case_big", "tune:case_small",
    ]


def test_fail_fast_skips_the_remaining_cases(tmp_path):
    state = run_two_case_node(tmp_path, fail_tuning_for="big")
    assert "tune:case_small" not in order_log()
    assert state.results_summary["failed_case"] == "big"
    assert state.results_summary["cases"]["small"] is None


def test_every_case_gets_its_own_directory_and_results(tmp_path):
    run_two_case_node(tmp_path)
    iter_dir = tmp_path / "problem" / "output" / "branches" / "b" / "iter1"
    for name in ("big", "small"):
        assert (iter_dir / f"case_{name}" / "results.json").exists()
        assert (iter_dir / f"case_{name}" / "inputs.hpp").exists()
        assert (iter_dir / f"case_{name}" / "tuner_output.txt").exists()
    assert not (iter_dir / "results.json").exists()


def test_a_problem_with_no_cases_keeps_todays_layout(tmp_path):
    state = run_two_case_node(tmp_path, cases=[])
    iter_dir = tmp_path / "problem" / "output" / "branches" / "b" / "iter1"
    assert (iter_dir / "results.json").exists()
    assert not list(iter_dir.glob("case_*"))
    assert order_log() == ["compile:iter1", "tune:iter1"]
    assert "cases" not in state.results_summary


def test_each_case_gets_its_own_budget(tmp_path):
    """The case's duration_s overrides the problem's, which is what makes a five-case
    problem affordable."""
    from nodes.run import _resolve_tuning_budget
    from utils.cases import case_list
    from tests.helpers.run_harness import PROBLEM_YAML

    cases = case_list(PROBLEM_YAML)
    assert _resolve_tuning_budget(PROBLEM_YAML, cases[0]) == (10.0, "problem.yaml")
    assert _resolve_tuning_budget(PROBLEM_YAML, cases[1]) == (5.0, "case")
