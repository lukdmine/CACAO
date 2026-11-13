"""Space size, exhaustive-time estimate and searcher choice."""

import subprocess
from pathlib import Path

from utils import space
from utils.space import (
    choose_searcher,
    count_configurations,
    estimate_exhaustive_s,
    parse_config_count,
    per_config_cost_s,
    seed_is_reachable,
)

P = 'tuner.AddParameter(kernel, "{n}", std::vector<uint64_t>{{{v}}});'


def test_the_count_line_is_parsed():
    assert parse_config_count("noise\nCACAO_CONFIG_COUNT 4096\n") == 4096
    assert parse_config_count("Tuning done") is None


def test_cost_is_measured_from_a_real_run(all_ok):
    cost = per_config_cost_s(all_ok)
    assert 0 < cost < 60
    assert estimate_exhaustive_s(100, all_ok) == 100 * cost


def test_cost_of_a_missing_or_empty_file_is_none(tmp_path):
    assert per_config_cost_s(tmp_path / "nope.json") is None
    empty = tmp_path / "results.json"
    empty.write_text('{"Results": []}')
    assert per_config_cost_s(empty) is None
    assert estimate_exhaustive_s(10, empty) is None


def test_searcher_choice_respects_the_margin():
    assert choose_searcher(899.0, 1000.0) == "exhaustive"
    assert choose_searcher(901.0, 1000.0) == "mcmc"
    assert choose_searcher(None, 1000.0) == "mcmc"


def test_a_seed_whose_parameter_was_dropped_is_unreachable():
    assert not seed_is_reachable({"TILE": 16, "UNROLL": 2}, P.format(n="TILE", v="16, 32"))


def test_a_seed_whose_value_was_removed_is_unreachable():
    assert not seed_is_reachable({"TILE": 8}, P.format(n="TILE", v="16, 32"))


def test_a_seed_missing_a_parameter_widening_added_is_unreachable():
    """KTT indexes a seed by every declared parameter; a short one reads past the end."""
    params_src = P.format(n="N", v="1") + "\n" + P.format(n="TILE", v="8, 16")
    assert not seed_is_reachable({"N": 1}, params_src)


def test_numeric_values_match_across_spellings():
    assert seed_is_reachable({"TILE": 16.0}, P.format(n="TILE", v="16, 32"))


def test_count_configurations_runs_count_mode(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="CACAO_CONFIG_COUNT 12\n", stderr="")

    monkeypatch.setattr(space.subprocess, "run", fake_run)
    n = count_configurations(
        tmp_path / "driver", tmp_path, gpu_index=0,
        kernel_file=tmp_path / "k.cu", ref_file=tmp_path / "r.cu",
    )
    assert n == 12
    assert seen["cmd"][-3:] == ["", "count", ""]


def test_count_configurations_is_none_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        space.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="CACAO_CONFIG_COUNT 5\n", stderr=""),
    )
    assert count_configurations(
        tmp_path / "d", tmp_path, gpu_index=0, kernel_file=Path("k"), ref_file=Path("r")
    ) is None


def test_count_for_iteration_stages_the_primary_case(monkeypatch, tmp_path):
    from utils.build import BuildResult

    meta = {"cases": [{"name": "a", "scalars": {}}, {"name": "b", "scalars": {}}]}
    staged, counted = [], []
    cdir = tmp_path / "case_a"

    monkeypatch.setattr(space, "load_inputs_spec", lambda p: "spec")
    monkeypatch.setattr(space, "load_regions", lambda d: {})
    monkeypatch.setattr(space, "reference_build_extras", lambda pd, case: ([], []))

    def fake_stage(problem_dir, iter_dir, spec, meta_, cases, case, regions):
        staged.append(case)
        return cdir

    monkeypatch.setattr(space, "stage_case", fake_stage)
    monkeypatch.setattr(
        space, "compile_framework",
        lambda d, **kw: BuildResult(True, cdir / "driver", "", []),
    )

    def fake_count(driver, cwd, **kw):
        counted.append((driver, cwd, kw))
        return 7

    monkeypatch.setattr(space, "count_configurations", fake_count)
    n = space.count_for_iteration(
        tmp_path / "iter1", meta, tmp_path, gpu_index=2, ref_file=tmp_path / "ref.cu"
    )
    assert n == 7
    assert [c.name for c in staged] == ["a"]
    (driver, cwd, kw), = counted
    assert cwd == cdir and kw["kernel_file"] == cdir / "kernels.cu"
    assert kw["gpu_index"] == 2


def test_count_for_iteration_skips_counting_when_the_build_fails(monkeypatch, tmp_path):
    from utils.build import BuildResult

    monkeypatch.setattr(space, "load_inputs_spec", lambda p: "spec")
    monkeypatch.setattr(space, "load_regions", lambda d: {})
    monkeypatch.setattr(space, "reference_build_extras", lambda pd, case: ([], []))
    monkeypatch.setattr(space, "stage_case", lambda *a: tmp_path)
    monkeypatch.setattr(
        space, "compile_framework", lambda d, **kw: BuildResult(False, None, "boom", [])
    )
    monkeypatch.setattr(
        space, "count_configurations", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    assert space.count_for_iteration(
        tmp_path / "iter1", {}, tmp_path, gpu_index=0, ref_file=tmp_path / "r.cu"
    ) is None
