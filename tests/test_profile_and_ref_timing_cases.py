"""NCU targets the primary case only; up-front reference timing covers every case."""

import asyncio

import pytest
import yaml

PYTHON_REF_PROBLEM = {
    "name": "p",
    "reference": {"type": "python", "function": "f", "file": "ref.py"},
    "cases": [{"name": "t512", "scalars": {"kT": 512}},
              {"name": "t127", "scalars": {"kT": 127}}],
}


def _write_problem(tmp_path, meta):
    (tmp_path / "problem.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return tmp_path


def test_reference_timing_is_attempted_for_every_case(tmp_path, monkeypatch):
    """A case with no baseline reports no speedup, which nulls the geomean — so every
    case has to be offered the chance to be timed."""
    import utils.ref_timing as ref_timing

    seen = []

    async def fake(problem_dir, output_dir, cfg, ref, case, case_key):
        seen.append((case.name, case_key))

    monkeypatch.setattr(ref_timing, "_time_reference_for_case", fake)
    problem = _write_problem(tmp_path, PYTHON_REF_PROBLEM)
    asyncio.run(ref_timing.time_python_reference(problem, tmp_path / "out"))

    assert seen == [("t512", None), ("t127", "t127")]


def test_the_primary_case_is_stored_under_the_flat_key(tmp_path, monkeypatch):
    """cases[0] maps to case_key None, which is the flat reference_time_us the four
    committed seeds already use."""
    import utils.ref_timing as ref_timing

    seen = []

    async def fake(problem_dir, output_dir, cfg, ref, case, case_key):
        seen.append(case_key)

    monkeypatch.setattr(ref_timing, "_time_reference_for_case", fake)
    problem = _write_problem(tmp_path, PYTHON_REF_PROBLEM)
    asyncio.run(ref_timing.time_python_reference(problem, tmp_path / "out"))

    assert seen[0] is None


def test_a_cuda_reference_is_not_timed_at_all(tmp_path, monkeypatch):
    import utils.ref_timing as ref_timing

    called = []

    async def fake(*a, **kw):
        called.append(1)

    monkeypatch.setattr(ref_timing, "_time_reference_for_case", fake)
    problem = _write_problem(
        tmp_path, {"name": "p", "reference": {"type": "cuda", "file": "ref_kernel.cu"}}
    )
    asyncio.run(ref_timing.time_python_reference(problem, tmp_path / "out"))

    assert called == []


def test_a_problem_with_no_cases_still_times_its_one_reference(tmp_path, monkeypatch):
    import utils.ref_timing as ref_timing

    seen = []

    async def fake(problem_dir, output_dir, cfg, ref, case, case_key):
        seen.append((case.name, case_key))

    monkeypatch.setattr(ref_timing, "_time_reference_for_case", fake)
    problem = _write_problem(
        tmp_path, {"name": "p", "reference": {"type": "python", "function": "f", "file": "ref.py"}}
    )
    asyncio.run(ref_timing.time_python_reference(problem, tmp_path / "out"))

    assert seen == [("default", None)]


def test_profile_resolves_the_primary_case_directory(tmp_path):
    """NCU diagnoses WHY a kernel is slow, and that mechanism is shape-invariant."""
    from utils.cases import case_dir, case_list

    cases = case_list(PYTHON_REF_PROBLEM)
    assert case_dir(tmp_path / "iter1", cases, cases[0]) == tmp_path / "iter1" / "case_t512"
