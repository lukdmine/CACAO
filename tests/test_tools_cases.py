"""Per-case behaviour of the authoring loop's tools.

BLK carries the `define` placement on purpose: the two cases then produce genuinely
different NVRTC input, which is the only situation where a per-case compile check is
more than a repeated one.
"""

import pytest
import yaml

from agentic.tools import Toolbox
from agentic.workspace import ToolError
from utils.build import BuildResult

TWO_CASE_PROBLEM = {
    "name": "twocase",
    "global_size_type": "cuda",
    "grid": {"x": "BLK", "y": "1", "z": "1"},
    "reference": {"type": "cuda", "function": "ref", "file": "ref_kernel.cu"},
    "validation": {"tolerance": 1e-4},
    "tuning": {"duration_s": 10},
    "cases": [
        {"name": "big", "scalars": {"BLK": 64}},
        {"name": "small", "scalars": {"BLK": 8}},
    ],
}

INPUTS = {
    "headers": [],
    "shared_setup": "",
    "args": [
        {"kind": "scalar", "name": "BLK", "dtype": "int", "value": 64,
         "placements": ["host", "define"]},
        {"kind": "buffer", "name": "out", "dtype": "float", "size": "BLK",
         "access": "write", "init": "zeros", "validate": True},
    ],
}


def _problem(tmp_path, meta, sub="problem"):
    problem = tmp_path / sub
    problem.mkdir()
    (problem / "problem.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    (problem / "inputs.yaml").write_text(yaml.safe_dump(INPUTS), encoding="utf-8")
    return problem


@pytest.fixture
def two_case_toolbox(filled_workspace, tmp_path):
    return Toolbox(
        filled_workspace,
        branch_path=tmp_path / "branch",
        output_dir=tmp_path / "out",
        problem_dir=_problem(tmp_path, TWO_CASE_PROBLEM),
        meta=TWO_CASE_PROBLEM,
    )


def _ok(defines):
    """A real NvrtcCheck, not a stand-in: format_checks and end_step both read fields a
    hand-rolled stub would silently not have."""
    from utils.nvrtc import NvrtcCheck

    return NvrtcCheck(label="cfg", config={}, ok=True, log="", options=list(defines))


def _fail(defines):
    from utils.nvrtc import NvrtcCheck

    return NvrtcCheck(
        label="cfg", config={}, ok=False,
        log='identifier "BLK" is undefined', options=list(defines),
    )


@pytest.fixture
def nvrtc_calls(monkeypatch):
    """Record the -D macro set handed to NVRTC on each check."""
    seen = []

    def fake_check_kernel(source, params, *, cuda_include, scalar_defines,
                          compute_capability=None):
        seen.append(tuple(scalar_defines))
        return [_ok(scalar_defines)]

    monkeypatch.setattr("utils.nvrtc.check_kernel", fake_check_kernel)
    monkeypatch.setattr(
        "utils.build.compile_framework",
        lambda d, **kw: BuildResult(True, d / "driver", "", []),
    )
    return seen


def test_check_compilation_uses_the_primary_case_only(two_case_toolbox, nvrtc_calls):
    """The inner loop stays fast: one NVRTC compile per check, at the primary shape."""
    two_case_toolbox.check_compilation()
    assert len(nvrtc_calls) == 1
    assert "-DBLK=64" in nvrtc_calls[0]


def test_end_step_checks_the_remaining_cases(two_case_toolbox, nvrtc_calls):
    """The complete gate: the non-primary cases are compiled before the step commits."""
    two_case_toolbox.mutations = 1
    two_case_toolbox.check_passed = True
    two_case_toolbox.end_step("done")
    assert nvrtc_calls == [("-DBLK=8",)]
    assert two_case_toolbox.ended is True


def test_end_step_refuses_when_a_non_primary_case_fails(two_case_toolbox, monkeypatch):
    """A kernel that only compiles at the primary shape must not commit."""

    def fake_check_kernel(source, params, *, cuda_include, scalar_defines,
                          compute_capability=None):
        return ([_ok(scalar_defines)] if "-DBLK=64" in scalar_defines
                else [_fail(scalar_defines)])

    monkeypatch.setattr("utils.nvrtc.check_kernel", fake_check_kernel)
    two_case_toolbox.mutations = 1
    two_case_toolbox.check_passed = True
    with pytest.raises(ToolError, match="small"):
        two_case_toolbox.end_step("done")
    assert two_case_toolbox.ended is False


def test_single_case_problem_still_checks_once(filled_workspace, tmp_path, nvrtc_calls):
    """No `cases:` block -> exactly today's behaviour."""
    meta = {k: v for k, v in TWO_CASE_PROBLEM.items() if k != "cases"}
    box = Toolbox(
        filled_workspace,
        branch_path=tmp_path / "b",
        output_dir=tmp_path / "o",
        problem_dir=_problem(tmp_path, meta, sub="p1"),
        meta=meta,
    )
    box.check_compilation()
    assert len(nvrtc_calls) == 1

    box.mutations = 1
    box.check_passed = True
    box.end_step("done")
    assert len(nvrtc_calls) == 1  # end_step adds nothing when there is one case


def test_tuning_landscape_rejects_an_unknown_case(two_case_toolbox):
    with pytest.raises(ToolError, match="No case named"):
        two_case_toolbox.tuning_landscape(iteration=1, case="nope")


def test_tuning_landscape_reads_the_named_cases_results(two_case_toolbox, tmp_path):
    import json

    iter_dir = tmp_path / "branch" / "iter1" / "case_small"
    iter_dir.mkdir(parents=True)
    (iter_dir / "results.json").write_text(json.dumps({
        "Metadata": {"TimeUnit": "Microseconds"},
        "Results": [{"KernelName": "K", "Status": "Ok",
                     "Configuration": [{"Name": "TILE", "Value": 16,
                                        "ValueType": "UnsignedInt"}],
                     "ComputationResults": [{"Duration": 12.5}],
                     "TotalDuration": 12.5}],
    }), encoding="utf-8")

    out = two_case_toolbox.tuning_landscape(iteration=1, case="small")
    assert "12.5" in out or "12.50" in out
