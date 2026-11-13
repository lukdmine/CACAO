"""Tool preconditions, rule enforcement, and access restriction.

Preconditions are checked here rather than by removing tools from the schema: an
error is a tool result the model can read and act on, whereas a vanished tool is a
shape it has to infer.
"""

import pytest

from agentic.tools import Toolbox, schemas_for
from agentic.workspace import ToolError
from conftest import make_toolbox
from utils.rules import ForbiddenPattern, Rules


@pytest.fixture
def toolbox(filled_workspace, tmp_path):
    return Toolbox(
        filled_workspace,
        branch_path=tmp_path / "branch",
        output_dir=tmp_path / "out",
        problem_dir=tmp_path / "problem",
        meta={},
    )


# -- gating ---------------------------------------------------------------


def test_end_step_requires_a_compilation_check(toolbox):
    toolbox.mutations = 1
    with pytest.raises(ToolError, match="check_compilation"):
        toolbox.end_step("done")
    assert toolbox.ended is False


def test_end_step_requires_a_change(toolbox):
    # The workspace is seeded with the previous iteration's files, so ending untouched
    # would spend an iteration re-measuring what was already measured.
    toolbox.check_passed = True
    with pytest.raises(ToolError, match="nothing was changed"):
        toolbox.end_step("done")


def test_end_step_requires_all_four_files(workspace, tmp_path):
    box = Toolbox(
        workspace,
        branch_path=tmp_path,
        output_dir=tmp_path,
        problem_dir=tmp_path,
        meta={},
    )
    box.check_passed = True
    box.mutations = 1
    with pytest.raises(ToolError, match="not written"):
        box.end_step("done")


def test_end_step_succeeds_once_satisfied(toolbox):
    toolbox.check_passed = True
    toolbox.mutations = 1
    assert toolbox.end_step("did a thing") == "Step complete."
    assert toolbox.ended is True and toolbox.end_summary == "did a thing"


def test_a_write_invalidates_a_previous_pass(toolbox):
    # Otherwise a model could pass a check, rewrite the kernel, and end on a stale
    # verdict — shipping code that was never compiled.
    toolbox.check_passed = True
    toolbox.write_file("kernels.cu", "// changed\n")
    assert toolbox.check_passed is False


def test_an_edit_invalidates_a_previous_pass(toolbox):
    toolbox.check_passed = True
    toolbox.edit_file("region_launcher.cpp", "default launcher", "custom")
    assert toolbox.check_passed is False


def test_compilation_check_refuses_incomplete_files(workspace, tmp_path):
    box = Toolbox(
        workspace,
        branch_path=tmp_path,
        output_dir=tmp_path,
        problem_dir=tmp_path,
        meta={},
    )
    with pytest.raises(ToolError, match="not written"):
        box.check_compilation()


# -- dispatch -------------------------------------------------------------


def test_dispatch_turns_precondition_errors_into_results(toolbox):
    result = toolbox.dispatch("edit_file", {"name": "kernels.cu", "old_text": "zzz", "new_text": "y"})
    assert result.startswith("ERROR:") and "not found" in result


def test_dispatch_reports_an_unknown_tool(toolbox):
    assert "unknown tool" in toolbox.dispatch("frobnicate", {})


def test_dispatch_reports_bad_arguments(toolbox):
    assert "bad arguments" in toolbox.dispatch("edit_file", {"wrong": 1})


def test_dispatch_survives_a_tool_raising(toolbox, monkeypatch):
    # A bug in one tool must not take the branch down.
    monkeypatch.setattr(
        Toolbox, "list_iterations", lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert "ERROR" in toolbox.dispatch("list_iterations", {})


def test_missing_artifacts_are_reported_not_raised(toolbox):
    assert "ERROR" in toolbox.dispatch("tuning_landscape", {"iteration": 3, "mode": "top"})
    assert "ERROR" in toolbox.dispatch("grep_tuner_output", {"iteration": 3, "pattern": "x"})


def test_grep_rejects_an_invalid_regex(toolbox, tmp_path):
    iter_dir = toolbox.branch_path / "iter1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tuner_output.txt").write_text("hello\n", encoding="utf-8")
    assert "invalid regular expression" in toolbox.dispatch(
        "grep_tuner_output", {"iteration": 1, "pattern": "("}
    )


def test_grep_finds_lines_with_numbers(toolbox):
    iter_dir = toolbox.branch_path / "iter1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tuner_output.txt").write_text("ok\nerror: bad\nok\n", encoding="utf-8")
    out = toolbox.dispatch("grep_tuner_output", {"iteration": 1, "pattern": "error"})
    assert "2: error: bad" in out


def test_read_file_rejects_a_path(toolbox):
    assert "ERROR" in toolbox.dispatch("read_file", {"name": "../secrets", "iteration": 1})


# -- rules ----------------------------------------------------------------


def test_forbidden_construct_fails_the_check_before_compiling(filled_workspace, tmp_path):
    box = Toolbox(
        filled_workspace,
        branch_path=tmp_path,
        output_dir=tmp_path,
        problem_dir=tmp_path,
        meta={},
        rules=Rules(forbid=[ForbiddenPattern("TILE", "no tiles for you")]),
    )
    result = box.check_compilation()
    assert "FAIL" in result and "no tiles for you" in result
    assert box.check_passed is False


def test_rules_are_checked_before_the_compiler_runs(filled_workspace, tmp_path, monkeypatch):
    # A rule violation must be reported as a rule violation, not surface as whatever
    # the compiler happens to say about the same code.
    import agentic.tools as tools_module

    def explode(*a, **k):
        raise AssertionError("compiler must not run on a rule violation")

    monkeypatch.setattr("utils.build.compile_framework", explode)
    box = Toolbox(
        filled_workspace,
        branch_path=tmp_path,
        output_dir=tmp_path,
        problem_dir=tmp_path,
        meta={},
        rules=Rules(forbid=[ForbiddenPattern("__global__")]),
    )
    assert "VIOLATED" in box.check_compilation()


def test_rules_come_from_problem_metadata_by_default(filled_workspace, tmp_path):
    box = Toolbox(
        filled_workspace,
        branch_path=tmp_path,
        output_dir=tmp_path,
        problem_dir=tmp_path,
        meta={"rules": {"forbid": [{"pattern": "TILE", "reason": "nope"}]}},
    )
    assert "VIOLATED" in box.check_compilation()


# -- cross-branch access --------------------------------------------------


def test_access_levels_gate_the_cross_branch_tools():
    names = lambda level: {s.__name__ for s in schemas_for(level)}
    assert "branch_log" not in names("index")
    assert "branch_log" in names("log")
    assert "branch_errors" not in names("log")
    assert "branch_errors" in names("errors")


def test_no_access_level_exposes_another_branch_code():
    for level in ("off", "index", "log", "errors"):
        exposed = {s.__name__ for s in schemas_for(level)}
        assert "read_branch_file" not in exposed


def test_unknown_access_level_falls_back_to_no_cross_branch_tools():
    assert {s.__name__ for s in schemas_for("nonsense")} == {
        s.__name__ for s in schemas_for("off")
    }


# -- binding only what can work -------------------------------------------


def test_history_tools_are_not_bound_without_history():
    """A live run's first iteration called grep_tuner_output three times and got
    "iter1 has no tuner_output.txt" every time. Two branches spent 22 and 23 of a
    25-call budget on tools that could not succeed."""
    names = {s.__name__ for s in schemas_for("errors", has_history=False, has_siblings=False)}
    assert "grep_tuner_output" not in names
    assert "tuning_landscape" not in names
    assert "list_iterations" not in names
    # The workspace exists from the first turn, so these always bind.
    assert {"write_file", "edit_file", "read_file", "check_compilation", "end_step"} <= names


def test_cross_branch_tools_are_not_bound_without_siblings():
    names = {s.__name__ for s in schemas_for("errors", has_history=True, has_siblings=False)}
    assert "branch_log" not in names and "branch_errors" not in names
    assert "tuning_landscape" in names


def test_everything_binds_once_there_is_something_to_read():
    names = {s.__name__ for s in schemas_for("errors", has_history=True, has_siblings=True)}
    assert {"grep_tuner_output", "tuning_landscape", "list_iterations",
            "branch_log", "branch_errors"} <= names


def test_failed_check_steers_toward_editing(filled_workspace, tmp_path, monkeypatch):
    """After a failed check the model rewrote whole files: write_file 11 and 12 times
    in single steps, against edit_file once across every recorded trace."""
    from utils.rules import ForbiddenPattern, Rules

    box = Toolbox(
        filled_workspace,
        branch_path=tmp_path,
        output_dir=tmp_path,
        problem_dir=tmp_path,
        meta={"grid": {"x": 1}, "reference": {"type": "cpu_c", "file": "ref_cpu.c"}},
        rules=Rules(forbid=[ForbiddenPattern("TILE")]),
    )
    # A rules violation short-circuits before the compiler, so drive the compile path.
    box.rules = Rules()
    monkeypatch.setattr(
        "utils.build.compile_framework",
        lambda *a, **k: type("R", (), {"ok": False, "stderr": "framework.cpp:3: error: x", "binary": None, "cmd": []})(),
    )
    monkeypatch.setattr("utils.nvrtc.check_kernel", lambda *a, **k: [])
    out = box.check_compilation()
    assert "FAIL" in out
    assert "edit_file" in out, "a failed check must steer toward patching, not rewriting"


def test_passing_check_does_not_nag_about_editing(filled_workspace, tmp_path, monkeypatch):
    box = Toolbox(
        filled_workspace, branch_path=tmp_path, output_dir=tmp_path,
        problem_dir=tmp_path, meta={"grid": {"x": 1}, "reference": {"type": "cpu_c", "file": "ref_cpu.c"}},
    )
    monkeypatch.setattr(
        "utils.build.compile_framework",
        lambda *a, **k: type("R", (), {"ok": True, "stderr": "", "binary": tmp_path / "driver", "cmd": []})(),
    )
    monkeypatch.setattr("utils.nvrtc.check_kernel", lambda *a, **k: [])
    out = box.check_compilation()
    assert "edit_file to fix them" not in out


_CPU_META = {"grid": {"x": 1}, "reference": {"type": "cpu_c", "file": "ref_cpu.c"}}


def _stub_passing_build(ws, monkeypatch):
    from utils.build import BuildResult
    from utils.nvrtc import NvrtcCheck

    monkeypatch.setattr(
        "utils.build.compile_framework",
        lambda *a, **k: BuildResult(True, ws.staging / "driver", "", []),
    )
    monkeypatch.setattr(
        "utils.nvrtc.check_kernel",
        lambda *a, **k: [NvrtcCheck("minimum values", {}, True, "")],
    )


def test_the_check_names_parameters_narrowed_since_the_previous_iteration(
    filled_workspace, tmp_path, monkeypatch
):
    _stub_passing_build(filled_workspace, monkeypatch)
    box = make_toolbox(
        filled_workspace, tmp_path, meta=_CPU_META,
        previous_params_src='tuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{8, 16, 32});\n',
    )
    out = box.check_compilation()
    assert "TILE" in out and "narrowed" in out


def test_no_report_without_a_previous_iteration(filled_workspace, tmp_path, monkeypatch):
    _stub_passing_build(filled_workspace, monkeypatch)
    out = make_toolbox(filled_workspace, tmp_path, meta=_CPU_META).check_compilation()
    assert "previous iteration" not in out


# -- rules and architecture -------------------------------------------------


def test_a_rules_violation_records_its_own_verdict(filled_workspace, tmp_path):
    """author_node reports last_check as the failure reason, so this branch must set
    it rather than leave the previous — possibly PASSing — verdict."""
    box = make_toolbox(
        filled_workspace,
        tmp_path,
        rules=Rules(forbid=[ForbiddenPattern(pattern="TILE", reason="no tiling")]),
    )
    box.last_check = "Compilation check: PASS\n\nstale"

    verdict = box.check_compilation()

    assert box.check_passed is False
    assert "VIOLATED" in verdict
    assert box.last_check == verdict
    assert "PASS" not in box.last_check.splitlines()[0]


@pytest.mark.parametrize(
    "meta, gpu_info, expected",
    [
        # gpu_info is empty whenever NCU is missing or permission-restricted; without
        # the problem.yaml fallback the check targets NVRTC's lowest arch and rejects
        # cp.async / wmma / bf16 kernels the real device compiles fine.
        ({"gpu": {"compute_capability": "8.6"}}, None, "8.6"),
        ({"gpu": {"compute_capability": "7.0"}}, {"compute_capability": "8.6"}, "8.6"),
        # NVRTC's own default is right when the architecture is unknown.
        ({}, None, None),
    ],
)
def test_compute_capability_resolution(
    filled_workspace, tmp_path, meta, gpu_info, expected
):
    box = make_toolbox(filled_workspace, tmp_path, meta=meta, gpu_info=gpu_info)
    assert box._compute_capability() == expected


# -- finalize tools -----------------------------------------------------------


def _finalize_box(filled_workspace, tmp_path, **ctx):
    from agentic.tools import FinalizeToolContext

    src = tmp_path / "branch" / "iter1"
    src.mkdir(parents=True, exist_ok=True)
    return make_toolbox(
        filled_workspace, tmp_path,
        finalize=FinalizeToolContext(
            source_iter_dir=src, budget_s=ctx.get("budget_s", 1000.0),
            gpu_index=0, ref_file=tmp_path / "ref.cu"),
    )


def test_space_size_refuses_before_a_passing_check(filled_workspace, tmp_path):
    box = _finalize_box(filled_workspace, tmp_path)
    with pytest.raises(ToolError, match="Run check_compilation and get a PASS first"):
        box.space_size()


def test_space_size_names_the_searcher(filled_workspace, tmp_path, monkeypatch):
    box = _finalize_box(filled_workspace, tmp_path)
    box.check_passed = True
    monkeypatch.setattr("utils.space.count_configurations", lambda *a, **k: 120)
    monkeypatch.setattr("utils.space.per_config_cost_s", lambda *a, **k: 0.5)
    out = box.space_size()
    assert "120 configurations" in out and "exhaustive" in out and "60" in out


def test_space_size_reports_an_unmeasurable_space(filled_workspace, tmp_path, monkeypatch):
    box = _finalize_box(filled_workspace, tmp_path)
    box.check_passed = True
    monkeypatch.setattr("utils.space.count_configurations", lambda *a, **k: None)
    with pytest.raises(ToolError):
        box.space_size()


def test_a_finalize_step_may_end_without_a_change(filled_workspace, tmp_path):
    """A space already as wide as plausible has nothing to widen; the seeded source
    files are then what gets retuned, and a forced edit would only add noise."""
    box = _finalize_box(filled_workspace, tmp_path)
    box.check_passed = True
    assert box.end_step("already as wide as plausible") == "Step complete."
    assert box.ended is True and box.mutations == 0


def test_a_finalize_step_still_needs_a_passing_check(filled_workspace, tmp_path):
    box = _finalize_box(filled_workspace, tmp_path)
    with pytest.raises(ToolError, match="check_compilation"):
        box.end_step("done")
    assert box.ended is False


def test_finalize_tools_refuse_outside_finalize(toolbox):
    with pytest.raises(ToolError):
        toolbox.parameter_history()
    with pytest.raises(ToolError):
        toolbox.space_size()


def test_finalize_binding_has_its_tools_and_no_sibling_tools():
    names = {s.__name__ for s in schemas_for("errors", finalize=True)}
    assert {"parameter_history", "space_size", "tuning_landscape"} <= names
    assert "branch_log" not in names and "branch_errors" not in names
    assert "space_size" not in {s.__name__ for s in schemas_for("errors")}
