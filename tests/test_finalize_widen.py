"""Stage B of finalize: re-widening a finalist's space with the authoring tool loop.

The step reads a finished branch and writes only into the finalist's own directory
under output/finalize/. The source result has to stand whatever widening does, so the
first thing tested is that nothing under the source branch changes.
"""

import hashlib
import textwrap
from pathlib import Path

import pytest
import yaml

import nodes.finalize_widen as widen_mod
from agentic.replay import ScriptedLLM
from conftest import call, pass_check

PROBLEM_YAML = textwrap.dedent(
    """\
    grid: {x: 1024}
    reference: {type: cuda, function: ref_kernel, file: ref_kernel.cu, block: {x: 8}}
    validation: {tolerance: 0.0001}
    tuning: {duration_s: 5}
    """
)

INPUTS_HPP = textwrap.dedent(
    """\
    struct Inputs { std::string defines; };
    inline Inputs DefineInputs(ktt::Tuner& tuner) {
        Inputs in; in.defines = " -DN=16"; return in;
    }
    """
)

INPUTS_YAML = textwrap.dedent(
    """\
    args:
    - {kind: scalar, name: N, dtype: int, value: 16, placements: [host, define]}
    - {kind: buffer, name: o, dtype: float, size: N, access: write, init: zeros, validate: true}
    """
)

# The tile size is a literal: authoring narrowed TILE to one value and then welded it in.
SOURCE_FILES = {
    "kernels.cu": 'extern "C" __global__ void k(float* o)\n{\n    o[0] = 16;\n}\n',
    "region_kernels.cpp": "ktt::KernelId kernel = 0;\n",
    "region_params.cpp": 'tuner.AddParameter(kernel, "N", std::vector<uint64_t>{1});\n',
    "region_launcher.cpp": "// default\n",
}

WIDEN_TURNS = [
    [call("parameter_history")],
    [
        call("edit_file", name="kernels.cu", old_text="o[0] = 16;", new_text="o[0] = TILE;"),
        call(
            "edit_file",
            name="region_params.cpp",
            old_text="{1});",
            new_text='{1});\ntuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{8, 16, 32});',
        ),
    ],
    [call("check_compilation")],
    [call("end_step", summary="TILE is a parameter again")],
]


@pytest.fixture
def env(tmp_path):
    """A problem, a finished source branch, and the finalist directory to widen into."""
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "problem.yaml").write_text(PROBLEM_YAML, encoding="utf-8")
    (problem / "inputs.hpp").write_text(INPUTS_HPP, encoding="utf-8")
    (problem / "ref_kernel.cu").write_text("// ref\n", encoding="utf-8")

    output = tmp_path / "output"
    source_branch = output / "branches" / "b"
    source_iter = source_branch / "iter2"
    source_iter.mkdir(parents=True)
    for name, text in SOURCE_FILES.items():
        (source_iter / name).write_text(text, encoding="utf-8")
    (source_branch / "iter2" / "state.json").write_text('{"iter_num": 2}', encoding="utf-8")

    return {
        "problem": problem,
        "output": output,
        "source_branch": source_branch,
        "source_iter": source_iter,
        "iter_dir": output / "finalize" / "b" / "iter1",
    }


def _use(monkeypatch, turns):
    monkeypatch.setattr(widen_mod, "get_llm_precise", lambda: ScriptedLLM(turns))


async def widen(env, **overrides):
    kwargs = dict(
        iter_dir=env["iter_dir"],
        source_branch=env["source_branch"],
        source_iter_dir=env["source_iter"],
        problem_dir=env["problem"],
        output_dir=env["output"],
        meta=yaml.safe_load(PROBLEM_YAML),
        problem_yaml=PROBLEM_YAML,
        gpu_info={},
        budget_s=1000.0,
        gpu_index=0,
        ref_file=env["problem"] / "ref_kernel.cu",
    )
    kwargs.update(overrides)
    return await widen_mod.widen_finalist(**kwargs)


def snapshot(root: Path) -> dict:
    """Relative path -> sha1 of contents, for every file under root."""
    return {
        str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def prompts_text() -> str:
    """The widen prompt's own text, without problem data that may carry any character."""
    from prompts.finalize import build_widen

    system, user = build_widen(
        {"current_files": "", "budget_s": 1000.0, "inputs_hpp": "", "problem_yaml": "", "rules_block": ""}
    )
    return system + user


async def test_a_widen_step_commits_into_the_finalist_dir(env, monkeypatch):
    pass_check(monkeypatch)
    _use(monkeypatch, WIDEN_TURNS)
    before = snapshot(env["source_branch"])

    result = await widen(env)

    assert result.ok, result.diagnosis
    assert result.summary == "TILE is a parameter again"
    iter_dir = env["iter_dir"]
    for name in (*SOURCE_FILES, "framework.cpp"):
        assert (iter_dir / name).exists(), name
    assert "o[0] = TILE;" in (iter_dir / "kernels.cu").read_text(encoding="utf-8")
    framework = (iter_dir / "framework.cpp").read_text(encoding="utf-8")
    assert '"TILE", std::vector<uint64_t>{8, 16, 32}' in framework
    assert not (iter_dir / ".staging").exists()
    # The source result stands whatever happens here, so the source must not change.
    assert snapshot(env["source_branch"]) == before


async def test_the_kernel_diff_is_counted(env, monkeypatch):
    pass_check(monkeypatch)
    _use(monkeypatch, WIDEN_TURNS)

    result = await widen(env)

    # One line changed in kernels.cu: one removal and one addition. The region edit
    # does not count; D3 limits the kernel, not the driver regions.
    assert result.diff_lines == 2


async def test_an_aborted_step_fails_with_its_diagnosis(env, monkeypatch):
    _use(monkeypatch, ["I will widen TILE.", "Still thinking."])

    result = await widen(env)

    assert not result.ok
    assert "never called a tool" in result.diagnosis
    assert "no file was written" in result.diagnosis
    assert result.diff_lines is None
    assert not (env["iter_dir"] / "kernels.cu").exists()
    assert not (env["iter_dir"] / "framework.cpp").exists()


async def test_a_step_that_never_passes_the_check_fails(env, monkeypatch):
    """author_node routes this to propose; finalize has no propose, so the source stands."""
    import config as _cfg
    from agentic.tools import Toolbox

    def failing_check(self):
        self.checks_run += 1
        self.check_passed = False
        self.last_check = "Compilation check: FAIL"
        return self.last_check

    monkeypatch.setattr(Toolbox, "check_compilation", failing_check)
    monkeypatch.setattr(_cfg, "STEP_TOOL_BUDGET", 4)
    _use(monkeypatch, WIDEN_TURNS[1:2] + [[call("check_compilation")]] * 4)

    result = await widen(env)

    assert not result.ok
    assert "ran out of tool calls" in result.diagnosis
    assert "compilation check did not pass" in result.diagnosis
    assert not (env["iter_dir"] / "framework.cpp").exists()


NO_CHANGE_TURNS = [
    [call("parameter_history")],
    [call("check_compilation")],
    [call("end_step", summary="already as wide as plausible")],
]


async def test_a_step_that_changes_nothing_retunes_the_source_files(env, monkeypatch):
    pass_check(monkeypatch)
    _use(monkeypatch, NO_CHANGE_TURNS)

    result = await widen(env)

    assert result.ok, result.diagnosis
    assert result.diff_lines == 0
    for name, text in SOURCE_FILES.items():
        assert (env["iter_dir"] / name).read_text(encoding="utf-8") == text, name
    assert (env["iter_dir"] / "framework.cpp").exists()


async def test_a_stalled_step_that_changed_nothing_after_a_passing_check_is_kept(env, monkeypatch):
    """end_step would have accepted it, so the stall loses nothing."""
    pass_check(monkeypatch)
    _use(monkeypatch, NO_CHANGE_TURNS[:2] + ["The space is already as wide as it gets.", "Done."])

    result = await widen(env)

    assert result.ok, result.diagnosis
    assert result.diff_lines == 0
    for name, text in SOURCE_FILES.items():
        assert (env["iter_dir"] / name).read_text(encoding="utf-8") == text, name


async def test_a_legacy_source_with_only_framework_cpp_seeds_the_workspace(env, monkeypatch):
    """An iteration written before region files only has kernels.cu and framework.cpp."""
    from utils.framework import assemble_framework_cpp

    source = env["source_iter"]
    for name in ("region_kernels.cpp", "region_params.cpp", "region_launcher.cpp"):
        (source / name).unlink()
    (source / "framework.cpp").write_text(
        assemble_framework_cpp(
            yaml.safe_load(PROBLEM_YAML),
            {
                "kernels": "ktt::KernelId kernel = 7;",
                "params": SOURCE_FILES["region_params.cpp"],
                # Two lines: a lone comment reads back as the empty-region placeholder.
                "launcher": "// legacy launcher\nci.RunKernel(def);",
            },
            cuda_include="/usr/include",
        ),
        encoding="utf-8",
    )
    pass_check(monkeypatch)
    # The region_params edit only applies if the region was recovered from framework.cpp.
    _use(monkeypatch, WIDEN_TURNS)

    result = await widen(env)

    assert result.ok, result.diagnosis
    iter_dir = env["iter_dir"]
    assert "ktt::KernelId kernel = 7;" in (iter_dir / "region_kernels.cpp").read_text(encoding="utf-8")
    assert "legacy launcher" in (iter_dir / "region_launcher.cpp").read_text(encoding="utf-8")
    assert '"TILE"' in (iter_dir / "region_params.cpp").read_text(encoding="utf-8")


async def test_the_widen_prompt_names_the_budget_and_the_tools(env, monkeypatch):
    pass_check(monkeypatch)
    _use(monkeypatch, WIDEN_TURNS)

    await widen(env)

    prompt = (env["iter_dir"] / "prompt_finalize.md").read_text(encoding="utf-8")
    assert "1000" in prompt
    assert "space_size" in prompt
    assert "parameter_history" in prompt
    # The seeded files are inlined, so the model edits them rather than rewriting.
    assert "o[0] = 16;" in prompt
    assert "\u2014" not in prompts_text(), "no em dashes in LLM-facing text"


async def test_the_widen_prompt_keeps_to_the_edit_only_rule_and_states_the_scalars(env, monkeypatch):
    """D3 allows kernels.cu edits only, and inputs.hpp's block points at a scalar
    contract, so the contract has to follow it."""
    (env["problem"] / "inputs.yaml").write_text(INPUTS_YAML, encoding="utf-8")
    pass_check(monkeypatch)
    _use(monkeypatch, WIDEN_TURNS)

    await widen(env)

    prompt = (env["iter_dir"] / "prompt_finalize.md").read_text(encoding="utf-8")
    assert "write_file when the change is structural" not in prompt
    assert "Change `kernels.cu` with `edit_file` only" in prompt.split("## Current files")[1]
    assert "See the scalar contract below" in prompt
    assert "## Problem scalars" in prompt and "`N` = 16" in prompt
    assert prompt.index("See the scalar contract below") < prompt.index("## Problem scalars")
    assert "MCMC then starts unseeded" in prompts_text()


def test_the_widen_prompt_carries_the_shared_group_rule_and_no_other_gpu_claim():
    from prompts._regions_reference import PARAMETER_GROUPS

    text = prompts_text()
    # Included verbatim, not paraphrased: the constraint-split rule fails silently.
    assert PARAMETER_GROUPS in text
    # Finalize tunes on this GPU only; nothing in it times another.
    assert "on others" not in text
    assert "\u2014" not in text
    # A space already as wide as plausible may end the step unchanged.
    assert "without changing anything" in text


def test_seed_unchanged_reproduces_the_source_files(env, monkeypatch):
    def no_llm():
        raise AssertionError("seed_unchanged must not call the LLM")

    monkeypatch.setattr(widen_mod, "get_llm_precise", no_llm)
    iter_dir = env["iter_dir"]

    widen_mod.seed_unchanged(
        iter_dir=iter_dir,
        source_iter_dir=env["source_iter"],
        problem_dir=env["problem"],
        meta=yaml.safe_load(PROBLEM_YAML),
    )

    for name, text in SOURCE_FILES.items():
        assert (iter_dir / name).read_text(encoding="utf-8") == text, name
    assert (iter_dir / "inputs.hpp").exists()
    framework = (iter_dir / "framework.cpp").read_text(encoding="utf-8")
    assert 'tuner.AddParameter(kernel, "N"' in framework
    assert "ktt::KernelId kernel = 0;" in framework
    assert not (iter_dir / ".staging").exists()
