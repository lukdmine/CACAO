"""The authoring node's contract with the rest of the pipeline.

Chiefly: that every way the step can fail is visible. A failed step ends the iteration
and hands decide a diagnosis; it does not re-run the work through the two single-shot
calls and report success. That silent path is what let a run lose its tool loop for
twenty iterations while every state file downstream looked healthy, so each way of
failing is tested for what it tells the branch rather than in aggregate.
"""

import textwrap

import pytest

import config as _cfg
import nodes.author as author_mod
from agentic.replay import ScriptedLLM
from state.types import StrategyInfo, WorkingState

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


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A problem directory, an output directory, and a branch, wired into the node."""
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "problem.yaml").write_text(PROBLEM_YAML, encoding="utf-8")
    (problem / "inputs.hpp").write_text(INPUTS_HPP, encoding="utf-8")
    (problem / "ref_kernel.cu").write_text("// ref\n", encoding="utf-8")

    output = tmp_path / "output"
    branch = output / "branches" / "b"
    branch.mkdir(parents=True)

    monkeypatch.setattr(author_mod, "get_problem_dir", lambda: problem)
    monkeypatch.setattr(author_mod, "get_output_dir", lambda: output)
    monkeypatch.setattr(_cfg, "AGENTIC_STEPS", True)
    return {"problem": problem, "output": output, "branch": branch}


def make_state(env, **overrides):
    fields = dict(
        analysis="",
        strategy=StrategyInfo(name="b", description="d"),
        iter_num=1,
        status="implementing",
        current_iter=1,
        max_iter=5,
        problem_yaml=PROBLEM_YAML,
        ref_kernel="// ref\n",
        branch_path=str(env["branch"]),
        plan="do the thing",
    )
    fields.update(overrides)
    return WorkingState(**fields)


def script(llm_turns):
    return ScriptedLLM(llm_turns)


def call(tool, **args):
    return {"tool": tool, "args": args}


def _passing_check(monkeypatch):
    from agentic.tools import Toolbox

    def fake(self):
        self.checks_run += 1
        self.check_passed = True
        self.last_check = "Compilation check: PASS"
        return self.last_check

    monkeypatch.setattr(Toolbox, "check_compilation", fake)


HAPPY_TURNS = [
    [
        call("write_file", name="kernels.cu", content='extern "C" __global__ void k(){}\n'),
        call("write_file", name="region_kernels.cpp", content="ktt::KernelId kernel = 0;\n"),
        call("write_file", name="region_params.cpp", content='tuner.AddParameter(kernel, "N", std::vector<uint64_t>{1});\n'),
        call("write_file", name="region_launcher.cpp", content="// default\n"),
    ],
    [call("check_compilation")],
    [call("end_step", summary="first implementation")],
]


async def test_successful_step_produces_the_artifacts_run_node_expects(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))

    state = await author_mod.author_node(make_state(env))

    iter_dir = env["branch"] / "iter1"
    assert state.status == "running"
    assert (iter_dir / "kernels.cu").exists()
    assert (iter_dir / "framework.cpp").exists()
    assert (iter_dir / "inputs.hpp").exists()
    # run_node compiles framework.cpp; propose/decide read these off the state.
    assert "__global__" in state.kernel_code
    assert "CACAO:PARAMS" in state.framework_cpp
    assert not (iter_dir / ".staging").exists()


async def test_assembled_driver_contains_the_authored_regions(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    state = await author_mod.author_node(make_state(env))
    assert 'tuner.AddParameter(kernel, "N"' in state.framework_cpp
    assert "ktt::KernelId kernel = 0;" in state.framework_cpp


async def test_step_writes_a_trace(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env))
    assert (env["branch"] / "iter1" / "step_trace.jsonl").exists()


async def test_retry_prompt_carries_the_previous_decision(env, monkeypatch):
    """propose and decide exist to tell this step what to change.

    A run lost three iterations to this: decide diagnosed the launcher correctly every
    time ("execute the full kernel pipeline each iteration"), the author step never saw
    it, and each retry reproduced the same mistake.
    """
    import json

    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env))

    # The worker, not this node, persists iteration state. Write what iteration 1 would
    # have left behind once propose and decide had run.
    state_path = env["branch"] / "iter1" / "state.json"
    snap = {"iter_num": 1, "status": "decided"}
    snap.update(
        {
            "feedback": "Update the launcher to run the full pipeline each round.",
            "proposal": "Loop expand_frontier until the changed flag stays zero.",
            "run_output": "Results differ for argument with id 1 at index 0",
            "decision": {
                "action": "retry",
                "iteration_summary": "single-launch flood fill, validation failed",
                "error_analysis": {
                    "error_type": "validation",
                    "root_cause": "only one frontier expansion is executed",
                    "suggested_fix": "wire the fixed-point launch sequence",
                },
            },
        }
    )
    state_path.write_text(json.dumps(snap), encoding="utf-8")

    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(
        make_state(env, iter_num=2, current_iter=2, mode="retry")
    )

    prompt = (env["branch"] / "iter2" / "prompt_author.md").read_text(encoding="utf-8")
    for fragment in (
        "Update the launcher to run the full pipeline",
        "only one frontier expansion is executed",
        "wire the fixed-point launch sequence",
        "Loop expand_frontier until",
        "Results differ for argument",
    ):
        assert fragment in prompt, f"retry prompt lost: {fragment!r}"


async def test_prompt_is_saved_and_stays_small(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env))
    prompt = (env["branch"] / "iter1" / "prompt_author.md").read_text(encoding="utf-8")
    # The single-shot prompts this replaces were 49 KB and 52 KB on a real problem.
    assert len(prompt) < 20_000, f"prompt grew to {len(prompt)} bytes"


# -- failing, one test per way the step can end empty-handed ---------------


@pytest.fixture
def spy_legacy(monkeypatch):
    """Catches any surviving route into the single-shot calls."""
    calls = []

    async def fake_legacy(state):
        calls.append(state.mode)
        state.status = "running"
        return state

    monkeypatch.setattr(author_mod, "_legacy", fake_legacy)
    return calls


async def test_a_provider_that_cannot_bind_tools_fails_the_iteration(env, monkeypatch, spy_legacy):
    monkeypatch.setattr(
        author_mod, "get_llm_precise", lambda: ScriptedLLM([], bind_error=NotImplementedError())
    )
    state = await author_mod.author_node(make_state(env))

    assert spy_legacy == []
    assert state.status == "deciding"
    assert "rejected the tool schemas" in state.run_output


async def test_a_model_that_will_not_call_tools_fails_the_iteration(env, monkeypatch, spy_legacy):
    # Prose, nudged, prose again.
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(["just prose", "still prose"]))
    state = await author_mod.author_node(make_state(env))

    assert spy_legacy == []
    assert state.status == "deciding"
    assert "never called a tool" in state.run_output
    assert "still prose" in state.run_output, "decide needs to see what the model said"


async def test_a_persistently_truncated_step_fails_with_the_reason(env, monkeypatch, spy_legacy):
    """The failure that started this: the branch has to be told the model ran out of
    room to answer, because the fix is to ask for less, not to change the strategy."""
    monkeypatch.setattr(_cfg, "STEP_TRUNCATION_RETRIES", 2)
    monkeypatch.setattr(
        author_mod,
        "get_llm_precise",
        lambda: script([{"text": "I'll write the kernel now.", "finish_reason": "length"}] * 8),
    )
    state = await author_mod.author_node(make_state(env))

    assert spy_legacy == []
    assert state.status == "deciding"
    assert "cut off at its output token limit 3 time(s)" in state.run_output
    assert "ask for less in one go" in state.run_output
    assert "failure to generate code, not a failure of the strategy" in state.run_output


async def test_the_budget_running_out_with_files_missing_fails_the_iteration(env, monkeypatch, spy_legacy):
    monkeypatch.setattr(_cfg, "STEP_TOOL_BUDGET", 1)
    monkeypatch.setattr(
        author_mod,
        "get_llm_precise",
        lambda: script([[call("write_file", name="kernels.cu", content="// only one\n")]] * 5),
    )
    state = await author_mod.author_node(make_state(env))

    assert spy_legacy == []
    assert state.status == "deciding"
    assert "missing" in state.run_output and "region_params.cpp" in state.run_output


async def test_a_failed_step_commits_nothing(env, monkeypatch):
    # A half-written workspace must not reach run_node: it would compile whatever
    # happened to land and report the result as this iteration's.
    monkeypatch.setattr(_cfg, "STEP_TOOL_BUDGET", 1)
    monkeypatch.setattr(
        author_mod,
        "get_llm_precise",
        lambda: script([[call("write_file", name="kernels.cu", content="// only one\n")]] * 5),
    )
    await author_mod.author_node(make_state(env))

    iter_dir = env["branch"] / "iter1"
    assert not (iter_dir / "kernels.cu").exists()
    assert not (iter_dir / "framework.cpp").exists()


async def test_a_failed_step_still_leaves_its_trace(env, monkeypatch):
    # The trace is the only record of why the step failed; losing it on failure loses
    # it exactly when it is wanted.
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(["prose", "prose"]))
    await author_mod.author_node(make_state(env))
    assert (env["branch"] / "iter1" / "step_trace.jsonl").exists()


async def test_agentic_steps_off_is_the_only_route_to_the_single_shot_calls(env, monkeypatch, spy_legacy):
    monkeypatch.setattr(_cfg, "AGENTIC_STEPS", False)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env))
    assert spy_legacy == ["fresh"]


async def test_a_truncated_step_that_recovers_is_not_a_failure(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(
        author_mod,
        "get_llm_precise",
        lambda: script([{"text": "", "finish_reason": "length"}] + HAPPY_TURNS),
    )
    state = await author_mod.author_node(make_state(env))
    assert state.status == "running"
    assert (env["branch"] / "iter1" / "kernels.cu").exists()


async def test_budget_exhausted_with_complete_files_routes_to_propose(env, monkeypatch):
    # The files exist but never compiled. Running the tuner on them would spend a GPU
    # slot to learn what the failed check already said, so this takes the route
    # run_node takes for a host-compile failure.
    from agentic.tools import Toolbox

    def failing_check(self):
        self.checks_run += 1
        self.check_passed = False
        self.last_check = "Compilation check: FAIL\n\nDriver (g++): FAIL\nsyntax error"
        return self.last_check

    monkeypatch.setattr(Toolbox, "check_compilation", failing_check)
    monkeypatch.setattr(_cfg, "STEP_TOOL_BUDGET", 6)
    monkeypatch.setattr(
        author_mod,
        "get_llm_precise",
        lambda: script(HAPPY_TURNS[:2] + [[call("check_compilation")]] * 4),
    )

    state = await author_mod.author_node(make_state(env))
    assert state.status == "proposing"
    assert "COMPILE ERROR" in state.run_output
    assert "syntax error" in state.run_output


# -- seeding ---------------------------------------------------------------


async def test_second_iteration_is_seeded_for_editing(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env))

    # Iteration 2 edits rather than rewrites; a rewrite is how unrelated parts regress.
    edit_turns = [
        [call("edit_file", name="kernels.cu", old_text="void k(){}", new_text="void k(){int a;}")],
        [call("check_compilation")],
        [call("end_step", summary="tweaked")],
    ]
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(edit_turns))
    state = await author_mod.author_node(make_state(env, iter_num=2, current_iter=2))

    assert state.status == "running"
    assert "int a;" in state.kernel_code


async def test_seeding_recovers_regions_from_a_legacy_framework_cpp(env, monkeypatch):
    """An output directory written before this node only has framework.cpp."""
    import yaml

    from utils.framework import assemble_framework_cpp

    iter1 = env["branch"] / "iter1"
    iter1.mkdir(parents=True, exist_ok=True)
    (iter1 / "kernels.cu").write_text('extern "C" __global__ void k(){}\n', encoding="utf-8")
    (iter1 / "framework.cpp").write_text(
        assemble_framework_cpp(
            yaml.safe_load(PROBLEM_YAML),
            {"kernels": "ktt::KernelId kernel = 7;", "params": "// p", "launcher": ""},
            cuda_include="/usr/include",
        ),
        encoding="utf-8",
    )

    seeded = author_mod._seed_from_previous(
        author_mod.Workspace(env["branch"] / "iter2"), env["branch"], 2
    )
    assert "kernels.cu" in seeded and "region_kernels.cpp" in seeded


async def test_missing_inputs_hpp_fails_loudly(env, monkeypatch):
    # The LLM does not own inputs.hpp and cannot author it, so routing this to the fix
    # loop would ask a branch to repair something it has no access to.
    (env["problem"] / "inputs.hpp").unlink()
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    with pytest.raises(FileNotFoundError, match="I/O boundary"):
        await author_mod.author_node(make_state(env))


async def test_first_iteration_binds_no_history_or_sibling_tools(env, monkeypatch):
    """Regression: schemas_for was imported but never passed to the loop, so
    CROSS_BRANCH_ACCESS had no effect on binding and every step was offered the full
    tool set. On a live run's first iteration the model then spent calls on
    grep_tuner_output ("iter1 has no tuner_output.txt") and branch_log against
    siblings that were themselves on iteration 1."""
    _passing_check(monkeypatch)
    captured = {}

    real = author_mod.run_agentic_step

    async def spy(*a, **kw):
        captured["schemas"] = [s.__name__ for s in (kw.get("schemas") or [])]
        return await real(*a, **kw)

    monkeypatch.setattr(author_mod, "run_agentic_step", spy)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env))

    bound = set(captured["schemas"])
    assert bound, "schemas must be passed to the loop, not left to the module default"
    for absent in ("grep_tuner_output", "tuning_landscape", "list_iterations",
                   "branch_log", "branch_errors"):
        assert absent not in bound, f"{absent} cannot return anything on iteration 1"
    assert {"write_file", "edit_file", "check_compilation", "end_step"} <= bound


async def test_second_iteration_binds_the_history_tools(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env))
    (env["branch"] / "iter1" / "state.json").write_text('{"iter_num": 1}', encoding="utf-8")

    captured = {}
    real = author_mod.run_agentic_step

    async def spy(*a, **kw):
        captured["schemas"] = [s.__name__ for s in (kw.get("schemas") or [])]
        return await real(*a, **kw)

    monkeypatch.setattr(author_mod, "run_agentic_step", spy)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    await author_mod.author_node(make_state(env, iter_num=2, current_iter=2))

    assert "grep_tuner_output" in set(captured["schemas"])
