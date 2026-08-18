"""The step loop: how it ends, and what it hands back when it ends badly.

Every outcome other than COMPLETED has to be something the caller can act on. A
provider that cannot drive tools must degrade to the previous single-shot path, never
take the branch down.
"""

import json

import pytest

from agentic.loop import StepOutcome, run_agentic_step
from agentic.replay import ScriptedLLM
from agentic.tools import Toolbox

pytestmark = pytest.mark.asyncio


@pytest.fixture
def toolbox(filled_workspace, tmp_path, monkeypatch):
    """A toolbox whose compile check passes without invoking g++ or NVRTC."""

    def fake_check(self):
        self.checks_run += 1
        self.check_passed = True
        self.last_check = "Compilation check: PASS"
        return self.last_check

    monkeypatch.setattr(Toolbox, "check_compilation", fake_check)
    return Toolbox(
        filled_workspace,
        branch_path=tmp_path / "branch",
        output_dir=tmp_path / "out",
        problem_dir=tmp_path / "problem",
        meta={},
    )


def call(tool, **args):
    return {"tool": tool, "args": args}


async def test_completes_on_end_step(toolbox, tmp_path):
    llm = ScriptedLLM(
        [
            [call("write_file", name="kernels.cu", content="// v2\n")],
            [call("check_compilation")],
            [call("end_step", summary="rewrote the kernel")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox, trace_path=tmp_path / "t.jsonl")
    assert result.outcome is StepOutcome.COMPLETED
    assert result.summary == "rewrote the kernel"
    assert result.tool_calls == 3 and result.turns == 3
    assert toolbox.ended is True


async def test_several_calls_in_one_turn_are_all_executed(toolbox):
    llm = ScriptedLLM(
        [
            [
                call("write_file", name="kernels.cu", content="// a\n"),
                call("write_file", name="region_launcher.cpp", content="// b\n"),
            ],
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED
    assert result.tool_calls == 4


async def test_no_tool_calls_on_the_first_turn_signals_fallback(toolbox):
    # The model is not driving the loop; the single-shot path will do better than a
    # nudge war with it.
    llm = ScriptedLLM(["here is some prose instead of a tool call"])
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.NO_TOOL_CALLS
    assert result.outcome.should_fall_back is True


async def test_stalling_mid_step_gets_one_nudge(toolbox):
    llm = ScriptedLLM(
        [
            [call("write_file", name="kernels.cu", content="// a\n")],
            "I think I am done.",  # stalls; nudged
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED


async def test_stalling_twice_gives_up(toolbox):
    # One nudge, not a loop: repeating it turns a confused model into a bill.
    llm = ScriptedLLM(
        [
            [call("write_file", name="kernels.cu", content="// a\n")],
            "done?",
            "still done?",
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.NO_TOOL_CALLS


async def test_bind_failure_signals_fallback(toolbox):
    llm = ScriptedLLM([], bind_error=NotImplementedError("no tool support"))
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.BIND_UNSUPPORTED
    assert result.outcome.should_fall_back is True


async def test_llm_error_signals_fallback(toolbox):
    class Exploding(ScriptedLLM):
        async def ainvoke(self, messages, **kwargs):
            raise RuntimeError("provider exploded")

    result = await run_agentic_step(Exploding([]), "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.LLM_ERROR
    assert "exploded" in result.error


async def test_budget_is_enforced(toolbox):
    llm = ScriptedLLM([[call("write_file", name="kernels.cu", content=f"// {i}\n")] for i in range(20)])
    result = await run_agentic_step(llm, "sys", "usr", toolbox, budget=3)
    assert result.outcome is StepOutcome.BUDGET_EXHAUSTED
    assert result.tool_calls == 3


async def test_budget_exhaustion_is_not_a_fallback_signal(toolbox):
    # Whether it is recoverable depends on whether the files came out complete, which
    # only the caller can see.
    assert StepOutcome.BUDGET_EXHAUSTED.should_fall_back is False


async def test_tool_errors_reach_the_model_and_it_can_recover(toolbox):
    llm = ScriptedLLM(
        [
            [call("edit_file", name="kernels.cu", old_text="nope", new_text="x")],
            [call("write_file", name="kernels.cu", content="// recovered\n")],
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED
    assert result.tools_used["edit_file"] == 1


async def test_ending_without_a_passing_check_is_refused(toolbox):
    llm = ScriptedLLM(
        [
            [call("write_file", name="kernels.cu", content="// a\n")],
            [call("end_step", summary="premature")],
            [call("check_compilation")],
            [call("end_step", summary="proper")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED
    assert result.summary == "proper"


async def test_trace_records_every_call_with_arguments(toolbox, tmp_path):
    trace = tmp_path / "step_trace.jsonl"
    llm = ScriptedLLM(
        [
            [call("write_file", name="kernels.cu", content="// traced\n")],
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    await run_agentic_step(llm, "sys", "usr", toolbox, trace_path=trace)

    records = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    events = [r["event"] for r in records]
    assert events[0] == "start" and events[-1] == "completed"

    writes = [r for r in records if r.get("tool") == "write_file"]
    # Full arguments, not a preview: the replay harness reconstructs the step from this.
    assert writes[0]["args"]["content"] == "// traced\n"


async def test_an_unwritable_trace_does_not_stop_the_step(toolbox, tmp_path):
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory", encoding="utf-8")
    llm = ScriptedLLM(
        [
            [call("write_file", name="kernels.cu", content="// a\n")],
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox, trace_path=blocked / "t.jsonl")
    assert result.outcome is StepOutcome.COMPLETED


async def test_tool_calls_without_ids_still_pair_with_results(toolbox):
    # Providers differ on whether an id comes back; an unpaired tool result makes the
    # next request malformed.
    llm = ScriptedLLM(
        [
            [{"tool": "write_file", "args": {"name": "kernels.cu", "content": "// a\n"}, "id": None}],
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED
