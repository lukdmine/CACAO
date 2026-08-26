"""The step loop: how it ends, and what it hands back when it ends badly.

The load-bearing case is a turn that reaches no tool call. There are two of those and
they need opposite corrections — a model cut off at its token cap has to be told to be
shorter, a model that stalled has to be told to call something — and for a long time
the loop read both as "this provider cannot use tools" and quit on the spot. Ending
the step there is what let one truncated reply take a run's tool loop offline, so the
tests here are mostly about not doing that.
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


async def test_prose_on_the_first_turn_is_nudged_not_abandoned(toolbox):
    # It used to return here. There is nothing to fall back to now, and a model that
    # opens with a preamble usually calls the tool once told to.
    llm = ScriptedLLM(
        [
            "Here is my plan, in prose.",
            [call("write_file", name="kernels.cu", content="// a\n")],
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED


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


async def test_bind_failure_aborts_the_step(toolbox):
    llm = ScriptedLLM([], bind_error=NotImplementedError("no tool support"))
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.BIND_UNSUPPORTED
    assert result.outcome.aborted is True


async def test_llm_error_aborts_the_step(toolbox):
    class Exploding(ScriptedLLM):
        async def ainvoke(self, messages, **kwargs):
            raise RuntimeError("provider exploded")

    result = await run_agentic_step(Exploding([]), "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.LLM_ERROR
    assert result.outcome.aborted is True
    assert "exploded" in result.error


async def test_budget_is_enforced(toolbox):
    llm = ScriptedLLM([[call("write_file", name="kernels.cu", content=f"// {i}\n")] for i in range(20)])
    result = await run_agentic_step(llm, "sys", "usr", toolbox, budget=3)
    assert result.outcome is StepOutcome.BUDGET_EXHAUSTED
    assert result.tool_calls == 3


async def test_budget_exhaustion_is_not_an_aborted_step(toolbox):
    # The model drove the loop and may have finished the files before running out of
    # calls; whether it did is something only the caller can see.
    assert StepOutcome.BUDGET_EXHAUSTED.aborted is False
    assert StepOutcome.COMPLETED.aborted is False


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


# -- truncation: the failure that used to end the step silently ------------


def truncated(text=""):
    """A reply the provider cut off at its output token cap."""
    return {"text": text, "finish_reason": "length"}


async def test_truncated_turn_is_reported_to_the_model_and_it_recovers(toolbox):
    """The case that cost a run its tool loop.

    kimi-k3 spent its whole 48k reply reasoning about a "write the kernel" prompt and
    was cut off before the tool call. The loop read "no tool calls" and quit; the model
    was never told, because there was no next turn.
    """
    llm = ScriptedLLM(
        [
            truncated("I'll implement the three changes: (1)... Writing the file first."),
            [call("write_file", name="kernels.cu", content="// recovered\n")],
            [call("check_compilation")],
            [call("end_step", summary="s")],
        ]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED
    assert result.truncations == 1


async def test_the_model_is_told_it_was_cut_off_not_that_it_ignored_the_tools(toolbox):
    # The corrections are not interchangeable. "Call more tools" is advice for a
    # problem a truncated model does not have.
    llm = ScriptedLLM(
        [truncated(), [call("write_file", name="kernels.cu", content="// a\n")],
         [call("check_compilation")], [call("end_step", summary="s")]]
    )
    await run_agentic_step(llm, "sys", "usr", toolbox)

    sent = [m.content for m in llm.last_messages if isinstance(m.content, str)]
    nudge = next(m for m in sent if "cut off" in m)
    assert "output token limit" in nudge
    assert "must open with a tool call" in nudge
    assert "end_step" not in nudge


async def test_repeated_truncation_escalates_the_correction(toolbox):
    llm = ScriptedLLM(
        [truncated(), truncated(),
         [call("write_file", name="kernels.cu", content="// a\n")],
         [call("check_compilation")], [call("end_step", summary="s")]]
    )
    await run_agentic_step(llm, "sys", "usr", toolbox)

    nudges = [m.content for m in llm.last_messages
              if isinstance(m.content, str) and "ut off" in m.content]
    assert len(nudges) == 2
    # The second asks for a smaller reply rather than repeating the first.
    assert "ONE file only" in nudges[1]
    assert nudges[0] != nudges[1]


async def test_persistent_truncation_ends_the_step(toolbox):
    # Corrected, then corrected again, then given up on — with the reason attached so
    # the caller can tell the branch what happened.
    llm = ScriptedLLM([truncated()] * 10)
    result = await run_agentic_step(llm, "sys", "usr", toolbox, truncation_retries=2)
    assert result.outcome is StepOutcome.TRUNCATED
    assert result.outcome.aborted is True
    assert result.truncations == 3
    assert "token limit" in result.error
    assert "cut off" in result.outcome.diagnosis


async def test_an_empty_reply_with_no_stated_reason_counts_as_truncation(toolbox):
    # cerit returns finish_reason on the raw response but TrackedLLM's retry ladder can
    # hand back a bare empty message. No content, no call, no reason given is the same
    # event, and the same correction applies.
    llm = ScriptedLLM(
        ["", [call("write_file", name="kernels.cu", content="// a\n")],
         [call("check_compilation")], [call("end_step", summary="s")]]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED
    assert result.truncations == 1


async def test_truncation_and_stalling_are_counted_separately(toolbox):
    # A step that was cut off once and later stalled has not used up its stall nudge.
    llm = ScriptedLLM(
        [truncated(),
         [call("write_file", name="kernels.cu", content="// a\n")],
         "I think I am done.",
         [call("check_compilation")], [call("end_step", summary="s")]]
    )
    result = await run_agentic_step(llm, "sys", "usr", toolbox)
    assert result.outcome is StepOutcome.COMPLETED
    assert result.truncations == 1


async def test_trace_records_truncation_and_its_correction(toolbox, tmp_path):
    trace = tmp_path / "step_trace.jsonl"
    llm = ScriptedLLM(
        [truncated("preamble only"),
         [call("write_file", name="kernels.cu", content="// a\n")],
         [call("check_compilation")], [call("end_step", summary="s")]]
    )
    await run_agentic_step(llm, "sys", "usr", toolbox, trace_path=trace)

    records = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    assistant = next(r for r in records if r["event"] == "assistant")
    # Without this the trace cannot tell a truncation from a refusal after the fact.
    assert assistant["finish_reason"] == "length"
    nudge = next(r for r in records if r["event"] == "truncation_nudge")
    assert nudge["attempt"] == 1
