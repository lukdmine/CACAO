"""TrackedLLM's empty-content retry must not fire underneath the agentic loop.

Both layers detect the same event — a reply cut off at the output token cap before it
reached a tool call — and both correct it. Stacked, they multiply: the loop's three
escalating nudges each cost the wrapper's five backoff retries underneath. The wrapper
is the one that has to yield, because the loop's correction is the better one (it
shrinks the unit of work) and the wrapper's appended nudge never reached the model
anyway: the loop keeps its own message list.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage

import config
from agentic.loop import StepOutcome, run_agentic_step
from agentic.tools import Toolbox

pytestmark = pytest.mark.asyncio


class AlwaysTruncated:
    """A provider that burns its whole output cap on reasoning, every time."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, *args, **kwargs):
        self.calls += 1
        return AIMessage(content="", response_metadata={"finish_reason": "length"})

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def toolbox(filled_workspace, tmp_path, monkeypatch):
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


@pytest.fixture
def no_sleep(monkeypatch):
    """Record what would have been slept instead of sleeping it."""
    slept = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds, *a, **kw):
        slept.append(seconds)
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return slept


async def test_bound_wrapper_hands_a_truncated_reply_straight_to_the_loop(
    toolbox, tmp_path, no_sleep
):
    provider = AlwaysTruncated()
    llm = config.TrackedLLM(provider).bind_tools([])

    result = await run_agentic_step(
        llm,
        "sys",
        "usr",
        toolbox,
        trace_path=tmp_path / "t.jsonl",
        truncation_retries=3,
    )

    assert result.outcome is StepOutcome.TRUNCATED
    # One initial turn plus three corrections. Without the guard the wrapper retried
    # each of those five more times: 24 provider calls and ~248 s of backoff.
    assert provider.calls == 4
    assert sum(no_sleep) == 0


async def test_unbound_wrapper_still_retries_an_empty_reply(no_sleep):
    """The retry is still right for the single-shot nodes, which have no loop above
    them to apply a correction."""
    provider = AlwaysTruncated()
    llm = config.TrackedLLM(provider)

    response = await llm.ainvoke([])

    assert response.content == ""
    # Initial call plus max_retries.
    assert provider.calls == 6
    assert sum(no_sleep) == pytest.approx(2 + 4 + 8 + 16 + 32)


async def test_bind_tools_marks_the_wrapper(no_sleep):
    llm = config.TrackedLLM(AlwaysTruncated())

    assert llm._tools_bound is False
    assert llm.bind_tools([])._tools_bound is True
