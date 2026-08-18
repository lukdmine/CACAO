"""Provider tool-capability detection.

bind_tools is a local call for every provider in this repo — langchain attaches the
schemas and returns without touching the network. So a provider that cannot drive the
loop is only discovered by calling it, and that discovery must be paid once per run
rather than once per iteration per branch.
"""

import pytest
from langchain_openai import ChatOpenAI

import config as _cfg
import nodes.author as author_mod
from agentic import capability
from agentic.replay import ScriptedLLM
from agentic.tools import SCHEMAS

from tests.test_author_node import HAPPY_TURNS, _passing_check, env, make_state, script  # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    capability.reset()
    yield
    capability.reset()


def test_bind_tools_does_not_raise_for_openai_compatible_endpoints():
    """The assumption the fallback design originally rested on, and why it was wrong.

    cerit is ChatOpenAI with a custom base_url. Binding attaches the schemas locally,
    so BIND_UNSUPPORTED can never fire for it: the real signals are NO_TOOL_CALLS and
    LLM_ERROR, which cost a request each.
    """
    model = ChatOpenAI(model="x", api_key="dummy", base_url="http://localhost:1")
    bound = model.bind_tools(SCHEMAS)
    assert len(bound.kwargs["tools"]) == len(SCHEMAS)


def test_one_failure_does_not_disable(monkeypatch):
    # A model that answered in prose once is not a model that cannot call tools.
    monkeypatch.setattr(_cfg, "get_provider", lambda: "cerit")
    monkeypatch.setattr(_cfg, "get_model", lambda: "kimi")
    capability.record_failure("no_tool_calls")
    assert capability.should_skip_loop() is False


def test_repeated_failures_disable_the_loop(monkeypatch):
    monkeypatch.setattr(_cfg, "get_provider", lambda: "cerit")
    monkeypatch.setattr(_cfg, "get_model", lambda: "kimi")
    for _ in range(capability.DISABLE_AFTER):
        capability.record_failure("no_tool_calls")
    assert capability.should_skip_loop() is True
    assert "cerit/kimi" in capability.status()["disabled"]


def test_success_clears_the_count(monkeypatch):
    monkeypatch.setattr(_cfg, "get_provider", lambda: "cerit")
    monkeypatch.setattr(_cfg, "get_model", lambda: "kimi")
    capability.record_failure("no_tool_calls")
    capability.record_success()
    capability.record_failure("no_tool_calls")
    assert capability.should_skip_loop() is False


def test_verdicts_are_per_provider_and_model(monkeypatch):
    provider, model = "cerit", "kimi"
    monkeypatch.setattr(_cfg, "get_provider", lambda: provider)
    monkeypatch.setattr(_cfg, "get_model", lambda: model)
    for _ in range(capability.DISABLE_AFTER):
        capability.record_failure("no_tool_calls")
    assert capability.should_skip_loop() is True

    # A different model on the same endpoint is different weights; re-evaluate it.
    model = "deepseek"
    assert capability.should_skip_loop() is False


async def test_repeated_prose_replies_stop_running_the_loop(env, monkeypatch):
    """The cerit case end to end: discovery is paid once, not every iteration."""
    attempts = []

    class Counting(ScriptedLLM):
        async def ainvoke(self, messages, **kwargs):
            attempts.append(1)
            return await super().ainvoke(messages, **kwargs)

    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: Counting(["prose, no tools"]))

    legacy_calls = []

    async def fake_legacy(state):
        legacy_calls.append(state.iter_num)
        state.status = "running"
        return state

    monkeypatch.setattr(author_mod, "_legacy", fake_legacy)

    for iteration in (1, 2, 3, 4):
        await author_mod.author_node(make_state(env, iter_num=iteration, current_iter=iteration))

    assert legacy_calls == [1, 2, 3, 4], "every iteration must still produce a kernel"
    # Iterations 3 and 4 skip the loop entirely once the verdict is in.
    assert len(attempts) == capability.DISABLE_AFTER
    assert capability.should_skip_loop() is True


async def test_a_working_provider_is_never_disabled(env, monkeypatch):
    _passing_check(monkeypatch)
    monkeypatch.setattr(author_mod, "get_llm_precise", lambda: script(HAPPY_TURNS))
    state = await author_mod.author_node(make_state(env))
    assert state.status == "running"
    assert capability.should_skip_loop() is False


async def test_our_own_tool_bug_is_not_blamed_on_the_provider(env, monkeypatch):
    """The real regression.

    A run died because check_compilation raised on an encoding issue. The model drove
    the loop correctly — wrote all four files, called the check, re-read them to debug,
    retried, then gave up when end_step refused. The capability detector read that
    dead-end as "this provider cannot call tools" and disabled the loop for the rest of
    the run, over our bug.
    """
    from agentic.tools import Toolbox

    def raising_check(self):
        raise UnicodeDecodeError("ascii", b"\xe2", 0, 1, "ordinal not in range(128)")

    monkeypatch.setattr(Toolbox, "check_compilation", raising_check)

    async def fake_legacy(state):
        state.status = "running"
        return state

    monkeypatch.setattr(author_mod, "_legacy", fake_legacy)
    monkeypatch.setattr(
        author_mod,
        "get_llm_precise",
        lambda: script(HAPPY_TURNS + [HAPPY_TURNS[1], HAPPY_TURNS[2]]),
    )

    for iteration in (1, 2, 3):
        await author_mod.author_node(make_state(env, iter_num=iteration, current_iter=iteration))

    assert capability.should_skip_loop() is False, (
        "an internal tool error must not be recorded as a provider capability failure"
    )


async def test_budget_exhaustion_is_not_a_capability_verdict(env, monkeypatch):
    # The model drove the loop; it just did not finish. Disabling on that would strand
    # a capable provider because one kernel was hard.
    monkeypatch.setattr(_cfg, "STEP_TOOL_BUDGET", 1)
    monkeypatch.setattr(
        author_mod,
        "get_llm_precise",
        lambda: script([[{"tool": "write_file", "args": {"name": "kernels.cu", "content": "// x\n"}}]] * 4),
    )

    async def fake_legacy(state):
        state.status = "running"
        return state

    monkeypatch.setattr(author_mod, "_legacy", fake_legacy)
    for iteration in (1, 2, 3):
        await author_mod.author_node(make_state(env, iter_num=iteration, current_iter=iteration))
    assert capability.should_skip_loop() is False
