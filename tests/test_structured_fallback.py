"""TrackedStructuredLLM's raw-JSON fallback.

It runs only when structured output has already failed, which is why nothing noticed
that splitting the LLM stack out of config.py left `json` behind: config.py imported
it at module level, llm/tracked.py did not, and the fallback raised NameError on the
one path that exists to rescue a failed call.
"""

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from llm.tracked import TrackedStructuredLLM

pytestmark = pytest.mark.asyncio


class Decision(BaseModel):
    action: str
    reason: str = ""


class Failing:
    """A structured wrapper that always fails, forcing the fallback."""

    async def ainvoke(self, *args, **kwargs):
        raise ValueError("could not coerce to schema")


class Raw:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    async def ainvoke(self, *args, **kwargs):
        self.calls += 1
        return AIMessage(content=self._content)


async def _fallback(content):
    raw = Raw(content)
    llm = TrackedStructuredLLM(Failing(), base_llm=raw, schema=Decision)
    return await llm._try_raw_fallback_async([]), raw


async def test_fenced_json_is_recovered():
    result, raw = await _fallback('```json\n{"action": "continue", "reason": "faster"}\n```')

    assert isinstance(result, Decision)
    assert result.action == "continue"
    assert raw.calls == 1


async def test_bare_json_object_is_recovered():
    result, _ = await _fallback('Here is my answer: {"action": "stop"} — done.')

    assert result.action == "stop"


async def test_no_json_at_all_returns_none_rather_than_raising():
    result, _ = await _fallback("I could not decide.")

    assert result is None


async def test_malformed_json_returns_none_rather_than_raising():
    result, _ = await _fallback('```json\n{"action": ,,, }\n```')

    assert result is None
