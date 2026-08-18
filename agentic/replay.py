"""Replay a recorded step offline.

A live run writes every tool call to ``step_trace.jsonl``. This module re-drives the
loop from that record with a scripted stand-in for the model, so a step that
misbehaved against a real provider becomes reproducible without spending tokens or
touching the network.

The tool *results* are recomputed, not replayed. That is the point: the recorded
model behaviour is the fixed input, and the current code is what is under test. A
replay that diverges from the recorded results is showing you a real difference.

The same ScriptedLLM drives the offline test suite, so tests and replay exercise one
code path rather than two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.messages import AIMessage

# One turn is either a list of tool calls or a plain text reply (no tool calls).
Turn = Union[List[Dict[str, Any]], str]


class ScriptedLLM:
    """Stand-in for a bound chat model that emits a fixed sequence of turns.

    Mirrors only what the loop uses: ``bind_tools`` and ``ainvoke``. Once the script
    runs out it keeps returning a bare text message, which the loop treats as "no tool
    calls" — the same thing a real model does when it stops driving.
    """

    def __init__(self, turns: Iterable[Turn], *, bind_error: Optional[Exception] = None):
        self.turns: List[Turn] = list(turns)
        self.bind_error = bind_error
        self.calls_seen: List[Dict[str, Any]] = []
        self.invocations = 0
        self.last_messages: List[Any] = []

    def bind_tools(self, tools, **kwargs):
        if self.bind_error is not None:
            raise self.bind_error
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages, **kwargs):
        self.invocations += 1
        self.last_messages = messages
        if not self.turns:
            return AIMessage(content="(script exhausted)")

        turn = self.turns.pop(0)
        if isinstance(turn, str):
            return AIMessage(content=turn)

        tool_calls = []
        for i, call in enumerate(turn):
            tc = {
                "name": call["tool"] if "tool" in call else call["name"],
                "args": call.get("args") or {},
                "id": call.get("id") or f"replay_{self.invocations}_{i}",
            }
            tool_calls.append(tc)
            self.calls_seen.append(tc)
        return AIMessage(content="", tool_calls=tool_calls)


def load_trace(trace_path) -> List[dict]:
    """Records from a step trace, skipping anything unparseable."""
    path = Path(trace_path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def turns_from_trace(trace_path) -> List[Turn]:
    """Group a trace's tool calls back into per-turn batches, in order."""
    grouped: Dict[int, List[dict]] = {}
    for record in load_trace(trace_path):
        if record.get("event") != "tool_call":
            continue
        grouped.setdefault(int(record.get("turn", 0)), []).append(
            {"tool": record.get("tool", ""), "args": record.get("args") or {}}
        )
    return [grouped[turn] for turn in sorted(grouped)]


def scripted_from_trace(trace_path) -> ScriptedLLM:
    return ScriptedLLM(turns_from_trace(trace_path))


def summarize_trace(trace_path) -> str:
    """Human-readable digest: what the model did, in order, and how it ended."""
    records = load_trace(trace_path)
    if not records:
        return f"No trace records in {trace_path}."

    lines = []
    for record in records:
        event = record.get("event")
        if event == "start":
            lines.append(f"start (budget {record.get('budget')})")
        elif event == "tool_call":
            result = str(record.get("result", ""))
            first = result.splitlines()[0] if result else ""
            lines.append(
                f"  {record.get('index'):>3}. [turn {record.get('turn')}] "
                f"{record.get('tool')} -> {first[:90]}"
            )
        elif event == "assistant":
            text = (record.get("text") or "").replace("\n", " ")
            if text:
                lines.append(f"  [turn {record.get('turn')}] said: {text[:110]}")
            if record.get("reasoning"):
                lines.append(f"  [turn {record.get('turn')}] reasoning: {len(record['reasoning'])} chars")
        elif event == "completed":
            lines.append(
                f"completed after {record.get('tool_calls')} calls / "
                f"{record.get('turns')} turns: {record.get('summary', '')}"
            )
        elif event:
            lines.append(f"{event}: " + json.dumps({k: v for k, v in record.items() if k != 'event'}, default=str)[:160])
    return "\n".join(lines)


async def replay_step(trace_path, toolbox, *, budget: int = 50) -> Any:
    """Re-drive a recorded step against the current code.

    Returns the StepResult, with the toolbox mutated exactly as a live run would leave
    it, so the resulting workspace can be inspected or diffed.
    """
    from agentic.loop import run_agentic_step

    llm = scripted_from_trace(trace_path)
    return await run_agentic_step(
        llm,
        "replay",
        "replay",
        toolbox,
        budget=budget,
        trace_path=None,
    )
