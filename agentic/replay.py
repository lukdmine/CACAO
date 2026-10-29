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

# One turn is a list of tool calls, a plain text reply (no tool calls), or a dict
# describing a reply in more detail — ``{"text": ..., "finish_reason": "length"}`` is
# how a turn cut off at the provider's token cap is scripted.
Turn = Union[List[Dict[str, Any]], str, Dict[str, Any]]

# Stands in for a stalled turn whose text the trace did not keep (it is recorded only
# when there was content or reasoning to record).
_STALL_TEXT = "(recorded turn: prose, no tool call)"


class ScriptedLLM:
    """Stand-in for a bound chat model that emits a fixed sequence of turns.

    Mirrors only what the loop uses: ``bind_tools`` and ``ainvoke``. Once the script
    runs out it keeps returning a bare text message, which the loop treats as "no tool
    calls" — the same thing a real model does when it stops driving.

    A dict turn carries the response metadata a plain string cannot. The truncation the
    loop now corrects for is invisible in the content alone — the tell is
    ``finish_reason``, and a turn cut off at the cap is written
    ``{"text": "", "finish_reason": "length"}``.
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
        if isinstance(turn, dict):
            if "tool" in turn or "name" in turn:
                # A tool-call turn is a *list* of calls. A bare call dict would fall
                # through as a reply with no content and no stated reason, which the
                # loop reads as a truncation — a mis-scripted test would pass while
                # exercising something else entirely.
                raise ValueError(
                    f"tool-call turn must be a list of calls, got a bare dict: {turn!r}"
                )
            metadata = {}
            if turn.get("finish_reason"):
                metadata["finish_reason"] = turn["finish_reason"]
            return AIMessage(
                content=turn.get("text", ""),
                response_metadata=metadata,
                additional_kwargs=turn.get("additional_kwargs") or {},
            )

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
    """Every recorded turn, in order: the tool-call batches and the empty-handed replies.

    Keeping only the tool calls made a truncated or stalled step — the failure mode the
    correction policy exists for — the one step a replay could not reproduce: the script
    ran out early and the loop ended NO_TOOL_CALLS instead.
    """
    batches: Dict[int, List[dict]] = {}
    replies: Dict[int, Dict[str, Any]] = {}

    for record in load_trace(trace_path):
        event = record.get("event")
        try:
            turn = int(record.get("turn", 0) or 0)
        except (TypeError, ValueError):
            continue

        if event == "tool_call":
            batches.setdefault(turn, []).append(
                {"tool": record.get("tool", ""), "args": record.get("args") or {}}
            )
        elif event == "assistant":
            reply = replies.setdefault(turn, {})
            reply["text"] = record.get("text") or ""
            reply["finish_reason"] = record.get("finish_reason") or ""
        elif event in ("truncation_nudge", "truncated"):
            # A turn that spent the whole cap on reasoning writes no assistant record,
            # so the nudge is the only trace of it.
            reply = replies.setdefault(turn, {})
            reply.setdefault("text", "")
            if not reply.get("finish_reason"):
                reply["finish_reason"] = "length"
        elif event in ("stall_nudge", "no_tool_calls"):
            reply = replies.setdefault(turn, {})
            if not reply.get("text"):
                # The text has to be non-empty, or a reply with no content and no
                # stated reason replays as a truncation rather than a stall.
                reply["text"] = _STALL_TEXT

    # A turn with both text and tool calls is a tool-call turn: the calls are what the
    # loop acts on, and the text is already in the record for the reader.
    return [
        batches[turn] if turn in batches else replies[turn]
        for turn in sorted(set(batches) | set(replies))
    ]


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
            if record.get("finish_reason") == "length":
                lines.append(f"  [turn {record.get('turn')}] ^ cut off at the output token cap")
        elif event == "truncation_nudge":
            lines.append(
                f"  [turn {record.get('turn')}] told it was cut off "
                f"(correction {record.get('attempt')})"
            )
        elif event == "stall_nudge":
            lines.append(f"  [turn {record.get('turn')}] stalled — nudged")
        elif event == "truncated":
            lines.append(
                f"gave up after {record.get('truncations')} replies cut off at the token cap"
            )
        elif event == "completed":
            lines.append(
                f"completed after {record.get('tool_calls')} calls / "
                f"{record.get('turns')} turns: {record.get('summary', '')}"
            )
        elif event:
            lines.append(f"{event}: " + json.dumps({k: v for k, v in record.items() if k != 'event'}, default=str)[:160])
    return "\n".join(lines)


async def replay_step(trace_path, toolbox, *, budget: Optional[int] = None) -> Any:
    """Re-drive a recorded step against the current code.

    Returns the StepResult, with the toolbox mutated exactly as a live run would leave
    it, so the resulting workspace can be inspected or diffed. The budget defaults to
    the configured one: a hardcoded copy diverges from every BUDGET_EXHAUSTED step the
    moment the knob is changed.
    """
    import config

    from agentic.loop import run_agentic_step

    if budget is None:
        budget = getattr(config, "STEP_TOOL_BUDGET", 50)
    llm = scripted_from_trace(trace_path)
    return await run_agentic_step(
        llm,
        "replay",
        "replay",
        toolbox,
        budget=budget,
        trace_path=None,
    )
