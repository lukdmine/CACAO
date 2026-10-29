"""The agentic step loop: bind tools, call the model, run what it asks for, repeat
until end_step or the budget runs out.

A turn with no tool call is corrected, not surrendered to — truncation and stalls get
different nudges (see _TRUNCATION_NUDGES). Every call is appended to a trace file for
offline replay (agentic/replay.py).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agentic.tools import SCHEMAS, Toolbox

# One predicate, shared with the retry wrapper that reads the same replies: the two
# copies drifted and Anthropic/Gemini truncations stopped being recognised on one side.
from llm.tracked import finish_reason as _finish_reason, was_truncated as _was_truncated
from utils.log import log

# Sent once if the model stops calling tools without ending the step. A single nudge:
# repeating it turns a confused model into an expensive loop.
_NUDGE = (
    "You have not finished. Either call more tools, or call end_step "
    "(which requires a passing check_compilation)."
)

# Sent when a turn was cut off at the output token cap before it reached a tool call.
# Escalating, because the corrections are not interchangeable: the first asks for the
# behaviour change that usually suffices, the later ones shrink the unit of work until
# whatever the model was trying to emit fits inside one reply.
_TRUNCATION_NUDGES = (
    "Your previous reply was cut off at the output token limit before you produced a "
    "tool call, so nothing was written. Do not plan, explain or summarise in prose "
    "first — your next reply must open with a tool call.",
    "Cut off again. Reduce the size of the reply: call write_file for ONE file only, "
    "or use edit_file to change just the part that differs. You have as many turns as "
    "you need — the files do not have to be written in a single reply.",
    "Cut off again. Emit the smallest useful call you can: a single edit_file on one "
    "region, or read_file to reorient. Anything committed now is better than another "
    "truncated reply.",
)

# How many truncated turns are corrected before the step gives up. Three, because the
# correction escalates and the third is qualitatively different from the first; a model
# that cannot answer inside its own cap after that is not going to.
TRUNCATION_RETRIES = 3


class StepOutcome(str, Enum):
    COMPLETED = "completed"
    NO_TOOL_CALLS = "no_tool_calls"
    TRUNCATED = "truncated"
    BIND_UNSUPPORTED = "bind_unsupported"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LLM_ERROR = "llm_error"

    @property
    def aborted(self) -> bool:
        """Whether the step ended before the model could author anything.

        BUDGET_EXHAUSTED is excluded: the model drove the loop and may well have
        finished the files before running out of calls, which only the caller can see.
        """
        return self in (
            StepOutcome.NO_TOOL_CALLS,
            StepOutcome.TRUNCATED,
            StepOutcome.BIND_UNSUPPORTED,
            StepOutcome.LLM_ERROR,
        )

    @property
    def diagnosis(self) -> str:
        """What to tell the branch when a step ends on this outcome."""
        return {
            StepOutcome.NO_TOOL_CALLS: (
                "the model answered in prose and never called a tool, so no kernel or "
                "driver region was written"
            ),
            StepOutcome.TRUNCATED: (
                "the model was cut off at its output token limit on every reply and "
                "never reached a tool call, so nothing was written. It is spending the "
                "whole reply budget reasoning before it acts"
            ),
            StepOutcome.BIND_UNSUPPORTED: (
                "the provider rejected the tool schemas, so the authoring step could "
                "not run at all"
            ),
            StepOutcome.LLM_ERROR: "the provider errored during the authoring step",
            StepOutcome.BUDGET_EXHAUSTED: (
                "the step ran out of tool calls before the files were complete"
            ),
        }.get(self, self.value)


@dataclass
class StepResult:
    outcome: StepOutcome
    turns: int = 0
    tool_calls: int = 0
    summary: str = ""
    error: str = ""
    text: str = ""
    truncations: int = 0
    tools_used: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome is StepOutcome.COMPLETED


class _Trace:
    """Append-only JSONL record of the step. Failures to write are never fatal."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text("", encoding="utf-8")
            except OSError as e:
                log(f"Could not open step trace {self.path}: {e}", "WARN")
                self.path = None

    def write(self, record: dict) -> None:
        if not self.path:
            return
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            self.path = None


def _normalize_tool_calls(message: AIMessage) -> List[Dict[str, Any]]:
    """Tool calls as (name, args, id) dicts.

    Providers differ on whether an id is present; a synthesized one keeps the
    tool-result pairing valid rather than letting the next request fail validation.
    """
    calls = getattr(message, "tool_calls", None) or []
    out = []
    for i, tc in enumerate(calls):
        if not isinstance(tc, dict):
            tc = {"name": getattr(tc, "name", ""), "args": getattr(tc, "args", {}), "id": getattr(tc, "id", None)}
        out.append(
            {
                "name": tc.get("name") or "",
                "args": tc.get("args") or {},
                "id": tc.get("id") or f"call_{i}",
            }
        )
    return out


async def run_agentic_step(
    llm,
    system_message: str,
    user_message: str,
    toolbox: Toolbox,
    *,
    budget: int = 50,
    trace_path: Optional[Path] = None,
    schemas: Optional[List[type]] = None,
    truncation_retries: int = TRUNCATION_RETRIES,
) -> StepResult:
    """Run one authoring step to completion, budget exhaustion, or an aborted outcome.

    ``schemas`` is what gets bound. Callers pass the set that can actually return
    something for this step (see agentic.tools.schemas_for); the module default exists
    for the replay harness and tests.

    ``truncation_retries`` is how many times a reply cut off at the token cap is
    corrected before the step gives up on the model reaching a tool call.
    """
    # Opened before the bind: _Trace truncates the file, so returning above it would
    # leave a resumed iteration holding the previous attempt's trace as if it were this
    # one's, with no record that binding failed.
    trace = _Trace(trace_path)
    trace.write({"event": "start", "budget": budget})

    try:
        bound = llm.bind_tools(schemas if schemas is not None else SCHEMAS)
    except Exception as e:
        log(f"Provider does not support tool binding: {e}", "WARN")
        trace.write({"event": "bind_unsupported", "error": str(e)})
        return StepResult(StepOutcome.BIND_UNSUPPORTED, error=str(e))

    messages: List[Any] = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ]
    result = StepResult(StepOutcome.BUDGET_EXHAUSTED)
    nudged = False
    truncations = 0

    while result.tool_calls < budget:
        result.turns += 1
        try:
            ai = await bound.ainvoke(messages)
        except Exception as e:
            log(f"LLM error during agentic step: {e}", "ERROR")
            trace.write({"event": "llm_error", "turn": result.turns, "error": str(e)})
            result.outcome = StepOutcome.LLM_ERROR
            result.error = str(e)
            return result

        messages.append(ai)
        content = ai.content if isinstance(ai.content, str) else ""
        if content:
            result.text = content

        calls = _normalize_tool_calls(ai)

        # The model's own text and reasoning, so the trace is a complete record of the
        # step rather than only what it called.
        reasoning = (getattr(ai, "additional_kwargs", None) or {}).get("reasoning_content")
        finish_reason = _finish_reason(ai)
        if content or reasoning:
            trace.write(
                {
                    "event": "assistant",
                    "turn": result.turns,
                    "text": content,
                    "reasoning": reasoning or "",
                    "finish_reason": finish_reason,
                    "tools": [c["name"] for c in calls],
                }
            )
        if not calls:
            # A turn that reached no tool call is corrected, not surrendered to. Which
            # correction depends on why: a model cut off mid-thought needs to be told
            # to be shorter, and telling it to "call more tools" instead is advice for
            # a problem it does not have.
            if _was_truncated(ai, content):
                truncations += 1
                result.truncations = truncations
                if truncations > truncation_retries:
                    trace.write(
                        {
                            "event": "truncated",
                            "turn": result.turns,
                            "truncations": truncations,
                        }
                    )
                    log(
                        f"Model cut off at its token limit {truncations}x without "
                        "reaching a tool call — ending the step",
                        "ERROR",
                    )
                    result.outcome = StepOutcome.TRUNCATED
                    result.error = (
                        f"reply cut off at the output token limit {truncations} times "
                        f"(finish_reason={finish_reason or 'unreported'})"
                    )
                    return result
                nudge = _TRUNCATION_NUDGES[
                    min(truncations, len(_TRUNCATION_NUDGES)) - 1
                ]
                log(
                    f"Turn {result.turns} was cut off at the token limit "
                    f"(finish_reason={finish_reason or 'unreported'}) — telling the "
                    f"model and retrying ({truncations}/{truncation_retries})",
                    "WARN",
                )
                trace.write(
                    {
                        "event": "truncation_nudge",
                        "turn": result.turns,
                        "attempt": truncations,
                        "text": nudge,
                    }
                )
                messages.append(HumanMessage(content=nudge))
                continue

            # Real prose and no call: the model stalled. One nudge, as ever.
            if nudged:
                trace.write({"event": "no_tool_calls", "turn": result.turns})
                result.outcome = StepOutcome.NO_TOOL_CALLS
                return result
            nudged = True
            trace.write({"event": "stall_nudge", "turn": result.turns})
            messages.append(HumanMessage(content=_NUDGE))
            continue

        for call in calls:
            if result.tool_calls >= budget:
                # Every tool call needs a matching result or the next request is
                # malformed; the loop exits after this message is complete.
                messages.append(
                    ToolMessage(
                        content="ERROR: step tool budget exhausted.",
                        tool_call_id=call["id"],
                    )
                )
                continue

            result.tool_calls += 1
            # Off the event loop: check_compilation blocks on g++ and on in-process
            # NVRTC compiles for tens of seconds, and the four branch coroutines share
            # one loop — a synchronous call here stalls the other three, including the
            # subprocess watchdogs whose clocks keep running regardless.
            output = await asyncio.to_thread(
                toolbox.dispatch, call["name"], call["args"]
            )
            result.tools_used[call["name"]] = result.tools_used.get(call["name"], 0) + 1
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))
            trace.write(
                {
                    "event": "tool_call",
                    "turn": result.turns,
                    "index": result.tool_calls,
                    "tool": call["name"],
                    "args": call["args"],
                    "result": output,
                }
            )

            if toolbox.ended:
                result.outcome = StepOutcome.COMPLETED
                result.summary = toolbox.end_summary
                trace.write(
                    {
                        "event": "completed",
                        "turns": result.turns,
                        "tool_calls": result.tool_calls,
                        "summary": result.summary,
                    }
                )
                return result

    trace.write(
        {
            "event": "budget_exhausted",
            "turns": result.turns,
            "tool_calls": result.tool_calls,
        }
    )
    log(
        f"Agentic step hit the {budget}-call budget "
        f"({result.turns} turns, checks passed: {toolbox.check_passed})",
        "WARN",
    )
    return result
