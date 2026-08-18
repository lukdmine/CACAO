"""The agentic step loop.

Drives one step: bind the tools, call the model, execute what it asks for, repeat
until it ends the step or the budget runs out. Every outcome other than COMPLETED is
something the caller can fall back from, so a provider that cannot do multi-turn tool
use degrades to the previous single-shot path instead of failing the branch.

Every tool call is appended to a trace file as it happens, with full arguments, so a
live run can be replayed offline (see agentic/replay.py) rather than re-run.
"""

from __future__ import annotations

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
from utils.log import log

# Sent once if the model stops calling tools without ending the step. A single nudge:
# repeating it turns a confused model into an expensive loop.
_NUDGE = (
    "You have not finished. Either call more tools, or call end_step "
    "(which requires a passing check_compilation)."
)


class StepOutcome(str, Enum):
    COMPLETED = "completed"
    NO_TOOL_CALLS = "no_tool_calls"
    BIND_UNSUPPORTED = "bind_unsupported"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LLM_ERROR = "llm_error"

    @property
    def should_fall_back(self) -> bool:
        """Whether the caller should run the single-shot path instead.

        BUDGET_EXHAUSTED is excluded: whether it is recoverable depends on whether the
        files ended up complete, which only the caller can see.
        """
        return self in (
            StepOutcome.NO_TOOL_CALLS,
            StepOutcome.BIND_UNSUPPORTED,
            StepOutcome.LLM_ERROR,
        )


@dataclass
class StepResult:
    outcome: StepOutcome
    turns: int = 0
    tool_calls: int = 0
    summary: str = ""
    error: str = ""
    text: str = ""
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
) -> StepResult:
    """Run one authoring step to completion, budget exhaustion, or a fallback signal.

    ``schemas`` is what gets bound. Callers pass the set that can actually return
    something for this step (see agentic.tools.schemas_for); the module default exists
    for the replay harness and tests.
    """
    try:
        bound = llm.bind_tools(schemas if schemas is not None else SCHEMAS)
    except Exception as e:
        log(f"Provider does not support tool binding: {e}", "WARN")
        return StepResult(StepOutcome.BIND_UNSUPPORTED, error=str(e))

    trace = _Trace(trace_path)
    trace.write({"event": "start", "budget": budget})

    messages: List[Any] = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ]
    result = StepResult(StepOutcome.BUDGET_EXHAUSTED)
    nudged = False

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
        if content or reasoning:
            trace.write(
                {
                    "event": "assistant",
                    "turn": result.turns,
                    "text": content,
                    "reasoning": reasoning or "",
                    "tools": [c["name"] for c in calls],
                }
            )
        if not calls:
            # No tool call on the first turn means the model is not driving the loop at
            # all — the caller's single-shot path will do better than a nudge war.
            if result.turns == 1 or nudged:
                trace.write({"event": "no_tool_calls", "turn": result.turns})
                result.outcome = StepOutcome.NO_TOOL_CALLS
                return result
            nudged = True
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
            output = toolbox.dispatch(call["name"], call["args"])
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
