"""
Shared scaffolding for LLM-calling nodes.

Absorbs banner printing, resume checks, prompt loading, LLM invocation,
debug-prompt saving, error handling, and state updates so each node
file only contains its unique logic (message building, post-processing).
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional, Type

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from config import get_llm_creative, get_llm_precise, get_provider, get_model
from utils.files import save_output
from utils.log import log


async def execute_llm_node(
    state,
    node_name: str,
    banner: str,
    system_message: str,
    user_message: str,
    *,
    resume_field: Optional[str] = None,
    resume_callback: Optional[Callable] = None,
    llm_mode: str = "creative",
    structured_schema: Optional[Type[BaseModel]] = None,
    output_dir: Optional[Path] = None,
    output_field: Optional[str] = None,
    output_filename: Optional[str] = None,
    raw_filename: Optional[str] = None,
    next_status: Optional[str] = None,
    post_process: Optional[Callable[[Any], Any]] = None,
    error_status: Optional[str] = None,
    error_field: Optional[str] = None,
):
    """Execute a full LLM node: banner, resume check, prompt, call, save, update.

    Returns (state, result) where result is None on skip or error.
    """
    # Banner
    print("\n" + "=" * 60)
    print(f"  NODE: {banner}")
    print("=" * 60)

    # Resume check
    if resume_field and getattr(state, resume_field, None):
        log(
            f"{node_name.capitalize()} already loaded (resume mode) - skipping LLM call",
            "SUCCESS",
        )
        if resume_callback:
            resume_callback(state)
        return state, None

    log(f"Sending to LLM ({llm_mode} mode, {get_provider()}/{get_model()})...")

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ]

    invoke_kwargs = {}
    # Custom logit processor (KimiK26ThinkingBudgetLogitProcessor) disabled for now.
    # if get_provider() == "cerit" and get_model() == "kimi-k2.6":
    #     invoke_kwargs["extra_body"] = {
    #         "custom_params": {"thinking_budget": CERIT_KIMI_THINKING_BUDGET},
    #         "custom_logit_processor": "true",
    #     }

    try:
        if llm_mode == "creative":
            ai_message = await get_llm_creative().ainvoke(messages, **invoke_kwargs)
            result = ai_message.content
        elif llm_mode == "structured":
            result = (
                await get_llm_precise()
                .with_structured_output(structured_schema)
                .ainvoke(messages, **invoke_kwargs)
            )
            ai_message = None
        else:
            ai_message = await get_llm_precise().ainvoke(messages, **invoke_kwargs)
            result = ai_message.content

        # Save thinking/reasoning content for debugging (e.g. DeepSeek, GLM thinking models)
        if ai_message is not None and output_dir:
            reasoning = (getattr(ai_message, "additional_kwargs", None) or {}).get(
                "reasoning_content"
            )
            if reasoning:
                save_output(output_dir, reasoning, f"thinking_{node_name}.md")
                log(f"Saved thinking content ({len(reasoning)} chars)", "INFO")

        # Save raw before post-processing
        if raw_filename and output_dir:
            save_output(
                output_dir,
                result if isinstance(result, str) else json.dumps(result, default=str),
                raw_filename,
            )

        if post_process:
            result = post_process(result)

        # Save debug prompt
        if output_dir:
            save_output(
                output_dir,
                f"# System Prompt\n\n{system_message}\n\n# User Message\n\n{user_message}",
                f"prompt_{node_name}.md",
            )

        # Save result file
        if output_filename and output_dir:
            content = (
                result
                if isinstance(result, str)
                else json.dumps(result, indent=2, default=str)
            )
            save_output(output_dir, content, output_filename)

        if output_field:
            setattr(state, output_field, result)
        if next_status:
            state.status = next_status

        return state, result

    except Exception as e:
        log(f"LLM error in {node_name}: {e}", "ERROR")
        if error_field:
            setattr(state, error_field, f"{node_name.capitalize()} failed: {e}")
        if error_status:
            state.status = error_status
        return state, None


def format_branching_status(branch_depth: int, max_depth: int) -> str:
    """Format the branching status block telling the LLM whether further branching is possible.

    Mirrors master.py's actual gate: a sub-branch is only spawned when
    ``branch_depth - 1 > 0``, so branching is genuinely available iff
    ``branch_depth > 1``. The block is emitted alongside other context blocks
    so the LLM can decide whether to suggest sub-strategies (propose) or pick
    the ``branch`` action (decide) without wasted work.
    """
    can_branch = branch_depth > 1
    levels_left = max(branch_depth - 1, 0)
    lines = [
        "## Branching Status:",
        f"- Current branch depth: {branch_depth} (configured max {max_depth}, counts down toward 0)",
    ]
    if can_branch:
        lines.append(
            f"- Further branching: AVAILABLE — sub-branches would have depth {branch_depth - 1} ({levels_left - 1} further levels possible after that)"
        )
    else:
        lines.append(
            "- Further branching: NOT AVAILABLE — depth budget exhausted; do not propose sub-strategies or pick the `branch` action"
        )
    return "\n".join(lines)


def format_ncu_context(ncu_metrics: Optional[dict]) -> str:
    """Format NCU metrics into a markdown section (shared by propose + decide)."""
    if not ncu_metrics:
        return ""
    lines = ["\n## NCU Metrics:"]
    for key, value in ncu_metrics.items():
        lines.append(
            f"- {key}: {value:.2f}" if isinstance(value, float) else f"- {key}: {value}"
        )
    return "\n".join(lines) + "\n"


def normalize_sub_strategies(sub_strats, parent_strategy) -> list:
    """Normalize sub-strategy objects/strings into dicts (shared logic in decide)."""
    parent_params = (
        parent_strategy.key_parameters
        if hasattr(parent_strategy, "key_parameters")
        else parent_strategy.get("key_parameters", [])
    )
    result = []
    for s in sub_strats:
        if isinstance(s, str):
            result.append(
                {
                    "name": s.replace(" ", "_").lower(),
                    "description": s,
                    "hypothesis": f"Exploring {s} variation",
                    "key_parameters": parent_params,
                }
            )
        elif hasattr(s, "model_dump"):
            result.append(s.model_dump())
        else:
            result.append(s)
    return result


# -------------------------------------------------------------------------
# Unified context builder
# -------------------------------------------------------------------------


def format_strategy(strategy) -> str:
    if not strategy:
        return ""
    name = (
        strategy.name if hasattr(strategy, "name") else strategy.get("name", "unknown")
    )
    desc = (
        strategy.description
        if hasattr(strategy, "description")
        else strategy.get("description", "")
    )
    hyp = (
        strategy.hypothesis
        if hasattr(strategy, "hypothesis")
        else strategy.get("hypothesis", "")
    )
    params = (
        strategy.key_parameters
        if hasattr(strategy, "key_parameters")
        else strategy.get("key_parameters", [])
    )
    return (
        f"## Strategy: {name}\n"
        f"**Description**: {desc}\n"
        f"**Hypothesis**: {hyp}\n"
        f"**Key Parameters**: {', '.join(params)}"
    )


def _format_current_iter(state, summary, tuner_tail) -> str:
    from utils.results import format_results_summary

    if not summary:
        return ""
    return (
        f"## Current Iteration ({state.iter_num}):\n\n"
        f"### Kernel Code:\n```cuda\n{state.kernel_code}\n```\n\n"
        f"### Tuning Parameters:\n```json\n{state.params_json}\n```\n\n"
        f"### Results Summary:\n{format_results_summary(summary)}\n\n"
        f"### Tuner Output (last 50 lines):\n```\n{tuner_tail or 'No output'}\n```\n"
        f"{format_ncu_context(state.ncu_metrics)}"
    )


def get_tuner_tail(run_output: str, max_lines: int = 50) -> str:
    if not run_output:
        return "No output"
    return "\n".join(run_output.split("\n")[-max_lines:])


def build_prompt_context(
    state, *, iteration_history_fields=None, summary=None, tuner_tail=None
) -> dict:
    """Extract common context from state into a dict for prompt functions.

    Works with both MainState and WorkingState — missing fields default to empty.
    """
    from state import (
        format_user_messages,
        format_iteration_history,
        format_iteration_summaries,
        format_parent_context,
        format_best_so_far,
    )
    from state.history import _parse_field_depths
    from config import HISTORY_ITERS

    branch_path = getattr(state, "branch_path", None)
    iter_num = getattr(state, "iter_num", 0)
    history_fields = iteration_history_fields or []
    field_depths = _parse_field_depths(history_fields, HISTORY_ITERS)

    ctx = {
        "problem_yaml": getattr(state, "problem_yaml", "") or "",
        "ref_kernel": getattr(state, "ref_kernel", "") or "",
        "analysis": getattr(state, "analysis", "") or "",
        "plan": getattr(state, "plan", "") if iter_num <= 1 else "",
        "strategy": getattr(state, "strategy", None),
        "kernel_code": getattr(state, "kernel_code", "") or "",
        "params_json": getattr(state, "params_json", "") or "",
        "proposal": getattr(state, "proposal", "") or "",
        "parent_context": format_parent_context(branch_path)
        if iter_num <= 1
        else "",
        "user_messages": format_user_messages(state.model_dump()),
        "iteration_summaries": format_iteration_summaries(branch_path, iter_num),
        "iteration_history": format_iteration_history(
            branch_path, iter_num, include=history_fields
        ),
        "best_so_far": format_best_so_far(branch_path, iter_num, field_depths),
    }
    if summary is not None:
        ctx["current_iteration"] = _format_current_iter(state, summary, tuner_tail)
    return ctx
