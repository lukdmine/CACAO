"""Decide node - reads the proposal and makes a structured action decision."""

from config import get_output_dir, MAX_BRANCH_DEPTH
from models.decision import OptimizationDecision
from utils.files import save_json, get_iter_dir
from utils.log import log
from utils.results import (
    get_results_summary,
    load_reference_time,
    format_results_summary,
)
from state.types import WorkingState
from state import (
    format_user_messages,
    format_iteration_summaries,
    format_existing_branches,
)
from state.persistence import load_iter_state_if_exists
from nodes._llm_helper import (
    execute_llm_node,
    normalize_sub_strategies,
    format_branching_status,
)
import prompts.decide


async def decide_node(state: WorkingState) -> WorkingState:
    iteration = state.iter_num
    strategy = state.strategy or {}
    branch_name = strategy.name if strategy else "default"
    branch_depth = state.branch_depth
    max_iter = state.max_iter

    iter_dir = get_iter_dir(state)

    # Load results summary
    summary = state.results_summary or get_results_summary(
        iter_dir / "results.json", load_reference_time(get_output_dir())
    )
    log(f"Results: {summary['num_successful']}/{summary['num_total']} successful")

    # Max iterations check — no LLM call needed
    if iteration >= max_iter:
        log(f"Max iterations ({max_iter}) reached", "WARN")
        decision_dict = {
            "action": "stop",
            "reasoning": f"Maximum iterations ({max_iter}) reached",
            "feedback": "Optimization stopped due to iteration limit",
        }
        save_json(iter_dir / "decision.json", decision_dict)
        state.decision = decision_dict
        state.feedback = decision_dict["feedback"]
        state.status = "success" if summary["has_success"] else "failed"
        state.next_status = state.status
        return state

    # Load previous iteration's feedback (what was asked to be implemented)
    prev_feedback = ""
    if iteration > 1 and state.branch_path:
        from pathlib import Path

        prev_state = load_iter_state_if_exists(Path(state.branch_path), iteration - 1)
        if prev_state and prev_state.decision:
            prev_feedback = prev_state.decision.get("feedback", "")

    # Build lightweight context — the proposal already contains the full analysis
    iteration_summaries = (
        format_iteration_summaries(state.branch_path, iteration)
        if state.branch_path
        else ""
    )

    ctx = {
        "iter_info": (
            f"## Iteration Info:\n"
            f"- Current iteration: {iteration}\n"
            f"- Max iterations: {max_iter}\n"
            f"- Strategy: {branch_name}"
        ),
        "results_summary_text": f"## Results Summary:\n{format_results_summary(summary)}",
        "proposal": getattr(state, "proposal", "") or "",
        "prev_feedback": prev_feedback,
        "iteration_summaries": iteration_summaries,
        "existing_branches": format_existing_branches(str(get_output_dir())),
        "branching_status": format_branching_status(branch_depth, MAX_BRANCH_DEPTH),
        "user_messages": format_user_messages(state.model_dump()),
    }
    system, user = prompts.decide.build(ctx)

    state, decision_obj = await execute_llm_node(
        state,
        "decide",
        f"Decide [{branch_name}] (iter {iteration}/{max_iter})",
        system,
        user,
        llm_mode="structured",
        structured_schema=OptimizationDecision,
        output_dir=iter_dir,
    )

    if decision_obj is None:
        # Fallback decision
        if summary["has_success"]:
            decision_dict = {
                "action": "stop",
                "reasoning": "LLM error, but some configurations succeeded",
                "feedback": "Stopping due to LLM error",
            }
            status = "success"
        else:
            decision_dict = {
                "action": "retry",
                "reasoning": "LLM error, no successful configurations",
                "feedback": "Fix compilation or runtime errors",
            }
            status = "implementing"
        state.decision = decision_dict
        state.feedback = decision_dict["feedback"]
        state.next_status = status
        state.status = "decided"
        return state

    decision_dict = decision_obj.model_dump()
    save_json(iter_dir / "decision.json", decision_dict)

    log(f"Decision: {decision_obj.action.upper()}", "SUCCESS")
    log(f"Reasoning: {decision_obj.reasoning[:100]}...")
    if decision_obj.error_analysis:
        log(f"Error type: {decision_obj.error_analysis.error_type}")
        log(f"Root cause: {decision_obj.error_analysis.root_cause[:80]}...")

    # Map decision to status
    action = decision_obj.action
    sub_strategies = None

    if action == "stop":
        status = "success" if summary["has_success"] else "failed"
    elif action == "retry":
        status = "configuring" if decision_obj.skip_implement else "implementing"
        if decision_obj.skip_implement:
            log("Retry with skip_implement — reconfiguring params only", "INFO")
    elif action == "continue":
        if decision_obj.skip_implement:
            status = "configuring"
            log("Skipping implement — reconfiguring params only", "INFO")
        else:
            status = "implementing"
    elif action == "branch":
        if branch_depth > 1 and decision_obj.sub_strategies:
            status = "branching"
            sub_strategies = normalize_sub_strategies(
                decision_obj.sub_strategies, strategy
            )
            log(f"Branch action: prepared {len(sub_strategies)} sub-strategies")
        else:
            log(
                "Branch requested but no depth remaining or no sub-strategies — continuing instead",
                "WARN",
            )
            status = "implementing"
    else:
        status = "failed"

    state.decision = decision_dict
    state.feedback = decision_obj.feedback
    state.next_status = status
    state.status = "decided"
    state.sub_strategies = sub_strategies
    return state
