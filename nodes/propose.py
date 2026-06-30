"""Propose node - analyzes results and proposes optimizations before the decide node."""

from config import get_output_dir, MAX_BRANCH_DEPTH
from utils.files import get_iter_dir
from utils.results import get_results_summary, load_reference_time
from state.types import WorkingState
from state import format_existing_branches
from nodes._llm_helper import (
    execute_llm_node,
    build_prompt_context,
    get_tuner_tail,
    format_branching_status,
)
import prompts.propose


async def propose_node(state: WorkingState) -> WorkingState:
    iteration = state.iter_num
    strategy = state.strategy or {}
    branch_name = strategy.name if strategy else "default"
    max_iter = state.max_iter

    if iteration >= max_iter:
        state.proposal = (
            f"Max iterations ({max_iter}) reached. No further proposals needed."
        )
        state.status = "deciding"
        return state

    iter_dir = get_iter_dir(state)
    ref_time = load_reference_time(get_output_dir())
    summary = get_results_summary(iter_dir / "results.json", ref_time)
    tuner_tail = get_tuner_tail(state.run_output)

    ctx = build_prompt_context(
        state,
        iteration_history_fields=[
            {"name": "kernel_code", "limit": 1},
            {"name": "params_json", "limit": 1},
            {"name": "results_summary"},
            {"name": "decision"},
            {"name": "feedback", "limit": 1},
        ],
        summary=summary,
        tuner_tail=tuner_tail,
    )
    ctx["branch_name"] = branch_name
    ctx["existing_branches"] = format_existing_branches(str(get_output_dir()))
    ctx["branching_status"] = format_branching_status(
        state.branch_depth, MAX_BRANCH_DEPTH
    )
    system, user = prompts.propose.build(ctx)

    state, _ = await execute_llm_node(
        state,
        "propose",
        f"Propose Optimizations [{branch_name}] (iter {iteration}/{max_iter})",
        system,
        user,
        output_dir=iter_dir,
        output_field="proposal",
        output_filename="proposal.md",
        next_status="deciding",
        error_field="proposal",
        error_status="deciding",
    )

    return state
