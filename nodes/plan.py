"""Plan node - creates a strategy-specific optimization plan."""

from pathlib import Path

from utils.log import log
from state.types import WorkingState
from nodes._llm_helper import execute_llm_node, build_prompt_context
import prompts.plan


async def plan_node(state: WorkingState) -> WorkingState:
    strategy = state.strategy or {}
    branch_name = strategy.name if strategy else "default"
    branch_path = Path(state.branch_path) if state.branch_path else None

    def on_resume(s):
        s.status = "implementing"

    ctx = build_prompt_context(state)
    system, user = prompts.plan.build(ctx)

    state, plan = await execute_llm_node(
        state,
        "plan",
        f"Create Plan [{branch_name}]",
        system,
        user,
        resume_field="plan",
        resume_callback=on_resume,
        output_dir=branch_path,
        output_field="plan",
        output_filename="plan.md",
        next_status="implementing",
        error_status="deciding",
        error_field="run_output",
    )

    if plan and branch_path:
        log(f"Plan saved to: {branch_path / 'plan.md'}", "SUCCESS")

    return state
