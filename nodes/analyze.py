"""Analyze node - analyzes the reference kernel to understand the algorithm."""

from config import get_output_dir
from utils.files import ensure_output_dir
from state.types import MainState
from nodes._llm_helper import execute_llm_node, build_prompt_context
import prompts.analyze


async def analyze_node(state: MainState) -> MainState:
    ensure_output_dir()

    ctx = build_prompt_context(state)
    system, user = prompts.analyze.build(ctx)

    state, analysis = await execute_llm_node(
        state,
        "analyze",
        "Analyze Reference Kernel",
        system,
        user,
        resume_field="analysis",
        output_dir=get_output_dir(),
        output_field="analysis",
        output_filename="analysis.md",
    )

    if analysis:
        print(
            f"\n--- Analysis Preview ---\n{analysis[:800]}{'...' if len(analysis) > 800 else ''}\n------------------------\n"
        )

    return state
