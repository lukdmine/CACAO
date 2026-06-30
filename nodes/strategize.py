"""Strategize node - identifies optimization strategies from the analysis."""

from config import get_output_dir
from models.strategy import StrategizeOutput
from utils.files import save_json
from utils.log import log
from state.types import MainState
from nodes._llm_helper import execute_llm_node, build_prompt_context
import prompts.strategize


async def strategize_node(state: MainState) -> MainState:
    def on_resume(s):
        for i, strat in enumerate(s.strategies, 1):
            log(
                f"  {i}. {strat.get('name', 'unknown')}: {strat.get('description', '')[:60]}..."
            )

    def process(result: StrategizeOutput):
        strategies = [s.model_dump() for s in result.strategies]
        save_json(
            get_output_dir() / "strategies.json",
            {"strategies": strategies, "reasoning": result.reasoning},
        )
        for i, s in enumerate(strategies, 1):
            log(f"  {i}. {s['name']}: {s['description'][:60]}...")
        return strategies

    ctx = build_prompt_context(state)
    system, user = prompts.strategize.build(ctx)

    state, strategies = await execute_llm_node(
        state,
        "strategize",
        "Identify Strategies",
        system,
        user,
        resume_field="strategies",
        resume_callback=on_resume,
        llm_mode="structured",
        structured_schema=StrategizeOutput,
        output_dir=get_output_dir(),
        output_field="strategies",
        post_process=process,
    )

    if strategies is None and not state.strategies:
        state.strategies = [
            {
                "name": "baseline_optimization",
                "description": "Apply standard CUDA optimizations (tiling, coalescing)",
                "hypothesis": "Standard optimizations should improve performance",
                "key_parameters": ["BLOCK_X", "BLOCK_Y", "TILE_SIZE"],
            }
        ]
        log("Using fallback single strategy", "WARN")
        save_json(get_output_dir() / "strategies.json", state.strategies)

    return state
