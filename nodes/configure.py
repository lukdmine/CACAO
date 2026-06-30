"""Configure node - generates the params.json tuning configuration."""

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm_precise
from models.params import TuningConfig
from utils.files import save_output, get_iter_dir
from utils.log import log
from state.types import WorkingState
from nodes._llm_helper import execute_llm_node, build_prompt_context
import prompts.configure


async def configure_node(state: WorkingState) -> WorkingState:
    iteration = state.iter_num
    strategy = state.strategy
    branch_name = strategy.name if strategy else "default"

    iter_dir = get_iter_dir(state)
    iter_dir.mkdir(parents=True, exist_ok=True)
    if not (iter_dir / "kernel.cu").exists() and state.kernel_code:
        save_output(iter_dir, state.kernel_code, "kernel.cu")

    # Build context
    ctx = build_prompt_context(
        state,
        iteration_history_fields=[
            {"name": "params_json", "limit": 1},
            {"name": "results_summary"},
            {"name": "decision"},
            {"name": "feedback", "limit": 1},
            {"name": "proposal", "limit": 1},
        ],
    )

    # Add node-specific context
    key_params = strategy.key_parameters if strategy else []
    if key_params:
        ctx["strategy_section"] = (
            f'## Strategy Key Parameters\nThe strategy "{strategy.name}" should focus on these parameters: {", ".join(key_params)}'
        )

    prev_context = ""
    if state.feedback:
        prev_context += f"## Previous Feedback:\n{state.feedback}\n"
    if state.run_output:
        tail = "\n".join(state.run_output.strip().split("\n")[-30:])
        prev_context += f"\n## Previous Run Output (last 30 lines):\n```\n{tail}\n```"
    if prev_context:
        ctx["prev_context"] = prev_context

    system, user = prompts.configure.build(ctx)

    def process_structured(result: TuningConfig) -> str:
        params_dict = result.model_dump(exclude_none=True)
        params_json = json.dumps(params_dict, indent=2)
        log(f"Parameters configured ({len(result.parameters)} params)", "SUCCESS")
        return params_json

    state, params_json = await execute_llm_node(
        state,
        "configure",
        f"Configure Parameters [{branch_name}] (iter {iteration})",
        system,
        user,
        llm_mode="structured",
        structured_schema=TuningConfig,
        output_dir=iter_dir,
        output_field="params_json",
        output_filename="params.json",
        next_status="running",
        post_process=process_structured,
    )

    if params_json is not None:
        lines = params_json.split("\n")[:30]
        print(f"\n--- params.json ---\n{chr(10).join(lines)}")
        if len(params_json.split("\n")) > 30:
            print("... (truncated)")
        print("-------------------\n")
    else:
        # Fallback: raw text mode — reuse prompt from above
        log("Falling back to raw text mode...", "WARN")
        try:
            llm = get_llm_precise()
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            raw_response = (await llm.ainvoke(messages)).content
            fallback_json = re.sub(r"```\w*\n?", "", raw_response).strip()
            json.loads(fallback_json)  # validate
            save_output(iter_dir, fallback_json, "params.json")
            save_output(iter_dir, raw_response, "params_raw.txt")
            state.params_json = fallback_json
            state.status = "running"
            log("Parameters configured (fallback mode)", "SUCCESS")
        except Exception as fallback_e:
            log(f"Fallback also failed: {fallback_e}", "ERROR")
            state.status = "deciding"
            state.run_output = f"Configuration failed (fallback: {fallback_e})"

    return state
