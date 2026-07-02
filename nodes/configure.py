"""Configure node — emits the three framework-file regions and assembles framework.cpp.

Framework-file mode: instead of params.json, the LLM writes the CACAO:KERNELS /
CACAO:PARAMS / CACAO:LAUNCHER region bodies (structured), which the engine splices
into the fixed skeleton via utils.framework.assemble_framework_cpp(). The iteration
directory is staged for the compile step (kernels.cu + inputs.hpp beside framework.cpp).
"""

import shutil

import yaml

from config import get_problem_dir
from models.regions import FrameworkRegions
from utils.files import save_output, get_iter_dir
from utils.framework import assemble_framework_cpp
from utils.log import log
from state.types import WorkingState
from nodes._llm_helper import execute_llm_node, build_prompt_context
import prompts.framework_configure


async def configure_node(state: WorkingState) -> WorkingState:
    iteration = state.iter_num
    strategy = state.strategy
    branch_name = strategy.name if strategy else "default"

    iter_dir = get_iter_dir(state)
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Stage the runtime NVRTC source (LLM-authored kernels) for the driver.
    if state.kernel_code:
        save_output(iter_dir, state.kernel_code, "kernels.cu")

    # inputs.hpp must sit beside framework.cpp for its #include at compile time.
    problem_dir = get_problem_dir()
    inputs_src = problem_dir / "inputs.hpp"
    inputs_hpp = ""
    if inputs_src.exists():
        inputs_hpp = inputs_src.read_text()
        shutil.copyfile(inputs_src, iter_dir / "inputs.hpp")
    else:
        log("inputs.hpp not found in problem dir — framework build will fail", "WARN")

    # Build context
    ctx = build_prompt_context(
        state,
        iteration_history_fields=[
            {"name": "results_summary"},
            {"name": "decision"},
            {"name": "feedback", "limit": 1},
            {"name": "proposal", "limit": 1},
        ],
    )
    ctx["inputs_hpp"] = inputs_hpp

    key_params = strategy.key_parameters if strategy else []
    if key_params:
        ctx["strategy_section"] = (
            f'## Strategy Key Parameters\nThe strategy "{strategy.name}" should focus '
            f'on these parameters: {", ".join(key_params)}'
        )

    prev_context = ""
    if state.feedback:
        prev_context += f"## Previous Feedback:\n{state.feedback}\n"
    if state.run_output:
        tail = "\n".join(state.run_output.strip().split("\n")[-30:])
        prev_context += f"\n## Previous Run Output (last 30 lines):\n```\n{tail}\n```"
    if prev_context:
        ctx["prev_context"] = prev_context

    system, user = prompts.framework_configure.build(ctx)

    meta = yaml.safe_load(state.problem_yaml) if state.problem_yaml else {}

    def process(result: FrameworkRegions) -> str:
        regions = {
            "kernels": result.kernels,
            "params": result.params,
            "launcher": result.launcher,
        }
        framework_cpp = assemble_framework_cpp(meta, regions)
        log("Framework regions configured; framework.cpp assembled", "SUCCESS")
        return framework_cpp

    state, framework_cpp = await execute_llm_node(
        state,
        "configure",
        f"Configure Framework [{branch_name}] (iter {iteration})",
        system,
        user,
        llm_mode="structured",
        structured_schema=FrameworkRegions,
        output_dir=iter_dir,
        output_field="framework_cpp",
        output_filename="framework.cpp",
        next_status="running",
        post_process=process,
    )

    if framework_cpp is not None:
        preview = "\n".join(framework_cpp.split("\n")[:40])
        print(f"\n--- framework.cpp (head) ---\n{preview}\n...\n----------------------------\n")
    else:
        log("Framework configuration failed (structured LLM error)", "ERROR")
        state.status = "deciding"
        state.run_output = "Framework configuration failed"

    return state
