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
from utils.framework import assemble_framework_cpp, resolve_cuda_include
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

    # inputs.hpp must sit beside framework.cpp for its #include at compile time. It is
    # generated from inputs.yaml once per run (utils.inputs.ensure_inputs_hpp), so its
    # absence here is not something this iteration can recover from: the LLM does not own
    # the file and cannot author it. Fail loudly rather than routing an impossible task to
    # the fix loop.
    problem_dir = get_problem_dir()
    inputs_src = problem_dir / "inputs.hpp"
    if not inputs_src.exists():
        raise FileNotFoundError(
            f"{inputs_src} not found — the problem's I/O boundary was never generated. "
            "It comes from inputs.yaml at the start of a run; check that inputs.yaml exists."
        )
    inputs_hpp = inputs_src.read_text()
    shutil.copyfile(inputs_src, iter_dir / "inputs.hpp")

    # Build context
    ctx = build_prompt_context(
        state,
        iteration_history_fields=[
            {"name": "results_summary"},
            {"name": "decision"},
            {"name": "feedback", "limit": 1},
            {"name": "proposal", "limit": 1},
            # Was a hand-rolled 30-line tail of the carried run_output. History's
            # formatter produces the same excerpt from the same data, and excerpts a
            # compile failure from the end that actually holds the diagnosis.
            {"name": "run_output", "limit": 1},
        ],
    )
    ctx["inputs_hpp"] = inputs_hpp

    key_params = strategy.key_parameters if strategy else []
    if key_params:
        ctx["strategy_section"] = (
            f'## Strategy Key Parameters\nThe strategy "{strategy.name}" should focus '
            f'on these parameters: {", ".join(key_params)}'
        )

    system, user = prompts.framework_configure.build(ctx)

    meta = yaml.safe_load(state.problem_yaml) if state.problem_yaml else {}

    def process(result: FrameworkRegions) -> str:
        regions = {
            "kernels": result.kernels,
            "params": result.params,
            "launcher": result.launcher,
        }
        framework_cpp = assemble_framework_cpp(meta, regions)
        log(
            f"Framework regions configured; framework.cpp assembled "
            f"(NVRTC include: {resolve_cuda_include()})",
            "SUCCESS",
        )
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
