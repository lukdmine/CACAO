"""Implement node - writes the optimized CUDA kernel."""

import re
from pathlib import Path

from config import get_output_dir, get_problem_dir
from utils.files import save_output, create_iter_dir
from utils.inputs import load_inputs_spec, scalar_contract_text
from utils.log import log
from state.types import WorkingState
from nodes._llm_helper import execute_llm_node, build_prompt_context
import prompts.implement
import prompts.fix_errors


def _select_prompt(mode: str):
    """Retry means the last attempt failed to compile or validate: fix_errors is the
    prompt that shows the diagnostics and asks for the smallest correct fix."""
    return prompts.fix_errors if mode == "retry" else prompts.implement


async def implement_node(state: WorkingState) -> WorkingState:
    iteration = state.iter_num
    branch_path = Path(state.branch_path) if state.branch_path else None
    strategy = state.strategy or {}
    branch_name = strategy.name if strategy else "default"
    max_iter = state.max_iter

    # Create iteration directory
    if branch_path:
        iter_dir = create_iter_dir(branch_path, iteration)
    else:
        iter_dir = get_output_dir() / f"iter{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
    # How this iteration was entered, decided once at creation (state/persistence.py).
    # Previously inferred from a decision blob copied forward from the previous
    # iteration; that no longer exists, and state.decision is always None here anyway --
    # decide runs last, implement runs first.
    is_retry = state.mode == "retry"
    is_followup = state.mode in ("retry", "followup")

    if is_retry:
        log(f"Retry mode: fixing errors from iteration {iteration - 1}", "WARN")
    elif is_followup:
        log(
            f"Follow-up implementation mode: applying requested changes from iteration {iteration - 1}",
            "INFO",
        )

    # Build context
    ctx = build_prompt_context(
        state,
        iteration_history_fields=[
            {"name": "kernel_code", "limit": 1},
            {"name": "results_summary"},
            {"name": "decision"},
            {"name": "feedback", "limit": 1},
            {"name": "proposal", "limit": 1},
            # A retry must see what it is fixing. This used to ride in on the carried
            # state.run_output, which no longer exists.
            {"name": "run_output", "limit": 1},
        ],
    )

    # Framework mode: show the I/O boundary so the kernel signature matches it.
    inputs_src = get_problem_dir() / "inputs.hpp"
    if inputs_src.exists():
        ctx["inputs_hpp"] = inputs_src.read_text()

    # State how THIS problem's scalars reach a kernel. inputs.hpp shows them as
    # `inline constexpr`, which reads like they are available everywhere — but NVRTC
    # compiles kernels.cu standalone and never sees that file. Whether a scalar is a -D
    # macro or a kernel argument is per-problem, and a generic prompt gets it wrong for
    # whichever problems do not match its example.
    inputs_yaml = get_problem_dir() / "inputs.yaml"
    if inputs_yaml.exists():
        try:
            ctx["scalar_contract"] = scalar_contract_text(load_inputs_spec(inputs_yaml))
        except Exception as e:
            log(f"Could not derive the scalar contract from inputs.yaml: {e}", "WARN")

    # Only the instruction. The feedback, the previous kernel, and the run output all come
    # from the iteration-history section that precedes this one in the prompt — sending
    # them again here duplicated ~16 KB per call.
    if is_followup:
        ctx["current_context"] = (
            "Fix the issues in the kernel using the feedback, tuner output, and kernel "
            "from the most recent iteration above."
            if is_retry
            else "Apply the requested kernel changes from the feedback in the most recent "
            "iteration above. Preserve correctness, keep all memory accesses in-bounds, "
            "and keep parameter usage consistent with the tuning configuration."
        )

    system, user = _select_prompt(state.mode).build(ctx)

    state, kernel_code = await execute_llm_node(
        state,
        "implement",
        f"Implement Kernel [{branch_name}] (iter {iteration}/{max_iter})",
        system,
        user,
        llm_mode="precise",
        output_dir=iter_dir,
        raw_filename="kernel_raw.txt",
        post_process=lambda code: (
            re.sub(r"```\w*\n?", "", code).replace("```", "").strip()
        ),
        error_status="deciding",
        error_field="run_output",
    )

    if kernel_code is None:
        state.kernel_code = ""
        state.iter_num = iteration
        return state

    # Sanity check
    if not kernel_code or len(kernel_code) < 20:
        log("LLM returned empty or trivially short kernel code", "ERROR")
        state.run_output = "Implementation failed: LLM returned empty or trivially short kernel code. No compilation possible."
        state.status = "deciding"
        state.iter_num = iteration
        return state
    if "__global__" not in kernel_code and "__device__" not in kernel_code:
        log(
            "LLM output does not contain __global__ or __device__ — may not be valid CUDA",
            "WARN",
        )

    save_output(iter_dir, kernel_code, "kernels.cu")
    state.kernel_code = kernel_code
    state.iter_num = iteration
    state.status = "configuring"
    log(f"Kernel implemented ({len(kernel_code)} chars)", "SUCCESS")
    log(f"Saved to: {iter_dir / 'kernels.cu'}")
    return state
