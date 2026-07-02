"""Implement node - writes the optimized CUDA kernel."""

import re
from pathlib import Path

from config import get_output_dir, get_problem_dir
from utils.files import save_output, create_iter_dir
from utils.log import log
from state.types import WorkingState
from nodes._llm_helper import execute_llm_node, build_prompt_context
import prompts.implement
import prompts.fix_errors


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
    # Determine prompt and mode
    decision = state.decision or {}
    decision_action = decision.get("action")
    is_retry = decision_action == "retry"
    is_followup = iteration > 1 and bool(state.feedback)

    if is_retry:
        log(f"Retry mode: fixing errors from iteration {iteration - 1}", "WARN")
        output_lines = (state.run_output or "").strip().split("\n")
        print("\n--- Last 20 lines of tuner output ---")
        for line in output_lines[-20:]:
            print(f"  {line}")
        print("-------------------------------------\n")
    else:
        if is_followup:
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
        ],
    )

    # Framework mode: show the I/O boundary so the kernel signature matches it.
    inputs_src = get_problem_dir() / "inputs.hpp"
    if inputs_src.exists():
        ctx["inputs_hpp"] = inputs_src.read_text()

    # Add node-specific follow-up/retry context
    if is_followup:
        action_guidance = (
            "Fix the issues in the kernel based on the feedback and results above."
            if is_retry
            else "Apply the requested kernel changes from the feedback above. Preserve correctness, keep all memory accesses in-bounds, and keep parameter usage consistent with the tuning configuration."
        )
        ctx["current_context"] = (
            f"## Current Feedback:\n{state.feedback or 'No specific feedback'}\n\n"
            f"## Decision Action:\n{decision_action or 'N/A'}\n\n"
            f"## Previous Implementation:\n```cuda\n{state.kernel_code or ''}\n```\n\n"
            f"{action_guidance}"
        )

    # Select prompt
    if is_retry:
        system, user = prompts.fix_errors.build(ctx)
    else:
        system, user = prompts.implement.build(ctx)

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
