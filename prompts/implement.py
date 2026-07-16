"""Implement prompt — writes optimized CUDA kernel."""

from nodes._llm_helper import format_strategy
from prompts._system_overview import SYSTEM_OVERVIEW, NVRTC_RULES
from prompts._tensor_core_reference import TENSOR_CORE_REFERENCE


def build(ctx: dict) -> tuple[str, str]:
    system = (
        """# Implement CUDA Kernel

You are a CUDA kernel developer. Your goal is to write the **fastest possible kernel** that maximizes GPU utilization and minimizes execution time.

Implement a highly optimized kernel based on the optimization plan.

"""
        + SYSTEM_OVERVIEW
        + """
## Input

You will receive:
1. The problem definition (problem.yaml) and I/O boundary (inputs.hpp)
2. The optimization plan (plan.md) - describes what to implement
3. Any previous implementation attempts and their errors (if retrying)

## Task

Write the **fastest possible CUDA kernel** that:
1. Implements the algorithm correctly
    - as **one or more** `extern "C" __global__` kernels — a single kernel, or a
      multi-kernel pipeline when the strategy calls for it (you wire the launch
      order and buffers in the next configure step)
2. Follows the optimization plan carefully
    — unless past iterations intentionally changed part of the approach; in that case, follow the most recent validated decisions rather than the original plan verbatim.
3. Uses tunable parameters (compile-time macros from KTT)
4. Maximizes memory bandwidth utilization
5. Maximizes compute throughput
6. Minimizes memory access latency through caching and prefetching

## Output Format

Output ONLY the CUDA kernel code. No markdown, no explanation.

## Example Structure

**IMPORTANT**: Document tunable parameters in a comment block at the top. Each kernel's
signature is your design: its arguments are bound positionally (next step) to inputs.hpp
buffers, scratch buffers, and any runtime scalar arguments. `-D`-style macro scalars stay
OUT of the signature.

```cuda
// =============================================================================
// TUNABLE PARAMETERS (provided by KTT as compiler defines)
// =============================================================================
// BLOCK_X    - Block size in X dimension (threads per block)
// BLOCK_Y    - Block size in Y dimension (threads per block)
// TILE_SIZE  - Tile size for shared memory caching
// =============================================================================
// PROBLEM SCALARS (the -D macros listed in inputs.hpp's `defines` string)
// =============================================================================
// M, N, K   - Matrix dimensions (compile-time constants)
// =============================================================================

extern "C" __global__ void kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    // Both parameters and scalars are compile-time constants!
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_Y + ty;
    const int col = blockIdx.x * BLOCK_X + tx;

    // ... implementation ...
}
```

## Critical Requirements

1. **Parameter names** - Tuning-parameter macros must match the names you declare in the configure step (PARAMS region)
2. **Bounds checking** - Handle edge cases when dimensions don't divide evenly
   - Never allow out-of-bounds writes; reject unsafe assumptions unless constraints guarantee them
3. **Shared memory and headers** - follow the overview above and the NVRTC rules below strictly

"""
        + NVRTC_RULES
        + TENSOR_CORE_REFERENCE
    )

    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem Definition:\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("inputs_hpp"):
        parts.append(
            "## Inputs (inputs.hpp) — the input/output buffers your kernels consume/produce "
            "(you'll bind kernel args to these by name next):\n```cpp\n"
            f"{ctx['inputs_hpp']}\n```"
        )
    if ctx.get("ref_kernel"):
        parts.append(
            f"## Reference Kernel:\n```{ctx.get('ref_language', 'cuda')}\n{ctx['ref_kernel']}\n```"
        )
    if ctx.get("plan"):
        parts.append(f"## Optimization Plan:\n{ctx['plan']}")
    strategy_text = format_strategy(ctx.get("strategy"))
    if strategy_text:
        parts.append(strategy_text)
    if ctx.get("iteration_summaries"):
        parts.append(ctx["iteration_summaries"])
    if ctx.get("parent_context"):
        parts.append(ctx["parent_context"])
    if ctx.get("best_so_far"):
        parts.append(ctx["best_so_far"])
    if ctx.get("iteration_history"):
        parts.append(ctx["iteration_history"])
    if ctx.get("current_context"):
        parts.append(
            ctx["current_context"]
            + "\n\n→ The feedback above is the **authoritative instruction** for this "
            "revision. Apply the requested change directly rather than re-planning "
            "from scratch. If it conflicts with the Optimization Plan, strategy, or "
            "past iterations, follow the feedback — it reflects the final decision "
            "after reviewing the proposal."
        )
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])
    parts.append(
        "Write the optimized CUDA kernel. Output ONLY the kernel code, no markdown.\n\n"
        "CRITICAL REMINDERS — verify before writing code:\n"
        "1. Static `__shared__ arr[MACRO_SIZE]` unless configure binds AddArgumentLocal.\n"
        "2. No glibc-reaching headers (`<stdint.h>`, `<cstdint>`, `<cuda.h>`, `<stdio.h>`).\n"
        '3. `extern "C"` on every kernel; signature is your design, macros stay out of it.\n'
        "4. Every global and shared memory access must be provably in-bounds."
    )

    return system, "\n\n".join(parts)
