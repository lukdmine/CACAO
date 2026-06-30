"""Plan prompt — creates strategy-specific optimization plan."""

from nodes._llm_helper import format_strategy
from prompts._tensor_core_reference import TENSOR_CORE_REFERENCE


def build(ctx: dict) -> tuple[str, str]:
    system = (
        """# Create Strategy-Specific Optimization Plan

You are a CUDA optimization expert. Your goal is to create a detailed implementation plan for a **specific optimization strategy**.

## Input

You will receive:
1. The problem definition (problem.yaml)
2. The kernel analysis (bottlenecks, memory patterns, opportunities)
3. The reference kernel implementation
4. **The assigned strategy** to implement

## Task

Create a detailed, actionable optimization plan specifically for the assigned strategy. This plan will guide the kernel implementation in the next step.

## Output Format

Write your plan as a markdown document:

```markdown
# Optimization Plan: [Strategy Name]

## Strategy Overview
[2-3 sentence summary of this specific optimization approach]

## Why This Strategy
[Explain why this strategy addresses the bottlenecks identified in the analysis]

## Target Performance
- Baseline: [estimated GFLOPS or bandwidth of reference]
- Target: [target GFLOPS or bandwidth with this strategy]
- Expected speedup: [Nx]

## Implementation Steps

### Step 1: [Specific Change]
**What**: [Concrete description]
**Why**: [How this helps performance]
**Code pattern**: [Pseudocode or key code structure]

### Step 2: [Specific Change]
...

## Proposed Parameter and Constraints

(This might be extended/edited in the configure step.)

| Parameter | Description | Values to Try |
|-----------|-------------|---------------|
| BLOCK_X   | Block size X | 8, 16, 32 |
| ...       | ...         | ...       |

- [CONSTRAINT 1]: e.g., TILE_SIZE must be divisible by BLOCK_X
- [CONSTRAINT 2]: Shared memory usage must be <= 48KB

## Critical Implementation Notes

[Specific details the implementer MUST know for correctness]

## Potential Pitfalls

- [Common mistake to avoid]
- [Edge case to handle]
```

## Important: No Host-Side Code

The kernel runs standalone — there is NO host-side driver code. The tuner launches the kernel and directly compares the output buffer against the reference. There is no opportunity for host-side post-processing (no cudaMemcpy-normalize-cudaMemcpy pattern). Any normalization, scaling, or finalization that the reference performs must happen **inside the kernel itself**. Plan accordingly: if the algorithm requires a final reduction or normalization step, include it as a kernel-side operation (e.g., have the last block apply it using an atomic counter).

## Important: No Cooperative Launches

KTT invokes kernels via standard `cudaLaunchKernel`, not `cudaLaunchCooperativeKernel`. Grid-wide synchronization — `grid.sync()`, cooperative groups that span the grid, `<cooperative_groups.h>` — does NOT work and will fail at runtime. Confine all cross-thread coordination to a single thread block (`__syncthreads`). If the algorithm naturally requires grid-wide coordination, express it via atomic counters in global memory or a persistent-thread pattern, not cooperative groups.

"""
        + TENSOR_CORE_REFERENCE
    )

    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem Definition:\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("ref_kernel"):
        parts.append(f"## Reference Kernel:\n```cuda\n{ctx['ref_kernel']}\n```")
    if ctx.get("analysis"):
        parts.append(f"## Kernel Analysis:\n{ctx['analysis']}")
    if ctx.get("parent_context"):
        parts.append(ctx["parent_context"])
    strategy_text = format_strategy(ctx.get("strategy"))
    if strategy_text:
        parts.append(strategy_text)
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])
    parts.append("Create a detailed implementation plan for this specific strategy.")

    return system, "\n\n".join(parts)
