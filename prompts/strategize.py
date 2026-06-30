"""Strategize prompt — identifies optimization strategies from analysis."""

from config import MAX_STRATEGIES


def build(ctx: dict) -> tuple[str, str]:
    system = f"""# Identify Optimization Strategies

You are a CUDA optimization expert. Your task is to identify discrete, independent optimization strategies based on the kernel analysis.

## Input

You will receive:
1. The problem definition (problem.yaml) with kernel interface and GPU specs
2. The kernel analysis (bottlenecks, memory patterns, opportunities)
3. The reference kernel implementation

## Task

Based on the analysis, identify **1-{MAX_STRATEGIES} distinct high-level optimization strategies** that could be explored in parallel and could lead to an optimal implementation. Each strategy represents a fundamentally different approach that warrants its own development branch with its own detailed plan.

Important constraints for every strategy:
- The solution must remain a **single CUDA kernel**
- Do not propose decomposing the work into multiple kernels or extra pipeline stages
- Assume **static shared memory only**; do not rely on `extern __shared__` / dynamic shared memory
- **No cooperative launches**: KTT invokes kernels via standard `cudaLaunchKernel`, so grid-wide synchronization (`grid.sync()`, `<cooperative_groups.h>`) is unavailable. All cross-thread coordination must fit inside a single block, or use atomic counters in global memory for cross-block handoff

## Strategy Guidelines

### What makes a good strategy?
- **Distinct approach**: Uses fundamentally different techniques
- **Self-contained**: Can be implemented independently without depending on other strategies
- **Addresses bottlenecks**: Targets the specific bottlenecks identified in the analysis
- **Feasible**: Actually implementable given the GPU hardware specs and the single-kernel constraint
- **Algorithmic diversity**: If the analysis identifies reducible work complexity (e.g. O(R) → O(1) via scan or recurrence), at least one strategy MUST exploit that reduction. Do not generate multiple strategies that all share the same per-element work complexity.

### What is NOT a separate strategy?
- Minor variations of the same approach (e.g., different tile sizes - these are parameters)
- Sequential optimizations that build on each other (combine into one strategy)
- Micro-optimizations that don't change the fundamental approach

## Examples

### Good strategy separation:
1. **Shared Memory Tiling**: Classic blocked approach with explicit data staging
2. **Tensor Core (WMMA)**: Use hardware matrix multiply units (if applicable)
3. **Warp-level Primitives**: Use shuffle operations to avoid shared memory

### Good: algorithmic + hardware diversity:
1. **Direct computation with shared memory tiling** — O(R) work per output, optimized memory access
2. **Prefix sum / scan in shared memory** — O(1) work per output after O(N) cooperative scan, trades compute for synchronization
3. **Sliding window with thread coarsening** — O(1) amortized per output via recurrence relation, each thread processes multiple consecutive outputs

### Bad strategy separation (too granular):
1. "Tiling with 64x64 tiles"
2. "Tiling with 128x128 tiles"
3. "Tiling with bank conflict avoidance"

These are all variations of the same tiling strategy and should be ONE strategy with parameters.

### Bad: same algorithm, different hardware technique:
1. "Shared memory tiling with direct sum"
2. "Shared memory tiling with vectorized loads"
3. "Shared memory tiling with warp shuffle"

These all do O(R) work per output — only the data movement differs. If the kernel is compute-bound after tiling, none of these will help. At least one strategy should reduce the per-element work.

## Output

You must output a JSON object with:
- `strategies`: List of 1-{MAX_STRATEGIES} strategies, each with:
  - `name`: Short identifier (lowercase, underscores, e.g., "shared_mem_tiling")
  - `description`: What this optimization approach does (1-2 sentences)
  - `hypothesis`: Why this might improve performance based on the analysis
  - `key_parameters`: List of main tuning parameters this approach would introduce
- `reasoning`: Why you chose these specific strategies based on the analysis

## Important

- If there's clearly only ONE good approach, output exactly ONE strategy (that's fine!)
- Don't artificially split - quality over quantity
- Each strategy will get its own detailed implementation plan in the next step
"""

    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem Definition:\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("ref_kernel"):
        parts.append(f"## Reference Kernel:\n```cuda\n{ctx['ref_kernel']}\n```")
    if ctx.get("analysis"):
        parts.append(f"## Kernel Analysis:\n{ctx['analysis']}")
    parts.append(
        "Based on this analysis, identify the optimization strategies to explore."
    )

    return system, "\n\n".join(parts)
