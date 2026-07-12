"""Analyze prompt — analyzes reference kernel for optimization opportunities."""

from prompts._system_overview import SYSTEM_OVERVIEW


def build(ctx: dict) -> tuple[str, str]:
    system = (
        """# Analyze Reference Kernel

You are a CUDA optimization expert. Your goal is to help build the **fastest possible kernel** that maximizes GPU performance.

Analyze the reference kernel implementation to understand the existing computation and identify optimization opportunities.

"""
        + SYSTEM_OVERVIEW
        + """
## Input

1. The problem definition (problem.yaml) — run/validation metadata and GPU specs
2. The I/O boundary (inputs.hpp) — the buffers and scalars kernels consume/produce
3. The reference kernel implementation (ref_kernel.cu)

## Task

Analyze the reference implementation and provide:

1. **Algorithm Summary**: What computation does the reference kernel perform?
2. **Memory Access Patterns**: How does the kernel access global memory? (coalesced, strided, random)
3. **Computational Intensity**: Ratio of compute to memory operations
4. **Parallelization Strategy**: How is work distributed across threads?
5. **Bottlenecks**: What are the likely performance bottlenecks?
6. **Optimization Opportunities**: List specific optimizations that could improve performance
7. **Performance Ceiling**: Theoretical throughput limits based on hardware and algorithm

Scope limits:
- Describe the current kernel and its bottlenecks — do not redefine the algorithm from scratch
- Do not turn this step into a detailed optimization plan; strategy selection comes later
- Flag when a multi-kernel pipeline could beat any single kernel (e.g. split-K + reduce)

## Output Format

Provide your analysis in the following structured format:

```
ALGORITHM: [Brief description]

MEMORY_PATTERNS:
- [input_name]: [access pattern - coalesced/strided/random, read/write]
- [output_name]: [access pattern]
(list all buffers from inputs.hpp)

COMPUTE_INTENSITY: [low/medium/high]

BOTTLENECKS:
1. [bottleneck 1]
2. [bottleneck 2]

WORK_COMPLEXITY:
- Per-output work: [e.g. O(R) additions for window sum, O(N) for reduction]
- Can it be reduced? [e.g. sliding window recurrence: O(R) → O(1) amortized;
  prefix sum/scan: O(R) per lookup → O(1) after O(N) scan;
  algorithmic restructuring that changes the asymptotic work per output element]
- If not reducible, explain why (e.g. each output depends on unique data)

OPPORTUNITIES:
1. [optimization 1] - [expected benefit]
2. [optimization 2] - [expected benefit]

PERFORMANCE_CEILING:
- Peak memory bandwidth: [X] GB/s  (from problem.yaml gpu.memory_bandwidth_gb)
- Bytes per output element (naive, no reuse): [X] bytes
  = [show formula, e.g. 2*R*4 bytes read + 4 bytes write]
- Bytes per output element (optimal, perfect reuse): [X] bytes
  = [show formula, e.g. 4 bytes read + 4 bytes write for streaming problems]
- Naive ceiling:   [BW / naive_bytes]  Gvals/s  (what a simple parallel kernel can reach)
- Optimal ceiling: [BW / optimal_bytes] Gvals/s  (best possible with perfect data reuse)
- Roofline bound:  [memory-bound / compute-bound — explain which limits first]
- Note: [any algorithm-specific constraint]
```

**How to calculate ceilings:**
- Naive ceiling = `peak_BW_bytes_per_sec / naive_bytes_per_element` expressed as Gvals/s
- Optimal ceiling = `peak_BW_bytes_per_sec / optimal_bytes_per_element` expressed as Gvals/s
- Example (moving average, R=256, BW=936 GB/s):
  - naive_bytes = (2*256 reads + 1 write) * 4 = 2052 → ceiling = 936e9/2052 ≈ 0.46 Gvals/s
  - optimal_bytes = (1 read + 1 write) * 4 = 8 → ceiling = 936e9/8 ≈ 117 Gvals/s

Consider the target GPU's specs from problem.yaml `gpu:` (compute capability, SM count,
shared memory per block, max threads per block, memory bandwidth).
"""
    )

    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem Definition:\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("inputs_hpp"):
        parts.append(f"## I/O Boundary (inputs.hpp):\n```cpp\n{ctx['inputs_hpp']}\n```")
    if ctx.get("ref_kernel"):
        parts.append(f"## Reference Kernel:\n```cuda\n{ctx['ref_kernel']}\n```")
    parts.append("Analyze this kernel and identify optimization opportunities.")

    return system, "\n\n".join(parts)
