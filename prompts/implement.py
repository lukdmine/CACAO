"""Implement prompt — writes optimized CUDA kernel."""

from nodes._llm_helper import format_strategy
from prompts._tensor_core_reference import TENSOR_CORE_REFERENCE


def build(ctx: dict) -> tuple[str, str]:
    system = (
        """# Implement CUDA Kernel

You are a CUDA kernel developer. Your goal is to write the **fastest possible kernel** that maximizes GPU utilization and minimizes execution time.

Implement a highly optimized kernel based on the optimization plan.

## Input

You will receive:
1. The problem definition (problem.yaml) - defines kernel interface
2. The optimization plan (plan.md) - describes what to implement
3. Any previous implementation attempts and their errors (if retrying)

## Task

Write the **fastest possible CUDA kernel** that:
1. Implements the algorithm correctly
    - as a **single kernel**, not a multi-kernel pipeline
2. Follows the optimization plan carefully
    — unless past iterations intentionally changed part of the approach; in that case, follow the most recent validated decisions rather than the original plan verbatim.
3. Uses tunable parameters (they are preprocessor defines from KTT)
4. Maximizes memory bandwidth utilization
5. Maximizes compute throughput
6. Minimizes memory access latency through caching and prefetching

## Output Format

Output ONLY the CUDA kernel code. No markdown, no explanation.

## Example Structure

**IMPORTANT**: Document tunable parameters in a comment block at the top. KTT passes them as compiler defines (`-DBLOCK_X=16`), so they're already compile-time constants you can use directly. Problem scalars from `problem.yaml` (like `M`, `N`, `K`) are also injected as compiler defines (`-DM=2048`) — do NOT include them in the function signature.

```cuda
// =============================================================================
// TUNABLE PARAMETERS (provided by KTT as compiler defines)
// =============================================================================
// BLOCK_X    - Block size in X dimension (threads per block)
// BLOCK_Y    - Block size in Y dimension (threads per block)
// TILE_SIZE  - Tile size for shared memory caching
// =============================================================================
// PROBLEM SCALARS (provided as compiler defines from problem.yaml)
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

1. **extern "C"** - Required for KTT to find the kernel by name
    - Keep the solution as a single externally visible kernel unless the reference interface itself requires otherwise
2. **Parameter names** - Must match exactly what will be in params.json
3. **Function signature** - Only vector (pointer) arguments, in `vectors:` order from problem.yaml. Scalars are compiler defines, NOT function parameters.
4. **Bounds checking** - Handle edge cases when dimensions don't divide evenly
    - Never allow out-of-bounds writes; reject unsafe assumptions unless constraints guarantee them
5. **Shared memory** - Use **STATIC** only (`__shared__ float tile[SIZE]`)
   - DO NOT use `extern __shared__`
   - If size > 48KB, compilation will fail (this is expected behavior for invalid configs)
6. **NVRTC-compatible code only** - Kernels are compiled at runtime using NVRTC (NVIDIA Runtime Compilation), which has limited header support but provides all CUDA device functionality as built-ins:
   - ❌ Do NOT `#include <cuda.h>` or `<cuda_runtime.h>` - these headers are not available
   - ❌ Do NOT `#include <stdint.h>` or `<cstdint>` - types like `uintptr_t`, `uint32_t` are NOT available
   - ✅ You CAN include: `<cuda_fp16.h>`, `<cuda_bf16.h>`, `<mma.h>` (device-side headers)
   - ✅ All CUDA device code works WITHOUT headers - NVRTC provides everything as built-ins:
     - Variables: `blockIdx`, `blockDim`, `threadIdx`, `gridDim`, `warpSize`
     - Synchronization: `__syncthreads()`, `__syncwarp()`, `__threadfence()`
     - Memory qualifiers: `__shared__`, `__global__`, `__device__`, `__constant__`
     - Math functions: `fmaf()`, `sqrtf()`, `__fdividef()`, `min()`, `max()`, etc.
     - Vector types: `float2`, `float4`, `int2`, `int4`, `make_float4()`, etc.
     - Warp intrinsics: `__shfl_sync()`, `__ballot_sync()`, `__any_sync()`, etc.
   - ✅ Use built-in scalar types: `int`, `unsigned int`, `long long`, `unsigned long long`, `float`, `double`
   - ✅ For pointer-to-integer casts (e.g., alignment checks), use `(unsigned long long)ptr` instead of `(uintptr_t)ptr`

## NVRTC Compatibility Rules

Kernels are compiled at runtime using NVRTC. Follow these rules strictly:

### 1. NO Host Headers

**CRITICAL**: NVRTC cannot include host-side headers. These will cause "catastrophic error: cannot open source file":

```cuda
// ❌ FORBIDDEN - Will crash compilation
#include <stdint.h>      // NO!
#include <cstdint>       // NO!
#include <cuda.h>        // NO!
#include <cuda_runtime.h> // NO!
#include <stdio.h>       // NO!
#include <cuda/wmma.h>   // NO! Use <mma.h> instead!
#include <cooperative_groups.h> // NO! KTT doesn't support cooperative launches

// ✅ ALLOWED - Device-side headers only
#include <mma.h>         // OK - Tensor Cores (nvcuda::wmma)
#include <cuda_fp16.h>   // OK - Half precision (__half)
#include <cuda_bf16.h>   // OK - Bfloat16 (__nv_bfloat16)
```

Use built-in types instead of stdint types:
- `uint32_t` → `unsigned int`
- `uint64_t` → `unsigned long long`
- `uintptr_t` → `unsigned long long`

### 2. Lambdas Cannot Have `__device__` Annotation

**Important**: `__device__` functions are fully supported! The restriction is only on lambdas.

NVRTC does not support explicit execution space annotations (`__device__`, `__host__`, `__global__`) on lambdas. The execution space is automatically inferred from the lambda's context.

```cuda
// ❌ WRONG - Lambda with __device__ annotation (will fail to compile)
auto load_tile = [&] __device__ () {
    // ...
};

// ✅ CORRECT - Lambda without annotation (infers __device__ from context)
auto load_tile = [&]() {
    // ...
};

// ✅ CORRECT - Use __device__ function (fully supported by NVRTC)
__device__ __forceinline__ void load_tile(...) {
    // ...
}
```

**Summary**: Use `__device__` functions freely. Only avoid `__device__` annotations on lambdas.
```

"""
        + TENSOR_CORE_REFERENCE
    )

    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem Definition:\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("ref_kernel"):
        parts.append(f"## Reference Kernel:\n```cuda\n{ctx['ref_kernel']}\n```")
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
        "1. NO `extern __shared__` — use ONLY static: `__shared__ float arr[COMPILE_TIME_SIZE]`. "
        "Dynamic shared memory crashes every config.\n"
        "2. NO host headers — `#include <stdint.h>`, `<cuda.h>`, `<cuda_runtime.h>`, `<stdio.h>` "
        "are forbidden. Use built-in types.\n"
        "3. Function signature: vector pointers ONLY. Scalars (M, N, K, etc.) are `#define` "
        "constants, NOT function parameters.\n"
        '4. `extern "C"` required on the kernel.\n'
        "5. Every global and shared memory access must be provably in-bounds."
    )

    return system, "\n\n".join(parts)
