"""Configure prompt — generates params.json tuning configuration."""


def build(ctx: dict) -> tuple[str, str]:
    system = """# Configure Tuning Parameters

You are configuring the KTT auto-tuner to find the **fastest possible kernel configuration**. The tuner will test all valid parameter combinations and measure their performance.

Generate the params.json file that defines the parameter search space.

## Input

You will receive:
1. The optimization plan - lists parameters and constraints
2. The implemented kernel - shows which parameters are used

## Task

Generate a params.json file that defines:
1. **Parameters**: All tunable parameters with their possible values
2. **Launch config**: Formula-based grid/block dimensions AND shared memory size
3. **Constraints**: Python expressions that filter invalid combinations

## Output Format

Output ONLY valid JSON. No markdown, no explanation.

```json
{
    "parameters": [
        {"name": "PARAM_NAME", "values": [val1, val2, val3]}
    ],

    "launch_config": {
        "grid_x": "M // TILE_M",
        "grid_y": "N // TILE_N",
        "block_x": "THREADS_X",
        "block_y": "THREADS_Y"
    },

    "constraints": [
        {"params": ["A", "B"], "expr": "A % B == 0"}
    ]
}
```

---

## Launch Configuration (RECOMMENDED)

Use `launch_config` for full control over kernel launch grid and block dimensions.

### Syntax

```json
"launch_config": {
    "grid_x": "M // TILE_M",
    "grid_y": "N // TILE_N",
    "grid_z": "1",
    "block_x": "THREADS_X",
    "block_y": "THREADS_Y",
    "block_z": "1"
}
```

### Formula Variables

Formulas can use:
- **Parameter names** (e.g., `TILE_M`, `THREADS_X`, `PAD_A`)
- **Problem scalars** from problem.yaml (e.g., `M`, `N`, `K`) — these are compile-time `#define` constants, available in both kernel code and launch config formulas
- **Python operators**: `//` (int division), `*`, `+`, `-`, `%`

### Key Fields

| Field | Description | Default |
|-------|-------------|---------|
| `grid_x` | Number of blocks in X | Problem's base grid X |
| `grid_y` | Number of blocks in Y | Problem's base grid Y |
| `grid_z` | Number of blocks in Z | 1 |
| `block_x` | Threads per block in X | 1 |
| `block_y` | Threads per block in Y | 1 |
| `block_z` | Threads per block in Z | 1 |

### Shared Memory (CRITICAL)

**ONLY static shared memory is supported.** Always declare shared memory with a fixed compile-time size:

```cuda
// kernel.cu — both KTT parameters and problem scalars are compile-time constants
__shared__ float smem[BLOCK_SIZE + 2 * R];  // R is a problem scalar define
```

**NEVER use `extern __shared__`** (dynamic shared memory). The tuner has no mechanism to pass a runtime shared memory size, so any kernel using `extern __shared__` will crash with `CUDA_ERROR_ILLEGAL_ADDRESS` on every configuration.

---

## CRITICAL RULES

### Rule 1: Every name used MUST be a defined parameter

**Launch config and constraints can ONLY reference parameter names that exist in the `parameters` list (plus problem scalars like M, N, K).**

❌ WRONG - TM is not defined as a parameter:
```json
{
    "parameters": [
        {"name": "TILE_M", "values": [64, 128]}
    ],
    "launch_config": {
        "block_x": "TM"
    }
}
```

✅ CORRECT - All names are defined parameters or problem scalars:
```json
{
    "parameters": [
        {"name": "TILE_M", "values": [64, 128]},
        {"name": "THREADS_X", "values": [8, 16, 32]}
    ],
    "launch_config": {
        "grid_x": "M // TILE_M",
        "block_x": "THREADS_X"
    }
}
```

### Rule 2: Constraints filter, they don't compute

Constraints eliminate invalid parameter combinations. They don't create new values.

If your kernel needs `THREADS_X = TILE_M / RM`, then:
1. Define all three as separate parameters with explicit value lists
2. Add constraint: `{"params": ["TILE_M", "THREADS_X", "RM"], "expr": "TILE_M == THREADS_X * RM"}`

The tuner will only test combinations where this relationship holds.

### Rule 3: Constraints MUST prevent illegal memory accesses

Only emit parameter combinations that keep **all global and shared memory accesses in-bounds**. A single out-of-bounds write can corrupt the run and poison the entire tuning iteration.

If the kernel does not contain explicit boundary guards, you must encode safety in the parameter constraints. In particular:
- Constrain tile sizes, work-per-thread values, and thread/block dimensions so computed indices never exceed tensor/matrix extents
- Require divisibility when the kernel assumes exact tiling (for example `M % TILE_M == 0`, `N % TILE_N == 0`, or `TILE_M == THREADS_X * TM`)
- Ensure shared-memory indexing stays within the statically allocated array shape

---

## Constraint Reference

Constraints are Python boolean expressions. Configuration is valid only if ALL constraints return True.

### Syntax

```json
{"params": ["A", "B", "C"], "expr": "A == B * C"}
```

- `params`: List **ONLY tuning parameter names** used in the expression. **DO NOT include problem scalar names** (like `M`, `N`, `STUDENTS`, `QUESTIONS`) — they are automatically available.
- `expr`: Python expression returning True/False. Can freely reference both tuning parameter names AND problem scalar names.

### CRITICAL: `params` must NOT contain scalar names

The tuner registers constraints with KTT using the `params` list. KTT only knows about tuning parameters, not problem scalars. If you put a scalar name (e.g. `STUDENTS`) in `params`, KTT will error out with "Kernel parameter does not exist".

❌ WRONG — scalar name in `params`:
```json
{"params": ["STUDENTS", "TILE_M"], "expr": "STUDENTS % TILE_M == 0"}
```

✅ CORRECT — only tuning parameter in `params`; scalar is auto-injected:
```json
{"params": ["TILE_M"], "expr": "STUDENTS % TILE_M == 0"}
```

### Common Patterns

| Need | Expression | params |
|------|------------|--------|
| A divisible by B | `"A % B == 0"` | `["A", "B"]` |
| Thread count limit | `"TX * TY <= 1024"` | `["TX", "TY"]` |
| Minimum threads | `"TX * TY >= 64"` | `["TX", "TY"]` |
| Tile relationship | `"TILE == THREADS * WORK"` | `["TILE", "THREADS", "WORK"]` |
| Scalar divisibility | `"M % TILE_M == 0"` | `["TILE_M"]` (M is a scalar) |
| Shared memory limit | `"4 * K * (M + N) <= 49152"` | `[]` if K,M,N are all scalars |
| Power of 2 | `"(X & (X - 1)) == 0"` | `["X"]` |
| Conditional | `"STAGES == 1 or MEM <= 24576"` | `["STAGES", "MEM"]` |

---

## Complete Example: GEMM Kernel with Static Shared Memory

For a tiled GEMM where each thread computes a TM×TN tile:

```json
{
    "parameters": [
        {"name": "TILE_M", "values": [64, 128]},
        {"name": "TILE_N", "values": [64, 128]},
        {"name": "TILE_K", "values": [8, 16, 32]},
        {"name": "THREADS_X", "values": [8, 16, 32]},
        {"name": "THREADS_Y", "values": [8, 16, 32]},
        {"name": "TM", "values": [4, 8]},
        {"name": "TN", "values": [4, 8]},
        {"name": "PAD_A", "values": [0, 1]},
        {"name": "PAD_B", "values": [0, 1]}
    ],
    "launch_config": {
        "grid_x": "M // TILE_M",
        "grid_y": "N // TILE_N",
        "block_x": "THREADS_X",
        "block_y": "THREADS_Y"
    },
    "constraints": [
        {"params": ["TILE_M", "THREADS_X", "TM"], "expr": "TILE_M == THREADS_X * TM"},
        {"params": ["TILE_N", "THREADS_Y", "TN"], "expr": "TILE_N == THREADS_Y * TN"},
        {"params": ["THREADS_X", "THREADS_Y"], "expr": "THREADS_X * THREADS_Y <= 1024"},
        {"params": ["THREADS_X", "THREADS_Y"], "expr": "THREADS_X * THREADS_Y >= 64"},
        {"params": ["TILE_K", "TILE_M", "TILE_N", "PAD_A", "PAD_B"], "expr": "4 * (TILE_M * (TILE_K + PAD_A) + TILE_K * (TILE_N + PAD_B)) <= 49152"},
        {"params": ["TILE_M"], "expr": "M % TILE_M == 0"},
        {"params": ["TILE_N"], "expr": "N % TILE_N == 0"}
    ]
}
```

**Why this works:**
- All parameters are explicitly defined with value lists
- `launch_config` uses formulas with parameters AND problem scalars (M, N)
- Shared memory in the kernel uses static sizing: `__shared__ float a[TILE_M][TILE_K + PAD_A]` etc.
- Constraints enforce relationships, memory-safety conditions, and memory limits (including the 48KB shared memory cap)
- The tuner tests all combinations, keeping only valid ones

---

## Checklist Before Output

1. ✅ Every parameter used in launch_config formulas is defined (or is a problem scalar)?
2. ✅ Every name in constraint `params` arrays is a **tuning parameter** (NOT a problem scalar)?
3. ✅ Thread counts have both upper (≤1024) and lower (≥64) bounds?
4. ✅ Shared memory is static (`__shared__ float smem[COMPILE_TIME_SIZE]`), NOT `extern __shared__`?
5. ✅ Shared memory size constrained (≤49152 bytes) via a constraint expression?
6. ✅ Constraints guarantee there are no out-of-bounds writes or other illegal memory accesses?
7. ✅ At least some parameter combinations will pass all constraints?
"""

    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem Definition:\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("plan"):
        parts.append(f"## Optimization Plan:\n{ctx['plan']}")
    if ctx.get("strategy_section"):
        parts.append(ctx["strategy_section"])
    if ctx.get("parent_context"):
        parts.append(ctx["parent_context"])
    if ctx.get("iteration_history"):
        parts.append(ctx["iteration_history"])
    if ctx.get("best_so_far"):
        parts.append(ctx["best_so_far"])
    if ctx.get("prev_context"):
        parts.append(ctx["prev_context"])
    if ctx.get("kernel_code"):
        parts.append(f"## Implemented Kernel:\n```cuda\n{ctx['kernel_code']}\n```")
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])
    parts.append("Generate the tuning parameter configuration for this kernel.")

    return system, "\n\n".join(parts)
