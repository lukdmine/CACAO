# `problem.yaml` Guide

This guide documents the `problem.yaml` format used by the tuner in [tuner.py](../tuner.py).

It covers:
- all supported top-level fields
- required vs optional fields
- allowed values and defaults
- expression syntax for sizes and grid dimensions
- CUDA-reference and CPU-reference variants

## Quick Example

```yaml
name: Averages Calculation 2019
description: Compute per-student and per-question averages.

gpu:
  index: 0

kernel:
  file: kernel.cu
  function: kernel

reference:
  type: cpu_c
  file: ref_cpu.c
  function: averages_reference

scalars:
  - name: STUDENTS
    dtype: int
    value: 4096
  - name: QUESTIONS
    dtype: int
    value: 1024

grid:
  x: STUDENTS
  y: QUESTIONS

vectors:
  - name: results
    dtype: int
    size: STUDENTS * QUESTIONS
    access: read
    init: random
    validate: false
  - name: avg_stud
    dtype: float
    size: STUDENTS
    access: write
    init: zeros
    validate: true
  - name: avg_que
    dtype: float
    size: QUESTIONS
    access: write
    init: zeros
    validate: true

validation:
  tolerance: 1.0e-6
```

---

## Top-Level Schema

| Field | Required | Type | Notes |
|---|---|---|---|
| `name` | yes | string | Human-readable problem name |
| `description` | recommended | string | Shown in UI and prompts |
| `gpu` | recommended | mapping | `gpu.index` affects runtime device selection; other fields are advisory metadata |
| `kernel` | yes | mapping | Kernel source file and entry function |
| `reference` | yes | mapping | Validation reference source and function |
| `scalars` | yes | list | Scalar constants injected as compiler defines (`-D` flags) |
| `grid` | yes | mapping | Base problem dimensions, not the tuned launch config |
| `vectors` | yes | list | Vector buffers used by the kernel/reference |
| `validation` | yes | mapping | Validation tolerance |

---

## `name`

```yaml
name: GEMM
```

- Type: string
- Required: yes
- Used by: tuner logs, UI, prompts

---

## `description`

```yaml
description: General Matrix-Matrix Multiplication: C = A * B
```

- Type: string
- Required: no, but strongly recommended
- Used by: UI and LLM context

---

## `gpu`

Example:

```yaml
gpu:
  index: 0
  model: NVIDIA GeForce RTX 3090
  compute_capability: "8.6"
  sm_count: 82
  max_threads_per_block: 1024
  shared_memory_per_block: 49152
  registers_per_block: 65536
  memory_bandwidth_gb: 936
```

### Supported fields

| Field | Required | Type | Runtime meaning |
|---|---|---|---|
| `index` | no | int | CUDA device index used by the tuner; defaults to `0` in most flows |
| `model` | no | string | Informational for prompts/UI |
| `compute_capability` | no | string | Informational for prompts/UI |
| `sm_count` | no | int | Informational for prompts/UI |
| `max_threads_per_block` | no | int | Informational for prompts/UI |
| `shared_memory_per_block` | no | int | Informational for prompts/UI |
| `registers_per_block` | no | int | Informational for prompts/UI |
| `memory_bandwidth_gb` | no | int/float | Informational for prompts/UI |

### Notes

- Only `gpu.index` is consumed directly by the runtime.
- The other fields help analysis/planning prompts and can also be injected automatically by the backend.
- Extra fields are generally tolerated and simply passed through as YAML metadata.

---

## `kernel`

```yaml
kernel:
  file: kernel.cu
  function: kernel
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `file` | yes | string | Path to the CUDA kernel source |
| `function` | yes | string | Kernel entry symbol name |

### Notes

- Paths may be relative to the problem directory.
- During optimization iterations, `kernel.file` is rewritten to the per-iteration local `kernel.cu` copy.
- The kernel source should use `extern "C" __global__ void ...` so KTT can locate the symbol.

---

## `reference`

The reference is mandatory for validation.

### CUDA reference

```yaml
reference:
  type: cuda
  file: ref_kernel.cu
  function: gemm_reference
  block_x: 8
  block_y: 8
```

### CPU C / C++ reference

```yaml
reference:
  type: cpu_c
  file: ref_cpu.c
  function: averages_reference
```

### Supported fields

| Field | Required | Type | Allowed values / behavior |
|---|---|---|---|
| `type` | no | string | `cuda` or `cpu_c`; defaults to `cuda` |
| `file` | yes | string | Path to reference source |
| `function` | yes | string | Reference function name |
| `block_x` | CUDA only | int | Optional, defaults to `8` |
| `block_y` | CUDA only | int | Optional, defaults to `8` |
| `block_z` | CUDA only | int | Optional, defaults to `1` |

### Reference file behavior

- `type: cuda`
  - `file` should point to a CUDA source file, usually `ref_kernel.cu`.
  - The reference is launched by KTT as another CUDA kernel.
- `type: cpu_c`
  - `file` should point to a C or C++ source file such as `.c`, `.cc`, `.cpp`, or `.cxx`.
  - The source is compiled into a shared object at runtime.
  - The function must be exported with `extern "C"` if using C++.

### CPU reference ABI

For `cpu_c`, the reference function receives only vector pointer arguments in `vectors:` order. Scalars are injected as `-D` compiler flags during compilation, so they are available as compile-time constants in the function body.

Standard example:

```c
void ref(const float* A, float* C)
```

---

## `scalars`

```yaml
scalars:
  - name: M
    dtype: int
    value: 2048
  - name: ALPHA
    dtype: float
    value: 1.0
```

**Scalar names must be UPPERCASE.** Lowercase names like `n` conflict with NVRTC built-in header parameter names, causing compilation failures.

Each item supports:

| Field | Required | Type | Default | Allowed values |
|---|---|---|---|---|
| `name` | yes | string | – | UPPERCASE identifier (e.g. `N`, `ALPHA`, `BLOCK_SIZE`) |
| `dtype` | no | string | `int` | See supported dtypes below |
| `value` | yes | number | – | Scalar literal |

### Supported scalar dtypes

Canonical types:
- `char`
- `short`
- `int`
- `long`
- `float`
- `double`

Accepted aliases:
- `int8` → `char`
- `int16` → `short`
- `int32` → `int`
- `int64` → `long`
- `float32` → `float`
- `float64` → `double`

### Notes

- Scalar values are also available inside `size:` and `grid:` expressions.
- If you want a scalar to be zero, just set `value: 0`.
- Scalars are always inputs in the current schema.

### How scalars reach the kernel

Scalars are injected as `-D` compiler flags (e.g., `-DM=2048 -DALPHA=1.5f`). They are **not** passed as function arguments. This means:
- Scalars are compile-time constants in the kernel
- They can be used for static shared memory sizing (e.g., `__shared__ float tile[M]`)
- The kernel function signature contains only vector (pointer) arguments

---

## `grid`

```yaml
grid:
  x: M
  y: N
  z: 1
```

| Field | Required | Type | Default | Meaning |
|---|---|---|---|---|
| `x` | yes | int or expression string | – | Base X problem extent |
| `y` | no | int or expression string | `1` | Base Y problem extent |
| `z` | no | int or expression string | `1` | Base Z problem extent |

### Expression rules

`grid.x`, `grid.y`, and `grid.z` are evaluated with the problem scalars as variables.

Examples:

```yaml
grid:
  x: M
  y: N
```

```yaml
grid:
  x: N // 32
  y: 1
```

Allowed operators depend on normal Python integer arithmetic, for example:
- `+`
- `-`
- `*`
- `//`
- `%`
- parentheses

The expression must evaluate to a non-negative integer.

### Important

- These are not the final launch dimensions used during tuning.
- Final runtime launch dimensions come from `params.json -> launch_config`.
- `grid` provides base problem dimensions for the tuner, references, and prompt context.

---

## `vectors`

```yaml
vectors:
  - name: mat_a
    dtype: float
    size: M * K
    access: read
    init: random
    validate: false
  - name: mat_c
    dtype: float
    size: M * N
    access: write
    init: zeros
    validate: true
```

Each vector item supports:

| Field | Required | Type | Default | Allowed values / behavior |
|---|---|---|---|---|
| `name` | yes | string | – | Buffer name |
| `dtype` | yes | string | – | Same dtype set as scalars |
| `size` | yes | int or expression string | – | Must evaluate to a non-negative integer |
| `access` | yes | string | – | Recommended: `read` or `write` |
| `init` | no | string | `zeros` | `random` or any non-`random` value for zero-initialization |
| `init_min` | no | number | see below | Minimum value for random initialization |
| `init_max` | no | number | see below | Maximum value for random initialization (inclusive) |
| `validate` | no | bool | `false` | Whether this vector is checked against the reference |

### `access`

Recommended values:
- `read` → KTT read-only buffer
- `write` → KTT write-only buffer

Current implementation note:
- only the exact string `read` is treated as read-only
- any other value is treated as write-only

So use only:
- `read`
- `write`

### `init`

Current implementation behavior:
- `random` → randomized initial contents
- anything else → zero-filled buffer

Recommended values:
- `random`
- `zeros`

Why random init is useful:
- it helps catch kernels that forget to write part of an output buffer

Why zero init is useful:
- it is convenient for reductions, accumulators, and debugging

### `init_min` / `init_max`

Optional bounds for random initialization. Only used when `init: random`.

- **Floats**: default range is `[-2.0, 2.0]`
- **Integers**: default range is `[-2, 2]` (inclusive)

Use these when your algorithm requires values in a specific range, e.g. array indices:

```yaml
- name: indices
  dtype: int
  size: N
  access: read
  init: random
  init_min: 0
  init_max: 99
```

### `size`

`size` can be a literal integer:

```yaml
size: 1024
```

or an expression using scalar names:

```yaml
size: M * N
size: N * 18
size: 20 * 18
```

The expression must evaluate to a non-negative integer.

### `validate`

- `true` means the vector is checked against the reference implementation.
- `false` means it is not checked.
- Multiple vectors may be validated.
- If no vector has `validate: true`, validation is effectively disabled.

---

## `tuning`

```yaml
tuning:
  duration_s: 300
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `duration_s` | no | int | Wall-clock budget (seconds) for one KTT tuning pass. Falls back to the system default (100 s) if omitted, and is overridden by the `--timeout` CLI flag. |

---

## `validation`

```yaml
validation:
  tolerance: 0.001
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `tolerance` | yes | float | Element-wise comparison tolerance used by KTT |

### Notes

- This field is required by the current tuner.
- Typical values seen in the repo:
  - `1e-6` for strict integer-derived float outputs
  - `1e-3` for moderate floating-point tolerance
  - `0.05` for looser comparisons

---

## Minimal Valid Configurations

### Minimal CUDA-reference problem

```yaml
name: My Problem
description: Minimal CUDA reference example.
gpu:
  index: 0
kernel:
  file: kernel.cu
  function: kernel
reference:
  type: cuda
  file: ref_kernel.cu
  function: reference
scalars:
  - name: N
    value: 1024
grid:
  x: N
vectors:
  - name: x
    dtype: float
    size: N
    access: read
    init: random
  - name: y
    dtype: float
    size: N
    access: write
    init: zeros
    validate: true
validation:
  tolerance: 1.0e-6
```

### Minimal CPU-reference problem

```yaml
name: My Problem
description: Minimal CPU reference example.
gpu:
  index: 0
kernel:
  file: kernel.cu
  function: kernel
reference:
  type: cpu_c
  file: ref_cpu.c
  function: reference
scalars:
  - name: N
    value: 1024
grid:
  x: N
vectors:
  - name: x
    dtype: float
    size: N
    access: read
    init: random
  - name: y
    dtype: float
    size: N
    access: write
    init: zeros
    validate: true
validation:
  tolerance: 1.0e-6
```

---

## Practical Recommendations

1. Use explicit dtypes everywhere.
2. Use only `read` and `write` for `vectors[].access`.
3. Use only `random` and `zeros` for `vectors[].init`.
4. Keep `size`, `grid.x`, and `grid.y` expressions simple and integer-valued.
5. Add GPU metadata when you know it; the LLM prompts use it.
6. Validate at least one output vector.
7. For CPU C++ references, wrap the function with `extern "C"`.

---

## Current Implementation Caveats

These reflect the current code behavior in [tuner.py](../tuner.py):

1. `vectors[].init` is permissive.
   - Only `random` is special.
   - Any other string currently becomes zero-initialization.

2. `vectors[].access` is permissive.
   - Only the exact value `read` is treated as read-only.
   - Any other value currently becomes write-only.

3. `reference.type` should be only `cuda` or `cpu_c`.
   - Any other value disables validation with a warning.

4. `validation.tolerance` is required.

If stricter schema validation is added later, invalid or ambiguous values may stop being accepted.