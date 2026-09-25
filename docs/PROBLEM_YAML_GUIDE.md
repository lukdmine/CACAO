# `problem.yaml` + `inputs.yaml` Guide

A problem is defined by **two** files:

- **`problem.yaml`** — metadata: name, GPU, base grid, the validation reference, tolerance,
  tuning budget, and optional kernel rules.
- **`inputs.yaml`** — the I/O boundary: scalars and buffers, how each scalar reaches the
  kernel, how each buffer is initialized, and which buffers are validated. This file is the
  source of truth; the C++ header `inputs.hpp` is **generated** from it at the start of
  every run (by `utils/inputs.py`) and must never be edited by hand — it is gitignored for
  that reason.

This guide covers both files: all supported fields, required vs optional, defaults, and
the expression syntax for sizes and grid dimensions.

## Quick Example

`problem.yaml`:

```yaml
name: GEMM
description: General matrix-matrix multiplication C = A * B

gpu:
  index: 0

global_size_type: opencl
grid:
  x: kSizeM
  y: kSizeN
  z: '1'

reference:
  type: cuda
  function: gemm_reference
  file: ref_kernel.cu
  block:
    x: 8
    y: 8
    z: 1

validation:
  tolerance: 1.0

tuning:
  duration_s: 100
```

`inputs.yaml`:

```yaml
args:
- kind: scalar
  name: kSizeM
  dtype: int
  value: 2048
  placements: [host, runtime]
- kind: scalar
  name: kSizeN
  dtype: int
  value: 2048
  placements: [host, runtime]
- kind: scalar
  name: kSizeK
  dtype: int
  value: 2048
  placements: [host, runtime]
- kind: buffer
  name: mat_a
  dtype: float
  size: kSizeM * kSizeK
  access: read
  init: random
  min: -2
  max: 2
  validate: false
- kind: buffer
  name: mat_b
  dtype: float
  size: kSizeN * kSizeK
  access: read
  init: random
  min: -2
  max: 2
  validate: false
- kind: buffer
  name: mat_c
  dtype: float
  size: kSizeM * kSizeN
  access: write
  init: zeros
  validate: true
```

---

## `problem.yaml` — Top-Level Schema

| Field | Required | Type | Notes |
|---|---|---|---|
| `name` | yes | string | Human-readable problem name |
| `description` | recommended | string | Shown in UI and LLM prompts |
| `gpu` | recommended | mapping | `gpu.index` selects the CUDA device; other fields are advisory |
| `global_size_type` | no | string | `cuda` (default) or `opencl` — how `grid` is interpreted |
| `grid` | yes | mapping | Base problem dimensions, used as the reference kernel's ndRange |
| `reference` | yes | mapping | Validation reference source and function |
| `validation` | no | mapping | Validation tolerance (defaults to `1e-4`) |
| `tuning` | no | mapping | Tuner wall-clock budget |
| `rules` | no | mapping or list | Constraints on what a kernel may do (see [`rules`](#rules)) |

The I/O boundary is **not** part of `problem.yaml` — there are no `scalars:`/`vectors:`
blocks here. Those live in `inputs.yaml` (below).

---

## `name`

```yaml
name: GEMM
```

- Type: string
- Required: yes
- Used by: logs, UI, prompts

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
| `index` | no | int | CUDA device index the tuner runs on; defaults to `0` |
| `model` | no | string | Informational for prompts/UI |
| `compute_capability` | no | string | Informational for prompts/UI |
| `sm_count` | no | int | Informational for prompts/UI |
| `max_threads_per_block` | no | int | Informational for prompts/UI |
| `shared_memory_per_block` | no | int | Informational for prompts/UI |
| `registers_per_block` | no | int | Informational for prompts/UI |
| `memory_bandwidth_gb` | no | int/float | Informational for prompts/UI |

### Notes

- Only `gpu.index` is consumed directly by the runtime.
- The other fields are auto-detected and injected by the engine at run start
  (`utils/gpu_info.py`), then cached in `output/context.json`; you can also write them
  by hand.

---

## `global_size_type`

```yaml
global_size_type: opencl   # or cuda
```

- Type: string
- Required: no; default `cuda`
- Meaning: how KTT interprets the ndRange (`grid`):
  - `opencl` — grid is the **total number of work-items**; KTT divides by the per-config
    local size. This is what most problems in this repo use.
  - `cuda` — grid is the **number of blocks**.

Getting this wrong launches a GEMM with 2048×2048 *blocks* instead of work-items, so set
it deliberately. It applies to the reference launch and to how the LLM's launcher region
should compute its own global size.

---

## `grid`

```yaml
grid:
  x: kSizeM
  y: kSizeN
  z: '1'
```

| Field | Required | Type | Default | Meaning |
|---|---|---|---|---|
| `x` | yes | int or expression string | – | Base X problem extent |
| `y` | no | int or expression string | – | Base Y problem extent |
| `z` | no | int or expression string | – | Base Z problem extent |

### Expression rules

Grid values are spliced **verbatim** into the generated C++ driver
(`utils/framework.py` builds a `ktt::DimensionVector` from them), where they are
evaluated by the C++ compiler against the `constexpr` host scalars from `inputs.hpp`.
That means:

- use **C++** expression syntax over host-placement scalar names: `kSizeM`,
  `kSizeM * 2`, `kSizeN / 32` — not Python (`//` is a comment in C++!)
- only scalars with the `host` placement exist in the driver's scope

### Important

- These are not the final launch dimensions used during tuning.
- The optimized kernel's launch configuration comes from the LLM-written launcher
  region and the tuning parameters; `grid` provides the base problem dimensions for the
  CUDA reference kernel's ndRange, and prompt context.

---

## `reference`

The reference is mandatory for validation.

### CUDA reference

```yaml
reference:
  type: cuda
  file: ref_kernel.cu
  function: gemm_reference
  block:
    x: 8
    y: 8
    z: 1
```

### CPU C / C++ reference

```yaml
reference:
  type: cpu_c
  file: ref_cpu.c
  function: averages_reference
```

### Python reference

```yaml
reference:
  type: python
  file: ref.py
  function: gemm_reference
```

### Supported fields

| Field | Required | Type | Allowed values / behavior |
|---|---|---|---|
| `type` | no | string | `cuda` (default), `cpu_c`, or `python` |
| `file` | yes | string | Path to reference source (`ref_kernel.cu`, `ref_cpu.c`, `ref.py`) |
| `function` | yes | string | Reference function name |
| `block` | CUDA only | mapping | Reference launch block `{x, y, z}`; defaults to `{x: 1}` when absent |

### Reference file behavior

- `type: cuda`
  - `file` points to a CUDA source, usually `ref_kernel.cu`.
  - The reference is launched by KTT as another CUDA kernel over the full *boundary*:
    every buffer **plus every `runtime`-placement scalar**, in `args` declaration order.
    `define`-placement scalars are also visible inside it as `-D` macros.
- `type: cpu_c`
  - `file` points to a `.c`, `.cc`, `.cpp`, or `.cxx` source.
  - The source is compiled into the driver as a separate translation unit.
  - The function takes one pointer per buffer, in `args` declaration order (buffers
    only). **All** scalars — regardless of kernel-side placements — reach the host
    compile as `-D` macros.
  - For C++ sources, export the function with `extern "C"`.
- `type: python`
  - `file` should point to `ref.py`, defining `function(scalars, buffers)`.
  - `scalars` is a dict of every scalar name → value; `buffers` is a dict of buffer
    name → flat `np.ndarray` (read/readwrite buffers hold the real input data, write
    buffers are zeros).
  - The return value (a `np.ndarray`, or anything with `.cpu().numpy()` such as a torch
    tensor) must have exactly as many elements as the validated buffer.
  - Third-party imports in `ref.py` (e.g. `torch`, see `problems/mmul_pytorch`) are
    optional dependencies of that problem only — they are never needed by the engine
    itself. Install with plain `python -m pip install torch` (PyPI's Linux wheels
    bundle their own CUDA runtime). The engine imports `ref.py` in a `python3`
    subprocess at startup and aborts the run with the import error if it fails.
  - An optional `prepare_input(scalars, buffers)` splits setup (e.g. host→device
    transfer) from the reference function, which then takes `prepared` instead. It is
    only needed for precise GPU-only reference timing; validation works without it,
    but the reference time then falls back to KTT's coarse wall clock of the whole
    Python process (startup + imports included), making speedup numbers meaningless —
    GPU references should always define it.

### CPU reference ABI

For `cpu_c`, the reference function receives only buffer pointer arguments, in `args:`
order. Scalars are injected as `-D` compiler flags during the host compilation, so they
are compile-time constants in the function body.

Standard example:

```c
void ref(const float* A, float* C)
```

See [REFERENCE_IMPLEMENTATION_GUIDE.md](REFERENCE_IMPLEMENTATION_GUIDE.md) for the full
per-type ABI contract.

---

## `tuning`

```yaml
tuning:
  duration_s: 300
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `duration_s` | no | int | Wall-clock budget (seconds) for one KTT tuning pass |

Resolution order for the effective budget: the `--timeout` CLI flag (or `timeout` in a
UI run config) wins; then `tuning.duration_s`; then the system default of **100 s**.

---

## `validation`

```yaml
validation:
  tolerance: 0.001
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `tolerance` | no | float | Element-wise comparison tolerance used by KTT; defaults to `1e-4` |

### Notes

- Typical values seen in the repo:
  - `1e-6` for strict integer-derived float outputs
  - `1e-3` for moderate floating-point tolerance
  - `0.05` for looser comparisons

---

## `rules`

Constraints on *how* the kernel may be written, as opposed to what it must compute.

The reference implementation defines correctness, not intent. A kernel that drops to
fp16 accumulation, or swaps a real reduction for tensor cores, is faster and still
validates whenever `validation.tolerance` is loose enough — and the run reports a
speedup that does not mean what it looks like. Rules are where you say so.

```yaml
rules:
  text:
    - "Accumulate in fp32. Reduced-precision accumulation is not a valid optimization."
  forbid:
    - pattern: "wmma::|mma\\.sync"
      reason: "tensor cores change the numerics this problem measures"
    - pattern: "\\b__half\\b|nv_bfloat16"
      reason: "fp16/bf16 storage is not permitted"
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `text` | no | list of strings | Stated in every prompt that writes or judges a kernel. Guidance the model applies to cases you did not enumerate. |
| `forbid` | no | list | Regular expressions checked against `kernels.cu` at compile time. |
| `forbid[].pattern` | yes | string (regex) | Matched against the kernel source. |
| `forbid[].reason` | no | string | Shown to the model when it matches. Worth writing — it turns a rejection into a correction. |

### Shorthand

A bare list is `text`:

```yaml
rules:
  - "Do not use tensor cores."
```

### How they are enforced

`text` rules are prompt-level. `forbid` patterns have teeth: they are checked before
the compiler runs, and a match fails the iteration's compilation check regardless of
correctness or speed. Anything you actually care about belongs in `forbid` — a rule
that is only asked for is a rule that loses to a speedup.

Both kinds appear in the prompt, including the patterns. A model that knows a check
exists writes conforming code the first time instead of discovering the constraint
through a failed compile.

### Notes

- Patterns are Python regular expressions. Escape backslashes for YAML (`"\\b"`).
- An invalid pattern is logged and skipped rather than taking the run down.
- Rules are re-read from `problem.yaml` every iteration, so editing them mid-run works.

---

# `inputs.yaml` — the I/O boundary

`inputs.yaml` declares the complete interface between the host driver and the kernel:
scalars, buffers, initializers, validation. The schema is enforced by
`models/inputs.py` (`InputsSpec`); `utils/inputs.py` renders it into `inputs.hpp`.

## Top-level fields

| Field | Required | Type | Notes |
|---|---|---|---|
| `args` | yes | list | Ordered scalar/buffer declarations — the order is the reference signature |
| `headers` | no | list of strings | Extra `#include` lines added to `inputs.hpp` (e.g. `"<cmath>"`) |
| `shared_setup` | no | string | Verbatim C++ emitted between the scalars and the generators — for coupled/derived data |

- `args` is **order-sensitive**: buffer and `runtime`-scalar order defines the
  reference kernel signature (the *boundary*). Host-only scalars are declared but never
  passed, so they can sit anywhere.
- At least one buffer must set `validate: true` — otherwise the spec is rejected.

## Scalars

```yaml
- kind: scalar
  name: kSizeM
  dtype: int
  value: 2048
  placements: [host, runtime]
```

| Field | Required | Type | Default | Notes |
|---|---|---|---|---|
| `kind` | yes | `scalar` | – | |
| `name` | yes | string | – | C++ identifier; must be UPPERCASE if `define` is in `placements` |
| `dtype` | no | `int` \| `float` | `int` | |
| `value` | yes | number | – | Scalar literal |
| `placements` | no | list | `[host, define]` | Any non-empty subset of `host`, `define`, `runtime` |

### Placements — how a scalar reaches the kernel

| Placement | Effect |
|---|---|
| `host` | Becomes `inline constexpr <dtype> <name> = <value>;` in `inputs.hpp`. Visible to the host driver only — usable in `size:`/`grid:` expressions. **Not reachable from a kernel.** |
| `define` | Passed to NVRTC as a `-DNAME=value` macro — a compile-time constant inside the kernel. Name must be UPPERCASE (lowercase `-D` macros collide with NVRTC built-in headers). |
| `runtime` | Passed as a kernel argument (a KTT scalar argument, bound positionally) and becomes part of the boundary — the reference signature takes it too. |

A `constexpr` in `inputs.hpp` is compiled into the host driver, not the kernel:
referencing a host-only scalar from a kernel is an undefined-identifier compile error.
The engine derives a per-problem "scalar contract" for the LLM prompt from these
placements (`utils/inputs.py: scalar_contract_text`).

## Buffers

```yaml
- kind: buffer
  name: mat_a
  dtype: float
  size: kSizeM * kSizeK
  access: read
  init: random
  min: -2
  max: 2
  validate: false
```

| Field | Required | Type | Default | Notes |
|---|---|---|---|---|
| `kind` | yes | `buffer` | – | |
| `name` | yes | string | – | C++ identifier |
| `dtype` | no | `int` \| `float` | `float` | |
| `size` | yes | string | – | C++ expression over host scalars, e.g. `kSizeM * kSizeK` |
| `access` | no | `read` \| `write` \| `readwrite` | `read` | |
| `init` | no | `random` \| `zeros` \| `custom` \| `file` | `random` | Initializer kind (below) |
| `min` / `max` | no | number | float: `[-1, 1]`; int: `[0, 1]` | Bounds for `init: random` only |
| `body` | `custom` only | string | – | Verbatim C++ function body; must `return std::vector<dtype>` of the right size |
| `file_name` | `file` only | string | – | Relative path inside the problem's `inputs/` directory |
| `validate` | no | bool | `false` | Whether this buffer is compared against the reference |

### `init`

- `random` — deterministic `std::mt19937` generator, uniform in `[min, max]`
  (defaults: float `[-1.0, 1.0]`, int `[0, 1]` inclusive). Random init helps catch
  kernels that forget to write part of an output buffer.
- `zeros` — zero-filled. Convenient for reductions, accumulators, and debugging.
- `custom` — your own C++ in `body:`; the generator calls it verbatim. For coupled or
  structured data (e.g. a banded matrix) that `random` cannot produce.
- `file` — reads a raw little-endian binary of the buffer's dtype from
  `inputs/<file_name>`. Byte count must equal `size * sizeof(dtype)` — checked at
  driver start. The web UI uploads these files via
  `POST /api/problems/{name}/inputs/{buffer}`.

### `access`

- `read` — KTT read-only buffer; the host keeps an input copy for reference calls.
- `write` — KTT write-only buffer.
- `readwrite` — both; input data is preserved and the buffer is also an output.

### `size`

A **C++ expression** over host-placement scalar names:

```yaml
size: kSizeM * kSizeK
size: N * 18
size: '1024'
```

The value must be a YAML string — a bare `size: 1024` parses as an integer and is
rejected by the schema.

It is compiled by C++ (in the generator and the reference launcher) and, for Python
references, also evaluated by Python — keep it to simple arithmetic (`+ - * / %`,
parentheses) that is valid in both languages.

Note Python-side evaluation uses `eval()` with `__builtins__` stripped, which is not a
strong sandbox — only run problems from sources you trust (see
[KNOWN_ISSUES.md](../KNOWN_ISSUES.md)).

---

# Minimal Valid Configurations

### Minimal CUDA-reference problem

`problem.yaml`:

```yaml
name: My Problem
description: Minimal CUDA reference example.
gpu:
  index: 0
grid:
  x: N
reference:
  type: cuda
  file: ref_kernel.cu
  function: reference
  block:
    x: 256
validation:
  tolerance: 1.0e-6
```

`inputs.yaml`:

```yaml
args:
- kind: scalar
  name: N
  dtype: int
  value: 1024
  placements: [host, define]
- kind: buffer
  name: x
  dtype: float
  size: N
  access: read
  init: random
- kind: buffer
  name: y
  dtype: float
  size: N
  access: write
  init: zeros
  validate: true
```

### Minimal CPU-reference problem

Identical, except:

```yaml
reference:
  type: cpu_c
  file: ref_cpu.c
  function: reference
```

---

# Practical Recommendations

1. Give every scalar all and only the placements it needs — `host` if it sizes buffers,
   `define` if the kernel needs it as a constant, `runtime` if it should vary per launch.
2. Validate at least one output buffer — the spec requires it.
3. Use `init: random` for inputs a kernel might only partially consume.
4. Keep `size` and `grid` expressions simple and integer-valued, in C++ syntax.
5. Add GPU metadata when you know it; the LLM prompts use it.
6. For CPU C++ references, wrap the function with `extern "C"`.
7. Put constraints you actually care about in `rules.forbid`, not `rules.text`.
