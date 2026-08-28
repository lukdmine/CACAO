# `problem.yaml`

One problem is a directory under `problems/` holding two spec files and a reference
implementation:

| File | Holds |
|---|---|
| `problem.yaml` | GPU, launch geometry, reference, validation, tuning budget, rules |
| `inputs.yaml` | the I/O boundary — every scalar and buffer. See [INPUTS_YAML_GUIDE.md](INPUTS_YAML_GUIDE.md) |
| `ref_kernel.cu` / `ref_cpu.c` / `ref.py` | the reference. See [REFERENCE_IMPLEMENTATION_GUIDE.md](REFERENCE_IMPLEMENTATION_GUIDE.md) |

`problem.yaml` carries no scalars, no buffers and no kernel filename. Those moved to
`inputs.yaml` and to the LLM-authored driver regions respectively.

## Example

```yaml
name: GEMM
description: Dense single-precision matrix multiply
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

## Schema

| Key | Required | Type | Notes |
|---|---|---|---|
| `name` | recommended | string | Shown in the UI and in prompts |
| `description` | recommended | string | Shown in the UI and in prompts |
| `gpu` | recommended | mapping | `gpu.index` selects the device; other keys are advisory metadata |
| `global_size_type` | recommended | string | `cuda` (default) or `opencl` — changes what `grid` means |
| `grid` | **yes** | mapping | `x`/`y`/`z`, pasted into the driver as C++ |
| `reference` | **yes** | mapping | Validation reference source and function |
| `validation` | no | mapping | `tolerance`, default `1e-4` |
| `tuning` | no | mapping | `duration_s`, default `20.0` |
| `cases` | no | list | Named input variants, all tuned and validated each iteration |
| `rules` | no | list, string or mapping | Constraints on what a kernel may do |

Unknown keys are ignored, not rejected.

### `gpu`

```yaml
gpu:
  index: 0
  compute_capability: '8.6'
```

`index` selects the device. `compute_capability` is used by the in-step NVRTC compile
check when auto-detection is unavailable — worth setting when `ncu` is missing or
permission-restricted, otherwise the check falls back to `compute_52` and rejects
`cp.async`, `wmma` and `bf16` kernels the device would compile fine.

### `global_size_type`

Decides how KTT reads `grid`:

- `cuda` (default) — `grid` is a **block count**. The launch is `grid` blocks of
  whatever block size the kernel's thread modifiers produce.
- `opencl` — `grid` is a **total work-item count**, which KTT divides by the block
  size to get the block count.

Every problem in this repo sets it explicitly. Getting it wrong changes the launch by
a factor of the block size.

### `grid`

```yaml
grid:
  x: kSizeM
  y: kSizeN
  z: '1'
```

`x`, `y` and `z` are pasted **verbatim into the generated C++** as
`ktt::DimensionVector(x, y, z)`. They are C++ expressions, not Python:

- Any host-placement scalar from `inputs.yaml` is in scope by name.
- Integer division is `/`, not `//` — `//` starts a C++ line comment and will silently
  truncate the rest of the line.
- Quote anything YAML would otherwise read as a number or a bool (`z: '1'`).

Missing components default to 1. This is the *base* geometry; the tuned launch comes
from `AddThreadModifier` in the LLM-authored `CACAO:PARAMS` region and from the
`CACAO:LAUNCHER` region.

### `reference`

```yaml
reference:
  type: cuda            # cuda | cpu_c | python
  file: ref_kernel.cu
  function: gemm_reference
  block: {x: 8, y: 8, z: 1}
```

| Field | Applies to | Notes |
|---|---|---|
| `type` | all | `cuda` (default), `cpu_c`, or `python` |
| `file` | all | Path relative to the problem directory |
| `function` | `cuda`, `python` | Entry symbol. Required for `cuda` |
| `block` | `cuda` only | Reference launch block, **nested** `{x, y, z}` |

`block` is a nested mapping. Flat `block_x` / `block_y` / `block_z` keys are silently
ignored, and an absent `block` defaults to `{x: 1}` — a one-thread block, which makes
the reference roughly a thousand times slower rather than failing. Set it.

An unrecognised `type` is treated as `cuda` and will fail on the missing `function`.

### `validation`

```yaml
validation:
  tolerance: 1.0
```

Element-wise tolerance for comparing validated buffers against the reference.
Optional — both readers default to `1e-4`, which is a silently strict comparison, so
set it deliberately.

A loose tolerance is how a kernel wins by quietly dropping precision. Pair it with
`rules.forbid` when that matters.

### `tuning`

```yaml
tuning:
  duration_s: 100
```

Wall-clock budget for one KTT tuning run, in seconds. Default `20.0`. `--timeout`
overrides it.

### `cases`

Named instantiations of the same inputs. Every case is tuned and validated each
iteration, so a kernel that is only correct at one shape fails the iteration instead of
being reported as a win.

```yaml
cases:
  - name: prefill          # the primary case: first in the list
    scalars: {T: 4096}
  - name: decode
    scalars: {T: 1}
    duration_s: 20         # a tail case catches a bug, it need not find an optimum
  - name: from_file
    files: {q: q_decode.bin}
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `name` | no | `default` | Becomes a directory name — letters, digits, `_`, `-` only. Must be unique |
| `scalars` | no | `{}` | Overrides for scalars declared in `inputs.yaml`, coerced to their dtype |
| `files` | no | `{}` | `file_name` overrides for `init: file` buffers |
| `duration_s` | no | `tuning.duration_s` | This case's tuning budget |

Omitting the block entirely gives one implicit case named `default` whose resolution is
an identity copy of `inputs.yaml` — which is what every problem without `cases:` gets,
and why the layout below is unchanged for them.

**Layout.** One case (declared or implicit) writes its artifacts at the iteration root,
exactly as before this feature. More than one, and each case gets
`iter_N/case_<name>/`, holding its own `inputs.hpp`, `framework.cpp`, `driver` and
`results.json`. The first case in the list is the primary, and is what run-level
summaries anchor on.

Only `scalars` and `files` vary per case. The grid expression, the kernel and the three
driver regions are shared: a case changes the numbers, not the program.

### `rules`

Constrains what a kernel may do. This is the mechanism that stops a branch winning by
dropping to fp16 accumulation, which still validates whenever `tolerance` is loose.

```yaml
rules:
  text:
    - Accumulate in fp32. Inputs may be fp16 but the accumulator may not.
  forbid:
    - pattern: 'wmma::'
      reason: this problem measures the CUDA-core path
    - '__half\s+acc'
```

| Field | Type | Notes |
|---|---|---|
| `text` | string or list of strings | Stated in the authoring prompt |
| `forbid` | list | Regexes matched against `kernels.cu` before the compiler runs |
| `forbid[].pattern` | string (regex) | Required for the mapping form |
| `forbid[].reason` | string | Reported with the violation |

Shorthands: a bare string is one `text` rule; a bare list is a list of `text` rules;
a bare string inside `forbid` is a pattern with no reason.

**How they are enforced.** `forbid` patterns are checked inside the agentic authoring
step, before compiling — a match fails the check with the reason attached. `text` goes
into the authoring prompt. Neither reaches the `propose` or `decide` prompts, and
with `AGENTIC_STEPS=False` neither is applied at all.

A malformed `rules` block warns and is ignored rather than failing the run.

## Minimal problem

```yaml
name: Vector scale
gpu:
  index: 0
global_size_type: cuda
grid:
  x: N / 256
reference:
  type: cpu_c
  file: ref_cpu.c
validation:
  tolerance: 1e-5
```

with an `inputs.yaml` declaring `N` and the buffers, and a `ref_cpu.c` defining the
reference function.

## What reads what

| Key | Read by |
|---|---|
| `gpu.index` | `nodes/run.py` |
| `gpu.compute_capability` | `agentic/tools.py` |
| `global_size_type`, `grid`, `reference`, `validation`, `tuning` | `utils/framework.py` |
| `reference.file`, `reference.type` | `utils/build.py`, `engine/master.py`, `nodes/author.py` |
| `cases` | `utils/cases.py`, `models/cases.py` |
| `rules` | `utils/rules.py` |
| `name`, `description` | the API and the prompts |
