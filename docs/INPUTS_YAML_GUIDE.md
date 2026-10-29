# `inputs.yaml`

The I/O boundary: every scalar and buffer a problem's kernels and reference see.
`utils/inputs.py` generates `inputs.hpp` from it at the start of every run.

`inputs.hpp` is a build artifact. It is gitignored, regenerated unconditionally, and
hand edits to it are lost. Edit `inputs.yaml`.

## Example

```yaml
headers: []
shared_setup: ''
args:
  - kind: scalar
    name: kSizeM
    dtype: int
    value: 2048
    placements: [host, runtime]
  - kind: scalar
    name: TILE
    dtype: int
    value: 32
    placements: [host, define]
  - kind: buffer
    name: mat_a
    dtype: float
    size: kSizeM * kSizeK
    access: read
    init: random
    min: -2.0
    max: 2.0
  - kind: buffer
    name: mat_c
    dtype: float
    size: kSizeM * kSizeN
    access: write
    init: zeros
    validate: true
```

## Top level

| Key | Type | Notes |
|---|---|---|
| `headers` | list of strings | Extra `#include`s emitted into `inputs.hpp` |
| `shared_setup` | string | Verbatim C++ placed before the generators |
| `args` | list | **Ordered.** Scalars and buffers, discriminated by `kind` |

**`args` order is the reference signature.** See "The boundary" below.

At least one buffer must set `validate: true`, or loading the spec raises.

A `cases:` block in `problem.yaml` overrides scalar `value`s and `init: file`
`file_name`s per case; everything else here is shared by every case. See
[PROBLEM_YAML_GUIDE.md](PROBLEM_YAML_GUIDE.md#cases).

## Scalars

```yaml
- kind: scalar
  name: kSizeM
  dtype: int
  value: 2048
  placements: [host, runtime]
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `name` | yes | — | Valid C++ identifier |
| `dtype` | no | `int` | `int` or `float` only — scalars cannot be `half` |
| `value` | yes | — | The constant |
| `placements` | no | `[host, define]` | One or more of `host`, `define`, `runtime` |

### `placements`

The field that decides how a scalar reaches the kernel. A scalar can carry more than
one placement.

| Placement | Becomes | Visible to |
|---|---|---|
| `host` | `inline constexpr` in `inputs.hpp` | the driver — sizes buffers and `grid` expressions |
| `define` | a `-D` macro handed to NVRTC | the kernel, as a compile-time constant |
| `runtime` | `AddArgumentScalar` + an entry in `boundary` | the kernel, as a **function parameter** |

Consequences:

- A `define` scalar must be **UPPERCASE**. `-D` macros are substituted textually into
  every NVRTC translation unit, and a lowercase one collides with CUDA's built-in
  headers. The validator enforces this only for `define`; `host` and `runtime` scalars
  may use any valid identifier.
- A `runtime` scalar is a real kernel parameter and must appear in the reference
  signature. A `define`-only scalar must not.
- A `host`-only scalar is invisible to kernels entirely. It exists to size buffers and
  grids.
- A scalar with `runtime` but not `host` cannot be used in a `size` expression.
- Empty `placements` is rejected — the scalar would be declared nowhere.

## Buffers

```yaml
- kind: buffer
  name: mat_a
  dtype: float
  size: kSizeM * kSizeK
  access: read
  init: random
  min: -2.0
  max: 2.0
  validate: false
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `name` | yes | — | Valid C++ identifier |
| `dtype` | no | `float` | `int`, `float` or `half` (see [below](#half-precision-buffers)) |
| `size` | yes | — | **String.** A C++ expression over `host` scalars |
| `access` | no | `read` | `read`, `write` or `readwrite` |
| `init` | no | `random` | `random`, `zeros`, `custom` or `file` |
| `min` / `max` | no | see below | `init: random` only |
| `body` | when `init: custom` | — | Verbatim C++ returning `std::vector<dtype>` |
| `file_name` | when `init: file` | — | Path under `problems/<name>/inputs/` |
| `validate` | no | `false` | Compare this buffer against the reference |

### `size`

Must be a **quoted string** — `size: 1024` is rejected, `size: '1024'` is fine.
Pydantic does not coerce an int to a str here.

It is a C++ expression, widened per operand to `size_t`. Only `host`-placement scalars
are in scope. As with `grid`, `//` is a comment, not integer division.

### `init`

- `random` — uniform over `[min, max]`, seeded per buffer name so runs are
  reproducible. Defaults are `[-1.0, 1.0]` for `float` and `[0, 1]` for `int`; an
  integer buffer left unbounded therefore holds only 0s and 1s.
- `zeros`
- `custom` — `body` is emitted verbatim as the generator body and must
  `return` a `std::vector<dtype>`. For coupled or derived inputs.
- `file` — reads `problems/<name>/inputs/<file_name>` as raw little-endian elements of
  `dtype`. The byte count must equal `size * sizeof(dtype)` exactly; it is checked at
  driver start, before any read. `file_name` must be a relative path with no `..`.
  The UI's upload endpoint writes into that directory.

`access` and `init` are closed sets: a typo raises a `ValidationError` at load rather
than being silently reinterpreted.

### `access`

`read`, `write` and `readwrite` map to KTT's argument access. A `readwrite` buffer that
is also a validation target is memcpy'd from the kept host copy into KTT's buffer
before the reference runs, so the reference sees the original input.

### Half-precision buffers

`dtype: half` is IEEE binary16. **The host and the kernel spell it differently, and both
spellings are the same two bytes:**

| Side | Type | Where it comes from |
|---|---|---|
| host (`inputs.hpp`, g++) | `half_float::half` | vendored by KTT; `Ktt.h` includes it |
| device (`kernels.cu`, NVRTC) | `__half` | `#include <cuda_fp16.h>` |

cacao only ever emits the host spelling — the kernel signature is written by whoever
writes the kernel, so a half buffer arrives as `const __half*` / `__half*` and the kernel
must include `<cuda_fp16.h>` itself. The engine already passes the CUDA include directory
to NVRTC (`utils/framework.resolve_cuda_include`), so no extra configuration is needed.

Consequences worth knowing before choosing it:

- **Scalars cannot be `half`.** A half `-D` macro has no sensible meaning. Pass a half
  *value* through a one-element buffer if you need one.
- **All three reference kinds work**, but `cpu_c` has one requirement. `utils/build.py`
  compiles `ref_cpu.c` with **g++, as C++**, and passes `-I KTT/Source`, so the generated
  declaration is `extern "C" void ref(const half_float::half* a, ...)` and your reference
  source must `#include <Utility/External/half.hpp>`. (`extern "C"` only suppresses name
  mangling; a C++ class parameter type is fine either side of it.) A `cuda` reference
  spells the same buffer `__half`, and a `python` reference sees numpy `float16`.
- **`init: random` draws in `float` and narrows on assignment**, because
  `std::uniform_real_distribution` requires a floating-point type and `half_float::half`
  is a class. `min`/`max` therefore mean exactly what they mean for `float`.
- **`init: file` expects raw little-endian binary16**, two bytes per element — numpy
  `float16` writes exactly this. The byte-count check is `size * 2`.
- **A validated half buffer is compared BIT-EXACTLY, and `validation.tolerance` is
  silently ignored for it.** This is the single most important thing on this page.
  `ResultValidator.inl` dispatches on `std::is_floating_point<T>`, which is **false** for
  `half_float::half` because it is a class, so half takes the integral overload —
  `if (result[i] != reference[i]) return false;` — instead of `ValidateSideBySide`.
  Verified: a candidate differing from the reference by **one ulp** is rejected with the
  tolerance set to `10.0`, about 5000x the gap.

  Consequences:
  - A `cuda` reference performing the *identical* arithmetic works fine, and is the
    intended pairing: both sides round the same way, so they match bit for bit.
  - A `python` reference is a trap. It computes in fp32 and rounds once at the end
    (`utils/python_ref_runner.py`), while the kernel rounds at every step. Those agree
    only by luck. Prefer fp16 in / **fp32 out** — leave the validated output `float` —
    unless your kernel provably reproduces the reference's rounding.
  - Values above 65504 become `inf` on the way in, with no warning.

  Widening the tolerance cannot rescue a half output; nothing about it is consulted.
  KTT does expose `Tuner::SetValueComparator(id, cmp)` for a per-argument approximate
  comparison, but the engine skeleton emits no comparator today.

- **Give the problem a `gpu.compute_capability`.** `__half`'s operators are gated on
  `__CUDA_ARCH__ >= 530`. When the capability is unknown, the NVRTC pre-check passes no
  architecture flag and NVRTC targets the lowest one its toolkit supports, which differs
  per toolkit. On CUDA 12.0 that is below 5.3, and the pre-check rejects an ordinary half
  kernel with *"more than one conversion function from `const __half` to a built-in type
  applies"*, a kernel that compiles fine for the real device. The engine normally
  auto-detects the capability, so this bites only when detection fails.

There is **no `bfloat16`**. KTT's `ArgumentDataType` has `Half` and no bf16 equivalent, so
a bf16 contract cannot be expressed today without a change to KTT itself.

## The boundary

`boundary` is what gets bound to the kernel and the reference, in `args` order:

> every buffer, plus every scalar carrying the `runtime` placement.

`host`-only and `define`-only scalars are declared but never passed, so they can sit
anywhere in `args` without disturbing the signature.

This differs by reference type:

- **`cuda`** — the reference kernel takes the full boundary: buffers *and* runtime
  scalars.
- **`cpu_c`** — the C function takes **buffers only**, as pointers. Scalars reach it as
  `-D` macros.
- **`python`** — `f(scalars, buffers)`.

See [REFERENCE_IMPLEMENTATION_GUIDE.md](REFERENCE_IMPLEMENTATION_GUIDE.md).

## Generated artifacts

`inputs.hpp` contains the host constants, one `gen_<name>()` per buffer, a `struct
Inputs` with the KTT argument ids, `boundary`, `validated`, and the `-D` string the
driver passes to NVRTC. `utils/inputs.py` is the generator; `models/inputs.py` is the
schema.
