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
| `dtype` | no | `int` | `int` or `float` only |
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
| `dtype` | no | `float` | `int` or `float` only |
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
