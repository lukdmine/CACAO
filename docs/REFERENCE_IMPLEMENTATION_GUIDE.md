# Reference Implementation Guide

This guide explains how to write the reference implementation (`ref_kernel.cu`,
`ref_cpu.c`, or `ref.py`) so validation is consistent across the three reference types.

The I/O boundary is declared in `inputs.yaml` (see
[PROBLEM_YAML_GUIDE.md](PROBLEM_YAML_GUIDE.md)). What reaches your reference function
depends on the reference `type:` in `problem.yaml` and on each scalar's `placements`.

## The boundary

`models/inputs.py` defines the *boundary*: every buffer, **plus every scalar with the
`runtime` placement**, in `args:` declaration order. Host-only scalars are declared but
never passed; `define`-placement scalars travel as `-D` macros instead of arguments.

Given:

```yaml
args:
- kind: scalar
  name: kSizeM
  dtype: int
  value: 2048
  placements: [host, runtime]
- kind: scalar
  name: ALPHA
  dtype: float
  value: 1.5
  placements: [host, define]
- kind: buffer
  name: A
  dtype: float
  size: kSizeM * kSizeM
  access: read
  init: random
- kind: buffer
  name: B
  dtype: float
  size: kSizeM
  access: write
  init: zeros
  validate: true
```

the boundary is `(int kSizeM, const float* A, float* B)`, and `ALPHA` is a compile-time
`#define` visible in any source compiled by the engine's kernel path.

## CUDA Reference Rules

For `reference.type: cuda`:

```yaml
reference:
  type: cuda
  file: ref_kernel.cu
  function: reference
  block:
    x: 256
    y: 1
    z: 1
```

Rules:
- use `extern "C" __global__`
- the signature is the full boundary: buffer pointers **and** `runtime`-placement
  scalars, in `args:` declaration order (KTT calls `SetArguments(refDef, in.boundary)`)
- `define`-placement scalars are compile-time `#define` constants — do not repeat them
  as function parameters
- host-only scalars are not visible at all
- keep the implementation simple and correct rather than optimized
- `block: {x, y, z}` sets the reference launch block shape (nested mapping; defaults to
  `{x: 1}` when absent). The reference's global size is the base `grid:` from
  `problem.yaml`, interpreted per `global_size_type`

Example (for the boundary above):

```cuda
extern "C" __global__ void reference(const int kSizeM, const float* A, float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kSizeM) {
        B[i] = A[i * kSizeM + i] * ALPHA;  // ALPHA is a -D macro
    }
}
```

## CPU Reference Rules

For `reference.type: cpu_c`:

```yaml
reference:
  type: cpu_c
  file: ref_cpu.c
  function: reference
```

Rules:
- the signature is buffer pointers only, in `args:` declaration order — scalars are
  never parameters here
- every scalar is available as a compile-time `#define` constant, whatever its
  `placements`. Note this differs from the CUDA reference: a `runtime`-placement
  scalar that is a *parameter* in `ref_kernel.cu` is still a macro in `ref_cpu.c`
- `const`-qualify read buffers; write/readwrite buffers are non-const pointers
- for C++ sources, export the function with `extern "C"`

How it runs: the engine compiles `ref_cpu.c` into the driver (a separate translation
unit, so the `-D` macros never touch the driver source) and the generated `inputs.hpp`
registers one KTT `SetReferenceComputation` per `validate: true` buffer. Each
registered computation calls the full C function: the validated buffer is the one KTT
hands in, read buffers come from kept host copies, readwrite buffers are copied so the
call cannot mutate the kept inputs, and any other output buffer gets a scratch
allocation.

Example C reference:

```c
#ifdef __cplusplus
extern "C" {
#endif

void reference(const float* A, float* B) {
    for (int i = 0; i < kSizeM; ++i) {   // kSizeM is a -D macro here
        B[i] = A[i * kSizeM + i] * ALPHA;
    }
}

#ifdef __cplusplus
}
#endif
```

## Python Reference Rules

For `reference.type: python`:

```yaml
reference:
  type: python
  file: ref.py
  function: reference
```

Unlike CUDA/`cpu_c`, the Python ABI passes everything **by name**. `ref.py` must define
the referenced function as:

```python
def reference(scalars, buffers):
    # scalars: {name: value} — every scalar, whatever its placements
    # buffers: {name: flat np.ndarray} — read/readwrite hold the real inputs,
    #          write buffers are zeros
    ...
    return result  # np.ndarray with exactly the validated buffer's element count
```

Rules:
- return one flat result matching the validated buffer's element count and order
  (e.g. row-major for matrices)
- anything with `.cpu().numpy()` (e.g. a torch tensor) is accepted and converted
  automatically
- keep it simple and correct; validation runs it once per evaluated config

How it runs: `inputs.hpp` registers one KTT `SetReferenceComputation` per validated
buffer; the driver's lambda dumps the input buffers to `cacao_in_<name>.bin` and
invokes `python3 -m utils.python_ref_runner`, which loads `ref.py`, calls the
function, and writes the result back for KTT to compare.

Optional dependencies: `numpy` and `pyyaml` are always available (engine deps).
Anything else `ref.py` imports (e.g. `torch`, see `problems/mmul_pytorch`) is an
optional dependency of that problem only — plain `python -m pip install torch`
is enough (PyPI's Linux wheels bundle their own CUDA runtime; the system CUDA
toolkit version is irrelevant to torch). At engine start, `ref.py` is imported in
a `python3` subprocess; if the import fails, the run aborts immediately with the
error instead of wasting LLM iterations on configs that all fail validation.

Optional GPU-only timing (`prepare_input`): if `ref.py` also defines

```python
def prepare_input(scalars, buffers):  # H2D transfer + setup — NOT timed
    ...
```

then the reference function takes `prepared` instead of `(scalars, buffers)` and
`utils/torch_ref_timer` can time just the device op with `torch.cuda.Event`
(min over several single calls), excluding transfers. Without `prepare_input`,
validation still works unchanged, but the reference time falls back to KTT's
coarse wall clock of the entire runner process (Python startup + imports +
H2D + kernel + D2H) — for a torch reference that is dominated by `import torch`
(~seconds), so every reported speedup becomes meaningless. Always define
`prepare_input` for GPU references; CPU/numpy references can skip it.

## Mapping from `inputs.yaml`

Given:

```yaml
args:
- kind: scalar
  name: M
  value: 1024        # default placements: [host, define]
- kind: scalar
  name: N
  value: 1024        # default placements: [host, define]
- kind: buffer
  name: A
  size: M * N
- kind: buffer
  name: B
  size: M * N
- kind: buffer
  name: C
  size: M * N
  validate: true
```

the expected signatures are:

- CUDA reference: `(const float* A, const float* B, float* C)` — pointers only, since
  both scalars are `define`-placement; with a `runtime` scalar it would appear in the
  signature at its `args:` position
- CPU reference: `(const float* A, const float* B, float* C)` — always pointers only
- Python reference: `reference({"M": 1024, "N": 1024}, {"A": ..., "B": ..., "C": ...})`

`M` and `N` are additionally `#define` constants in the CUDA/kernel compile path and in
the host-compiled CPU reference.

## Output Buffers in CPU References

For CPU references:
- `read` buffers contain the initialized input data
- the validated output buffer is provided as the callback output buffer
- non-validated write buffers are provided as temporary zeroed buffers

That means the CPU reference should compute all outputs it logically owns, even if only
one output is currently validated.

## Common Mistakes

1. Adding scalar parameters to the CPU reference signature — there, scalars are always
   `#define` constants, not arguments.
2. Omitting a `runtime`-placement scalar from the CUDA reference signature — it IS part
   of the boundary, bound positionally.
3. Reordering arguments relative to `args:` in `inputs.yaml`.
4. Forgetting `extern "C"` for C++ sources (kernel or CPU reference) so KTT/ld cannot
   find the symbol.
5. Validating one output while leaving other output buffers semantically unimplemented.
6. Referencing a host-only scalar from the kernel or CUDA reference — it exists only in
   the host driver; use the `define` or `runtime` placement instead.

## Checklist

- [ ] `reference.function` matches the reference symbol name in the source file
- [ ] signature matches the boundary for this reference type (see above)
- [ ] `define`/`runtime`/host-only scalars are used the way their placements declare
- [ ] validated outputs are fully written
- [ ] for CUDA references, `reference.block` is a sensible launch block
- [ ] tolerance matches expected numeric error
