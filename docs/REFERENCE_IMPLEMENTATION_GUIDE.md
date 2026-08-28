# Writing a reference implementation

The reference is the ground truth every candidate kernel is validated against. Three
kinds, selected by `reference.type` in [`problem.yaml`](PROBLEM_YAML_GUIDE.md):

| `type` | File | Runs as |
|---|---|---|
| `cuda` | `ref_kernel.cu` | A CUDA kernel KTT launches |
| `cpu_c` | `ref_cpu.c` | A C function linked into the driver |
| `python` | `ref.py` | A Python function in a subprocess |

The argument order in every case comes from the `args` list in
[`inputs.yaml`](INPUTS_YAML_GUIDE.md), never from `problem.yaml`.

## The boundary

`boundary` is every buffer, plus every scalar carrying the `runtime` placement, in
`args` order. What the reference receives depends on the type:

| `type` | Receives |
|---|---|
| `cuda` | the full boundary — buffers **and** runtime scalars, as kernel parameters |
| `cpu_c` | buffers only, as pointers. Scalars arrive as `-D` macros |
| `python` | `(scalars, buffers)` — two dicts |

The common mistake is assuming scalars are always `-D` macros. That is true for
`cpu_c` only. For a CUDA reference, a scalar with `runtime` in its `placements` **is a
kernel parameter** and must appear in the signature; a `define`-only scalar must not.

## `type: cuda`

```yaml
reference:
  type: cuda
  file: ref_kernel.cu
  function: gemm_reference
  block: {x: 8, y: 8, z: 1}
```

```cuda
// inputs.yaml args: kSizeM [host,runtime], kSizeN [host,runtime],
//                   TILE [host,define], mat_a, mat_b, mat_c
extern "C" __global__ void gemm_reference(
    const int kSizeM, const int kSizeN,     // runtime scalars, in args order
    const float* mat_a, const float* mat_b, // buffers, in args order
    float* mat_c)
{
    ...
}
```

Rules:

- `extern "C"` and `__global__`.
- Parameters follow `args` order, filtered to the boundary.
- `block` is a **nested** `{x, y, z}` mapping. Flat `block_x`/`block_y`/`block_z` are
  silently ignored, and an absent `block` defaults to `{x: 1}` — one thread per block,
  which makes the reference about a thousand times slower without failing.
- The reference is launched with the same `grid` as the candidate.
- Keep it obviously correct. It is not tuned and its speed is not the baseline.

## `type: cpu_c`

```yaml
reference:
  type: cpu_c
  file: ref_cpu.c
```

```c
// Scalars are -D macros. Only buffers are parameters, in args order.
void savings_reference(const float* deposits, float* balances) {
    for (int c = 0; c < CLIENTS; c++)
        for (int p = 0; p < PERIODS; p++)
            balances[c * PERIODS + p] = ...;
}
```

Rules:

- Pointer parameters only, one per buffer, in `args` order. No scalar parameters.
- Every `define`-placement scalar is available as a macro. `utils/build.py` compiles
  `ref_cpu.c` to a separate object file with those `-D` flags and links it into the
  driver, so the macros never reach `framework.cpp`.
- Declared `extern "C"` by the generated `inputs.hpp`; the file itself is plain C.
- Read-only buffers arrive `const`.

## `type: python`

```yaml
reference:
  type: python
  file: ref.py
  function: gdn_reference
```

```python
import torch

def prepare_input(scalars, buffers):
    """Optional. H2D and reshaping, not timed."""
    return {
        "q": torch.from_numpy(buffers["q"]).cuda().view(scalars["B"], scalars["T"], -1),
        ...
    }

def gdn_reference(prepared):
    out = ...
    return out            # torch tensor or numpy array
```

Rules:

- The function must be **defined in `ref.py`**, not imported into it. The runner
  resolves it by name and then rejects anything whose `__module__` is not `"ref"`, so
  `from my_oracle import gdn_reference` fails with "Reference function not found".
- With `prepare_input`: it is called as `prepare_input(scalars, buffers)`, its result
  is passed to the reference as a single argument, and a returned device tensor is
  converted with `.cpu().numpy()` automatically.
- Without `prepare_input`: the reference is called as `f(scalars, buffers)` and its
  result goes through `np.asarray(...)`. A CUDA tensor raises there — return a numpy
  array or add `prepare_input`.
- The result must have exactly `size` elements for the validated buffer.
- KTT computes the reference **once per `Tune` call** and caches it for every
  configuration, so the subprocess cost is paid once per iteration, not per config.
- `torch` is an optional per-problem dependency (`pip install torch`). A `ref.py` that
  cannot be imported aborts the run at engine start rather than failing mid-tune.

### Speedup for non-CUDA references

For `cpu_c` and `python`, the reference is not a competitive GPU baseline, so the
reported speedup is not a meaningful figure of merit unless
`output/reference_time.json` is seeded with a real one. It can be written by hand and
survives archiving. For `python` references the engine tries to measure the GPU op
precisely up front; when that fails it falls back to KTT's coarse wall-clock time,
which is much larger and inflates every speedup on the run.

## Checklist

- [ ] `reference.type`, `file` and (for `cuda`/`python`) `function` match reality
- [ ] Parameter order matches `args` in `inputs.yaml`
- [ ] `cuda`: runtime scalars are parameters; `define` scalars are not
- [ ] `cuda`: `block` is nested `{x, y, z}` and is not the default one-thread block
- [ ] `cpu_c`: buffers only, no scalar parameters
- [ ] `python`: the function is defined in `ref.py`, and returns numpy or uses `prepare_input`
- [ ] At least one buffer in `inputs.yaml` sets `validate: true`
- [ ] `validation.tolerance` is set deliberately, not left at the 1e-4 default
