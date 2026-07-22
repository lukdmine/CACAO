"""Shared tensor-core API reference block.

Imported by plan, implement, and propose prompts. Keep single-sourced so
the shape tables don't drift.
"""

TENSOR_CORE_REFERENCE = """## Tensor Cores (Ampere sm_80 / sm_86)

**`nvcuda::wmma` fragment API (`<mma.h>`) — only these shapes are specialized.** Any other `(M, N, K)`/dtype combo is a **hard NVRTC compile error**: `error: incomplete type is not allowed` on the `wmma::fragment<>` declaration, usually followed by `no instance of function template "nvcuda::wmma::fill_fragment"`. Seeing that = pick a row from this table or drop to inline-PTX `mma.sync`. (The tuner-log pair `kernel duration was 0us` + `Results differ` is a DIFFERENT failure: the kernel launched but wrote nothing — dead bounds guard, zero-sized grid, or wrong argument binding.)

| Input dtype (template arg)   | Supported `(M, N, K)`                |
|------------------------------|--------------------------------------|
| `__half` / `__nv_bfloat16`   | 16x16x16, 32x8x16, 8x32x16           |
| `wmma::precision::tf32`      | 16x16x8 (only)                       |
| `signed char` / `unsigned char` (int8) | 16x16x16, 32x8x16, 8x32x16 |
| `double`                     | 8x8x4                                |

**Correct template parameters** (the accumulator dtype is separate from the input dtype — raw `float` is only valid for the accumulator):

```cpp
#include <mma.h>
using namespace nvcuda;   // REQUIRED: mma.h declares wmma inside namespace nvcuda.
                          // Without it (or an nvcuda::wmma:: prefix on every use) the
                          // first fragment declaration fails with
                          //   error: name followed by "::" must be a class or namespace name
                          //   error: type name is not allowed
                          // and every later wmma call cascades from it.

// fp16 input, float accumulator (shape 16x16x16):
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,        wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

// tf32 input: use the TAG TYPE wmma::precision::tf32, NOT raw float.
// Memory is still stored as float; the template parameter is the tag.
wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16, 16, 8, float> c;
```

**Converting float→half (`<cuda_fp16.h>`) — two similarly named functions, different arity:**
```cpp
__half2 __floats2half2_rn(float a, float b);   // TWO scalars  ("floatS")
__half2 __float22half2_rn(float2 a);           // ONE float2   ("float2")
__half  __float2half_rn(float a);              // one scalar -> one half
```
Passing two scalars to `__float22half2_rn` gives `no suitable constructor exists to convert
from "float" to "float2"` plus `too many arguments in function call`.

For any shape NOT in this table (e.g. fp16 `m16n8k8`, tf32 `m16n8k4`), use inline-PTX `mma.sync` instead — Ampere supports fp16/bf16 `m16n8k{8,16}`, tf32 `m16n8k{4,8}`, int8 `m16n8k{16,32}`. Pair with `ldmatrix.sync.aligned`. For async global→shared copies, `<cuda_pipeline.h>` (`__pipeline_memcpy_async`/`__pipeline_commit`/`__pipeline_wait_prior`) works under NVRTC and emits `cp.async` with no inline PTX; `<cuda/pipeline>` + `<cuda/barrier>` also compile. **Never pass an unsupported shape or dtype to `wmma::fragment<>`.**

Unavailable: Hopper `wgmma`/`tcgen05`/TMA (sm_90+); grid-wide sync — `cg::this_grid().sync()` COMPILES but KTT launches via `cuLaunchKernel` (not the cooperative API), so it is invalid at runtime. `<cooperative_groups.h>` itself works at block/warp scope (`cg::tiled_partition<32>`, `cg::reduce`, `cg::memcpy_async`).
"""
