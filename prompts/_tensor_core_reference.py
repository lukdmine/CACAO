"""Shared tensor-core API reference block.

Imported by plan, implement, and propose prompts. Keep single-sourced so
the shape tables don't drift.
"""

TENSOR_CORE_REFERENCE = """## Tensor Cores (Ampere sm_80 / sm_86)

**`nvcuda::wmma` fragment API (`<mma.h>`) — only these shapes are specialized.** Any other `(M, N, K)` does NOT fail to compile; the kernel silently no-ops (tuner log: `kernel duration was 0us` + `Results differ`). That signature = unsupported wmma shape.

| Input dtype (template arg)   | Supported `(M, N, K)`                |
|------------------------------|--------------------------------------|
| `__half` / `__nv_bfloat16`   | 16x16x16, 32x8x16, 8x32x16           |
| `wmma::precision::tf32`      | 16x16x8 (only)                       |
| `signed char` / `unsigned char` (int8) | 16x16x16, 32x8x16, 8x32x16 |
| `double`                     | 8x8x4                                |

**Correct template parameters** (the accumulator dtype is separate from the input dtype — raw `float` is only valid for the accumulator):

```cpp
// fp16 input, float accumulator (shape 16x16x16):
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,        wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

// tf32 input: use the TAG TYPE wmma::precision::tf32, NOT raw float.
// Memory is still stored as float; the template parameter is the tag.
wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16, 16, 8, float> c;
```

For any shape NOT in this table (e.g. fp16 `m16n8k8`, tf32 `m16n8k4`), use inline-PTX `mma.sync` instead — Ampere supports fp16/bf16 `m16n8k{8,16}`, tf32 `m16n8k{4,8}`, int8 `m16n8k{16,32}`. Pair with `ldmatrix.sync.aligned` and `cp.async` (no `<cuda/pipeline>` needed). **Never pass an unsupported shape or dtype to `wmma::fragment<>`.**

Unavailable: Hopper `wgmma`/`tcgen05`/TMA (sm_90+); `<cuda/pipeline>`, `<cuda/barrier>`, `<cooperative_groups.h>` (not validated here).
"""
