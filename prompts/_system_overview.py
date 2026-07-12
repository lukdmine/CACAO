"""Shared system-overview blocks — the single source of truth for facts that
apply across nodes. Import these instead of restating the execution model or
NVRTC constraints per prompt; per-prompt paraphrases drift (the single-kernel
prohibition survived two days past the framework migration that lifted it).

SYSTEM_OVERVIEW  — execution model; for every LLM node that reasons about kernels.
NVRTC_RULES      — compile-environment facts; for nodes that write or debug kernel code
                   (implement, fix_errors, propose).
"""

SYSTEM_OVERVIEW = """## How This System Works

- **Loop**: an LLM iteratively writes `kernels.cu` (one or more `extern "C" __global__`
  kernels), a KTT C++ driver compiles them at runtime via NVRTC and autotunes tuning
  parameters on the target GPU; results and profiles feed the next iteration.
- **Multi-kernel pipelines are allowed**: stage order and scratch buffers are wired in the
  configure step (composite kernel + launcher). Each kernel launch is a free grid-wide
  barrier. Prefer a single kernel when it suffices — extra stages add launch overhead.
- **Validation is engine-owned**: each config's designated output buffer is compared against
  the reference kernel. The final write must land in that buffer.
- **Tuning parameters** arrive as compile-time macros: KTT prepends `#define NAME value` to
  the kernel source per config. Never `#define` a tuning-parameter name in the kernel.
  Problem scalars are macros (default) or runtime kernel arguments — each kernel's signature
  is the implementer's design; arguments are bound positionally in the configure step.
- **Shared memory**: static `__shared__ arr[MACRO_SIZE]` is the default. `extern __shared__`
  (dynamic) works ONLY when the configure step binds an `AddArgumentLocal` argument;
  unbound dynamic shared memory = 0 bytes = out-of-bounds.
- **No grid-wide sync**: kernels launch via plain `cuLaunchKernel`. `cg::this_grid().sync()`
  compiles but fails at runtime ("unspecified launch failure"). Block/warp-scope cooperative
  groups work fine. Cross-block coordination: atomics in global memory, or a pipeline split.
"""

NVRTC_RULES = """## NVRTC Compilation Rules (kernels.cu is compiled by NVRTC at runtime)

- Headers that transitively reach glibc FAIL ("cannot open source file ..."): `<stdint.h>`,
  `<cstdint>`, `<stdio.h>`, `<cuda.h>`. Allowed device headers: `<mma.h>`, `<cuda_fp16.h>`,
  `<cuda_bf16.h>`, `<cuda_pipeline.h>` (`__pipeline_memcpy_async`/`_commit`/`_wait_prior` =
  cp.async with no inline PTX), `<cooperative_groups.h>` (block/warp scope only).
  `<cuda_runtime.h>` compiles but adds nothing — skip it.
- Everything device-side is built in without headers: `blockIdx`, `__syncthreads()`,
  `fmaf()`, `float4`/`make_float4()`, `__shfl_sync()`, ... Use built-in scalar types
  (`int`, `unsigned int`, `unsigned long long`, `float`, `double`); `uint32_t`/`uintptr_t`
  do NOT exist — cast pointers via `(unsigned long long)ptr`.
- Macros are preprocessor-substituted EVERYWHERE — never reuse a tuning-parameter or
  problem-scalar macro name as a variable, parameter, or function identifier
  (causes cryptic "expected a )" / "identifier undefined" errors).
- Lambdas must NOT carry `__device__` annotations (execution space is inferred);
  plain `__device__` functions are fully supported.
- Static `__shared__` > 48KB fails at launch, not in the NVRTC log (invalid config — expected).
- `extern "C"` is required on every `__global__` kernel.
"""
