"""Rules that keep a kernel tunable on a GPU other than the one it was written on.

The finalize step widens the winning kernels' parameter ranges and re-tunes them (on
this GPU), and checks that they compile for other architectures; any later retune on
another card searches the same space. That only works if the kernel exposes its choices
as parameters: a decision baked into ``kernels.cu`` as a literal cannot be searched, and
a device fact baked into a value list is wrong on every other card.
"""

TOGGLE_RULES = """## Keep every change switchable

Each change relative to the previous kernel goes behind a `{0, 1}` parameter named
`USE_<NAME>` (e.g. `USE_VEC_LOADS`, `USE_DOUBLE_BUFFER`), declared with
`AddParameter(kernel, "USE_VEC_LOADS", std::vector<uint64_t>{0, 1})` and read with
`#if USE_VEC_LOADS` in `kernels.cu`. The exception is a structural change (a new
decomposition, a different grid shape, a launcher rewrite), which cannot be a toggle.
The proposal states which kind each change is; the implementer follows that
classification. A first kernel has no previous version, so it needs toggles only for its
optional optimizations, not for the base algorithm or every small choice.

A later stage keeps the toggles that help and drops the rest; a change welded into the
kernel cannot be dropped, so a regression in it is permanent. Change one thing per
retry, so a failure names its cause.

To narrow a parameter, give it one value in `AddParameter`. Never replace it with a
literal in `kernels.cu`: the literal removes the knob on every GPU, not just this one.
"""

DEVICE_RULES = """## Device facts are read at runtime, never written into the kernel

`AddParameter` value lists stay literal brace lists (`std::vector<uint64_t>{2, 4, 8}`).
The compile pre-check reads them with a regex and skips anything computed, so a list
built from a device query is skipped or pasted verbatim into the `#define`, and the
kernel fails to compile or runs with the wrong value.

Device facts enter only through `AddConstraint` lambdas and the launcher, via
`tuner.GetCurrentDeviceInfo()`: `GetMaxComputeUnits()`,
`GetCudaComputeCapabilityMajor()` / `GetCudaComputeCapabilityMinor()`,
`GetLocalMemorySize()`. Shared memory per block is capped at 48 KB on every GPU (the
limit KTT enforces), so never size for a larger opt-in carve-out.

A persistent grid is `GetMaxComputeUnits() * BLOCKS_PER_SM` blocks, computed in the
launcher, with `BLOCKS_PER_SM` a literal-list parameter. No GPU's SM count may appear as
a number.

Lambdas capture explicitly, so `tuner` is not in scope inside them. Read the fact into a
local in the region body and capture that:
```cpp
const uint64_t sms = tuner.GetCurrentDeviceInfo().GetMaxComputeUnits();
tuner.SetLauncher(kernel, [def, sms](ktt::ComputeInterface& ci) {
    const auto& cfg = ci.GetCurrentConfiguration();
    const uint64_t bps =
        ktt::ParameterPair::GetParameterValue<uint64_t>(cfg.GetPairs(), "BLOCKS_PER_SM");
    ci.RunKernel(def, ktt::DimensionVector(sms * bps * 256), ktt::DimensionVector(256));
});
```
The example is for `global_size_type: opencl`, where `RunKernel`'s global size counts
threads (blocks x 256 here). With `cuda` (the default) it counts blocks, so the call is
`ci.RunKernel(def, ktt::DimensionVector(sms * bps), ktt::DimensionVector(256))`.
"""

TENSOR_CORE_FALLBACK = """**Portability: a tensor-core path is a toggle with a CUDA-core fallback.** Put it behind
`USE_TENSOR_CORES` and write the plain CUDA-core version in the `#else` of the same `#if`:
```cpp
#if USE_TENSOR_CORES
    // wmma / mma.sync path
#else
    // scalar or vectorized CUDA-core path computing the same result
#endif
```
`#include <mma.h>` and every wmma/mma use must sit inside the `#if USE_TENSOR_CORES`
branch, or the fallback still fails to compile on a GPU without tensor cores. A GPU older
than sm_70, or a shape the fragment table lacks, then still has a kernel that
compiles and validates; with no `#else`, those configurations fail outright.
"""
