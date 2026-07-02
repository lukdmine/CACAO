# Framework-File Autotuning — Specification

> Status: **DRAFT / design spec.** Branch: `framework-file-autotuning`.
> Both the design record and the contract handed to the LLM when it fills a
> framework file.

---

## 1. Purpose

Today CACAO autotunes a **single** CUDA kernel through a fixed declarative path:
`problem.yaml` + LLM-authored `params.json`, interpreted by one generic Python
harness (`tuner.py`) that hard-codes *one kernel, vector-only arguments, and a
single string-`eval` launcher*.

Too rigid for:

- **Predefined / structured inputs** (lookup tables, permutation maps, banded
  matrices, data from disk) — `tuner.py` only fills buffers `random`/`zeros`.
- **Multi-kernel pipelines / dependency graphs** — the launcher runs one definition.
- **Dynamic shared memory, runtime scalar args, native thread modifiers** — all in
  KTT, all disabled by the current wrapper.

The **framework file** replaces that path with a **plain KTT C++ driver** — the
same shape as `KTT/Examples/ClTuneGemm/ClTuneGemm.cpp` — that is part
engine-generated, part user-authored, part LLM-authored:

- The **Python engine** codegens the *fixed skeleton* (tuner setup, argument
  registration, validation wiring, tune/save). Pure KTT calls emitted as text.
- The **user** authors the *input data* — generator functions producing named,
  typed buffers (maps / predefined / random / file-loaded).
- The **LLM** fills three regions — *kernel wiring*, *parameter space*, and the
  *launcher lambda* — which together express arbitrary multi-kernel pipelines.

**No `pyktt`, and no CACAO-owned C++ library.** The driver includes only
`<Ktt.h>` + the C++ standard library and links against `libktt.so`.

---

## 2. Locked design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Framework file is **C++**, compiled to a binary, linked against `libktt.so`. | Full KTT C++ API; matches `KTT/Examples/*.cpp`. No pyktt. |
| D2 | User input definitions are real C++ **generator functions** (not YAML data). | Build maps / predefined / file-loaded data with arbitrary host code. |
| D3 | The **launcher lambda is written directly by the LLM** (free-form), not compiled from a declarative graph. | It's KTT's only native mechanism anyway; max expressiveness (data-dependent loops, conditional pipelines). The LLM owns topo + sync correctness. |
| D4 | **Validation is engine-owned.** The LLM never writes the correctness check. | A functionally-wrong pipeline must *fail*, not report a fake speedup. |
| D5 | Framework path **fully replaces** the legacy YAML/`tuner.py`/`params.json` path. **No backward compatibility.** | Rewriting the tuner layer; one path ⇒ simpler engine. `tuner.py`, `params.json` removed; `problem.yaml` slimmed. Existing problems migrated or dropped. |
| D6 | Validation uses **one CUDA reference kernel** (for now). | Single simplest mechanism. CPU-reference deferred. |
| D7 | `inputs.hpp` is a **separate, form-editable** file (with the reference kernel). Its **structured spec is canonical**; `inputs.hpp` is generated from it. | Clean ownership; LLM never edits it; users author data in the UI. |
| D8 | **One `kernels.cu`** holds all `__global__`s; **searcher + stop condition FIXED** (not LLM-chosen). | Simplicity; LLM owns the parameter space, not the search meta-strategy. |
| D9 | **No CACAO C++ helper library.** The Python engine **codegens the fixed skeleton** as pure-KTT text; the LLM fills only 3 regions. | Nothing C++ to maintain; validation stays engine-controlled (D4). Building blocks (`AddArgumentVector`, `SetReferenceKernel`, `ParameterPair::GetParameterValue`, `<random>`) are already KTT/stdlib. |

---

## 3. Terminology

- **Kernel definition** (`KernelDefinitionId`) — one `__global__` + default geometry.
- **Kernel** (`KernelId`) — *simple* (one def) or *composite* (several defs + launcher). Parameters/constraints/modifiers attach here.
- **Launcher** — a `void(ktt::ComputeInterface&)` lambda; the `RunKernel` sequence inside it *is* the pipeline/DAG.
- **Argument** (`ArgumentId`) — a GPU buffer or scalar. **Pipeline edges are shared arguments**: a buffer written by stage A and read by B *is* the A→B dependency.
- **Scratch buffer** — an intermediate argument between stages; **LLM-owned** (created in the KERNELS region).

---

## 4. File layout

Three tiers, by owner. **The problem definition is only what the user authors** —
`problem.yaml` + `ref_kernel.cu`. `kernels.cu` and `framework.cpp` are
**generated artifacts of the optimization loop**, never user-authored or distributed.

**Tier 1 — problem definition (USER, form-editable, distributed):**
```
problems/<name>/
├── problem.yaml        # PURE metadata: name, gpu.index, tuning.duration_s,
│                       #   validation.tolerance, global_size_type, grid, reference{...}
└── ref_kernel.cu       # the single CUDA reference kernel (correctness oracle)
```

**Tier 2 — the I/O boundary (USER-owned, form-generated, shared across iterations):**
```
problems/<name>/inputs.hpp     # scalars + data generators + KTT arg registration +
                               # which buffer is validated — all via DefineInputs() (§6)
```

**Tier 3 — per iteration (generated; never user-touched):**
```
output/branches/<name>/iter_N/
├── kernels.cu          # LLM-authored candidate kernel(s) for this iteration
├── framework.cpp       # engine skeleton + 3 LLM regions (references inputs.hpp)
├── driver              # compiled binary
└── results.json        # KTT SaveResults(JSON) — parsed by utils/results.py (unchanged)
```

`inputs.hpp` owns the **entire argument boundary**; the engine skeleton is
**argument-agnostic** and consumes it via `DefineInputs()` (§8). `problem.yaml`
carries no inputs/scalars. `inputs.hpp` + `ref_kernel.cu` are visible to each
iteration's build.

> The Phase 0 `problems/mmul/framework/` directory colocates all tiers in one
> place as a **build test only** — it is not the production layout.

---

## 5. The three ownership bands

`framework.cpp` is one `main()`; each line originates from exactly one owner:

```
┌─ ENGINE (Python codegen — pure KTT, argument-agnostic) ───────────────────┐
│  #include <Ktt.h>, "inputs.hpp"                                            │
│  parse argv (platform, device, duration, tolerance, output, kernels, ref) │
│  ktt::Tuner tuner(...); SetGlobalSizeType(...); SetTimeUnit(Microseconds)  │
│  const ktt::DimensionVector ndRange(<grid>);      // from problem.yaml      │
│  Inputs in = DefineInputs(tuner);     // USER (inputs.hpp): whole boundary  │
│  … reference def;  tuner.SetArguments(refDef, in.boundary) …               │
│                                                                            │
│    // ===== BEGIN CACAO:KERNELS =====   <<< LLM >>>   (defines `kernel`)    │
│    // ===== BEGIN CACAO:PARAMS  =====   <<< LLM >>>                         │
│    // ===== BEGIN CACAO:LAUNCHER =====  <<< LLM >>>                         │
│                                                                            │
│  tuner.SetValidationMethod(SideBySideComparison, tolerance);               │
│  tuner.SetReferenceKernel(in.validated, refKernel, {});                    │
│  tuner.SetSearcher(kernel, make_unique<RandomSearcher>());                 │
│  tuner.Tune(kernel, make_unique<TuningDuration>(duration)); SaveResults…   │
└────────────────────────────────────────────────────────────────────────────┘
```

- **ENGINE** — codegen'd text (Python templates). Owns tuner setup, the reference
  + validation wiring, search/stop/save. **Argument-agnostic**: it never names an
  individual argument — it calls `DefineInputs()` and uses `in.boundary` /
  `in.validated`.
- **USER** — `inputs.hpp`: scalars + generators + KTT arg registration + which
  buffer validates, all via `DefineInputs()` (§6).
- **LLM** — the three regions (§9), referencing `in.<name>` and scalar consts.

Everything is direct KTT + stdlib; there is no `cacao::` namespace.

---

## 6. Authoring inputs

### 6.1 The form generates `inputs.hpp`

The create-problem form presents a table; each row is one input (name, dtype,
access, size, `validate`, and an `init`: preset `random{min,max}` / `zeros`, or a
**custom** generator body). An optional **shared-setup** block handles
coupled/derived inputs (a sparse-CSR triple, a matrix + its transpose). The form
emits `inputs.hpp` **directly** — there is no separate YAML input spec. (A
raw-editor escape hatch lets power users edit `inputs.hpp` verbatim.)

### 6.2 `inputs.hpp` shape (the boundary contract)

`inputs.hpp` declares scalars as host consts, one generator per buffer, and a
`DefineInputs()` that registers every boundary argument and exposes them through
an `Inputs` struct the skeleton consumes:

```cpp
#pragma once
#include <vector>
#include <random>
#include <Ktt.h>

inline constexpr int N = 1048576;                     // scalar host const (§7)

inline std::vector<float> gen_data() { /* random / zeros / custom body over `v` */ }
inline std::vector<float> gen_out()  { return std::vector<float>(N, 0.f); }

struct Inputs {
    ktt::ArgumentId data, out;                        // named -> LLM uses in.data / in.out
    ktt::ArgumentId validated;                        // buffer checked vs the reference
    std::vector<ktt::ArgumentId> boundary;            // args in reference-signature order
};

inline Inputs DefineInputs(ktt::Tuner& t) {
    Inputs in;
    in.data = t.AddArgumentVector(gen_data(), ktt::ArgumentAccessType::ReadOnly);
    in.out  = t.AddArgumentVector(gen_out(),  ktt::ArgumentAccessType::WriteOnly);
    in.validated = in.out;
    in.boundary  = {in.data, in.out};
    return in;
}
```

The `Inputs` struct **is** the contract: its fields are the arg handles the LLM
references (`in.data`); `validated` is the buffer checked against the reference;
`boundary` is the ordered list the engine passes to the reference's
`SetArguments`. Rules: exactly one `validated`; sizes are expressions over
scalars; generators use scalars, never tuning params (§7). Working example:
`problems/mmul/framework/inputs.hpp`.

---

## 7. Scalar model

A scalar is declared **once** (a form row) and the engine fans it out to every
place that needs it. There are two compilers — **host** (`g++`, driver +
`inputs.hpp`) and **device** (NVRTC, `kernels.cu`) — and a scalar lands in both:

| Place | Form | How |
|-------|------|-----|
| **Host const** | `inline constexpr long N = …;` in `inputs.hpp` | Generators **and** the launcher use `N` directly (driver `#include`s `inputs.hpp`). |
| **Device `-D` define** | `tuner.SetCompilerOptions("… -DN=1048576")` | Kernels use `N` as a compile-time macro (enables unroll/specialize). |
| **Runtime arg** *(opt-in)* | `auto nId = tuner.AddArgumentScalar<int>(N);` | Only when the launcher mutates it per-launch (e.g. reduction's shrinking `n`); kernel takes it as a parameter, launcher calls `UpdateScalarArgument`. |

Default = host const + `-D` define (matches today's behavior). Floats are emitted
with an `f` suffix in defines. In Shape B all three placements live in
`inputs.hpp`/`DefineInputs` (the host `constexpr`; an optional
`SetCompilerOptions("-DN=…")` for a define; an optional `AddArgumentScalar` for a
runtime arg) — the engine skeleton stays argument-agnostic.

⚠️ **Hard rule:** generators may reference **scalars**, never **tuning
parameters** — inputs are built once, *before* tuning; parameters only exist
per-trial inside the launcher. A buffer sized by a parameter is an LLM **scratch**
buffer (§9.1), not a user input.

---

## 8. The engine-generated driver skeleton

The engine assembles `framework.cpp` from a template + `problem.yaml` metadata +
the three LLM region bodies. It is **argument-agnostic** — the arg boundary comes
from `DefineInputs()`. Real generated single-kernel example (`mmul`):

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <Ktt.h>
#include "inputs.hpp"

int main(int argc, char** argv) {
    ktt::PlatformIndex platform = std::stoul(argv[1]);
    ktt::DeviceIndex   device   = std::stoul(argv[2]);
    const double       duration  = std::stod(argv[3]);
    const double       tolerance = std::stod(argv[4]);
    const std::string  output = argv[5], kernelFile = argv[6], refFile = argv[7];

    ktt::Tuner tuner(platform, device, ktt::ComputeApi::CUDA);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::OpenCL);   // from problem.yaml global_size_type
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    tuner.SetCompilerOptions("-I/usr/local/cuda/include");

    const ktt::DimensionVector ndRange(kSizeM, kSizeN);     // from problem.yaml grid

    // inputs — user-owned; the ENTIRE argument boundary:
    Inputs in = DefineInputs(tuner);

    // reference kernel (engine-owned validation):
    auto refDef = tuner.AddKernelDefinitionFromFile("gemm_reference", refFile,
        ndRange, ktt::DimensionVector(8, 8));
    auto refKernel = tuner.CreateSimpleKernel("Reference", refDef);
    tuner.SetArguments(refDef, in.boundary);

    // ===== BEGIN CACAO:KERNELS =====   (LLM; MUST define `ktt::KernelId kernel`)
    // ===== BEGIN CACAO:PARAMS  =====   (LLM)
    // ===== BEGIN CACAO:LAUNCHER =====  (LLM)

    tuner.SetValidationMethod(ktt::ValidationMethod::SideBySideComparison, tolerance);
    tuner.SetReferenceKernel(in.validated, refKernel, ktt::KernelConfiguration());
    tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
    auto results = tuner.Tune(kernel, std::make_unique<ktt::TuningDuration>(duration));
    tuner.SaveResults(results, output, ktt::OutputFormat::JSON);
    return 0;
}
```

**Name contract the LLM relies on:** `tuner`; `in` (the `Inputs` from
`DefineInputs` — boundary args are `in.<name>`, e.g. `in.data`); `ndRange`;
`kernelFile`; and scalar host consts from `inputs.hpp`. The KERNELS region must
define `kernel`. (`std::make_unique` — the C++ API takes `unique_ptr`, verified
Phase 0.)

---

## 9. LLM contract — the three regions

Pure KTT + the names above. No helper layer.

### 9.1 `CACAO:KERNELS`
```cpp
auto defA = tuner.AddKernelDefinitionFromFile("stage_a", kernelFile,
                ktt::DimensionVector(), ktt::DimensionVector());
auto defB = tuner.AddKernelDefinitionFromFile("stage_b", kernelFile,
                ktt::DimensionVector(), ktt::DimensionVector());

std::vector<float> tmp(N);                                   // scratch = a DAG edge (LLM-owned)
auto tmpId = tuner.AddArgumentVector(tmp, ktt::ArgumentAccessType::ReadWrite);

ktt::KernelId kernel = tuner.CreateCompositeKernel("Pipeline", {defA, defB});  // REQUIRED name

tuner.SetArguments(defA, {in.data, tmpId});                  // in.* = user boundary args
tuner.SetArguments(defB, {tmpId, in.out});
```

### 9.2 `CACAO:PARAMS`
```cpp
tuner.AddParameter(kernel, "TILE",    std::vector<uint64_t>{16, 32, 64});
tuner.AddParameter(kernel, "THREADS", std::vector<uint64_t>{64, 128, 256});
tuner.AddConstraint(kernel, {"TILE", "THREADS"},
    [](const std::vector<uint64_t>& v){ return v[0] % v[1] == 0; });
tuner.AddThreadModifier(kernel, {defB}, ktt::ModifierType::Local,
    ktt::ModifierDimension::X, "THREADS", ktt::ModifierAction::Multiply);
```

### 9.3 `CACAO:LAUNCHER`
```cpp
tuner.SetLauncher(kernel, [defA, defB](ktt::ComputeInterface& ci) {
    const auto& cfg = ci.GetCurrentConfiguration();
    const uint64_t tile = ktt::ParameterPair::GetParameterValue(cfg.GetPairs(), "TILE");
    ci.RunKernel(defA, ktt::DimensionVector((N + tile - 1) / tile), ktt::DimensionVector(tile));
    ci.RunKernel(defB);                                     // synchronous; runs after A
});
```

### 9.4 Hard requirements
1. `CACAO:KERNELS` **must** assign `ktt::KernelId kernel` (the skeleton footer uses it).
2. Every launched definition **must** have `SetArguments` matching that `__global__`'s parameter order.
3. A **composite** kernel (>1 def) **must** have a launcher; there is no auto-launch.
4. Any `RunKernelAsync` **must** be joined (`WaitForComputeAction`/`SynchronizeQueue`) before the launcher returns, or KTT records **incorrect durations** (`ComputeInterface.h:55,70`).
5. The LLM **must not** emit validation, `Tune`, `SaveResults`, or edit the skeleton.
6. Parameter names become `-D` defines on **all** definitions of the kernel; namespace names that differ across kernels (`A_TILE`, `B_TILE`).

---

## 10. Multi-kernel pipelines & dependency graphs

KTT has **no declarative DAG API** — the launcher body *is* the graph.

**Correctness (edges) — free.** Emit stages in topological order as synchronous
`RunKernel` calls; each blocks, so every producer finishes before its consumer.
Edges are **shared buffers** (a scratch/user buffer written then read). A wrong
order ⇒ wrong output ⇒ caught by validation; it is not auto-detected.

**Concurrency (independent branches) — optional, via CUDA streams:**
```cpp
auto qs = ci.GetAllQueues();
auto a  = ci.RunKernelAsync(branchA, qs[0]);   // -> ComputeActionId
auto bb = ci.RunKernelAsync(branchB, qs[1]);
ci.WaitForComputeAction(a); ci.WaitForComputeAction(bb);   // join (required, §9.4.4)
ci.RunKernel(joinKernel);
```

**Timing & validation come free:** KTT sums each launch's duration into the
config total, and `utils/results.py` already sums `ComputationResults[].Duration`
— an N-stage pipeline needs **no parser change**. Reference: `Covariance.cpp`
(mean→reduce→covar) and `Bicg.cpp` (parameter-selected pipeline shape).

### 10.1 Runtime scalar arguments & iterative launchers (the reduction pattern)

For values that are **data-dependent and change between launches** — *not* swept
by the tuner — use runtime scalar arguments: add them with `AddArgumentScalar`,
bind them in `SetArguments`, and mutate them in the launcher with
`UpdateScalarArgument`. The canonical case is an iterative reduction that runs one
definition `log n` times over shrinking data (idiom: `Reduction.py:25-64`).

Kernel (`kernels.cu`) — takes the dynamic values as parameters:
```cpp
extern "C" __global__ void reduce(float* src, float* dst, int n, int inOff, int outOff) { /*…*/ }
```

`CACAO:KERNELS` — register the runtime scalars (LLM-owned) alongside a scratch dst:
```cpp
auto def = tuner.AddKernelDefinitionFromFile("reduce", kernelFile,
               ktt::DimensionVector(), ktt::DimensionVector());

std::vector<float> scratch(N);
auto dstId    = tuner.AddArgumentVector(scratch, ktt::ArgumentAccessType::ReadWrite);
auto nId      = tuner.AddArgumentScalar(int(N));   // dynamic — mutated each launch
auto inOffId  = tuner.AddArgumentScalar(0);
auto outOffId = tuner.AddArgumentScalar(0);

ktt::KernelId kernel = tuner.CreateSimpleKernel("Reduction", def);
tuner.SetArguments(def, {in.data, dstId, nId, inOffId, outOffId});   // in.data = user input
```

`CACAO:LAUNCHER` — mutate the scalars and ping-pong buffers each iteration:
```cpp
tuner.SetLauncher(kernel, [def, src = in.data, dstId, nId, inOffId, outOffId]
                          (ktt::ComputeInterface& ci) {
    const ktt::DimensionVector local  = ci.GetCurrentLocalSize(def);
    ktt::DimensionVector       global = ci.GetCurrentGlobalSize(def);
    ci.RunKernel(def, global, local);                        // first pass

    int n = int(global.GetSizeX() / local.GetSizeX());       // partials remaining
    int inOff = 0, outOff = n;
    while (n > 1) {
        ci.SwapArguments(def, src, dstId);                   // ping-pong src/dst
        global.SetSizeX(((n - 1) / local.GetSizeX() + 1) * local.GetSizeX());
        ci.UpdateScalarArgument(nId,      &n);               // C++ form: (id, const void*)
        ci.UpdateScalarArgument(inOffId,  &inOff);
        ci.UpdateScalarArgument(outOffId, &outOff);
        ci.RunKernel(def, global, local);
        n = int((n + local.GetSizeX() - 1) / local.GetSizeX());
        inOff = outOff; outOff += n;
    }
});
```

Notes:
- **C++ signature is `UpdateScalarArgument(id, const void* data)`** — pass `&value`.
  The typed `UpdateScalarArgumentInt/Float/…` names are *pyktt-only*; the framework
  file is C++.
- These are **runtime args, not tuning parameters** — they carry data-dependent
  state, so they must never be `AddParameter`. A *tuned* value (tile, unroll
  factor) still goes in `CACAO:PARAMS` as a `-D` define (§9.2).
- The technique **composes with multi-kernel pipelines**: the same
  `UpdateScalarArgument` / `SwapArguments` calls sit between the `RunKernel`s of a
  composite launcher.

---

## 11. Validation (engine-owned)

The engine emits, argument-agnostically (via the `Inputs` struct from §6):
1. the reference kernel definition (`ref_kernel.cu`, `reference.function`, `reference.block`);
2. `tuner.SetArguments(refDef, in.boundary)` — the user-declared boundary args, in
   order. Scratch buffers are LLM-local and never in `in.boundary`, so they never
   reach the reference;
3. `SetValidationMethod(SideBySideComparison, tolerance)` +
   `SetReferenceKernel(in.validated, refKernel, {})`.

The reference computes the final output from the **original** inputs. The LLM
emits no validation code; `in.validated` is declared by the *user* in
`inputs.hpp`, not the LLM (D4).

---

## 12. Constraints carried over (NVRTC)

Kernels are still NVRTC-compiled inside KTT:
- `extern "C" __global__` (symbol lookup by name).
- **No host headers** (`<cuda_runtime.h>`, `<stdint.h>`, …). Allowed device headers: `<mma.h>`, `<cuda_fp16.h>`, `<cuda_bf16.h>`.
- Static `__shared__` works; **dynamic shared memory now available** via `AddArgumentLocal<T>(size)` + `UpdateLocalArgument` in the launcher.

---

## 13. KTT C++ API reference (LLM-usable subset)

Anchored to `KTT/Source/Api/ComputeInterface.h` and `KTT/Source/Python/PythonTuner.cpp`.

**`ktt::Tuner`** (in LLM regions): `AddKernelDefinitionFromFile(name,file,global,local)→KernelDefinitionId`; `CreateSimpleKernel(name,def)→KernelId`; `CreateCompositeKernel(name,{defs}[,launcher])→KernelId`; `SetLauncher(kernel,launcher)`; `AddParameter(kernel,name,std::vector<uint64_t>)` (+ Int/Double/Bool/String); `AddConstraint(kernel,{names},fn)`; `AddThreadModifier(kernel,{defs},ModifierType{Global,Local},ModifierDimension{X,Y,Z},name(s),ModifierAction{Add,Subtract,Multiply,Divide,DivideCeil})`; `AddArgumentVector(vec,AccessType)→ArgumentId`; `AddArgumentScalar(v)`, `AddArgumentLocal<T>(size)`; `SetArguments(def,{ids})`.

**`ktt::ComputeInterface`** (in launcher): `RunKernel(def[,global,local])` (sync); `RunKernelAsync(def,queue[,g,l])→ComputeActionId` + `WaitForComputeAction(id)`; `GetDefaultQueue()`,`GetAllQueues()`,`SynchronizeQueue(q)`,`SynchronizeQueues()`; `GetCurrentConfiguration()` (`.GetPairs()`→`ParameterPair`); `GetCurrentGlobalSize(def)`/`GetCurrentLocalSize(def)`; `SwapArguments`,`ChangeArguments`,`UpdateScalarArgument`,`UpdateLocalArgument`,`ResizeBuffer`,`ClearBuffer`.

**Helpers/enums:** `ktt::DimensionVector(x[,y[,z]])`; `ktt::ParameterPair::GetParameterValue(pairs,"NAME")`; `ktt::ArgumentAccessType{ReadOnly,WriteOnly,ReadWrite}`.

**⚠️ C++ vs pyktt (verified Phase 0):** the engine-emitted skeleton uses
`std::unique_ptr` where pyktt takes values — `SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>())`
and `Tune(kernel, std::make_unique<ktt::TuningDuration>(seconds))`. `AddArgumentScalar(x)`
deduces its type from the argument. These are engine-owned lines; the LLM regions
use `AddParameter`/`AddConstraint`/`AddThreadModifier`/`SetArguments`/`RunKernel` as shown.

---

## 14. Compilation, execution, feedback

**Host compile** — ✅ **verified in Phase 0** (GEMM, RTX 2080 Ti, CUDA 12.5, g++ 11.4).
Kernels stay NVRTC-compiled inside KTT; only the driver is host-compiled, and it
needs no CUDA host includes (NVRTC pulls them at runtime via `SetCompilerOptions`):
```bash
g++ -std=c++17 -m64 -O3 -I<repo>/KTT/Source \
    framework.cpp <repo>/libktt.so -Wl,-rpath,<repo> -o driver
```
`libcuda`/`libnvrtc`/`libOpenCL` come **transitively** from `libktt.so` (its
`NEEDED` entries) — no explicit `-lcuda -lnvrtc`.

**Run:** `./driver <platform> <device> <duration_s> <tolerance> <output_base> <kernels.cu> <ref_kernel.cu>`
— KTT writes `<output_base>.json` (it appends the extension for `OutputFormat::JSON`).
Runtime scalars (duration, tolerance, indices, paths) are **argv**; only
compile-time/structural values are codegen'd into the source.

**Two error classes → LLM feedback:**
1. **Host-compile error** (bad C++ in an LLM region) — from `g++` stderr. *New* failure mode → fix/propose node.
2. **NVRTC / tune-time error** (bad kernel or launcher exception) — driver stdout / `ResultStatus{CompilationFailed, ComputationFailed, ValidationFailed, DeviceLimitsExceeded}`. As today.

**Results:** `results.json` (`OutputFormat::JSON`) — parsed by existing `utils/results.py`, unchanged.

---

## 15. Integration with CACAO

Status chain unchanged (`planning → implementing → configuring → running →
profiling → proposing → deciding`).

| Node | Change |
|------|--------|
| `implement.py` | May emit **multiple** `__global__`s in one `kernels.cu`; drops "single kernel only". Interface rules §12. |
| `configure.py` | **No `params.json`.** Emits the **three region bodies**; `utils/framework.py` assembles `framework.cpp` = skeleton (from `problem.yaml` metadata) + regions. |
| `run.py` | Adds a **compile step** (§14) before executing; feeds host-compile errors back. Result parsing unchanged. |
| `profile.py` | ✅ Runs the compiled driver in **profile mode** (LoadResults → fastest valid config → single `Run`) under NCU; validation skipped so only agent kernels launch. `parse_ncu_csv` returns per-kernel metrics (prefixed keys for pipelines). |
| `propose.py`/`decide.py` | Unchanged; also react to host-compile errors. |

**Removed:** `tuner.py`, `params.json`, `models/params.py`, the scalars/vectors/grid
sections of `problem.yaml`.

**New/changed engine pieces:** the codegen module `utils/framework.py` (✅ Phase 1)
assembles `framework.cpp` from `problem.yaml` metadata + the three region bodies;
a compile step (Phase 2) builds + captures errors. `state/types.py` gains
`framework_cpp`, `kernels_code` (multi-kernel); drops `params_json`.

**Frontend:** the create-problem form gains an **inputs table** (per-row preset /
custom generator), a **shared-setup** block, and editors for `inputs.hpp` +
`ref_kernel.cu`. The form emits `inputs.hpp` directly (Shape B, §6); `problem.yaml`
stays pure metadata. `api/problems.py` create/update handle `inputs.hpp` +
`ref_kernel.cu`; `generate_types.py` re-run after model changes.

---

## 16. Implementation plan (phased)

- **Phase 0 — de-risk the build. ✅ DONE.** Hand-authored `framework.cpp` +
  `inputs.hpp` + `kernels.cu` + `ref_kernel.cu` for `mmul` in
  `problems/mmul/framework/`; compiled against `libktt.so`, tuned 195 configs on
  the RTX 2080 Ti, `results.json` parsed by `utils/results.py` unchanged. Flags
  locked (§14); C++/pyktt API diffs recorded (§13).
- **Phase 1 — codegen. ✅ DONE.** `utils/framework.py` assembles an
  argument-agnostic `framework.cpp` from `problem.yaml` metadata + the three
  region bodies (Shape B: `inputs.hpp` owns the boundary via `DefineInputs()`;
  skeleton wires reference/validation off `in.boundary`/`in.validated`).
  Regenerated `mmul`, compiled, tuned 139 configs, `results.json` parsed by
  `utils/results.py` unchanged.
- **Phase 2 — compile module. ✅ (core).** `utils/build.py`:
  `compile_framework()` (Phase-0 flags, structured g++ error capture) +
  `driver_command()`. Tested: compiles + runs `mmul` end-to-end, and a broken
  driver yields `ok=False` with captured diagnostics. `run.py` wiring lands with
  Phase 3 (once `configure.py` emits `framework.cpp` per iteration; the current
  `params.json` path stays live until then).
- **Phase 3 — LLM regions. ✅ DONE.** `configure.py` emits the three region bodies
  (structured `FrameworkRegions`) → assembles `framework.cpp` (3a); `run.py`
  compiles + runs the driver, routing compile errors to propose (3b);
  `implement.py` allows multi-kernel + LLM-designed signatures, sees `inputs.hpp`
  (3c). **Validated live (3d):** a real LLM ran implement→configure→run on `mmul`
  framework mode — compilable kernel, correct `in.*` boundary binding + runtime
  scalars, 41/41 configs validated on GPU.
- **Phase 4 — validation & feedback.** Reference wiring; host-compile-error loop.
- **Phase 5 — pipeline problem.** Add a 2-stage problem (transpose→GEMM, or
  gather/map→reduce) to exercise composite kernels + scratch.
- **Phase 6 — frontend + profiling + docs.** Inputs table + editors; per-stage
  NCU; update `CLAUDE.md`, guides.
