"""Prompt for the framework-file configure node.

Teaches the LLM to fill the three regions of a KTT C++ autotuning driver
(CACAO:KERNELS / CACAO:PARAMS / CACAO:LAUNCHER). The engine assembles these into
framework.cpp around a fixed skeleton. See docs/FRAMEWORK_FILE_SPEC.md.
"""

SYSTEM = r"""You configure a **KTT C++ autotuning driver** by writing three C++ region bodies
that the engine splices into a fixed `main()`. You do NOT write includes, `main()`,
validation, or the tuning loop — only the three region bodies, as plain C++
statements.

# What the engine already provides (in scope for your regions)
- `ktt::Tuner tuner` — the tuner.
- `Inputs in` — the problem's I/O boundary, from the user's inputs.hpp. Reference
  each boundary argument as `in.<name>` (e.g. `in.mat_a`, `in.kSizeM_`). You are
  shown inputs.hpp below — use those exact field names.
- `const ktt::DimensionVector ndRange(...)` — the base global size (from the grid).
- `std::string kernelFile` — path to kernels.cu (the kernel source, shown below).
- Problem scalars are host consts (e.g. `kSizeM`) from inputs.hpp — usable directly.
The engine wires validation (`in.validated`, `in.boundary`), the searcher, `Tune`,
and `SaveResults` AFTER your regions. Never emit those.

# Region 1 — CACAO:KERNELS
For each `__global__` in kernels.cu, add a definition; create the kernel; register
scratch buffers for pipeline intermediates; bind arguments per definition.
- MUST declare `ktt::KernelId kernel = ...` (the skeleton uses it).
- Single kernel: `CreateSimpleKernel("Name", def)`. Pipeline: `CreateCompositeKernel("Name", {defA, defB, ...})`.
- `tuner.SetArguments(def, {...})` for EVERY definition, in that `__global__`'s
  exact parameter order, using `in.<name>` handles and scratch ids.
- Intermediate/scratch buffers (LLM-owned): `auto tmpId = tuner.AddArgumentVector(std::vector<float>(size), ktt::ArgumentAccessType::ReadWrite);`
Example (single kernel):
```cpp
const ktt::KernelDefinitionId def = tuner.AddKernelDefinitionFromFile(
    "gemm_fast", kernelFile, ndRange, ktt::DimensionVector());
ktt::KernelId kernel = tuner.CreateSimpleKernel("Gemm", def);
tuner.SetArguments(def, {in.kSizeM_, in.kSizeN_, in.kSizeK_, in.mat_a, in.mat_b, in.mat_c});
```

# Region 2 — CACAO:PARAMS
Tuning parameters become `-D` compile-time macros in the kernel (KTT recompiles per
config). Add them, plus constraints and thread modifiers.
```cpp
tuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{16, 32, 64});
tuner.AddConstraint(kernel, {"TILE", "THREADS"},
    [](const std::vector<uint64_t>& v){ return v[0] % v[1] == 0; });
tuner.AddThreadModifier(kernel, {def}, ktt::ModifierType::Local,
    ktt::ModifierDimension::X, "TILE", ktt::ModifierAction::Multiply);
```
Every macro your kernel uses via `#ifdef`/as a constant must be declared here.
Add constraints that guarantee legal, in-bounds configs (divisibility, shared mem).

# Region 3 — CACAO:LAUNCHER
- **Single kernel using thread modifiers: leave EMPTY** (KTT's default launcher runs it).
- Multi-kernel pipeline or custom/iterative launch: set a launcher whose `RunKernel`
  sequence IS the schedule (topological order; blocking calls satisfy dependencies):
```cpp
tuner.SetLauncher(kernel, [defA, defB](ktt::ComputeInterface& ci) {
    const auto& cfg = ci.GetCurrentConfiguration();
    const uint64_t tile = ktt::ParameterPair::GetParameterValue(cfg.GetPairs(), "TILE");
    ci.RunKernel(defA, ktt::DimensionVector(/*grid*/), ktt::DimensionVector(/*block*/));
    ci.RunKernel(defB);                        // runs after A (synchronous)
});
```
For data-dependent iteration, use runtime scalar args + `ci.UpdateScalarArgument(id, &v)`
and `ci.SwapArguments(def, a, b)` between launches. The FINAL write must land in the
validated output buffer (`in.<validated>`), or validation fails.

# Hard requirements
1. CACAO:KERNELS must define `ktt::KernelId kernel`.
2. Every launched definition needs `SetArguments` matching its `__global__` order.
3. A composite kernel (>1 def) MUST have a launcher; there is no auto-launch.
4. Any `RunKernelAsync` must be joined (`WaitForComputeAction`/`SynchronizeQueue`) before the launcher returns.
5. Do NOT emit validation, `Tune`, `SaveResults`, includes, or `main()`.
6. Parameter names become `-D` macros on ALL definitions of the kernel — namespace
   names that mean different things in different kernels (e.g. `A_TILE`, `B_TILE`).

# KTT C++ API you may use
`tuner`: AddKernelDefinitionFromFile(name,file,global,local); CreateSimpleKernel(name,def);
CreateCompositeKernel(name,{defs}); SetLauncher(kernel,lambda); AddParameter(kernel,name,
std::vector<uint64_t>{...}); AddConstraint(kernel,{names},fn); AddThreadModifier(kernel,{defs},
ModifierType{Global,Local},ModifierDimension{X,Y,Z},name(s),ModifierAction{Add,Subtract,
Multiply,Divide,DivideCeil}); AddArgumentVector(vec,AccessType); AddArgumentScalar(v);
AddArgumentLocal<T>(size); SetArguments(def,{ids}).
`ci` (launcher): RunKernel(def[,g,l]); RunKernelAsync(def,queue)+WaitForComputeAction(id);
GetAllQueues(); SynchronizeQueue(q); GetCurrentConfiguration().GetPairs();
ktt::ParameterPair::GetParameterValue(pairs,"NAME"); SwapArguments; UpdateScalarArgument(id,&v);
ResizeBuffer. `ktt::DimensionVector(x[,y[,z]])`.

Return the three region bodies. Emit only C++ statements for each — no code fences,
no `main()`, no includes."""


def build(ctx: dict) -> tuple[str, str]:
    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem metadata (problem.yaml):\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("inputs_hpp"):
        parts.append(
            "## Inputs (inputs.hpp) — reference these boundary handles as `in.<name>`:\n"
            f"```cpp\n{ctx['inputs_hpp']}\n```"
        )
    if ctx.get("kernel_code"):
        parts.append(
            "## Kernels you wrote (kernels.cu) — match SetArguments to each signature, "
            "declare every `-D` macro it uses as a parameter:\n"
            f"```cuda\n{ctx['kernel_code']}\n```"
        )
    if ctx.get("plan"):
        parts.append(f"## Optimization Plan:\n{ctx['plan']}")
    if ctx.get("strategy_section"):
        parts.append(ctx["strategy_section"])
    if ctx.get("parent_context"):
        parts.append(ctx["parent_context"])
    if ctx.get("iteration_history"):
        parts.append(ctx["iteration_history"])
    if ctx.get("best_so_far"):
        parts.append(ctx["best_so_far"])
    if ctx.get("prev_context"):
        parts.append(ctx["prev_context"])
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])
    parts.append(
        "Write the three region bodies (kernels, params, launcher) to autotune this "
        "kernel via the KTT C++ driver."
    )
    return SYSTEM, "\n\n".join(parts)
