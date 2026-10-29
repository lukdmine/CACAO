"""Prompt for the framework-file configure node.

Teaches the LLM to fill the three regions of a KTT C++ autotuning driver
(CACAO:KERNELS / CACAO:PARAMS / CACAO:LAUNCHER). The engine assembles these into
framework.cpp around a fixed skeleton.

The execution model, the region contract and the launcher contract are imported, not
restated. This prompt used to carry its own copy of all three, and every fix since the
split landed only in the shared ones: its launcher section still taught a single round of
launches for a problem that runs to a fixed point, and offered no way to read a flag back
at all — which fails every configuration on validation with nothing in the output that
points at the cause.
"""

from prompts._launcher_reference import LAUNCHER_RULES
from prompts._regions_reference import BASE_BLOCK_SIZE, PARAMETER_GROUPS
from prompts._system_overview import SYSTEM_OVERVIEW

_INTRO = r"""You configure a **KTT C++ autotuning driver** by writing three C++ region bodies
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
"""

_REGION_KERNELS = r"""# Region 1 — CACAO:KERNELS
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
"""

_REGION_PARAMS = r"""# Region 2 — CACAO:PARAMS
Declare every tuning parameter the kernel uses; they reach it as the `#define`s described
above.
```cpp
tuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{16, 32, 64});
tuner.AddParameter(kernel, "B_TILE", std::vector<uint64_t>{16, 32}, "stage_b");  // 4th arg = group
tuner.AddConstraint(kernel, {"TILE", "THREADS"},
    [](const std::vector<uint64_t>& v){ return v[0] % v[1] == 0; });
// ONE parameter → name + action:
tuner.AddThreadModifier(kernel, {def}, ktt::ModifierType::Local,
    ktt::ModifierDimension::X, "TILE", ktt::ModifierAction::Multiply);
// SEVERAL parameters → vector of names + lambda. There is NO {names}+action overload:
// a braced name-list with a ModifierAction compiles but corrupts the parameter name.
tuner.AddThreadModifier(kernel, {def}, ktt::ModifierType::Global, ktt::ModifierDimension::X,
    std::vector<std::string>{"BM", "BN"},
    [](const uint64_t base, const std::vector<uint64_t>& v){ return base * v[0] / v[1]; });
```
Add constraints that guarantee legal, in-bounds configs (divisibility, shared mem).
"""

_REGION_LAUNCHER_HEADING = "# Region 3 — CACAO:LAUNCHER\n\n"

_REQUIREMENTS = r"""# Hard requirements
1. CACAO:KERNELS must define `ktt::KernelId kernel`.
2. Every launched definition needs `SetArguments` matching its `__global__` order
   (`AddArgumentLocal` ids are dynamic shared memory, not parameters — they go LAST).
3. A composite kernel (>1 def) MUST have a launcher; there is no auto-launch.
4. Do NOT emit validation, `Tune`, `SaveResults`, includes, or `main()`.
5. Parameter names become macros on ALL definitions of the kernel — namespace
   names that mean different things in different kernels (e.g. `A_TILE`, `B_TILE`).
6. The FINAL writes must land in the validated output buffer(s) (`in.validated`
   entries), or validation fails.

# KTT C++ API you may use
`tuner`: AddKernelDefinitionFromFile(name,file,global,local); CreateSimpleKernel(name,def);
CreateCompositeKernel(name,{defs}); SetLauncher(kernel,lambda); AddParameter(kernel,name,
std::vector<uint64_t>{...}[,group]) — spell the vector type; only uint64_t params may drive
constraints/modifiers; AddConstraint(kernel,{names},fn) with fn=bool(const std::vector<uint64_t>&);
AddThreadModifier(kernel,{defs},ModifierType{Global,Local},ModifierDimension{X,Y,Z}, then
"NAME",ModifierAction{Add,Subtract,Multiply,Divide,DivideCeil} — or for several params
std::vector<std::string>{names},lambda(uint64_t base, const std::vector<uint64_t>& vals)→uint64_t;
AddArgumentVector(vec,ktt::ArgumentAccessType::{ReadOnly,WriteOnly,ReadWrite}); AddArgumentScalar(v);
AddArgumentLocal<T>(bytes) = dynamic shared memory (id LAST in SetArguments; kernel reads it
via `extern __shared__`); SetArguments(def,{ids}).
`ci` (launcher): RunKernel(def) / RunKernel(def,global,local); GetCurrentConfiguration().GetPairs();
ktt::ParameterPair::GetParameterValue<uint64_t>(pairs,"NAME"); UpdateBuffer(id,ptr,bytes);
DownloadBuffer(id,ptr,bytes); SwapArguments(def,a,b); UpdateScalarArgument(id,&v);
ResizeBuffer(id,bytes,preserveData). `ktt::DimensionVector(x[,y[,z]])`.

Return the three region bodies. Emit only C++ statements for each — no code fences,
no `main()`, no includes."""

SYSTEM = (
    _INTRO
    + "\n"
    + SYSTEM_OVERVIEW
    + "\n"
    + _REGION_KERNELS
    + "\n"
    + BASE_BLOCK_SIZE
    + "\n"
    + _REGION_PARAMS
    + PARAMETER_GROUPS
    + "\n"
    + _REGION_LAUNCHER_HEADING
    + LAUNCHER_RULES
    + "\n"
    + _REQUIREMENTS
)


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
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])
    parts.append(
        "Write the three region bodies (kernels, params, launcher) to autotune this "
        "kernel via the KTT C++ driver."
    )
    return SYSTEM, "\n\n".join(parts)
