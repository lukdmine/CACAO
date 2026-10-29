"""Prompt for the agentic authoring step (kernel + framework regions).

Short by design. The single-shot prompts this replaces ran 49 KB and 52 KB because
everything that might be needed had to be present before the model could ask for it:
iteration history with full kernel bodies, tuner output, NCU dumps, the best previous
kernel. With tools, all of that is one call away, so the prompt carries only what
cannot be derived — the contract, the strategy, the I/O boundary, and the rules.

The execution model, the compile environment, the region contract and the launcher
contract are imported rather than restated. They are the single source of truth, and
prompts/_system_overview.py documents what per-prompt paraphrases cost last time.
"""

from prompts._format import format_reference, format_strategy, inputs_hpp_block
from prompts._launcher_reference import LAUNCHER_RULES
from prompts._regions_reference import BASE_BLOCK_SIZE, PARAMETER_GROUPS
from prompts._system_overview import SYSTEM_OVERVIEW, NVRTC_RULES

_FILES = """# Author the kernel and its tuning driver

You write a CUDA kernel and the KTT driver regions that tune it, using tools. Work
until the code compiles, then end the step.

## Files you own
- `kernels.cu` — one or more `extern "C" __global__` kernels.
- `region_kernels.cpp` — kernel definitions, `ktt::KernelId kernel = ...`, scratch
  buffers, `SetArguments` per definition in that kernel's exact parameter order:

  ```cpp
  const ktt::KernelDefinitionId defA = tuner.AddKernelDefinitionFromFile(
      "expand", kernelFile, ndRange, ktt::DimensionVector());
  // Scratch/intermediate buffers are yours to declare — pipeline temporaries, and
  // the convergence flag an iterative launcher reads back:
  auto changedId = tuner.AddArgumentVector(std::vector<unsigned int>(1, 0u),
                                           ktt::ArgumentAccessType::ReadWrite);
  ktt::KernelId kernel = tuner.CreateCompositeKernel("Flood", {defA, defB});
  tuner.SetArguments(defA, {in.heights, in.flooded, changedId});
  ```
- `region_params.cpp` — `AddParameter` / `AddConstraint` / `AddThreadModifier`.
- `region_launcher.cpp` — the launch schedule; see the launcher contract below.

The engine splices the three regions into a fixed `main()`. You do not write
includes, `main()`, validation, the searcher, or the tuning loop. In scope: `tuner`,
`Inputs in` (use `in.<name>`), `ndRange`, `kernelFile`.

"""

_HOW_TO_WORK = """## How to work

Open with a tool call, not a plan. Prose before the first call buys nothing, and a
reply that spends its whole token allowance thinking gets cut off before the call
lands — nothing is written and the turn is wasted. You have many turns and a large
tool budget, so keep each reply small: one file per `write_file` is fine, and a
targeted `edit_file` is better than a rewrite.

1. `check_compilation()` compiles the driver with g++ and the kernel with NVRTC,
   exactly as the tuner will. Use it — a failure here costs nothing, the same
   failure after `end_step` costs a whole iteration.
2. `edit_file` for changes to a working kernel. Rewriting a file to change one
   constant is how unrelated parts regress.
3. `end_step` requires a passing `check_compilation`. Any write invalidates the
   previous pass.
4. Investigate before rewriting: `tuning_landscape` shows every measured
   configuration of an earlier iteration, not just its best. If the fastest value of
   a parameter sits at the edge of its declared range, widen the range instead of
   restructuring the kernel.

"""

_CLOSING = """Write the kernel and the three regions, get a passing
`check_compilation()`, then `end_step`."""

# A config_only step is entered with a kernel that already compiles and validates; the
# task text says to keep it. Closing on "write the kernel" is the last thing the model
# reads, and it rewrites the working implementation the scope existed to protect.
_CLOSING_CONFIG_ONLY = """Update the three driver regions — leave `kernels.cu` as it
is — get a passing `check_compilation()`, then `end_step`."""


def build(ctx: dict) -> tuple[str, str]:
    # LAUNCHER_RULES prevents the mistakes that compile cleanly and fail only validation.
    system = (
        _FILES
        + BASE_BLOCK_SIZE
        + "\n"
        + _HOW_TO_WORK
        + PARAMETER_GROUPS
        + "\n"
        + SYSTEM_OVERVIEW
        + NVRTC_RULES
        + LAUNCHER_RULES
    )

    parts = []

    # Rules first: they constrain everything below.
    if ctx.get("rules_block"):
        parts.append(ctx["rules_block"])

    if ctx.get("problem_yaml"):
        parts.append(f"## Problem (problem.yaml):\n```yaml\n{ctx['problem_yaml']}\n```")

    if ctx.get("inputs_hpp"):
        parts.append(inputs_hpp_block(ctx["inputs_hpp"]))
    if ctx.get("cases_block"):
        parts.append(ctx["cases_block"])

    if ctx.get("scalar_contract"):
        parts.append(ctx["scalar_contract"])

    strategy_text = format_strategy(ctx.get("strategy"))
    if strategy_text:
        parts.append(strategy_text)
    if ctx.get("plan"):
        parts.append(f"## Plan:\n{ctx['plan']}")

    if ctx.get("parent_context"):
        parts.append(ctx["parent_context"])

    if ctx.get("ref_kernel"):
        parts.append(format_reference(ctx))

    if ctx.get("iteration_summaries"):
        parts.append(ctx["iteration_summaries"])
    if ctx.get("branch_index"):
        parts.append(ctx["branch_index"])

    if ctx.get("iteration_history"):
        parts.append(ctx["iteration_history"])

    if ctx.get("current_files"):
        parts.append(ctx["current_files"])

    if ctx.get("task"):
        parts.append(f"## This iteration:\n{ctx['task']}")
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])

    parts.append(
        _CLOSING_CONFIG_ONLY
        if ctx.get("authoring_scope") == "config_only"
        else _CLOSING
    )
    return system, "\n\n".join(parts)
