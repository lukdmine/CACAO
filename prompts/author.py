"""Prompt for the agentic authoring step (kernel + framework regions).

Short by design. The single-shot prompts this replaces ran 49 KB and 52 KB because
everything that might be needed had to be present before the model could ask for it:
iteration history with full kernel bodies, tuner output, NCU dumps, the best previous
kernel. With tools, all of that is one call away, so the prompt carries only what
cannot be derived — the contract, the strategy, the I/O boundary, and the rules.

SYSTEM_OVERVIEW and NVRTC_RULES are imported rather than restated. They are the
single source of truth for the execution model and the compile environment, and
prompts/_system_overview.py documents what per-prompt paraphrases cost last time.
"""

from nodes._llm_helper import format_strategy
from prompts._launcher_reference import LAUNCHER_RULES
from prompts._system_overview import SYSTEM_OVERVIEW, NVRTC_RULES

_SYSTEM = """# Author the kernel and its tuning driver

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

  The 4th `AddKernelDefinitionFromFile` argument is the BASE block size.
  `ktt::DimensionVector()` is **(1,1,1)**, not "auto" — pass the real block size, or
  scale it with `ModifierType::Local` modifiers in every dimension the block uses.
- `region_params.cpp` — `AddParameter` / `AddConstraint` / `AddThreadModifier`.
- `region_launcher.cpp` — the launch schedule; see the launcher contract below.

The engine splices the three regions into a fixed `main()`. You do not write
includes, `main()`, validation, the searcher, or the tuning loop. In scope: `tuner`,
`Inputs in` (use `in.<name>`), `ndRange`, `kernelFile`.

## How to work

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

Every parameter macro the kernel uses must be declared in `region_params.cpp`, and
every parameter named by an `AddConstraint` must share one parameter group — KTT
drops a constraint split across groups silently.

"""

_CLOSING = """Write the kernel and the three regions, get a passing
`check_compilation()`, then `end_step`."""


def build(ctx: dict) -> tuple[str, str]:
    # LAUNCHER_RULES is the one block that cannot be dropped for brevity: the mistakes
    # it prevents compile cleanly and surface only as validation failures, which cost a
    # full iteration each to diagnose.
    system = _SYSTEM + SYSTEM_OVERVIEW + NVRTC_RULES + LAUNCHER_RULES

    parts = []

    # Rules first: they constrain everything below, including anything a past
    # iteration or a proposal asks for.
    if ctx.get("rules_block"):
        parts.append(ctx["rules_block"])

    # Carries the base grid, the validation tolerance, and the detected GPU (compute
    # capability, SM count, shared memory per block) that the worker enriches it with.
    # Occupancy and shared-memory budgeting are guesswork without it, and it is small.
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem (problem.yaml):\n```yaml\n{ctx['problem_yaml']}\n```")

    if ctx.get("inputs_hpp"):
        parts.append(
            "## I/O boundary (inputs.hpp) — compiled into the host driver, NOT into "
            "your kernel. NVRTC never sees this file, so nothing here is visible to a "
            "kernel unless it is a `-D` macro or an argument.\n"
            f"```cpp\n{ctx['inputs_hpp']}\n```"
        )
    if ctx.get("scalar_contract"):
        parts.append(ctx["scalar_contract"])

    strategy_text = format_strategy(ctx.get("strategy"))
    if strategy_text:
        parts.append(strategy_text)
    if ctx.get("plan"):
        parts.append(f"## Plan:\n{ctx['plan']}")

    # A sub-branch's inheritance: what its parent had reached when it split. Present
    # only on the first iteration of a spawned branch, which would otherwise start from
    # the strategy text alone and rediscover the parent's ground.
    if ctx.get("parent_context"):
        parts.append(ctx["parent_context"])

    # Reference source only on the first iteration: afterwards the branch's own kernel
    # is the thing being changed, and it is one read_file away.
    if ctx.get("ref_kernel"):
        parts.append(
            f"## Reference implementation:\n```{ctx.get('ref_language', 'cuda')}\n"
            f"{ctx['ref_kernel']}\n```"
        )

    if ctx.get("iteration_summaries"):
        parts.append(ctx["iteration_summaries"])
    if ctx.get("branch_index"):
        parts.append(ctx["branch_index"])

    # The previous iteration's analysis: what was measured, what decide concluded went
    # wrong, and the fix it asked for. Without this the step re-derives the problem from
    # raw output and tends to repeat whatever it did last time.
    if ctx.get("iteration_history"):
        parts.append(ctx["iteration_history"])

    if ctx.get("current_files"):
        parts.append(ctx["current_files"])

    if ctx.get("task"):
        parts.append(f"## This iteration:\n{ctx['task']}")
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])

    parts.append(_CLOSING)
    return system, "\n\n".join(parts)
