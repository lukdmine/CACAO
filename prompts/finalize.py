"""Prompts of the finalize step, one ``build_*`` per LLM call.

Each returns ``(system, user)``. Stage A (``build_select``) chooses the finalists and
stage B (``build_widen``) re-widens each one's space, and ``build_overview`` writes the
prose of the closing run overview.
"""

from typing import Tuple

from prompts._format import inputs_hpp_block
from prompts._portability import DEVICE_RULES, TENSOR_CORE_FALLBACK
from prompts._regions_reference import PARAMETER_GROUPS


def build_select(ctx: dict) -> Tuple[str, str]:
    """Stage A: complete the code-forced finalist set up to the cap.

    ctx: ``branches`` (pre-rendered table of every candidate), ``forced`` (names with
    the category that forced them) and ``cap`` (the most finalists allowed).
    """
    system = (
        "You choose which optimized CUDA kernels deserve a final, longer tuning run. "
        "The goal is a small set of kernels that differ in a way a user would care "
        "about (tensor cores or not, precision, minimum GPU architecture), not four "
        "near-copies of the fastest one."
    )
    user = f"""Candidate branches, best first:

{ctx["branches"]}

Already selected by rule (they stay regardless of your answer):
{ctx["forced"]}

Choose branches to add so that the total reaches at most {ctx["cap"]} finalists. Prefer
kernels that use a clearly different strategy from those already selected, and skip a
branch that is a slower variant of one already chosen. Adding nothing is acceptable.

For every branch you name (including any you want to keep from the selected list),
give a one-sentence reason. Use the branch names exactly as listed."""
    return system, user


_WIDEN_GOAL = """# Widen the tuning space of a finished kernel

The kernel in your workspace compiles and validates. Optimization narrowed its tuning
space to what suited one GPU, often to a single value or to a literal in the kernel. A
long tuning run on this GPU follows this step, and it can only search what the space
declares. Give this kernel the widest plausible space: every value that could win on
some GPU, and none that cannot.

## Kernel edits are limited

`kernels.cu` may change in two ways only:
1. Turn a literal back into a parameter: replace the number with a macro, declare it
   with `AddParameter`, and keep the literal's value in its list.
2. Add architecture guards and fallbacks: an `#if __CUDA_ARCH__ >= ...` path with a
   version that compiles on older GPUs, or a CUDA-core `#else` for a tensor-core path.

The algorithm stays as it is. Do not restructure, re-tile, reorder or optimize the
kernel. The size of your kernel diff is recorded and reported. The three region files
may change as the space needs.

## Building the space

- Keep the source's best configuration inside the space: every parameter it set is
  still declared, its value is still in the list, and no constraint excludes it. A
  space too large to exhaust is searched by MCMC, which starts from that configuration
  only if the space still contains it.
- Adding a parameter the source did not have (a literal turned back into one) leaves
  that configuration without a value for it, so MCMC then starts unseeded.
- Restore values and parameters the history shows were tried and later dropped, and
  add plausible values around the best ones.
- Merge parameter groups when the merged space fits the budget: only a merged group
  measures combinations across groups. The group rules below still hold.
- The budget is {budget} s of tuning. A space whose exhaustive estimate fits the budget
  is searched exhaustively; a larger one is searched by MCMC.

## How to work

1. Call `parameter_history` first. It lists every value each parameter had in this
   kernel's branch, the best time each value measured, and the parameters that were
   dropped (a dropped parameter is usually a literal in `kernels.cu` now).
2. Change `kernels.cu` with `edit_file` only; rewriting it is outside the limits
   above. Either tool fits the region files.
3. Run `check_compilation` until it passes. It also names any parameter you narrowed
   against the source.
4. Call `space_size` before `end_step`. It counts the configurations, estimates the
   exhaustive tuning time against the budget, and names the searcher that will run.
5. Call `end_step` with a one-line summary of what you widened. A space that is already
   as wide as plausible may end without changing anything; say so in the summary.

"""


def build_widen(ctx: dict) -> Tuple[str, str]:
    """Stage B: re-widen one finalist's tuning space within the D3 limits.

    ctx: ``current_files`` (the seeded workspace, rendered), ``budget_s`` (the tuner
    budget the space is sized against), ``inputs_hpp``, ``problem_yaml`` and
    ``rules_block``. ``scalar_contract`` (how the problem's scalars reach the kernel) is
    optional, as is ``source_iter`` (the source iteration's number), with which the
    model can read the source's measured configurations without searching.
    """
    budget = f"{float(ctx['budget_s']):.0f}"
    system = (
        _WIDEN_GOAL.replace("{budget}", budget)
        + PARAMETER_GROUPS
        + "\n"
        + DEVICE_RULES
        + "\n"
        + TENSOR_CORE_FALLBACK
    )

    parts = []
    # Rules first: they constrain everything below.
    if ctx.get("rules_block"):
        parts.append(ctx["rules_block"])
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem (problem.yaml):\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("inputs_hpp"):
        parts.append(inputs_hpp_block(ctx["inputs_hpp"]))
    if ctx.get("scalar_contract"):
        parts.append(ctx["scalar_contract"])
    if ctx.get("source_iter") is not None:
        n = ctx["source_iter"]
        parts.append(
            f"## Source\nThe files below are iteration {n} of the kernel's branch. "
            f"`tuning_landscape(iteration={n})` shows its measured configurations, "
            "best first."
        )
    if ctx.get("current_files"):
        parts.append(ctx["current_files"])
    parts.append(
        f"Widen the space for a {budget} s budget within the limits above: "
        "`parameter_history` first, `check_compilation` until it passes, `space_size`, "
        "then `end_step`."
    )
    return system, "\n\n".join(parts)


def build_overview(ctx: dict) -> Tuple[str, str]:
    """Stage E: the short prose of the run overview.

    ctx: ``facts`` (rendered facts table), ``branches`` (every branch's strategy and its
    last stated reasoning, with or without a valid result) and ``errors`` (failure
    analyses of the failed branches and of those that never validated, may be empty).
    """
    system = (
        "You write a short overview of an automated CUDA kernel optimization run for the "
        "person who started it. Be concrete and plain. Every number you write must be "
        "copied from the facts table; never compute, estimate or round a number "
        "yourself. If a number is not in the table, describe it in words instead."
    )
    errors = ctx.get("errors") or "(none)"
    user = f"""Facts (the only source of numbers):

{ctx["facts"]}

Branches, with each one's strategy and the last reasoning it recorded:

{ctx["branches"]}

Why failed branches, and branches that never validated, failed:
{errors}

Write:
- headline: one sentence naming the best result and what made it fast. If no branch
  has a valid result, say what kept them from one instead.
- findings: at most 5 short items on what worked and why.
- dead_ends: at most 3 short items on strategies that did not pay off, and why.
- caveats: at most 3 short items a reader should know before trusting the result.

Leave a list empty when you have nothing concrete to say."""
    return system, user
