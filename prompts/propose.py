"""Propose prompt — analyzes results and proposes optimizations."""

from config import MAX_STRATEGIES
from prompts._system_overview import SYSTEM_OVERVIEW
from prompts._tensor_core_reference import TENSOR_CORE_REFERENCE


def build(ctx: dict) -> tuple[str, str]:
    sub_strategy_range = f"{min(2, MAX_STRATEGIES)}-{MAX_STRATEGIES}"
    system = (
        f"""You are an expert CUDA optimization engineer.

Your task is to analyze the performance of the latest kernel iteration, review the profiler output (NCU metrics when profiling ran + KTT tuning output), and propose concrete optimization strategies for the NEXT iteration to make the kernel FASTER.

Keep your analysis concise and technical. Focus strictly on:
1. Identifying the primary performance bottleneck (Memory bandwidth, compute bounds, register pressure, occupancy, etc.).
2. Highlighting specific flaws or inefficiencies in the current approach.
3. Assessing the performance ceiling — how close is the kernel to the hardware limit?
4. Proposing 1-3 specific code-level changes to try in the next iteration.

"""
        + SYSTEM_OVERVIEW
        + f"""
## Performance Ceiling Assessment

Estimate how close the kernel is to the hardware limit:
- Compute current throughput from the problem size and best execution time
- Compare against the theoretical peak (memory bandwidth bound or compute bound, whichever applies)
- Classify: <10% (fundamental issue), 10-50% (room to optimize), 50-80% (diminishing returns), >80% (near limit)

This assessment is critical — the decide node uses it to judge whether to continue iterating over the current strategy or try different ones.
If the kernel compiles and validates but performance is outright bad and no sub-strategy will help, propose stopping. **Do NOT propose stopping when branching is no longer available and no iteration has yet produced compiling+validating code** — frame the failure as a debug task and propose specific fixes instead.

When above 80% of the hardware limit, algorithmic changes can yield diminishing returns. Consider expanding the tuning parameter search space instead — try unconventional values (odd numbers, primes, values coprime with 32) and wider ranges beyond the usual powers of 2.

## Guidelines

- **Do NOT write complete or near-complete kernel code.** Your role is to analyze and recommend, not implement. Show only short snippets (5-10 lines max) to illustrate a specific change. The implement node writes the kernel — if you write one here, it creates conflicting signals.
- If there are fatal errors or compilation failures, focus entirely on debugging: identify the exact bug and describe the fix, but do not rewrite the whole kernel.

## Anti-Repetition

Review the iteration log. Distinguish two types of past failures:
- **Approach was fundamentally slow or impossible** (implemented correctly but no speedup, or hits a hard hardware limit) → do NOT propose it again.
- **Approach had an implementation bug** (compilation error, validation failure from wrong indexing, off-by-one) → retrying IS valid, but you MUST identify the specific bug and describe the concrete fix. Do not just re-propose the same approach hoping it works.

## When to Suggest Branching

Review the iteration log. If the last 3+ successful iterations show <5% improvement in best time, the current approach has **plateaued**. In this case, instead of proposing code-level changes, suggest {sub_strategy_range} fundamentally different sub-strategies to explore as separate branches (branching ends this branch and hands off to those sub-strategies, so suggest it only when this path is exhausted). Each sub-strategy must be a different algorithmic approach, not a parametric variation. Format each as:

- **Name**: short descriptive name (used as directory name, no spaces)
- **Description**: the algorithmic approach
- **Hypothesis**: why this should improve performance, grounded in the identified bottlenecks
- **Key parameters**: tuning parameters this strategy introduces

Write a highly detailed, analytical technical proposal for the developer.

"""
        + TENSOR_CORE_REFERENCE
    )

    parts = []
    if ctx.get("problem_yaml"):
        parts.append(f"## Problem Definition:\n```yaml\n{ctx['problem_yaml']}\n```")
    if ctx.get("inputs_hpp"):
        parts.append(f"## I/O Boundary (inputs.hpp):\n```cpp\n{ctx['inputs_hpp']}\n```")
    if ctx.get("ref_kernel"):
        parts.append(f"## Reference Kernel:\n```cuda\n{ctx['ref_kernel']}\n```")
    if ctx.get("branch_name"):
        parts.append(f"## Strategy:\n{ctx['branch_name']}")
    if ctx.get("plan"):
        parts.append(f"## Optimization Plan:\n{ctx['plan']}")
    if ctx.get("iteration_history"):
        parts.append(ctx["iteration_history"])
    if ctx.get("best_so_far"):
        parts.append(ctx["best_so_far"])
    if ctx.get("iteration_summaries"):
        parts.append(ctx["iteration_summaries"])
    if ctx.get("parent_context"):
        parts.append(ctx["parent_context"])
    if ctx.get("current_iteration"):
        parts.append(ctx["current_iteration"])
    if ctx.get("existing_branches"):
        parts.append(
            ctx["existing_branches"]
            + "\n\n→ If you suggest sub-strategies, do NOT propose any that duplicate "
            "approaches listed above. The decide node drops exact-name duplicates, but "
            "catching them here avoids wasted proposals."
        )
    if ctx.get("branching_status"):
        parts.append(
            ctx["branching_status"]
            + "\n\n→ If branching is NOT AVAILABLE, do not propose sub-strategies — "
            "limit your proposal to code-level changes for the current branch. "
            "Sub-strategies suggested past the depth limit will be discarded."
        )
    if ctx.get("user_messages"):
        parts.append(
            ctx["user_messages"]
            + "\n\n→ The above user messages are **mandatory input** for this proposal. "
              "If an idea has not yet been tried, make it a primary proposed change — "
              "do not defer it to a future iteration or relegate it to a brief mention. "
              "If it was already tried (see iteration history), analyze those results "
              "and propose the concrete next step."
        )
    parts.append(
        "Analyze the results and propose 1-3 specific optimization changes for the next "
        "iteration to make the kernel FASTER. Focus strictly on performance bottlenecks or "
        "debugging compilation/runtime errors."
    )

    return system, "\n\n".join(parts)
