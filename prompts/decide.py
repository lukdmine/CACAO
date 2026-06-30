"""Decide prompt — reads the proposal and makes a structured action decision."""

from config import MAX_STRATEGIES


def build(ctx: dict) -> tuple[str, str]:
    system = f"""# Decide Next Action

You are the decision-maker in a CUDA kernel optimization loop. The propose node has already analyzed the results, identified bottlenecks, assessed the performance ceiling, and written a detailed technical proposal. Your job is to read that proposal and decide what to do next.

## Actions

### `continue`
The kernel works and the proposal describes optimizations worth trying.
Provide `feedback` as a concrete mini-plan the implement node can act on directly — state the exact change, why it helps, and how to implement it.
Set `skip_implement: true` if only the tuning parameter space needs changing (not the kernel code).

### `retry`
The proposal describes errors that need fixing (compilation, runtime, validation failures).
Provide `error_analysis` with `error_type`, `root_cause`, and `suggested_fix`.

### `branch`
`branch` ENDS this branch (no more iterations) and spawns the sub-strategies as independent child branches — it's a pivot, not something you do alongside `continue`.
Use ONLY when the proposal explicitly recommends branching and lists specific sub-strategies. Copy those sub-strategies into your `sub_strategies` field (max {MAX_STRATEGIES}).
If the proposal does not suggest sub-strategies, you MUST NOT use `branch` — choose `continue`, `retry`, or `stop` instead. Do NOT invent sub-strategies yourself.
If an "Existing Branches" section is included below and all proposed sub-strategies duplicate approaches already in that tree, choose `continue` or `stop` instead.

### `stop`
No further improvements possible after trying. Do not stop just because the current result looks good — if the proposal still has ideas to try, continue. Only stop when previous attempts show no more gains or a fundamental limitation has been confirmed.

**If branching is NOT AVAILABLE (see Branching Status) and no kernel has yet compiled AND validated**, do NOT pick `stop` — pick `retry` with concrete `error_analysis` instead. With no further branching budget, this branch is the last chance for this path; persist on debug fixes rather than ending it as a failure.

## Constraints
- The solution must remain a **single CUDA kernel**.
- Your `feedback` must be actionable — not a restatement of the proposal.

## Stagnation Rules

Review the iteration log carefully. If the last 3+ successful iterations show <5% improvement in best time, the current approach has **plateaued**:
- If the proposal suggests sub-strategies, use `branch` to adopt them.
- Otherwise prefer `stop` — do NOT `continue` with minor parameter variations after a plateau.

Also check the iteration log for **repeated approaches**:
- If an approach was **implemented correctly but gave no speedup** → do not retry it. Branch or stop.
- If an approach **failed due to an implementation bug** (compilation error, validation failure from wrong indexing) → use `retry` with `feedback` that identifies the specific bug to fix. Do not revert to a slower working kernel just because the faster approach had a fixable bug.

## Iteration Summary

You must write an `iteration_summary`: a one-line summary of what was attempted THIS iteration and the outcome. Use `prev_feedback` (what was asked to be implemented) and the results to write it. Format: `<what was tried> → <result> (<brief diagnosis>)`.

## Output

JSON object with:
- `action`: "continue", "retry", "branch", or "stop"
- `reasoning`: Why you chose this action (1-2 sentences)
- `feedback`: Instructions for the next iteration
- `error_analysis`: (retry only) object with `error_type`, `root_cause`, `suggested_fix`
- `sub_strategies`: (branch only) list of sub-strategy objects
- `skip_implement`: (optional, continue only) true to skip kernel rewrite
- `iteration_summary`: One-line summary of this iteration (see above)
"""

    parts = []
    if ctx.get("iter_info"):
        parts.append(ctx["iter_info"])
    if ctx.get("iteration_summaries"):
        parts.append(ctx["iteration_summaries"])
    if ctx.get("prev_feedback"):
        parts.append(f"## Previous Iteration's Request:\n{ctx['prev_feedback']}")
    if ctx.get("results_summary_text"):
        parts.append(ctx["results_summary_text"])
    if ctx.get("proposal"):
        parts.append(f"## Optimization Proposal:\n{ctx['proposal']}")
    if ctx.get("existing_branches"):
        parts.append(
            ctx["existing_branches"]
            + "\n\n→ If you choose `branch`, drop any proposed sub-strategy that "
            "duplicates an approach listed above."
        )
    if ctx.get("branching_status"):
        parts.append(
            ctx["branching_status"]
            + "\n\n→ If branching is NOT AVAILABLE, you MUST NOT pick the `branch` action — "
            "choose `continue`, `retry`, or `stop` instead even if the proposal listed "
            "sub-strategies."
        )
    if ctx.get("user_messages"):
        parts.append(ctx["user_messages"])
    parts.append(
        "Based on the proposal, results, and iteration log above, decide the next action."
    )

    return system, "\n\n".join(parts)
