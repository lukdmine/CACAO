"""The three policies propose and decide have to agree on.

They read the same iteration log and their conclusions must match: a proposal that declares
the approach exhausted is worthless if decide's own copy of the threshold still reads
"continue", and the branch then spends its remaining budget on variations the proposal
already ruled out. Each node keeps only the sentence naming its own action.
"""

PLATEAU_RULE = (
    "If the last 3+ successful iterations show <5% improvement in best time, the current "
    "approach has **plateaued**."
)

ANTI_REPETITION = """## Repeated Approaches

Review the iteration log and classify every past failure:
- **Fundamentally slow or impossible** — implemented correctly but no speedup, or it hits a
  hard hardware limit. That approach is finished.
- **An implementation bug** — a compilation error, or a validation failure from wrong
  indexing or an off-by-one. Trying it again IS valid, but only with the specific bug
  identified and the concrete fix described. Never re-run an approach hoping it works, and do
  not fall back to a slower working kernel just because the faster one had a fixable bug.
"""

LAST_CHANCE_RULE = (
    "**When branching is NOT AVAILABLE (see Branching Status) and no iteration has yet "
    "produced compiling AND validating code, ending this path is not an option.** With no "
    "branching budget left this branch is the last chance for it: treat the failure as a "
    "debug task with concrete fixes, not as a result."
)
