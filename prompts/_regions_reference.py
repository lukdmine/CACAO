"""The region contract shared by both authoring paths.

Two rules the agentic author prompt and the legacy configure prompt each have to teach,
kept in one place because both failures are invisible to the compiler: a base block size
left at the default launches one thread per block, and a constraint split across parameter
groups is dropped without a warning.
"""

BASE_BLOCK_SIZE = """**The 4th `AddKernelDefinitionFromFile` argument is the BASE block size.**
`ktt::DimensionVector()` is **(1,1,1)**, NOT "auto" — either pass the literal block size, or
scale the base with `ModifierType::Local` modifiers in every dimension the block uses.
Neither, and every launch gets 1 thread per block.
"""

PARAMETER_GROUPS = """**Parameter groups (4th `AddParameter` arg, default `""`).** KTT
enumerates each group SEPARATELY and the per-group counts ADD instead of multiplying, holding
every other group at the best configuration found so far. Two stages of 3 params x 4 values
cost 64+64 grouped vs 4096 ungrouped; the gap is what makes a multi-kernel pipeline tunable at
all. So: parameters that affect only ONE definition belong in that definition's own group.
Parameters that must agree ACROSS definitions — a tile size two stages share, one shared-memory
budget they both draw on — belong together in ONE group. A single-kernel pipeline wants no
groups: leave the argument off and everything lands in the default group.

Every parameter macro the kernel uses must be declared in the params region. HARD RULE: every
parameter named by an `AddConstraint` must sit in the SAME group. KTT evaluates a constraint
only once a group holds ALL of its parameters, so one split across groups is dropped SILENTLY
— no warning, and the illegal configurations it was meant to exclude reach the GPU and fail at
launch.
"""
