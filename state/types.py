"""
State type definitions for the CUDA Agentic Optimizer.

Three cleanly separated types:
- Context: shared problem data (stored once in output/context.json)
- BranchManifest: thin branch identity and control (branch.json)
- IterState: per-iteration work products and status (iter_N/state.json)

Plus one input type:
- BranchConfig: frontend-owned knobs the worker only reads (branch_config.json)

Also retained:
- MainState: minimal state for the launcher (analyze, strategize, dispatch)
- BranchResult: summary of a completed branch
- SubStrategyDict: sub-strategy descriptor for recursive branching
"""

from typing import Dict, List, Literal, Optional, get_args
from pydantic import BaseModel, Field

from models.finalize import ArchReport, KernelTags

# Pydantic models for core state


class MainState(BaseModel):
    """
    Minimal state for the main launcher.

    Handles the initial linear flow:
      1. analysis  – Analyze the problem
      2. strategies – Generate optimization strategies
    """

    problem_yaml: str
    ref_kernel: str
    analysis: str = ""
    strategies: List[dict] = Field(default_factory=list)


class Context(BaseModel):
    """
    Shared problem context, stored once in ``output/context.json``.

    Written during the analysis phase and read by all branches/nodes.
    Source files (problem.yaml, reference kernel) are always read fresh
    from disk by the worker — they are NOT cached here.
    """

    analysis: str
    gpu_info: Optional[dict] = None  # cached auto-detected GPU specs


class StrategyInfo(BaseModel):
    """Strategy descriptor stored in branch manifests."""

    name: str = "unknown"
    description: str = ""
    hypothesis: str = ""
    key_parameters: List[str] = Field(default_factory=list)


class BranchConfig(BaseModel):
    """Frontend-owned branch settings (``branch_config.json``).

    Its own file because the API writes it and the worker only reads it — one writer
    per file. The single worker touch is the legacy ``max_iter`` migration, which
    creates the file when it is absent and never overwrites one.
    """

    max_iter: int


class BranchManifest(BaseModel):
    """
    Thin branch manifest stored as ``branch.json``.

    Contains identity, control, and aggregated results — everything here is
    produced by the worker, which is its only writer. Settings the user edits
    live in ``BranchConfig`` instead.
    No work products — those live in ``iter_N/state.json``.
    """

    strategy: StrategyInfo = Field(default_factory=StrategyInfo)
    # Branch-level: nodes/plan.py runs once per branch, so every iteration of a branch
    # shares one plan. It lived on IterState and was stored identically in each
    # iter_N/state.json.
    plan: str = ""
    branch_depth: int = 0
    path_iters_consumed: int = (
        0  # Total iterations used by ancestors (path budget mode)
    )
    # Path budget this root-to-leaf path runs on; 0 means use config.PATH_BUDGET.
    # Grows by whatever an ancestor was granted beyond its allocated share.
    path_budget_total: int = 0
    current_iter: int = 1
    status: str = (
        "initialized"  # initialized | running | success | failed | branching | stopped
    )
    best_time_us: Optional[float] = None
    speedup: Optional[float] = None
    sub_strategies_cache: Optional[List[dict]] = None
    pre_stop_status: Optional[str] = None  # Saved status before stop, for resuming


class IterState(BaseModel):
    """
    Per-iteration state stored in ``iter_N/state.json``.

    Each iteration owns all of its work products, decision, and user messages.
    Updated after every node step within the iteration.
    """

    iter_num: int
    status: str  # planning | authoring | implementing | configuring | running | profiling | proposing | deciding | decided
    next_status: Optional[str] = (
        None  # status for the *next* iteration (implementing | configuring | success | failed | branching)
    )
    # What the authoring step may change; "config_only" is the agentic form of
    # skip_implement.
    authoring_scope: Literal["full", "config_only"] = "full"
    # How this iteration was entered; set by create_initial_iter_state from the previous
    # decision.
    mode: Literal["fresh", "retry", "followup"] = "fresh"
    kernel_code: str = ""
    framework_cpp: str = ""  # assembled framework.cpp driver (framework-file mode)
    run_output: str = ""
    ncu_metrics: Optional[dict] = None
    decision: Optional[dict] = None
    feedback: str = ""
    results_summary: Optional[dict] = None
    user_messages: List[dict] = Field(default_factory=list)
    proposal: str = ""

    def dump_for_disk(self) -> dict:
        """Dump the state for saving to disk, excluding None values to keep JSON clean."""
        return self.model_dump(exclude_none=True)


class WorkingState(Context, BranchManifest, IterState):
    """
    Unified view composed of Context, BranchManifest, and IterState.
    Used exclusively as the input/output type for optimization nodes.

    ``problem_yaml``, ``ref_kernel`` and ``max_iter`` are read fresh from disk
    by the worker each iteration — never persisted or cached. ``max_iter`` has
    no default on purpose: a compose that forgot to supply it should fail
    validation loudly rather than hand the nodes a budget of zero.
    """

    problem_yaml: str = ""  # populated from source file by worker
    ref_kernel: str = ""  # populated from source file by worker
    branch_path: str = ""  # set at runtime by worker, not persisted
    max_iter: int  # read from branch_config.json by worker, not persisted here
    sub_strategies: Optional[List[dict]] = None  # passed from decide node to master


class BranchResult(BaseModel):
    """Summary of a completed branch."""

    branch_name: str
    strategy: dict
    best_config: Optional[dict] = None
    best_time_us: Optional[float] = None
    speedup: Optional[float] = None
    iterations: int
    status: str  # success | failed | stopped


class SubStrategyDict(BaseModel):
    """Sub-strategy descriptor for recursive branching (dict form for state serialization)."""

    name: str
    description: str
    hypothesis: str
    key_parameters: List[str]


# -------------------------------------------------------------------------
# Status transition enforcement
# -------------------------------------------------------------------------

# The whole status vocabulary in one place: the iteration FSM, the branch-level
# statuses no iteration ever holds, and "idle" (api/tree.py, for a problem with an
# output directory but no branches yet). The transition table below lists only the
# statuses a node runs from, so it is not the set to read this off. The frontend keeps
# two more hand-written copies (generate_types.py's BranchStatus, statusColors.ts) which
# have already drifted from it and from each other; they should be generated from here.
BRANCH_STATUSES: tuple[str, ...] = (
    "idle",
    "initialized",
    "planning",
    "authoring",
    "implementing",
    "configuring",
    "running",
    "profiling",
    "proposing",
    "deciding",
    "decided",
    "success",
    "failed",
    "branching",
    "stopped",
)

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "initialized": {"running", "failed"},
    # "implementing" and "configuring" stay valid alongside "authoring": output
    # directories written before the merge resume from whichever status they stopped at,
    # and AGENTIC_STEPS=False runs those two nodes, which set them.
    "planning": {"authoring", "implementing", "deciding"},
    # proposing: the step ended without a passing compile check, so the tuner run is
    # skipped and the failure goes straight to analysis — the same route run.py takes
    # for a host-compile failure.
    "authoring": {"running", "proposing", "deciding"},
    "implementing": {"configuring", "deciding"},
    "configuring": {"running", "deciding"},
    "running": {"profiling", "proposing", "deciding"},
    "profiling": {"proposing", "deciding"},
    "proposing": {"deciding"},
    "deciding": {"decided", "success", "failed", "branching"},
}


def validate_transition(old_status: str, new_status: str):
    """Raise ValueError if the status transition is not allowed."""
    allowed = _ALLOWED_TRANSITIONS.get(old_status)
    if allowed is None:
        return  # Terminal/unknown status — skip validation
    if new_status not in allowed:
        raise ValueError(
            f"Invalid status transition: '{old_status}' -> '{new_status}'. "
            f"Allowed: {allowed}"
        )


# --- Finalize (output/finalize/finalize.json) ---

FinalizeStatus = Literal[
    "selecting", "widening", "tuning", "verifying", "overview", "done", "failed"
]
FINALIZE_STATUSES = get_args(FinalizeStatus)
STAGE_STATUSES = ("pending", "done", "failed", "skipped")


class Variant(BaseModel):
    """One kernel version of a finalist: the branch's own best, or its finalized rework."""

    dir: str  # relative to output/
    summary: Optional[dict] = None  # aggregate_case_summaries shape
    tags: Optional[KernelTags] = None
    arch: Optional[ArchReport] = None
    ncu_metrics: Optional[dict] = None
    # The searcher a finalized variant was tuned with: "exhaustive", "mcmc (seeded)",
    # "mcmc (unseeded)" or, when only some cases had a seed, the cases that did. Empty
    # on a source variant, which its branch tuned with the random searcher.
    searcher: str = ""


class FinalistState(BaseModel):
    name: str
    source_branch: str  # relative to output/
    source_iter: int
    reason: str = ""
    forced: bool = False
    stages: Dict[str, str] = Field(
        default_factory=lambda: {"widen": "pending", "tune": "pending", "verify": "pending"}
    )
    source: Variant
    finalized: Optional[Variant] = None
    reported: Literal["source", "finalized"] = "source"
    reported_dir: str = ""  # == source.dir or finalized.dir
    kernel_diff_lines: Optional[int] = None
    error: str = ""


class OverviewFact(BaseModel):
    label: str
    value: str


class Overview(BaseModel):
    headline: str
    findings: List[str] = Field(default_factory=list)
    dead_ends: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    facts: List[OverviewFact] = Field(default_factory=list)
    markdown: str = ""


class FinalizeManifest(BaseModel):
    """Written only by the finalize process; the API reads it."""

    status: FinalizeStatus = "selecting"
    attempt: int = 1
    trigger: Literal["auto", "manual"] = "auto"
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    # The finalize process, so the API can tell a live pass from the leftovers of a killed
    # one (auto-finalize runs inside the normal run, so the registry cannot say).
    pid: Optional[int] = None
    source_fingerprint: str = ""
    finalists: List[FinalistState] = Field(default_factory=list)
    category_bests: Dict[str, str] = Field(default_factory=dict)  # "tensor_cores=true" -> finalist name
    overview: Optional[Overview] = None
