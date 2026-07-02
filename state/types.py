"""
State type definitions for the CUDA Agentic Optimizer.

Three cleanly separated types:
- Context: shared problem data (stored once in output/context.json)
- BranchManifest: thin branch identity and control (branch.json)
- IterState: per-iteration work products and status (iter_N/state.json)

Also retained:
- MainState: minimal state for the launcher (analyze, strategize, dispatch)
- BranchResult: summary of a completed branch
- SubStrategyDict: sub-strategy descriptor for recursive branching
"""

from typing import List, Optional
from pydantic import BaseModel, Field

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


class BranchManifest(BaseModel):
    """
    Thin branch manifest stored as ``branch.json``.

    Contains identity, control, and aggregated results.
    No work products — those live in ``iter_N/state.json``.
    """

    strategy: StrategyInfo = Field(default_factory=StrategyInfo)
    branch_depth: int = 0
    path_iters_consumed: int = (
        0  # Total iterations used by ancestors (path budget mode)
    )
    current_iter: int = 1
    max_iter: int
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
    status: str  # planning | implementing | configuring | running | profiling | proposing | deciding | decided
    next_status: Optional[str] = (
        None  # status for the *next* iteration (implementing | configuring | success | failed | branching)
    )
    plan: str = ""
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

    ``problem_yaml`` and ``ref_kernel`` are read fresh from disk by the
    worker each iteration — never persisted or cached.
    """

    problem_yaml: str = ""  # populated from source file by worker
    ref_kernel: str = ""  # populated from source file by worker
    branch_path: str = ""  # set at runtime by worker, not persisted
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

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "initialized": {"running", "failed"},
    "planning": {"implementing", "deciding"},
    "implementing": {"configuring", "deciding"},
    "configuring": {"running", "deciding"},
    "running": {"profiling", "deciding"},
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
