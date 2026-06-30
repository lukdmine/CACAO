"""
State management for the CUDA Agentic Optimizer.

Provides:
  - **types** – ``MainState``, ``Context``, ``BranchManifest``, ``IterState``, ``BranchResult``, ``SubStrategyDict``
  - **persistence** – save/load for context, branch manifests, and iteration states
  - **control** – file-based control signals, re-queue, and branch revert
  - **history** – LLM prompt formatting for iteration history and parent context
"""

# --- Types ---
from state.types import (
    MainState,
    Context,
    BranchManifest,
    IterState,
    BranchResult,
    SubStrategyDict,
    StrategyInfo,
    WorkingState,
    validate_transition,
)

# --- Persistence ---
from state.persistence import (
    save_context,
    load_context_for_branch,
    save_branch_manifest,
    load_branch_manifest,
    save_iter_state,
    load_iter_state,
    load_iter_state_if_exists,
    create_initial_iter_state,
)

# --- Control ---
from state.control import (
    read_control_signal,
    write_control_signal,
    write_requeue,
    read_requeue,
    revert_branch_on_disk,
    continue_branch_on_disk,
)

# --- History / formatting ---
from state.history import (
    format_user_messages,
    format_iteration_history,
    format_iteration_summaries,
    format_existing_branches,
    format_parent_context,
    format_best_so_far,
)

__all__ = [
    # Types
    "MainState",
    "Context",
    "BranchManifest",
    "IterState",
    "BranchResult",
    "SubStrategyDict",
    "StrategyInfo",
    "WorkingState",
    "validate_transition",
    # Persistence
    "save_context",
    "load_context_for_branch",
    "save_branch_manifest",
    "load_branch_manifest",
    "save_iter_state",
    "load_iter_state",
    "load_iter_state_if_exists",
    "create_initial_iter_state",
    # Control
    "read_control_signal",
    "write_control_signal",
    "write_requeue",
    "read_requeue",
    "revert_branch_on_disk",
    "continue_branch_on_disk",
    # History
    "format_user_messages",
    "format_iteration_history",
    "format_iteration_summaries",
    "format_existing_branches",
    "format_parent_context",
    "format_best_so_far",
]
