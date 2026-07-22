"""Load/save helpers for Context, BranchManifest, and IterState."""

import json
from pathlib import Path
from typing import Optional

from state.types import Context, BranchManifest, IterState


# --- Atomic JSON write helper ---


def _atomic_write_json(path: Path, data: dict):
    """Write JSON atomically via tmp+rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --- Context (output/context.json) ---


def save_context(output_dir: Path, context: Context):
    """Save shared context to ``output/context.json``. Called once after analysis."""
    _atomic_write_json(output_dir / "context.json", context.model_dump())


def load_context_for_branch(branch_path: Path) -> Context:
    """
    Load context by walking up from a branch path to find output/context.json.

    Handles both top-level branches (output/branches/X) and sub-branches
    (output/branches/X/branches/Y) by searching upward for context.json.
    """
    path = Path(branch_path)
    while path != path.parent:
        context_file = path / "context.json"
        if context_file.exists():
            with context_file.open("r") as f:
                return Context.model_validate(json.load(f))
        path = path.parent
    raise FileNotFoundError(f"No context.json found above {branch_path}")


# --- Branch Manifest (branch_dir/branch.json) ---


def save_branch_manifest(branch_path: Path, manifest: BranchManifest):
    """Save branch manifest to ``branch.json``."""
    _atomic_write_json(
        Path(branch_path) / "branch.json", manifest.model_dump(exclude_none=True)
    )


def load_branch_manifest(branch_path: Path) -> BranchManifest:
    """Load branch manifest from ``branch.json``."""
    with (Path(branch_path) / "branch.json").open("r") as f:
        return BranchManifest.model_validate(json.load(f))


# --- Iteration State (branch_dir/iter_N/state.json) ---


def save_iter_state(branch_path: Path, iter_num: int, state: IterState):
    """Save iteration state to ``iter_N/state.json``."""
    iter_dir = Path(branch_path) / f"iter{iter_num}"
    _atomic_write_json(iter_dir / "state.json", state.dump_for_disk())


def load_iter_state(
    branch_path: Path, iter_num: int, optional: bool = False
) -> Optional[IterState]:
    """Load iteration state from ``iter_N/state.json``. Returns None if optional and missing."""
    state_file = Path(branch_path) / f"iter{iter_num}" / "state.json"
    if optional and not state_file.exists():
        return None
    with state_file.open("r") as f:
        return IterState.model_validate(json.load(f))


def load_iter_state_if_exists(branch_path: Path, iter_num: int) -> Optional[IterState]:
    """Alias for ``load_iter_state(..., optional=True)``."""
    return load_iter_state(branch_path, iter_num, optional=True)


def create_initial_iter_state(
    iter_num: int, prev_state: Optional[IterState] = None
) -> IterState:
    """Create the next iteration's state. It owns only what it will produce itself.

    Previous-iteration context (the last kernel, feedback, decision, run output) is NOT
    copied: state/history.py already loads it from disk per-field with configurable
    depth, and every consuming node already requests it. Copying it forward duplicated
    that — the same kernel reached the implement prompt twice — and left the frontend
    rendering the previous iteration's decision as this one's.

    What cannot come from history is a value to branch on, since history returns
    formatted markdown: hence ``mode``, computed once here from the previous decision.
    """
    if prev_state is None:
        return IterState(iter_num=iter_num, status="planning")

    status = prev_state.next_status if prev_state.next_status else "implementing"
    action = (prev_state.decision or {}).get("action")
    if action == "retry":
        mode = "retry"
    elif prev_state.feedback:
        mode = "followup"
    else:
        mode = "fresh"

    return IterState(
        iter_num=iter_num,
        status=status,
        mode=mode,
        # skip_implement routes straight to configuring, so no node will write a kernel
        # this iteration: the previous one IS this iteration's kernel. Adoption, not
        # carried context.
        kernel_code=prev_state.kernel_code if status == "configuring" else "",
    )


