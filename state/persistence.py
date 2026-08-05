"""Load/save helpers for Context, BranchConfig, BranchManifest, and IterState."""

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from state.types import Context, BranchConfig, BranchManifest, IterState

logger = logging.getLogger(__name__)


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


# --- Branch config (branch_dir/branch_config.json) ---

# Last resort for a branch with no config file and no legacy manifest field.
# Reaching it means a branch runs on a budget nobody chose, which silently
# truncates it, so it is logged rather than applied quietly.
_MAX_ITER_FALLBACK = 5


def save_branch_config(branch_path: Path, config: BranchConfig):
    """Save frontend-owned branch settings to ``branch_config.json``."""
    _atomic_write_json(
        Path(branch_path) / "branch_config.json", config.model_dump()
    )


def _migrate_legacy_max_iter(branch_path: Path) -> Optional[BranchConfig]:
    """Move a pre-split ``max_iter`` out of ``branch.json`` into its own file.

    Returns the migrated config, or None if there was nothing to migrate.

    Reading the legacy field is not enough on its own: ``BranchManifest`` no
    longer declares ``max_iter``, so the next ``save_branch_manifest`` writes a
    dict without it and the value is gone for good. The migration therefore has
    to persist, and has to happen before that save — see its call site there.
    """
    branch_path = Path(branch_path)
    if (branch_path / "branch_config.json").exists():
        return None

    try:
        with (branch_path / "branch.json").open("r") as f:
            legacy = json.load(f).get("max_iter")
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(legacy, int):
        return None

    config = BranchConfig(max_iter=legacy)
    try:
        save_branch_config(branch_path, config)
    except OSError as e:
        # Read-only or vanished directory: still report the budget we found.
        logger.warning("Could not migrate max_iter for %s: %s", branch_path, e)
    return config


def load_branch_config(branch_path: Path) -> BranchConfig:
    """Load ``branch_config.json``, migrating a legacy manifest field if needed.

    Runs that started before ``max_iter`` moved out of the manifest have no
    config file, so ``branch.json`` is read and migrated — a resumed
    pre-existing run keeps the budget it was started with rather than silently
    resetting to the default.
    """
    branch_path = Path(branch_path)
    try:
        with (branch_path / "branch_config.json").open("r") as f:
            return BranchConfig.model_validate(json.load(f))
    except (OSError, json.JSONDecodeError, ValidationError):
        pass

    migrated = _migrate_legacy_max_iter(branch_path)
    if migrated is not None:
        return migrated

    logger.warning(
        "No max_iter for %s (neither branch_config.json nor a legacy manifest "
        "field) — falling back to %d, which may cut the branch short",
        branch_path,
        _MAX_ITER_FALLBACK,
    )
    return BranchConfig(max_iter=_MAX_ITER_FALLBACK)


def grant_one_more_iteration(branch_path: Path, current_iter: int) -> BranchConfig:
    """Raise ``max_iter`` just enough to let an exhausted branch run again.

    Reviving a branch that stopped because it hit its budget needs the budget
    lifted, or decide_node terminates it again immediately. A user who already
    raised the limit past ``current_iter`` keeps their value — the whole point
    of the config file being theirs to set.
    """
    config = load_branch_config(branch_path)
    if current_iter >= config.max_iter:
        config.max_iter = current_iter + 1
        save_branch_config(branch_path, config)
    return config


# --- Branch Manifest (branch_dir/branch.json) ---


def save_branch_manifest(branch_path: Path, manifest: BranchManifest):
    """Save branch manifest to ``branch.json``.

    This write is what destroys a legacy ``max_iter``: the field is no longer on
    the model, so it silently drops out of the file. Rescue it into its own file
    first — some save sites (utils/resume.py, for one) run before anything has
    read the config, so migrating only on read is too late.
    """
    branch_path = Path(branch_path)
    _migrate_legacy_max_iter(branch_path)
    _atomic_write_json(
        branch_path / "branch.json", manifest.model_dump(exclude_none=True)
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


