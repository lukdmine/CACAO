"""Load/save helpers for Context, BranchConfig, BranchManifest, and IterState."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from state.types import Context, BranchConfig, BranchManifest, FinalizeManifest, IterState

logger = logging.getLogger(__name__)


# --- Atomic JSON write helper ---


def atomic_write_json(path: Path, data: dict):
    """Write JSON atomically via tmp+rename, leaving no temp file behind on failure.

    The shared implementation: anything writing JSON into the output tree calls this
    rather than repeating the sequence and forgetting the cleanup.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_json_if_absent(path: Path, data: dict) -> bool:
    """Create *path* atomically; return False and touch nothing if it already exists.

    ``os.link`` is the atomic form of create-if-absent: the loser of a race cannot
    half-overwrite the winner's file. Its temp name carries the pid so two processes
    creating the same target never share one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


# --- Context (output/context.json) ---


def save_context(output_dir: Path, context: Context):
    """Save shared context to ``output/context.json``. Called once after analysis."""
    atomic_write_json(output_dir / "context.json", context.model_dump())


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
            with context_file.open("r", encoding="utf-8") as f:
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
    atomic_write_json(
        Path(branch_path) / "branch_config.json", config.model_dump()
    )


def _migrate_legacy_max_iter(branch_path: Path) -> Optional[BranchConfig]:
    """Move a pre-split ``max_iter`` out of ``branch.json`` into its own file.

    Returns the migrated config, or None if there was nothing to migrate.

    Reading the legacy field is not enough on its own: ``BranchManifest`` no
    longer declares ``max_iter``, so the next ``save_branch_manifest`` writes a
    dict without it and the value is gone for good. The migration therefore has
    to persist, and has to happen before that save — see its call site there.

    It creates the file and never overwrites one: this is the single write the
    worker process makes to a file the API otherwise owns, so it must not be able
    to land on top of a budget the user just set.
    """
    branch_path = Path(branch_path)
    if (branch_path / "branch_config.json").exists():
        return None

    try:
        with (branch_path / "branch.json").open("r", encoding="utf-8") as f:
            legacy = json.load(f).get("max_iter")
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(legacy, int):
        return None

    config = BranchConfig(max_iter=legacy)
    try:
        _write_json_if_absent(
            branch_path / "branch_config.json", config.model_dump()
        )
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
        with (branch_path / "branch_config.json").open("r", encoding="utf-8") as f:
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

    Call it from the server process only: ``branch_config.json`` has one writer,
    and the worker revert/continue paths reach it through a signal the API grants
    the budget for before sending.
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
    first — some save sites (engine/resume.py, for one) run before anything has
    read the config, so migrating only on read is too late.
    """
    branch_path = Path(branch_path)
    _migrate_legacy_max_iter(branch_path)
    atomic_write_json(
        branch_path / "branch.json", manifest.model_dump(exclude_none=True)
    )


def load_branch_manifest(branch_path: Path) -> BranchManifest:
    """Load branch manifest from ``branch.json``."""
    with (Path(branch_path) / "branch.json").open("r", encoding="utf-8") as f:
        return BranchManifest.model_validate(json.load(f))


# --- Iteration State (branch_dir/iter_N/state.json) ---


def save_iter_state(branch_path: Path, iter_num: int, state: IterState):
    """Save iteration state to ``iter_N/state.json``."""
    iter_dir = Path(branch_path) / f"iter{iter_num}"
    atomic_write_json(iter_dir / "state.json", state.dump_for_disk())


def load_iter_state(
    branch_path: Path, iter_num: int, optional: bool = False
) -> Optional[IterState]:
    """Load iteration state from ``iter_N/state.json``. Returns None if optional and missing."""
    state_file = Path(branch_path) / f"iter{iter_num}" / "state.json"
    if optional and not state_file.exists():
        return None
    with state_file.open("r", encoding="utf-8") as f:
        return IterState.model_validate(json.load(f))


def load_iter_state_if_exists(branch_path: Path, iter_num: int) -> Optional[IterState]:
    """Alias for ``load_iter_state(..., optional=True)``."""
    return load_iter_state(branch_path, iter_num, optional=True)


def create_initial_iter_state(
    iter_num: int, prev_state: Optional[IterState] = None
) -> IterState:
    """Create the next iteration's state. It owns only what it will produce itself.

    Previous-iteration context is loaded per-field from disk by state/history.py, not
    copied forward. Only ``mode`` has to travel, because history returns formatted
    markdown rather than a value to branch on.
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
        # decide still emits "configuring" for skip_implement; the merged authoring step
        # reads the same intent from here, so that prompt and its schema stay untouched.
        authoring_scope="config_only" if status == "configuring" else "full",
    )




# --- Finalize manifest (output_dir/finalize/finalize.json) ---


def finalize_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "finalize"


def save_finalize_manifest(output_dir: Path, manifest: FinalizeManifest) -> None:
    atomic_write_json(finalize_dir(output_dir) / "finalize.json", manifest.model_dump())


def load_finalize_manifest(output_dir: Path) -> Optional[FinalizeManifest]:
    """None when absent, and None plus a WARN when the file does not parse."""
    path = finalize_dir(output_dir) / "finalize.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return FinalizeManifest.model_validate(json.load(f))
    except (OSError, ValueError, ValidationError) as e:
        logger.warning("WARN: could not read %s: %s", path, e)
        return None
