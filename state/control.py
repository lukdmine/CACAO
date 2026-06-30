"""
Control signals and re-queue helpers for branch orchestration.

Provides the file-based communication channel between the server/master
and running worker coroutines:
  - stop / resume / revert / message / config signals
  - file-based re-queue for reverted branches
  - disk surgery for reverting a branch to a prior iteration
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from state.persistence import (
    load_branch_manifest,
    save_branch_manifest,
    load_iter_state,
    save_iter_state,
)

import logging

logger = logging.getLogger(__name__)


def _inject_user_message(iter_state, message: str):
    """Append a user message and reset iter status to deciding."""
    if not message:
        return
    msgs = list(iter_state.user_messages or [])
    msgs.append({"content": message, "timestamp": datetime.now().isoformat()})
    iter_state.user_messages = msgs
    iter_state.status = "deciding"


def _stop_sub_branches(branch_path: Path):
    """Signal stop to all sub-branches and remove the branches directory."""
    branches_dir = branch_path / "branches"
    if not branches_dir.is_dir():
        return
    for branch_file in branches_dir.rglob("branch.json"):
        write_control_signal(branch_file.parent, {"action": "stop"})
    shutil.rmtree(branches_dir, ignore_errors=True)


def read_control_signal(branch_path: Path) -> Optional[dict]:
    """Read and consume a ``control.json`` signal from a branch directory.

    Uses atomic rename to prevent lost signals: if the rename succeeds,
    this caller owns the file and no concurrent writer can interfere.
    """
    control_file = branch_path / "control.json"
    consumed = branch_path / f".control_consumed_{os.getpid()}.json"
    try:
        os.rename(control_file, consumed)
    except (FileNotFoundError, OSError):
        return None
    try:
        with consumed.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    finally:
        consumed.unlink(missing_ok=True)


def write_control_signal(branch_path: Path, signal: dict):
    """Write a control signal to a branch directory (atomic via tmp+rename)."""
    branch_path.mkdir(parents=True, exist_ok=True)
    tmp = branch_path / "control.json.tmp"
    final = branch_path / "control.json"
    with tmp.open("w") as f:
        json.dump(signal, f)
    tmp.replace(final)


# -------------------------------------------------------------------------
# File-based re-queue
# -------------------------------------------------------------------------


def write_requeue(output_dir: Path, branch_path: Path):
    """Drop a re-queue file so a worker picks up this branch."""
    requeue_dir = output_dir / "requeue"
    requeue_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{branch_path.name}_{int(time.time() * 1000)}.json"
    tmp = requeue_dir / f"{filename}.tmp"
    final = requeue_dir / filename
    with tmp.open("w") as f:
        json.dump({"branch_path": str(branch_path)}, f)
    tmp.replace(final)


def read_requeue(output_dir: Path) -> List[Path]:
    """Consume all re-queue files and return the branch paths."""
    requeue_dir = output_dir / "requeue"
    if not requeue_dir.is_dir():
        return []
    paths = []
    for f in sorted(requeue_dir.iterdir()):
        if f.suffix == ".json":
            try:
                with f.open("r") as fh:
                    data = json.load(fh)
                paths.append(Path(data["branch_path"]))
                f.unlink()
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("Malformed requeue file %s: %s — removing", f.name, e)
                f.unlink(missing_ok=True)
    return paths


# -------------------------------------------------------------------------
# Revert utility
# -------------------------------------------------------------------------


def revert_branch_on_disk(
    branch_path: Path,
    target_iter: int,
    message: Optional[str] = None,
) -> dict:
    """
    Revert a branch to the state after *target_iter*.

    Restores branch manifest to point at target_iter, deletes later
    iteration directories and sub-branches, and returns the manifest dict.
    """
    branch_path = Path(branch_path)
    manifest = load_branch_manifest(branch_path)
    iter_state = load_iter_state(branch_path, target_iter)

    # Update branch manifest
    manifest.current_iter = target_iter
    manifest.status = "running"
    manifest.pre_stop_status = None

    if manifest.current_iter >= manifest.max_iter:
        manifest.max_iter = manifest.current_iter + 1

    _inject_user_message(iter_state, message)
    # Re-run propose → decide so the agent reconsiders with fresh analysis
    iter_state.status = "proposing"
    iter_state.proposal = ""
    save_iter_state(branch_path, target_iter, iter_state)

    _stop_sub_branches(branch_path)

    # Delete iteration directories after target
    for d in branch_path.iterdir():
        if d.is_dir() and d.name.startswith("iter"):
            try:
                n = int(d.name[4:])
                if n > target_iter:
                    shutil.rmtree(d)
            except ValueError:
                pass

    save_branch_manifest(branch_path, manifest)
    return manifest.model_dump()


def continue_branch_on_disk(
    branch_path: Path,
    message: Optional[str] = None,
) -> dict:
    """
    Re-open a dead branch (success/failed/branching) so it continues
    from its current iteration without deleting any work.

    Sets status to ``running``, bumps ``max_iter`` if needed, and
    optionally appends a user message to the current iteration.
    """
    branch_path = Path(branch_path)
    manifest = load_branch_manifest(branch_path)
    old_status = manifest.status
    logger.info(
        "continue_branch_on_disk: %s  status %s -> running",
        branch_path.name,
        old_status,
    )

    manifest.status = "running"
    manifest.pre_stop_status = None

    if manifest.current_iter >= manifest.max_iter:
        manifest.max_iter = manifest.current_iter + 1

    iter_state = load_iter_state(branch_path, manifest.current_iter)
    if message:
        _inject_user_message(iter_state, message)
    # Re-run propose → decide so the agent reconsiders with fresh analysis
    iter_state.status = "proposing"
    iter_state.proposal = ""
    save_iter_state(branch_path, manifest.current_iter, iter_state)

    if old_status == "branching":
        _stop_sub_branches(branch_path)

    save_branch_manifest(branch_path, manifest)
    return manifest.model_dump()
