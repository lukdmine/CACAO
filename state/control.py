"""
Control signals and re-queue helpers for branch orchestration.

Provides the file-based communication channel between the server/master
and running worker coroutines:
  - stop / resume / revert / message / config signals
  - file-based re-queue for reverted branches
  - disk surgery for reverting a branch to a prior iteration
"""

import itertools
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from state.persistence import (
    atomic_write_json,
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
    """Remove the sub-branch tree. Deletion is the stop.

    Signalling each child first cannot work: the signals live inside the directory
    being removed, and a worker polls once per node — an LLM call or a tuner run
    later. The child notices instead by finding its own directory gone, which
    engine/worker.py checks before every save (a save would mkdir it back).
    """
    branches_dir = branch_path / "branches"
    if not branches_dir.is_dir():
        return
    shutil.rmtree(branches_dir, ignore_errors=True)


# Signals queue as one file each under this directory. A single ``control.json`` slot
# lost whichever of two signals arrived first, and with the worker polling once per node
# — minutes apart — a stop followed by a message is the ordinary case, not a race.
_CONTROL_DIR = "control"

# Per-process tiebreaker for two signals written within the same nanosecond.
_signal_seq = itertools.count()


def _pending_signals(branch_path: Path) -> List[Path]:
    """Queued signal files, oldest first. Names sort chronologically by construction."""
    queue = branch_path / _CONTROL_DIR
    queued = sorted(queue.glob("*.json")) if queue.is_dir() else []
    legacy = branch_path / "control.json"  # written by a run that predates the queue
    return ([legacy] if legacy.is_file() else []) + queued


def read_control_signal(branch_path: Path) -> Optional[dict]:
    """Read and consume the oldest pending signal for a branch, or None if there is none.

    Claiming is an atomic rename: whoever renames a file owns it, so a concurrent
    reader or writer cannot interfere with a signal already being processed.
    """
    branch_path = Path(branch_path)
    consumed = branch_path / f".control_consumed_{os.getpid()}.json"
    for candidate in _pending_signals(branch_path):
        try:
            os.rename(candidate, consumed)
        except OSError:
            continue  # vanished, or another reader got there first
        try:
            with consumed.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        finally:
            consumed.unlink(missing_ok=True)
    return None


def write_control_signal(branch_path: Path, signal: dict):
    """Queue a control signal for a branch. Never displaces an unread one."""
    name = f"{time.time_ns():020d}_{os.getpid()}_{next(_signal_seq)}.json"
    atomic_write_json(Path(branch_path) / _CONTROL_DIR / name, signal)


def clear_control_signals(branch_path: Path):
    """Drop every pending signal, consumed or not.

    For a caller that has just done the work a queued signal would have asked for:
    replaying it afterwards would undo that work.
    """
    branch_path = Path(branch_path)
    shutil.rmtree(branch_path / _CONTROL_DIR, ignore_errors=True)
    (branch_path / "control.json").unlink(missing_ok=True)


# -------------------------------------------------------------------------
# File-based re-queue
# -------------------------------------------------------------------------


def write_requeue(output_dir: Path, branch_path: Path):
    """Drop a re-queue file so a worker picks up this branch."""
    filename = f"{branch_path.name}_{int(time.time() * 1000)}.json"
    atomic_write_json(
        output_dir / "requeue" / filename, {"branch_path": str(branch_path)}
    )


def read_requeue(output_dir: Path) -> List[Path]:
    """Consume all re-queue files and return the branch paths."""
    requeue_dir = output_dir / "requeue"
    if not requeue_dir.is_dir():
        return []
    paths = []
    for f in sorted(requeue_dir.iterdir()):
        if f.suffix == ".json":
            try:
                with f.open("r", encoding="utf-8") as fh:
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


def _recompute_branch_best(branch_path: Path, through_iter: int) -> tuple:
    """Best time and speedup across iterations 1..through_iter, recomputed from disk.

    ``best_time_us`` is a running minimum nothing ever raises, so a revert that deletes
    the iteration which set it would leave the branch advertising a time whose
    results.json is gone. Returns (best_time_us, speedup), either possibly None.
    """
    best_time = None
    best_speedup = None
    for n in range(1, through_iter + 1):
        state = load_iter_state(branch_path, n, optional=True)
        if state is None:
            continue
        summary = state.results_summary or {}
        t_us, speedup = summary.get("best_time_us"), summary.get("speedup")
        if t_us is not None and (best_time is None or t_us < best_time):
            best_time = t_us
        if speedup is not None and (best_speedup is None or speedup > best_speedup):
            best_speedup = speedup
    return best_time, best_speedup


def revert_branch_on_disk(
    branch_path: Path,
    target_iter: int,
    message: Optional[str] = None,
) -> dict:
    """
    Revert a branch to the state after *target_iter*.

    Restores branch manifest to point at target_iter, deletes later
    iteration directories and sub-branches, and returns the manifest dict.

    Does not touch ``branch_config.json``: it has one writer, the server, which
    grants the extra iteration an exhausted branch needs before it gets here (or
    before it signals the worker to run this).
    """
    branch_path = Path(branch_path)
    manifest = load_branch_manifest(branch_path)
    iter_state = load_iter_state(branch_path, target_iter)

    # Update branch manifest
    manifest.current_iter = target_iter
    manifest.status = "running"
    manifest.pre_stop_status = None

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

    # Only now that the later iterations are gone: the surviving ones are the whole
    # truth about this branch, and the manifest has to agree with them.
    manifest.best_time_us, manifest.speedup = _recompute_branch_best(
        branch_path, target_iter
    )

    save_branch_manifest(branch_path, manifest)
    return manifest.model_dump()


def continue_branch_on_disk(
    branch_path: Path,
    message: Optional[str] = None,
) -> dict:
    """
    Re-open a dead branch (success/failed/branching) so it continues
    from its current iteration without deleting any work.

    Sets status to ``running`` and optionally appends a user message to the current
    iteration. Lifting an exhausted branch's budget is the server's to do — see
    ``revert_branch_on_disk`` — because ``branch_config.json`` has one writer.
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
