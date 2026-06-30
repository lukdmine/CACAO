"""Shared state and utilities for the API package."""

import json
import multiprocessing
import os
import signal
import threading
from pathlib import Path

import yaml
from fastapi import HTTPException

from dataclasses import dataclass

PROBLEMS_DIR = Path(__file__).parent.parent / "problems"


@dataclass
class RunEntry:
    """Tracks a running optimizer process and which GPU it uses."""

    process: multiprocessing.Process
    gpu_index: int


# Track running optimizations (process-based for clean stopping)
_running_problems: dict[str, RunEntry] = {}
_running_lock = threading.Lock()

# Per-branch mutex to prevent concurrent mutations
_branch_locks: dict[str, threading.Lock] = {}
_branch_locks_guard = threading.Lock()


# ── PID file helpers (survive server reload) ──────────────────────────────────


def write_pid_file(problem_dir: Path, pid: int):
    """Write optimizer PID to output/run.pid."""
    pid_file = problem_dir / "output" / "run.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def remove_pid_file(problem_dir: Path):
    """Remove the PID file (called on stop or normal exit)."""
    (problem_dir / "output" / "run.pid").unlink(missing_ok=True)


def _pid_file_alive(problem_dir: Path) -> bool:
    """Check if a PID file exists and the process is still alive."""
    pid_file = problem_dir / "output" / "run.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pid_file.unlink(missing_ok=True)  # stale
        return False


# ── Running state queries ─────────────────────────────────────────────────────


def _is_running_unlocked(name: str) -> bool:
    """is_problem_running without acquiring the lock. Caller must hold _running_lock."""
    entry = _running_problems.get(name)
    if entry is not None:
        if entry.process.is_alive():
            return True
        del _running_problems[name]
        return False
    # Fallback: PID file survives server reload
    problem_dir = PROBLEMS_DIR / name
    if problem_dir.is_dir():
        return _pid_file_alive(problem_dir)
    return False


def is_problem_running(name: str) -> bool:
    """Check if a problem has a running optimizer (in-memory dict + PID file fallback)."""
    with _running_lock:
        return _is_running_unlocked(name)


def _check_gpu_orphans(gpu_index: int, skip_name: str | None = None):
    """Raise 409 if a PID-file-only optimizer (server-reload orphan) holds the GPU.

    Must run without holding _running_lock (it reads problem.yaml files).
    """
    if not PROBLEMS_DIR.is_dir():
        return
    for problem_dir in PROBLEMS_DIR.iterdir():
        if not problem_dir.is_dir() or problem_dir.name == skip_name:
            continue
        with _running_lock:
            if problem_dir.name in _running_problems:
                continue  # in-memory entry is checked under the lock by the caller
        if not _pid_file_alive(problem_dir):
            continue
        py = problem_dir / "problem.yaml"
        if not py.exists():
            continue
        try:
            cfg = load_yaml(py) or {}
        except Exception:
            continue
        if (cfg.get("gpu") or {}).get("index", 0) == gpu_index:
            raise HTTPException(
                status_code=409,
                detail=f"GPU {gpu_index} is in use by orphan optimizer '{problem_dir.name}' (PID file). Stop it or use a different GPU.",
            )


def terminate_run(name: str, problem_dir: Path):
    """Kill the optimizer for ``name``, whether tracked in-memory or only via PID file.

    No-op if nothing is running. Always removes the PID file on exit.
    """
    with _running_lock:
        entry = _running_problems.pop(name, None)
    if entry:
        terminate_optimizer_process(entry.process)
    else:
        pid_file = problem_dir / "output" / "run.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                pass
    remove_pid_file(problem_dir)


def get_branch_lock(branch_path: Path) -> threading.Lock:
    key = str(branch_path)
    with _branch_locks_guard:
        if key not in _branch_locks:
            _branch_locks[key] = threading.Lock()
        return _branch_locks[key]


def terminate_optimizer_process(proc: multiprocessing.Process, timeout: float = 5.0):
    """Terminate an optimizer process and its subprocesses."""
    if proc.pid is None:
        return

    current_pgid = os.getpgid(os.getpid())
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None

    if pgid == current_pgid:
        pgid = None

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        proc.terminate()

    proc.join(timeout=timeout)

    if proc.is_alive():
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.join(timeout=1)


def terminate_all_optimizers():
    """Terminate all running optimizer processes. Called on server shutdown."""
    for entry in list(_running_problems.values()):
        try:
            terminate_optimizer_process(entry.process, timeout=3.0)
        except Exception:
            pass
    _running_problems.clear()

    # Also kill any orphaned processes via PID files
    if PROBLEMS_DIR.is_dir():
        for pid_file in PROBLEMS_DIR.rglob("output/run.pid"):
            try:
                pid = int(pid_file.read_text().strip())
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                pass
            pid_file.unlink(missing_ok=True)


def get_problem_dir(name: str) -> Path:
    problem_dir = PROBLEMS_DIR / name
    if not problem_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Problem '{name}' not found")
    return problem_dir


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def load_problem_inputs(problem_dir: Path) -> tuple[str, str]:
    """Load problem.yaml and the configured reference source code."""
    from utils.files import load_file

    problem_yaml = load_file(problem_dir / "problem.yaml")
    config = yaml.safe_load(problem_yaml) or {}
    ref_file = config.get("reference", {}).get("file", "ref_kernel.cu")
    return problem_yaml, load_file(problem_dir / ref_file)


def resolve_branch_path(name: str, branch_id: str) -> Path:
    problem_dir = get_problem_dir(name)
    branch_path = problem_dir / "output" / "branches" / branch_id
    if not branch_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Branch '{branch_id}' not found")
    return branch_path


def is_branch_live(name: str, branch_path: Path) -> bool:
    """Check if the optimizer process is running and the branch has a live worker."""
    if not is_problem_running(name):
        return False
    state = load_json(branch_path / "branch.json")
    return state.get("status") not in ("success", "failed", "branching")
