"""Shared state and utilities for the API package."""

import json
import multiprocessing
import os
import signal
import threading
import time
from pathlib import Path

import yaml
from fastapi import HTTPException

from utils.problem import gpu_index as problem_gpu_index, reference_spec

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

# Exit status of the last optimizer to finish, per problem. Whoever first notices a
# dead child drops its RunEntry, and with it the only copy of ``exitcode`` — so a run
# that died in startup was indistinguishable from one that finished its branches.
_last_exit_codes: dict[str, int] = {}

# Per-branch mutex to prevent concurrent mutations
_branch_locks: dict[str, threading.Lock] = {}
_branch_locks_guard = threading.Lock()


# ── PID file helpers (survive server reload) ──────────────────────────────────


def write_pid_file(problem_dir: Path, pid: int):
    """Write optimizer PID to output/run.pid."""
    pid_file = problem_dir / "output" / "run.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid), encoding="utf-8")


def remove_pid_file(problem_dir: Path):
    """Remove the PID file (called on stop or normal exit)."""
    (problem_dir / "output" / "run.pid").unlink(missing_ok=True)


def _pid_file_alive(problem_dir: Path) -> bool:
    """Check if a PID file exists and the process is still alive."""
    pid_file = problem_dir / "output" / "run.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)  # signal 0 = existence check
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pid_file.unlink(missing_ok=True)  # stale
        return False


# ── Running state queries ─────────────────────────────────────────────────────


def _reap_unlocked(name: str):
    """Drop a finished run from the registry, keeping its exit code. Caller holds the lock."""
    entry = _running_problems.pop(name, None)
    if entry is not None and entry.process.exitcode is not None:
        _last_exit_codes[name] = entry.process.exitcode


def last_exit_code(name: str) -> int | None:
    """How the last optimizer for ``name`` ended, or None if none has ended here.

    Non-zero means it died rather than finished — the UI has no other way to tell,
    since the registry entry is gone by the time the next poll asks.
    """
    with _running_lock:
        return _last_exit_codes.get(name)


def _is_running_unlocked(name: str) -> bool:
    """is_problem_running without acquiring the lock. Caller must hold _running_lock."""
    entry = _running_problems.get(name)
    if entry is not None:
        if entry.process.is_alive():
            return True
        _reap_unlocked(name)
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
            cfg = load_yaml(py)
        except Exception:
            continue
        if problem_gpu_index(cfg) == gpu_index:
            raise HTTPException(
                status_code=409,
                detail=f"GPU {gpu_index} is in use by orphan optimizer '{problem_dir.name}' (PID file). Stop it or use a different GPU.",
            )


def _kill_pid_group(pid: int, timeout: float = 5.0) -> bool:
    """SIGTERM a process group, escalate to SIGKILL, and report whether it is gone.

    The in-memory path already escalates (terminate_optimizer_process). This one is
    what actually runs after a server restart, when the PID file is the only tracking
    left — and a worker inside a CUDA call does not die on the first SIGTERM.
    """
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return True  # already gone, or not ours to signal

    # Never killpg our own group. _run_target's setsid() can fail, and a PID read out
    # of a stale run.pid may have been recycled into the server's group — either way
    # the group here would be uvicorn plus every other live run. Signal just the PID.
    if pgid == os.getpgrp():
        send = lambda sig: os.kill(pid, sig)  # noqa: E731
    else:
        send = lambda sig: os.killpg(pgid, sig)  # noqa: E731

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            send(sig)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        deadline = time.monotonic() + (timeout if sig == signal.SIGTERM else 1.0)
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except (PermissionError, OSError):
                return False
            time.sleep(0.1)
    return False


def terminate_run(name: str, problem_dir: Path, raise_on_failure: bool = True):
    """Kill the optimizer for ``name``, whether tracked in-memory or only via PID file.

    No-op if nothing is running. The PID file is removed only once the process group is
    confirmed gone: while it survives, it is the one thing that still reports the run as
    running, and every disk-editing endpoint gates on that.
    """
    with _running_lock:
        entry = _running_problems.pop(name, None)
        if entry is not None and entry.process.exitcode is not None:
            _last_exit_codes[name] = entry.process.exitcode
    if entry:
        terminate_optimizer_process(entry.process)
    else:
        pid_file = problem_dir / "output" / "run.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pid = None
            if pid is not None and not _kill_pid_group(pid):
                if raise_on_failure:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Optimizer for '{name}' (PID {pid}) did not stop; leaving the PID file in place",
                    )
                # A stale run.pid owned by another user makes killpg fail forever.
                # /stop should say so; delete must not be blocked by it.
                from utils.log import log

                log(
                    f"Could not stop optimizer for '{name}' (PID {pid}); continuing anyway",
                    "WARN",
                )
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
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                pass
            pid_file.unlink(missing_ok=True)


def get_problem_dir(name: str) -> Path:
    """The directory for ``name``, which must be a child of problems/.

    Containment, not just is_dir(): uvicorn percent-decodes the path before routing, so
    ``/api/problems/%2e%2e`` reaches here as the literal ``..`` and used to resolve to
    the repo root — which delete_problem then handed to shutil.rmtree.
    """
    base = PROBLEMS_DIR.resolve()
    problem_dir = (PROBLEMS_DIR / name).resolve()
    if (
        problem_dir == base
        or not problem_dir.is_relative_to(base)
        or not problem_dir.is_dir()
    ):
        raise HTTPException(status_code=404, detail=f"Problem '{name}' not found")
    return problem_dir


def load_yaml(path: Path) -> dict:
    """A parsed YAML mapping. ``{}`` for an empty or comment-only file — safe_load
    returns None there, and every caller here calls .get on the result."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def load_problem_inputs(problem_dir: Path) -> tuple[str, str]:
    """Load problem.yaml and the configured reference source code.

    Also regenerates inputs.hpp from the canonical inputs.yaml, since this runs once per
    run before any node reads the header. Raises FileNotFoundError if inputs.yaml or the
    reference source is missing — the run cannot proceed without an I/O boundary, and
    load_file returns "" for a missing path, which would otherwise hand every prompt an
    empty reference and burn the whole budget on a problem the model was never shown.
    cli.py makes the same check in check_prerequisites.
    """
    from utils.files import load_file
    from utils.inputs import ensure_inputs_hpp

    problem_yaml = load_file(problem_dir / "problem.yaml")
    config = yaml.safe_load(problem_yaml) or {}
    ref_file = reference_spec(config).file
    ref_path = problem_dir / ref_file
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference source '{ref_file}' not found in {problem_dir}"
        )
    ensure_inputs_hpp(problem_dir)
    return problem_yaml, load_file(ref_path)


def resolve_branch_path(name: str, branch_id: str) -> Path:
    """The directory for ``branch_id``, which must be under this problem's branches/.

    The branch routes declare ``{branch_id:path}``, so slashes and ``..`` segments both
    survive routing — and delete_branch / change_decision rmtree what comes back.
    """
    problem_dir = get_problem_dir(name)
    base = (problem_dir / "output" / "branches").resolve()
    branch_path = (base / branch_id).resolve()
    if (
        branch_path == base
        or not branch_path.is_relative_to(base)
        or not branch_path.is_dir()
    ):
        raise HTTPException(status_code=404, detail=f"Branch '{branch_id}' not found")
    return branch_path


def is_branch_live(name: str, branch_path: Path) -> bool:
    """Check if the optimizer process is running and the branch has a live worker."""
    if not is_problem_running(name):
        return False
    state = load_json(branch_path / "branch.json")
    return state.get("status") not in ("success", "failed", "branching")
