"""
File I/O utilities and directory management.

Handles the hierarchical branch/iteration directory structure.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union


from config import get_output_dir
from utils.log import log


def ensure_output_dir() -> Path:
    """Ensure the output directory exists."""
    get_output_dir().mkdir(parents=True, exist_ok=True)
    return get_output_dir()


def create_branch_dir(parent: Path, name: str) -> Path:
    """Create a branch directory, suffixing the name if it is taken.

    The caller must take the branch's name from the returned path, not from what it
    asked for: everything that resolves a branch by name reads the manifest, so a
    manifest saying "tiled" beside a directory called "tiled_2" is unresolvable.
    """
    original_name = name
    counter = 2
    branch_dir = parent / name
    while branch_dir.exists():
        branch_dir = parent / f"{original_name}_{counter}"
        counter += 1

    branch_dir.mkdir(parents=True, exist_ok=True)
    return branch_dir


def get_parent_branch_dir(branch_path: Union[str, Path, None]) -> Optional[Path]:
    """
    Get a branch's parent branch directory, derived from the nesting structure.

    Sub-branches live at ``<parent_branch>/branches/<name>``, so the parent is
    two levels up. Top-level branches sit directly under ``output/branches``
    (two levels up is the output dir, which has no branch.json) and return None.
    """
    if not branch_path:
        return None
    branch_path = Path(branch_path)
    candidate = branch_path.parent.parent
    if branch_path.parent.name == "branches" and (candidate / "branch.json").exists():
        return candidate
    return None


def write_pid_file(problem_dir: Path, pid: int):
    """Write the optimizer's PID to output/run.pid.

    Here rather than in api/helpers, which imports FastAPI, so cli --finalize can claim
    the file too: the API refuses to start a run or a finalize while it names a live
    process.
    """
    pid_file = Path(problem_dir) / "output" / "run.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid), encoding="utf-8")


def remove_pid_file(problem_dir: Path):
    """Remove the PID file (called on stop or normal exit)."""
    (Path(problem_dir) / "output" / "run.pid").unlink(missing_ok=True)


def save_output(path: Path, content: str, filename: Optional[str] = None) -> Path:
    """
    Save content to a file.

    Args:
        path: Directory path or full file path
        content: Content to save
        filename: Filename (if path is a directory)

    Returns:
        Path to saved file
    """
    if filename:
        path = path / filename

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_json(path: Path, data: dict) -> Path:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_file(path: Path) -> str:
    """Load a file if it exists, return empty string otherwise."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def get_iter_dir(state) -> Path:
    """``branch_path/iterN`` for a WorkingState or an equivalent dict.

    Deliberately unannotated: importing state.types for a type the body duck-types
    anyway put utils above state in the layering, for nothing.
    """
    if hasattr(state, "iter_num"):
        iteration = getattr(state, "iter_num", 1) or 1
        branch_path_str = getattr(state, "branch_path", None)
    else:
        iteration = state.get("iter_num", 1)
        branch_path_str = state.get("branch_path")

    if branch_path_str:
        return Path(branch_path_str) / f"iter{iteration}"
    else:
        return get_output_dir() / f"iter{iteration}"


# Files under an archived run that the engine can regenerate: the tuner's input and
# reference dumps (~25 MB per iteration) and the compiled driver. Everything else in an
# iteration — the kernel, the driver regions, the results, the decisions, the step
# trace — is irreplaceable and is never touched by pruning. ``.staging/`` counts as
# irreplaceable: it survives on disk only when an authoring step aborted before
# committing, so it holds the only copy of that attempt's kernel and driver regions.
_REGENERABLE = ("cacao_in_*.bin", "cacao_ref_*.bin", "driver")
_REGENERABLE_DIRS = ("__pycache__",)

_ARCHIVE_DIR_RE = re.compile(r"output_(\d+)$")


def _next_archive_path(output_dir: Path) -> Path:
    """``archive/output_N`` beside the output directory, N one past the highest.

    Numbered rather than timestamped because the number is something you can say out
    loud and type; the directory's mtime carries the date anyway. Highest-plus-one
    never reuses a number, so a reference to "archive/output_3" stays valid.
    """
    root = output_dir.parent / "archive"
    highest = 0
    if root.is_dir():
        for entry in root.iterdir():
            match = _ARCHIVE_DIR_RE.match(entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return root / f"output_{highest + 1}"


def _describe_run(output_dir: Path) -> str:
    """A one-line summary of what a run produced, for the archive log line.

    Best-effort: this runs at the moment a run is about to be displaced, and a
    malformed state file is not a reason to fail the archive.
    """
    try:
        branches = list(output_dir.glob("**/branch.json"))
        iterations = list(output_dir.glob("**/iter*/state.json"))
        best = None
        for manifest in branches:
            try:
                value = json.loads(manifest.read_text(encoding="utf-8")).get("best_time_us")
            except (OSError, ValueError):
                continue
            if value and (best is None or value < best):
                best = value
        parts = [f"{len(branches)} branch(es)", f"{len(iterations)} iteration(s)"]
        if best:
            parts.append(f"best {best:.3f} us")
        return ", ".join(parts)
    except OSError:
        return "contents unknown"


def archive_output_dir():
    """Move the previous run's output aside, and start the next one with an empty dir.

    Renamed, never deleted: the frontend's Run button and ``cli.py`` without
    ``--resume`` both land here without asking, and a rename is O(1) at any size (the
    archive is on the same filesystem), so irreversible deletion buys nothing. A
    directory with no ``branches/`` held no iteration work and is removed rather than
    numbered. ``reference_time.json`` is carried into the fresh directory because it
    may be a hand-seeded baseline, and this is the one place it could be lost.

    Returns the archive path, or None when nothing was archived.
    """
    from utils.results import _REFERENCE_TIME_FILE

    output_dir = get_output_dir()
    preserved = {}
    for name in (_REFERENCE_TIME_FILE,):
        path = output_dir / name
        if path.is_file():
            preserved[name] = path.read_bytes()

    archived = None
    if output_dir.exists():
        if (output_dir / "branches").is_dir():
            archived = _next_archive_path(output_dir)
            archived.parent.mkdir(parents=True, exist_ok=True)
            summary = _describe_run(output_dir)
            os.rename(output_dir, archived)
            log(
                f"Archived previous run -> {archived.parent.name}/{archived.name}  ({summary})",
                "SUCCESS",
            )
        else:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in preserved.items():
        (output_dir / name).write_bytes(data)
    return archived


def prune_archives(problem_dir: Path) -> tuple:
    """Strip regenerable files from archived runs. Returns (files, bytes freed).

    An archived run is ~2 GB and ~99% of that is the tuner's input dumps and compiled
    drivers, both of which the engine rebuilds from scratch on the next run. The part
    worth keeping — every kernel, driver region, result, decision and step trace — is a
    few hundred KB per iteration.

    Deliberately manual, and deliberately never touching the current ``output/``.
    Automatic deletion is the behaviour this module was changed to stop doing.
    """
    root = problem_dir / "archive"
    if not root.is_dir():
        return (0, 0)

    removed = freed = 0
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not _ARCHIVE_DIR_RE.match(run.name):
            continue
        for pattern in _REGENERABLE:
            for path in run.glob(f"**/{pattern}"):
                if path.is_file():
                    try:
                        size = path.stat().st_size
                        path.unlink()
                    except OSError:
                        continue
                    removed += 1
                    freed += size
        for name in _REGENERABLE_DIRS:
            for path in run.glob(f"**/{name}"):
                if path.is_dir():
                    freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    removed += sum(1 for f in path.rglob("*") if f.is_file())
                    shutil.rmtree(path, ignore_errors=True)
    return (removed, freed)


# A clone copies the problem definition, never a run: output/ and archive/ are run
# history, and copying them also leaves the clone looking already-run.
_CLONE_EXCLUDE = ("run.pid", "run.log", "requeue", "output", "archive")


def clone_problem_dir(source_dir: Path, target_dir: Path) -> Path:
    """Copy a problem's definition files into a new directory.

    symlinks=True so a stale link in the source cannot crash the copy.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    shutil.copytree(
        source_dir,
        target_dir,
        symlinks=True,
        ignore=shutil.ignore_patterns(*_CLONE_EXCLUDE),
    )
    return target_dir
