"""
File I/O utilities and directory management.

Handles the hierarchical branch/iteration directory structure.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Any, Union, Dict


from config import get_output_dir
from state.types import WorkingState
from utils.log import log


def ensure_output_dir() -> Path:
    """Ensure the output directory exists."""
    get_output_dir().mkdir(parents=True, exist_ok=True)
    return get_output_dir()


def create_branch_dir(
    parent: Path,
    name: str,
    strategy: dict,
    depth: int,
) -> Path:
    """
    Create a branch directory with strategy metadata.

    Args:
        parent: Parent directory (e.g., output/branches or output/branches/tiled/branches)
        name: Branch name (e.g., "shared_mem_tiling")
        strategy: Strategy dict from strategize_node
        depth: Current branch depth

    Returns:
        Path to created branch directory
    """
    original_name = name
    counter = 2
    branch_dir = parent / name
    while branch_dir.exists():
        name = f"{original_name}_{counter}"
        branch_dir = parent / name
        counter += 1

    branch_dir.mkdir(parents=True, exist_ok=True)

    # Save strategy metadata (use **strategy first so we can override the name key)
    strategy_info = {**strategy, "name": name, "depth": depth}
    save_json(branch_dir / "strategy.json", strategy_info)

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


def create_iter_dir(branch_path: Path, iter_num: int) -> Path:
    """
    Create an iteration directory within a branch.

    Args:
        branch_path: Path to branch directory
        iter_num: Iteration number (1-indexed)

    Returns:
        Path to created iteration directory
    """
    iter_dir = branch_path / f"iter{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    return iter_dir


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


def load_json(path: Path) -> Optional[dict]:
    """Load a JSON file if it exists."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_iter_dir(state: Union[Dict[str, Any], WorkingState]) -> Path:
    """
    Get the current iteration directory path from a working state.

    Shared helper used by all node files to avoid duplication.
    Accepts either a dict or a Pydantic ``WorkingState`` model.

    Args:
        state: Working state (needs ``iter_num`` and ``branch_path``)

    Returns:
        Path to the iteration directory (e.g. ``branch_path/iter3``)
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
# trace — is irreplaceable and is never touched by pruning.
_REGENERABLE = ("cacao_in_*.bin", "cacao_ref_*.bin", "driver")
_REGENERABLE_DIRS = ("__pycache__", ".staging")

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

    This used to be ``clean_output_dir`` and it used to ``rmtree``. It cost a 23-hour
    run: the frontend's Run button and ``cli.py`` without ``--resume`` both land here,
    and neither asks first, so one misclick destroyed every kernel four branches had
    produced. A rename costs nothing — it is O(1) regardless of size, the archive is on
    the same filesystem — and turns that misclick into an inconvenience.

    A directory with no ``branches/`` never held iteration work, so it is removed rather
    than archived; otherwise a run aborted during analysis would leave a numbered
    archive holding a log and nothing else.

    ``reference_time.json`` is carried into the fresh directory. It may be seeded by
    hand before the first run, and for a ``python`` or ``cpu_c`` reference it usually
    must be: timing the reference itself is meaningless as a speedup denominator,
    because a GPU kernel "beats" a single-threaded C loop or a torch eager call by two
    orders of magnitude. The incumbent is measured externally and dropped in instead.
    Everything downstream already respects that seed — ``save_reference_time`` is
    first-write-wins via ``O_CREAT|O_EXCL``, and reference timing no-ops when the file
    exists — so this is the one place a hand-measured baseline could be lost.

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
