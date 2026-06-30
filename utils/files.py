"""
File I/O utilities and directory management.

Handles the hierarchical branch/iteration directory structure.
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Any, Union, Dict


from config import get_output_dir
from state.types import WorkingState


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
    path.write_text(content)
    return path


def save_json(path: Path, data: dict) -> Path:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_file(path: Path) -> str:
    """Load a file if it exists, return empty string otherwise."""
    if path.exists():
        return path.read_text()
    return ""


def load_json(path: Path) -> Optional[dict]:
    """Load a JSON file if it exists."""
    if path.exists():
        with open(path) as f:
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


def clean_output_dir():
    """Remove and recreate the output directory."""
    if get_output_dir().exists():
        shutil.rmtree(get_output_dir())
    get_output_dir().mkdir(parents=True, exist_ok=True)
