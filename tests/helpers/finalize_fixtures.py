"""Synthetic ``output/`` trees for the finalize tests.

``make_branch`` writes what the worker would: a ``branch.json`` and one
``iter<N>/state.json`` per summary. Nothing else (no kernels, no results.json), so a
test built on it exercises the paths that read ``state.results_summary``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from state.types import BranchManifest, IterState, StrategyInfo


def valid_summary(speedup: Optional[float], time_us: float) -> dict:
    """An aggregate summary of an iteration that ran and validated."""
    return {
        "has_success": True,
        "num_successful": 1,
        "num_total": 1,
        "best_config": {},
        "best_time_us": time_us,
        "speedup": speedup,
        "failed_case": None,
    }


def invalid_summary() -> dict:
    """An aggregate summary of an iteration with no valid configuration."""
    return {
        "has_success": False,
        "num_successful": 0,
        "num_total": 1,
        "best_config": None,
        "best_time_us": None,
        "speedup": None,
        "failed_case": None,
    }


def make_branch(
    output_dir: Path,
    name: str,
    status: str,
    summaries: List[Optional[dict]],
    *,
    parent: Optional[Path] = None,
) -> Path:
    """Write a branch under ``output_dir/branches`` (or under ``parent/branches``).

    ``summaries[i]`` becomes ``iter{i+1}/state.json``'s ``results_summary`` (None leaves
    it unset). Returns the branch directory.
    """
    root = (parent if parent is not None else Path(output_dir)) / "branches"
    branch = root / name
    branch.mkdir(parents=True, exist_ok=True)
    manifest = BranchManifest(
        strategy=StrategyInfo(name=name), status=status, current_iter=len(summaries)
    )
    (branch / "branch.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")
    for n, summary in enumerate(summaries, start=1):
        it = branch / f"iter{n}"
        it.mkdir(exist_ok=True)
        state = IterState(iter_num=n, status="decided", results_summary=summary)
        (it / "state.json").write_text(
            json.dumps(state.dump_for_disk()), encoding="utf-8"
        )
    return branch
