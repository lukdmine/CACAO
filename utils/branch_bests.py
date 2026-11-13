"""Each branch's best valid iteration, whatever the branch's status.

``nodes/merge.find_branch_results`` keeps a status filter, which suits the merge step
but would hide a run that was interrupted or whose last iterations failed after an
earlier one had validated. Finalize wants the best kernel that ever validated, so a
branch is eligible on the strength of one iteration alone.

Reads ``branch.json`` and ``iter*/state.json`` as plain JSON (as state/crossbranch.py
does) so nothing under utils/ depends on the ``state`` package.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils.results import iteration_results_summary, load_reference_time

_ITER_DIR = re.compile(r"^iter(\d+)$")


@dataclass(frozen=True)
class BranchBest:
    name: str  # manifest.strategy.name
    path: Path  # absolute branch dir
    rel: str  # path relative to output/
    status: str
    best_iter: int
    summary: dict  # that iteration's aggregate summary
    score: Optional[float]  # summary["speedup"]
    time_us: Optional[float]  # summary["best_time_us"]
    description: str = ""  # manifest.strategy.description


def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def is_valid_summary(summary: Optional[dict]) -> bool:
    """Same test nodes/decide.py applies: something validated and no case failed."""
    return bool(
        summary and summary.get("has_success") and summary.get("failed_case") is None
    )


def iteration_summary(
    problem_dir: Path, iter_dir: Path, ref_time: Optional[float]
) -> Optional[dict]:
    """The iteration's aggregate summary: state.json's when present (it spans every
    case), else rebuilt from the primary case's results.json."""
    state = _load_json(Path(iter_dir) / "state.json")
    if state and state.get("results_summary"):
        return state["results_summary"]
    if not (Path(iter_dir)).is_dir():
        return None
    return iteration_results_summary(problem_dir, iter_dir, ref_time)


def summary_rank_key(summary: Optional[dict]) -> tuple:
    """Sort key, best first: a speedup beats none, larger beats smaller, and with no
    speedups the shorter time wins."""
    summary = summary or {}
    speedup = summary.get("speedup")
    return (
        speedup is None,
        -(speedup or 0),
        summary.get("best_time_us") or float("inf"),
    )


def outranks(a: Optional[dict], b: Optional[dict]) -> bool:
    """Whether summary ``a`` beats ``b``.

    When both carry a re-measured score the comparison uses it: two single-shot times
    a few percent apart are ordered by noise. With one side missing it, both fall back
    to their single-shot numbers, since a precise time against a single-shot one is not
    a like-for-like comparison.
    """
    a, b = a or {}, b or {}
    for key, better in (("speedup_precise", max), ("precise_best_time_us", min)):
        x, y = a.get(key), b.get(key)
        if x and y:
            return x != y and better(x, y) == x
    return summary_rank_key(a) < summary_rank_key(b)


def rank_key(b: BranchBest) -> tuple:
    return summary_rank_key(b.summary)


def _iter_numbers(branch: Path) -> List[int]:
    nums = []
    for d in branch.iterdir():
        m = _ITER_DIR.match(d.name)
        if m and d.is_dir():
            nums.append(int(m.group(1)))
    return sorted(nums)


def _best_of(
    branch: Path, problem_dir: Path, ref_time: Optional[float]
) -> Optional[tuple]:
    best = None
    for n in _iter_numbers(branch):
        summary = iteration_summary(problem_dir, branch / f"iter{n}", ref_time)
        if not is_valid_summary(summary):
            continue
        if best is None or summary_rank_key(summary) < summary_rank_key(best[1]):
            best = (n, summary)
    return best


def collect_branch_bests(output_dir: Path, problem_dir: Path) -> List[BranchBest]:
    """Every branch in the tree with at least one valid iteration, best first."""
    output_dir = Path(output_dir)
    ref_time = load_reference_time(output_dir)
    found: List[BranchBest] = []

    def scan(root: Path):
        if not root.is_dir():
            return
        for d in sorted(root.iterdir()):
            manifest = _load_json(d / "branch.json") if d.is_dir() else None
            if manifest is None:
                continue
            best = _best_of(d, problem_dir, ref_time)
            if best is not None:
                n, summary = best
                found.append(
                    BranchBest(
                        name=(manifest.get("strategy") or {}).get("name") or d.name,
                        path=d.resolve(),
                        rel=d.relative_to(output_dir).as_posix(),
                        status=manifest.get("status", ""),
                        best_iter=n,
                        summary=summary,
                        score=summary.get("speedup"),
                        time_us=summary.get("best_time_us"),
                        description=(manifest.get("strategy") or {}).get("description")
                        or "",
                    )
                )
            scan(d / "branches")

    scan(output_dir / "branches")
    return sorted(found, key=rank_key)


def source_fingerprint(bests: List[BranchBest]) -> str:
    """sha1 over what finalize's inputs are: which iteration of which branch, and its
    score (its time when there is no speedup), so a rerun can tell the sources moved."""
    rows = sorted(
        (
            b.rel,
            b.best_iter,
            round(b.score if b.score is not None else (b.time_us or 0.0), 6),
        )
        for b in bests
    )
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()
