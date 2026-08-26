"""Reading other branches' results.

Branches run as parallel coroutines and, apart from a sub-branch inheriting its
parent's last iteration, are invisible to each other. In one real run that cost about
four iterations: three branches each independently tried async double-buffering and
reverted it, each independently found ``__launch_bounds__``, and each separately
diagnosed the same API outage.

What is shared, and what is not, is deliberate. The one-line iteration summaries carry
*what was tried and what happened* without carrying the kernel, so they prune dead ends
without collapsing parallel strategies into one bet. Kernel bodies and framework
regions are reachable only by explicitly naming a branch, an iteration, and a file --
never surfaced automatically.

Everything here is read-only and tolerant of concurrent writers: a sibling's worker
rewrites branch.json wholesale, so a partial read is expected, not exceptional. Only
iterations strictly below a branch's ``current_iter`` are read, so a half-decided
iteration is never observed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# A prompt-resident index has to stay small enough to be worth its space. Rows are
# sorted fastest-first, so a deep tree keeps the branches worth investigating and drops
# the tail rather than truncating arbitrarily.
_MAX_INDEX_ROWS = 8


def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _branches_root(output_dir) -> Path:
    return Path(output_dir) / "branches"


def iter_branches(output_dir) -> List[Tuple[str, Path, dict]]:
    """Every branch in the tree as (name, path, manifest).

    Nested branches get a slash-joined name ('parent/child') so a sub-branch can be
    named unambiguously; the intermediate 'branches' directories are not part of it.
    """
    found: List[Tuple[str, Path, dict]] = []

    def scan(root: Path, prefix: str = ""):
        if not root.is_dir():
            return
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            manifest = _load_json(d / "branch.json")
            if manifest is None:
                continue
            name = f"{prefix}{d.name}"
            found.append((name, d, manifest))
            scan(d / "branches", prefix=f"{name}/")

    scan(_branches_root(output_dir))
    return found


def resolve_branch(output_dir, name: str) -> Optional[Path]:
    """Path for a branch named as in the index. Accepts the bare leaf name too."""
    branches = iter_branches(output_dir)
    for bname, path, _ in branches:
        if bname == name:
            return path
    leaf_matches = [p for bname, p, _ in branches if bname.rsplit("/", 1)[-1] == name]
    return leaf_matches[0] if len(leaf_matches) == 1 else None


def branch_names(output_dir) -> List[str]:
    return [name for name, _, _ in iter_branches(output_dir)]


# -------------------------------------------------------------------------
# The always-on index
# -------------------------------------------------------------------------


def branch_index(output_dir, exclude_path=None) -> str:
    """One line per sibling branch: name, best time, iterations, status.

    This is the only cross-branch content that enters a prompt unasked. Without the
    relative timing there is no signal that another branch is worth a tool call, and
    the pull tools go unused; with the full summaries the prompt grows by everything
    the tools were meant to keep out.
    """
    exclude = Path(exclude_path).resolve() if exclude_path else None
    rows = []
    for name, path, manifest in iter_branches(output_dir):
        if exclude is not None and path.resolve() == exclude:
            continue
        best = manifest.get("best_time_us")
        iters = max(int(manifest.get("current_iter") or 1) - 1, 0)
        rows.append(
            {
                "name": name,
                "best": best if isinstance(best, (int, float)) else None,
                "iters": iters,
                "status": manifest.get("status", "?"),
            }
        )

    if not rows:
        return ""

    # Fastest first: the ordering that makes the useful row the one already read.
    rows.sort(key=lambda r: (r["best"] is None, r["best"] if r["best"] is not None else 0))
    shown = rows[:_MAX_INDEX_ROWS]

    width = max(len(r["name"]) for r in shown)
    lines = ["## Other branches:  (branch_log(name) for what each one tried)"]
    for r in shown:
        best = f"best {r['best']:,.0f} µs" if r["best"] is not None else "no result yet"
        lines.append(
            f"- {r['name']:<{width}} | {best} | {r['iters']} iters | {r['status']}"
        )
    if len(rows) > len(shown):
        lines.append(f"- ... and {len(rows) - len(shown)} more")
    return "\n".join(lines)


# -------------------------------------------------------------------------
# Pulled detail
# -------------------------------------------------------------------------


def _completed_iters(branch_path: Path, manifest: dict) -> List[dict]:
    """Decided iterations only, oldest first.

    Stops below current_iter: that iteration is being written by its own worker right
    now and its decision does not exist yet.
    """
    current = int(manifest.get("current_iter") or 1)
    snaps = []
    for n in range(1, current):
        snap = _load_json(branch_path / f"iter{n}" / "state.json")
        if snap is not None:
            snaps.append(snap)
    return snaps


def branch_log(output_dir, name: str) -> str:
    """What a branch tried, one line per iteration, with the outcome.

    Deliberately excludes kernel code: this is the pruning signal, not a template to
    copy. Use read_branch_file to look at an iteration that turns out to matter.
    """
    path = resolve_branch(output_dir, name)
    if path is None:
        available = ", ".join(branch_names(output_dir)) or "(none)"
        return f"Unknown branch '{name}'. Available: {available}."

    manifest = _load_json(path / "branch.json") or {}
    snaps = _completed_iters(path, manifest)
    if not snaps:
        return f"Branch '{name}' has no completed iterations yet."

    strategy = manifest.get("strategy") or {}
    lines = [f"## Branch '{name}' — {strategy.get('description', '')}".rstrip()]
    for snap in snaps:
        n = snap.get("iter_num", "?")
        decision = snap.get("decision") or {}
        summary = decision.get("iteration_summary", "")
        rs = snap.get("results_summary") or {}
        t = rs.get("best_time_us")
        if summary:
            lines.append(f"- iter {n}: {summary}")
        elif t:
            lines.append(f"- iter {n}: {t:.0f} µs (decision: {decision.get('action', '?')})")
        else:
            lines.append(f"- iter {n}: failed (decision: {decision.get('action', '?')})")
    return "\n".join(lines)


def list_iterations(branch_path) -> str:
    """Index of this branch's own iterations: which ran, which worked, how fast.

    Answers 'what was the last iteration that compiled at all', which the prompt's
    two-iteration history window and the single best_so_far entry cannot: after a run
    of failures the useful iteration to diff against is neither recent nor fastest.
    """
    branch_path = Path(branch_path)
    manifest = _load_json(branch_path / "branch.json") or {}
    snaps = _completed_iters(branch_path, manifest)
    if not snaps:
        return "No completed iterations in this branch yet."

    rows = ["| iter | outcome | best µs | configs ok | decision |", "|---|---|---|---|---|"]
    for snap in snaps:
        rs = snap.get("results_summary") or {}
        decision = snap.get("decision") or {}
        t = rs.get("best_time_us")
        ok, total = rs.get("num_successful"), rs.get("num_total")
        rows.append(
            f"| {snap.get('iter_num', '?')} "
            f"| {'ran' if t else 'failed'} "
            f"| {f'{t:,.1f}' if t else '—'} "
            f"| {ok if ok is not None else '—'}/{total if total is not None else '—'} "
            f"| {decision.get('action', '?')} |"
        )
    return "\n".join(rows)


def branch_errors(output_dir, name: str) -> str:
    """What went wrong in another branch, and what it concluded about the cause.

    Deliberately the deepest cross-branch read available. There is no route to a
    sibling's kernel or framework regions, because the value of running four branches
    is that they are four independent attempts: a branch that can read the current
    leader's code converges on it, and the parallelism buys one bet instead of four.

    Failure analyses do not have that effect. "cp.async double buffering regressed and
    was reverted" removes a dead end without supplying a solution, which is the whole
    reason cross-branch sharing is worth having.
    """
    path = resolve_branch(output_dir, name)
    if path is None:
        available = ", ".join(branch_names(output_dir)) or "(none)"
        return f"Unknown branch '{name}'. Available: {available}."

    manifest = _load_json(path / "branch.json") or {}
    entries = []
    for snap in _completed_iters(path, manifest):
        decision = snap.get("decision") or {}
        analysis = decision.get("error_analysis") or {}
        if not analysis:
            continue
        n = snap.get("iter_num", "?")
        parts = [f"- iter {n}: {analysis.get('error_type', 'error')}"]
        if analysis.get("root_cause"):
            parts.append(f"  cause: {analysis['root_cause']}")
        if analysis.get("suggested_fix"):
            parts.append(f"  fix applied: {analysis['suggested_fix']}")
        entries.append("\n".join(parts))

    if not entries:
        return f"Branch '{name}' recorded no failure analyses."
    return f"## Failures recorded in branch '{name}':\n" + "\n".join(entries)
