"""Read-only views of other branches' results, tolerant of concurrent writers.

Only decided iterations are read — those below a running branch's ``current_iter``, and
its last one once the branch has terminated — so an iteration a worker is mid-write on
is never observed. No function here exposes another branch's kernel or framework
regions, at any access level: parallel branches are parallel bets, and one that can
read the leader's code converges on it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

# A prompt-resident index has to stay small enough to be worth its space. Rows are
# sorted fastest-first, so a deep tree keeps the branches worth investigating and drops
# the tail rather than truncating arbitrarily.
_MAX_INDEX_ROWS = 8

# After these the worker stops advancing current_iter, so it names the last decided
# iteration rather than one in flight.
_TERMINAL_STATUSES = ("success", "failed", "branching")


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
        iters = _completed_count(manifest)
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


def _completed_count(manifest: dict) -> int:
    """How many iterations of this branch have been decided.

    current_iter is an exclusive bound only while the branch runs; a finished branch
    keeps it pointing at its last decided iteration.
    """
    current = int(manifest.get("current_iter") or 1)
    if manifest.get("status") in _TERMINAL_STATUSES:
        return max(current, 0)
    return max(current - 1, 0)


def _completed_iters(branch_path: Path, manifest: dict) -> List[dict]:
    """Decided iterations only, oldest first."""
    snaps = []
    for n in range(1, _completed_count(manifest) + 1):
        snap = _load_json(branch_path / f"iter{n}" / "state.json")
        if snap is not None:
            snaps.append(snap)
    return snaps


def branch_log(output_dir, name: str) -> str:
    """What a branch tried, one line per iteration, with the outcome.

    Deliberately excludes kernel code: this is the pruning signal, not a template to
    copy. branch_errors is the deepest read there is; nothing reaches a sibling's files.
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


# A failure analysis is free-form model output, written by a prompt that carried the
# compiler's stderr — which quotes the failing branch's source. Uncapped, one call
# splices a sibling's code into this branch's context, which is both the budget blow-up
# the pull tools exist to avoid and a breach of the no-code-sharing rule. Every other
# cross-branch surface is bounded; so is this one.
_MAX_ERROR_ENTRIES = 6
_MAX_ERROR_FIELD_CHARS = 400

_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)


def _condense(text) -> str:
    """A prompt-sized line of an analysis field: fences dropped, whitespace flat, capped."""
    if not isinstance(text, str):
        return ""
    # An unclosed fence takes the rest of the field with it — that tail is the code.
    stripped = _FENCED_BLOCK.sub(" ", text).split("```", 1)[0]
    flat = " ".join(stripped.split())
    if len(flat) > _MAX_ERROR_FIELD_CHARS:
        flat = flat[:_MAX_ERROR_FIELD_CHARS].rstrip() + " …"
    return flat


def branch_errors(output_dir, name: str) -> str:
    """What went wrong in another branch and what it concluded caused it.

    The deepest cross-branch read there is — a failure analysis prunes a dead end
    without supplying a solution. Capped and condensed on the way out, so what the
    reader gets is bounded regardless of what the writing branch's model produced.
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
        parts = [f"- iter {n}: {_condense(analysis.get('error_type')) or 'error'}"]
        cause = _condense(analysis.get("root_cause"))
        if cause:
            parts.append(f"  cause: {cause}")
        fix = _condense(analysis.get("suggested_fix"))
        if fix:
            parts.append(f"  fix applied: {fix}")
        entries.append("\n".join(parts))

    if not entries:
        return f"Branch '{name}' recorded no failure analyses."

    shown = entries[-_MAX_ERROR_ENTRIES:]
    header = f"## Failures recorded in branch '{name}':"
    if len(entries) > len(shown):
        header += f"  (most recent {len(shown)}; {len(entries) - len(shown)} earlier omitted)"
    return header + "\n" + "\n".join(shown)
