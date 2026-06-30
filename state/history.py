"""
Iteration history and context formatting for LLM prompts.

These helpers build markdown sections that give the LLM awareness of past
iterations, parent branch context, and user messages.

Iteration data lives in ``iter_N/state.json``. User messages are per-iteration.
"""

import json
from pathlib import Path
from typing import Optional, Set

from utils.log import log
from utils.results import get_results_summary, load_reference_time

# Fields that can be requested per-node via the ``include`` parameter.
_ALL_HISTORY_FIELDS: Set[str] = {
    "plan",
    "kernel_code",
    "params_json",
    "run_output",
    "ncu_metrics",
    "decision",
    "feedback",
    "results_summary",
    "user_messages",
    "proposal",
}


def format_user_messages(iter_state: dict) -> str:
    """Format user messages from an iteration state for LLM prompts."""
    msgs = iter_state.get("user_messages", [])
    if not msgs:
        return ""
    lines = ["\n## User Messages:"]
    for m in msgs:
        lines.append(f"- {m['content']}")
    return "\n".join(lines)


def _load_past_iter_states(
    branch_path: Path,
    current_iter: int,
    max_iters: Optional[int] = None,
) -> list[dict]:
    """
    Load completed iteration states (excluding the current in-progress one).

    Also enriches each state with ``results_summary`` from
    ``iter{N}/results.json`` if not already present.
    """
    from config import get_output_dir

    branch_path = Path(branch_path)
    states = []

    for n in range(1, current_iter):
        state_file = branch_path / f"iter{n}" / "state.json"
        if not state_file.exists():
            continue
        try:
            with state_file.open("r") as f:
                snap = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log(f"Skipping corrupt state iter{n}: {e}", "WARN")
            continue

        # Enrich with results_summary if missing
        if not snap.get("results_summary"):
            results_path = branch_path / f"iter{n}" / "results.json"
            if results_path.exists():
                ref_time = load_reference_time(get_output_dir())
                snap["results_summary"] = get_results_summary(results_path, ref_time)

        states.append(snap)

    if max_iters is not None and len(states) > max_iters:
        states = states[-max_iters:]

    return states


def _preview(text: str, max_lines: int) -> str:
    lines = text.strip().split("\n")
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n... ({len(lines) - max_lines} more lines)"
    return preview


def _tail(text: str, max_lines: int) -> str:
    return "\n".join(text.strip().split("\n")[-max_lines:])


def _fmt_ncu(metrics: dict) -> str:
    return "**NCU Metrics:**\n" + "\n".join(
        f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}"
        for k, v in metrics.items()
    )


def _fmt_results(rs: dict) -> str:
    lines = [
        f"- Configs: {rs.get('num_successful', 0)}/{rs.get('num_total', 0)} passed"
    ]
    if rs.get("best_time_us") is not None:
        lines.append(f"- Best time: {rs['best_time_us']:.2f} µs")
    if rs.get("speedup") is not None:
        lines.append(f"- Speedup: {rs['speedup']:.2f}x")
    if rs.get("best_config"):
        cfg = ", ".join(f"{k}={v}" for k, v in rs["best_config"].items())
        lines.append(f"- Best config: {cfg}")
    return "**Results:**\n" + "\n".join(lines)


def _fmt_decision(d: dict) -> str:
    lines = [f"- Action: {d.get('action', '?')}"]
    if d.get("reasoning"):
        lines.append(f"- Reasoning: {d['reasoning']}")
    if d.get("performance_assessment"):
        lines.append(f"- Performance: {d['performance_assessment']}")
    if d.get("error_analysis"):
        ea = d["error_analysis"]
        lines.append(
            f"- Error: {ea.get('error_type', '?')} — {ea.get('root_cause', '')}"
        )
    return "**Decision:**\n" + "\n".join(lines)


_FIELD_FORMATTERS = {
    "plan": lambda v: f"**Plan:**\n{_preview(v, 5)}",
    "kernel_code": lambda v: f"**Kernel Code:**\n```cuda\n{v}\n```",
    "params_json": lambda v: f"**Tuning Parameters:**\n```json\n{v}\n```",
    "run_output": lambda v: (
        f"**Run Output (last 30 lines):**\n```\n{_tail(v, 30)}\n```"
    ),
    "ncu_metrics": _fmt_ncu,
    "results_summary": _fmt_results,
    "decision": _fmt_decision,
    "feedback": lambda v: f"**Feedback:** {v}",
    "proposal": lambda v: f"**Optimization Proposal:**\n{v}",
    "user_messages": lambda v: (
        "**User Messages:**\n" + "\n".join(f"- {m['content']}" for m in v)
    ),
}


def _format_iter_state(snap: dict, include: set) -> str:
    """Format a single iteration snapshot with only the requested fields."""
    parts = [f"### Iteration {snap.get('iter_num', '?')}"]
    for field, fmt in _FIELD_FORMATTERS.items():
        if field in include and snap.get(field):
            parts.append(fmt(snap[field]))
    return "\n\n".join(parts)


def _parse_field_depths(include, default_depth: int) -> dict[str, int]:
    """Parse history_fields into {field: max_depth} dict.

    Accepts either:
      - list of dicts: [{"name": "field", "limit": N}, {"name": "field2"}, ...]
      - iterable of strings: {"field1", "field2"} (all use default_depth)
    """
    depths = {}
    if isinstance(include, (list, tuple)):
        for item in include:
            field = item["name"]
            depths[field] = item.get("limit", default_depth)
    else:
        for field in include:
            depths[field] = default_depth
    return {f: d for f, d in depths.items() if f in _ALL_HISTORY_FIELDS}


def format_iteration_history(
    branch_path: str,
    current_iter: int,
    include,
    max_iters: Optional[int] = None,
) -> str:
    """
    Format past iteration history for LLM context.

    Args:
        branch_path: Path to the branch directory.
        current_iter: Current iteration number.
        include:      Fields to include. Either:
                      - list of dicts ``[{"name": "field", "limit": N}, ...]``
                      - iterable of strings ``{"field1", "field2"}`` (all default depth)
        max_iters:    Global max past iterations (``None`` = config default).

    Returns:
        Formatted markdown string, or ``""`` if no history.
    """
    from config import HISTORY_ITERS

    if max_iters is None:
        max_iters = HISTORY_ITERS

    field_depths = _parse_field_depths(include, max_iters)
    if not field_depths:
        return ""

    # Load enough snapshots to cover the deepest field
    load_depth = max(field_depths.values())
    snapshots = _load_past_iter_states(branch_path, current_iter, load_depth)
    if not snapshots:
        return ""

    total = len(snapshots)
    sections = ["\n## Iteration History:"]
    for idx, snap in enumerate(snapshots):
        age = total - 1 - idx  # 0 = most recent past iteration
        fields = {f for f, depth in field_depths.items() if age < depth}
        if fields:
            sections.append("---\n\n" + _format_iter_state(snap, fields))

    if len(sections) <= 1:
        return ""

    return "\n\n".join(sections)


def format_best_so_far(
    branch_path: str, current_iter: int, history_field_depths: Optional[dict] = None
) -> str:
    """
    Format the best-performing past iteration as a standalone context section.

    Returns kernel code, tuning params, and config from whichever iteration
    achieved the lowest best_time_us.  Avoids duplicating content that is
    already visible in iteration history by checking ``history_field_depths``
    (a ``{field: max_depth}`` dict matching the calling node's history config).
    """
    if history_field_depths is None:
        history_field_depths = {}
    from config import INCLUDE_BEST_SO_FAR

    if not INCLUDE_BEST_SO_FAR or not branch_path or current_iter <= 1:
        return ""

    snapshots = _load_past_iter_states(Path(branch_path), current_iter)
    if not snapshots:
        return ""

    # Find the snapshot with the best (lowest) time
    best_snap = None
    best_time = None
    for snap in snapshots:
        rs = snap.get("results_summary") or {}
        t = rs.get("best_time_us")
        if t is not None and (best_time is None or t < best_time):
            best_time = t
            best_snap = snap

    if best_snap is None:
        return ""

    rs = best_snap.get("results_summary", {})
    iter_num = best_snap.get("iter_num", "?")

    parts = [f"## Best Iteration So Far (iter {iter_num})"]

    # Results summary (always shown — small and not duplicated in history)
    lines = [f"- Best time: {best_time:.2f} µs"]
    if rs.get("speedup") is not None:
        lines.append(f"- Speedup: {rs['speedup']:.2f}x")
    if rs.get("best_config"):
        cfg = ", ".join(f"{k}={v}" for k, v in rs["best_config"].items())
        lines.append(f"- Best config: {cfg}")
    if rs.get("num_successful") is not None:
        lines.append(
            f"- Configs: {rs['num_successful']}/{rs.get('num_total', 0)} passed"
        )
    parts.append("\n".join(lines))

    # Check which fields are already visible for this iteration in history,
    # given the per-field depth limits from the calling node.
    total = len(snapshots)
    best_idx = next(
        (i for i, s in enumerate(snapshots) if s.get("iter_num") == iter_num), None
    )
    age = (total - 1 - best_idx) if best_idx is not None else total  # 0 = most recent

    def _field_visible_in_history(field: str) -> bool:
        depth = history_field_depths.get(field, 0)
        return age < depth

    kernel_visible = _field_visible_in_history("kernel_code")
    params_visible = _field_visible_in_history("params_json")

    if kernel_visible and params_visible:
        parts.append(
            f"*(See iteration {iter_num} in the history above for kernel code and params.)*"
        )
    else:
        if kernel_visible:
            parts.append(
                f"*(Kernel code for this iteration is shown in iteration {iter_num} above.)*"
            )
        elif best_snap.get("kernel_code"):
            parts.append(f"**Kernel Code:**\n```cuda\n{best_snap['kernel_code']}\n```")

        if params_visible:
            parts.append(
                f"*(Tuning params for this iteration are shown in iteration {iter_num} above.)*"
            )
        elif best_snap.get("params_json") and best_snap["params_json"] != "{}":
            parts.append(
                f"**Tuning Parameters:**\n```json\n{best_snap['params_json']}\n```"
            )

    return "\n\n".join(parts)


def format_iteration_summaries(branch_path: str, current_iter: int) -> str:
    """
    Collect one-line iteration summaries from all past decide outputs.

    These are cheap (one line each) and always included for ALL past
    iterations regardless of HISTORY_ITERS, giving the LLM full memory
    of what was tried and what happened.
    """
    if not branch_path or current_iter <= 1:
        return ""

    snapshots = _load_past_iter_states(Path(branch_path), current_iter)
    if not snapshots:
        return ""

    lines = []
    for snap in snapshots:
        n = snap.get("iter_num", "?")
        decision = snap.get("decision") or {}
        summary = decision.get("iteration_summary", "")
        if summary:
            lines.append(f"- iter {n}: {summary}")
        else:
            # Fallback: build a minimal summary from results
            rs = snap.get("results_summary") or {}
            t = rs.get("best_time_us")
            action = decision.get("action", "?")
            if t:
                lines.append(f"- iter {n}: → {t:.0f}µs (decision: {action})")
            else:
                lines.append(f"- iter {n}: → failed (decision: {action})")

    if not lines:
        return ""

    return "## Iteration Log:\n" + "\n".join(lines)


def format_existing_branches(output_dir: str) -> str:
    """
    Format a tree of all branches with names and short descriptions.

    Used by the decide node to avoid proposing duplicate sub-strategies.
    """
    if not output_dir:
        return ""

    branches_root = Path(output_dir) / "branches"
    if not branches_root.is_dir():
        return ""

    lines = []

    def _scan(bdir: Path, depth: int = 0):
        for d in sorted(bdir.iterdir()):
            if not d.is_dir():
                continue
            bfile = d / "branch.json"
            if not bfile.exists():
                continue
            try:
                with bfile.open() as f:
                    bm = json.load(f)
                s = bm.get("strategy", {})
                name = s.get("name", d.name)
                desc = s.get("description", "")
                indent = "  " * depth
                prefix = "├─ " if depth > 0 else ""
                lines.append(f"{indent}{prefix}{name} — {desc}")
            except (json.JSONDecodeError, OSError):
                continue
            sub = d / "branches"
            if sub.is_dir():
                _scan(sub, depth + 1)

    _scan(branches_root)

    if not lines:
        return ""

    return "## Existing Branches:\n" + "\n".join(lines)


def format_parent_context(branch_path: Optional[str]) -> str:
    """
    Format the parent branch's last iteration as context for a new sub-branch.

    The parent is derived from the directory nesting of ``branch_path``;
    top-level branches have no parent and produce no output.
    """
    from utils.files import get_parent_branch_dir

    parent_path = get_parent_branch_dir(branch_path)
    if parent_path is None:
        return ""

    # Find the latest iter_N/state.json in the parent
    iter_states = []
    for d in sorted(parent_path.iterdir()):
        if d.is_dir() and d.name.startswith("iter"):
            state_file = d / "state.json"
            if state_file.exists():
                with state_file.open("r") as f:
                    iter_states.append(json.load(f))

    if not iter_states:
        return ""

    iter_states.sort(key=lambda s: s.get("iter_num", 0))
    snap = iter_states[-1]

    include = _ALL_HISTORY_FIELDS
    formatted = _format_iter_state(snap, include)

    # Get parent strategy name
    parent_name = "unknown"
    branch_file = parent_path / "branch.json"
    if branch_file.exists():
        try:
            with branch_file.open("r") as f:
                ps = json.load(f)
            parent_name = ps.get("strategy", {}).get("name", "unknown")
        except (json.JSONDecodeError, OSError):
            pass

    return (
        f"\n## Parent Branch Context ('{parent_name}' — last iteration):\n{formatted}"
    )
