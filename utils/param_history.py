"""What changed in the tuning space between two iterations' PARAMS regions.

A rewrite that trims a parameter's values (or drops the parameter) shrinks the space
the tuner searches, and nothing else in the authoring step says so: the kernel still
compiles and the narrower space still tunes. Naming it at the compile check lets the
model see the cost of the cut while it can still undo it.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.nvrtc import parse_parameters

Change = Dict[str, Tuple[List[str], List[str]]]


def _key(value: str):
    """Numeric where it parses, so "16" and "16.0" are one value."""
    try:
        return float(value)
    except ValueError:
        return value


def _keys(values: List[str]) -> set:
    return {_key(v) for v in values}


@dataclass
class ParamDiff:
    narrowed: Change = field(default_factory=dict)
    widened: Change = field(default_factory=dict)
    dropped: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)

    def render(self, include_widened: bool = False) -> str:
        """Plain-text report; "" when there is nothing to report."""
        lines: List[str] = []
        for name, (old, new) in self.narrowed.items():
            lines.append(f"- {name} narrowed: {{{', '.join(old)}}} -> {{{', '.join(new)}}}")
        for name in self.dropped:
            lines.append(f"- {name} dropped (it is no longer tuned)")
        if include_widened:
            for name, (old, new) in self.widened.items():
                lines.append(f"- {name} widened: {{{', '.join(old)}}} -> {{{', '.join(new)}}}")
        return "\n".join(lines)


def diff_parameters(old_src: str, new_src: str) -> ParamDiff:
    old = parse_parameters(old_src)
    new = parse_parameters(new_src)
    diff = ParamDiff()
    for name, old_values in old.items():
        if name not in new:
            diff.dropped.append(name)
            continue
        new_values = new[name]
        old_set, new_set = _keys(old_values), _keys(new_values)
        if new_set < old_set:
            diff.narrowed[name] = (old_values, new_values)
        elif new_set > old_set:
            diff.widened[name] = (old_values, new_values)
    diff.added = [name for name in new if name not in old]
    return diff


def _iteration_dirs(branch_path: Path) -> List[Tuple[int, Path]]:
    """``iterN`` directories ordered by N, so iter10 follows iter9."""
    found = []
    for d in Path(branch_path).glob("iter*"):
        m = re.fullmatch(r"iter(\d+)", d.name)
        if m and d.is_dir():
            found.append((int(m.group(1)), d))
    return sorted(found)


def _params_of(iter_dir: Path) -> Optional[str]:
    """An iteration's PARAMS region: its region file, else the one spliced into framework.cpp."""
    from utils.framework import extract_regions

    region = iter_dir / "region_params.cpp"
    try:
        if region.exists():
            return region.read_text(encoding="utf-8")
        framework = iter_dir / "framework.cpp"
        if framework.exists():
            return extract_regions(framework.read_text(encoding="utf-8")).get("params")
    except OSError:
        pass
    return None


def _best_by_value(iter_dir: Path, problem_dir: Path) -> Dict[str, Dict[object, float]]:
    """{param: {value key: best time in us}} from the iteration's primary-case results."""
    from utils.cases import primary_case_dir
    from utils.landscape import load_records

    best: Dict[str, Dict[object, float]] = {}
    for rec in load_records(primary_case_dir(problem_dir, iter_dir) / "results.json"):
        if not rec.ok:
            continue
        for name, value in rec.config.items():
            slot = best.setdefault(name, {})
            k = _key(value)
            if k not in slot or rec.duration < slot[k]:
                slot[k] = rec.duration
    return best


# A summary tool, not a file read: a long branch with many parameters would otherwise put
# tens of KB into the model's context on every call.
_MAX_CHARS = 12_000


def _span(first: int, last: int) -> str:
    return f"iter{first}" if first == last else f"iter{first}-iter{last}"


def parameter_history(
    branch_path: Path, current_params_src: str, problem_dir: Path, meta: dict
) -> str:
    """Every value each parameter has had across a branch's iterations, and the dropped ones.

    Finalize widens or restores a space that authoring narrowed; this is what shows
    which values were tried and what the best of each measured. Consecutive iterations
    that declared a parameter identically (same values, same best times) share one line
    naming their range, and the output is capped at ``_MAX_CHARS`` with a note.

    ``meta`` is part of the planned interface; the primary-case lookup currently reads
    problem.yaml itself and does not use it.
    """
    current = parse_parameters(current_params_src)
    history = []  # (iteration number, declared params, best time per value)
    for n, d in _iteration_dirs(branch_path):
        src = _params_of(d)
        if src is None:
            continue
        history.append((n, parse_parameters(src), _best_by_value(d, problem_dir)))

    lines: List[str] = []
    for name, values in current.items():
        lines.append(f"{name}: now {{{', '.join(values)}}}")
        runs: List[list] = []  # [first iter, last iter, rendered values]
        open_run = False  # only iterations that all declared the parameter may merge
        for n, declared, best in history:
            if name not in declared:
                open_run = False
                continue
            times = best.get(name, {})
            shown = []
            for v in declared[name]:
                t = times.get(_key(v))
                shown.append(f"{v} ({t:,.1f} us)" if t is not None else v)
            text = ", ".join(shown)
            if open_run and runs[-1][2] == text:
                runs[-1][1] = n
            else:
                runs.append([n, n, text])
            open_run = True
        for first, last, text in runs:
            lines.append(f"  {_span(first, last)}: {text}")

    tail: List[str] = []
    last_seen: Dict[str, int] = {}
    for n, declared, _ in history:
        for name in declared:
            last_seen[name] = n
    dropped = [(name, n) for name, n in last_seen.items() if name not in current]
    if dropped:
        tail.append("No longer declared:")
        tail += [f"- {name} (last declared in iter{n})" for name, n in dropped]

    if not lines and not tail:
        return "No parameters are declared."
    note = f"... (truncated at {_MAX_CHARS} characters; later entries omitted)"
    tail_text = "\n".join(tail)
    if len(tail_text) > _MAX_CHARS:
        tail_text = tail_text[: _MAX_CHARS - len(note) - 1].rsplit("\n", 1)[0] + "\n" + note
    body = "\n".join(lines)
    room = _MAX_CHARS - len(tail_text) - 1
    if len(body) > room:
        keep = body[: max(room - len(note) - 1, 0)].rsplit("\n", 1)[0] if room > len(note) else ""
        body = (keep + "\n" + note) if keep else note
    return "\n".join(part for part in (body, tail_text) if part)
