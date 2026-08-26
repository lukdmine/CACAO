"""Queries over a tuning run's full configuration set.

``results.json`` is the largest artifact a run produces — 347 KB / 103 configurations
in one real iteration — and ``get_results_summary`` reduces it to the single fastest
configuration plus a pass/total count. That is enough to report a result and not
enough to decide what to do next: whether the optimum sits at the edge of a declared
range (widen it) or on a plateau (the parameters do not touch the bottleneck), and
which parameter values fail outright, are both properties of the set, not of its
maximum.

Everything here returns compact markdown sized for a tool response. Nothing returns
the raw file: putting it in a prompt is what these functions exist to avoid.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from utils.results import get_computation_duration

# A tool response that grows without bound defeats the purpose. Sweeps over a
# parameter with a pathological number of distinct values get truncated with a note.
_MAX_ROWS = 24


class Record:
    """One configuration's outcome, flattened out of KTT's result shape."""

    __slots__ = ("config", "status", "duration", "registers", "local_mem")

    def __init__(self, raw: dict):
        self.config: Dict[str, str] = {
            p.get("Name", "?"): str(p.get("Value")) for p in raw.get("Configuration", [])
        }
        self.status: str = str(raw.get("Status", "Unknown"))
        self.duration: Optional[float] = (
            get_computation_duration(raw) if self.status == "Ok" else None
        )
        if self.duration is not None and self.duration == float("inf"):
            self.duration = None

        # Register count and local (spill) memory are per-kernel; the maximum across
        # the composite is what bounds occupancy, so that is what gets reported.
        regs, local = 0, 0
        for cr in raw.get("ComputationResults") or []:
            cd = cr.get("CompilationData") or {}
            regs = max(regs, int(cd.get("RegistersCount") or 0))
            local = max(local, int(cd.get("LocalMemorySize") or 0))
        self.registers: int = regs
        self.local_mem: int = local

    @property
    def ok(self) -> bool:
        return self.status == "Ok" and self.duration is not None


def load_records(results_path: Path) -> List[Record]:
    """Parse results.json into Records. Returns [] for a missing or corrupt file."""
    results_path = Path(results_path)
    if not results_path.exists():
        return []
    try:
        with results_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    return [Record(r) for r in (data.get("Results") or [])]


def parameter_names(records: List[Record]) -> List[str]:
    for r in records:
        if r.config:
            return list(r.config.keys())
    return []


def _fmt_us(v: Optional[float]) -> str:
    return f"{v:,.1f}" if v is not None else "—"


# -------------------------------------------------------------------------
# Queries
# -------------------------------------------------------------------------


def top_configs(results_path: Path, n: int = 10) -> str:
    """The n fastest configurations, with register pressure and spill memory.

    The gap between rank 1 and rank n is the signal: a sharp optimum rewards more
    search, a flat one says the parameters are not what limits this kernel.
    """
    records = load_records(results_path)
    if not records:
        return "No results available for this iteration."

    ok = sorted([r for r in records if r.ok], key=lambda r: r.duration)
    if not ok:
        return (
            f"No configuration succeeded ({len(records)} attempted). "
            "Use mode='failures' to see why."
        )

    n = max(1, min(n, _MAX_ROWS))
    names = parameter_names(records)
    head = ["| # | time µs | regs | spill B | " + " | ".join(names) + " |"]
    head.append("|---" * (4 + len(names)) + "|")
    for i, r in enumerate(ok[:n], 1):
        cells = " | ".join(r.config.get(k, "—") for k in names)
        head.append(
            f"| {i} | {_fmt_us(r.duration)} | {r.registers or '—'} | "
            f"{r.local_mem or 0} | {cells} |"
        )

    best, worst = ok[0].duration, ok[-1].duration
    spread = f"{worst / best:.2f}x" if best else "—"
    summary = (
        f"\n{len(ok)}/{len(records)} configurations succeeded. "
        f"Best {_fmt_us(best)} µs, slowest successful {_fmt_us(worst)} µs "
        f"(spread {spread})."
    )
    return "\n".join(head) + "\n" + summary


def param_sweep(results_path: Path, param: str) -> str:
    """Best and median time per value of one parameter, plus its failure count.

    A monotone column that has not turned by the last declared value means the range
    is clipped and the fix is the params region, not the kernel.
    """
    records = load_records(results_path)
    if not records:
        return "No results available for this iteration."

    names = parameter_names(records)
    if param not in names:
        return f"Unknown parameter '{param}'. Available: {', '.join(names) or '(none)'}."

    buckets: Dict[str, List[Record]] = {}
    for r in records:
        buckets.setdefault(r.config.get(param, "—"), []).append(r)

    def sort_key(v: str):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    rows = ["| " + param + " | n | ok | best µs | median µs |", "|---|---|---|---|---|"]
    for value in sorted(buckets, key=sort_key)[:_MAX_ROWS]:
        group = buckets[value]
        good = [r.duration for r in group if r.ok]
        best = min(good) if good else None
        med = statistics.median(good) if good else None
        rows.append(
            f"| {value} | {len(group)} | {len(good)} | {_fmt_us(best)} | {_fmt_us(med)} |"
        )

    note = ""
    if len(buckets) > _MAX_ROWS:
        note = f"\n\n({len(buckets) - _MAX_ROWS} further values omitted.)"

    ranked = [
        (v, min((r.duration for r in g if r.ok), default=None)) for v, g in buckets.items()
    ]
    ranked = [(v, t) for v, t in ranked if t is not None]
    edge = ""
    if len(ranked) >= 2:
        ranked.sort(key=lambda vt: sort_key(vt[0]))
        best_value = min(ranked, key=lambda vt: vt[1])[0]
        if best_value in (ranked[0][0], ranked[-1][0]):
            edge = (
                f"\n\nThe best value ({param}={best_value}) is at the edge of the "
                "declared range — consider extending it in the params region."
            )
    return "\n".join(rows) + note + edge


def failure_breakdown(results_path: Path) -> str:
    """Status counts, and which parameter values never produce a working kernel.

    'N/M passed' says something is wrong. This says what.
    """
    records = load_records(results_path)
    if not records:
        return "No results available for this iteration."

    statuses: Dict[str, int] = {}
    for r in records:
        statuses[r.status] = statuses.get(r.status, 0) + 1

    lines = [f"{len(records)} configurations:"]
    for status, count in sorted(statuses.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {status}: {count}")

    failing = [r for r in records if not r.ok]
    if not failing:
        return "\n".join(lines) + "\n\nEvery configuration succeeded."

    lines.append("\nParameter values that never produced a working configuration:")
    found = False
    for name in parameter_names(records):
        buckets: Dict[str, List[Record]] = {}
        for r in records:
            buckets.setdefault(r.config.get(name, "—"), []).append(r)
        dead = [
            f"{name}={v} ({len(g)} configs)"
            for v, g in buckets.items()
            if g and not any(r.ok for r in g)
        ]
        if dead:
            found = True
            lines.append("- " + "; ".join(sorted(dead)))
    if not found:
        lines.append("- none; failures are spread across all parameter values")
    return "\n".join(lines)


def describe(results_path: Path, mode: str = "top", param: Optional[str] = None, n: int = 10) -> str:
    """Dispatch for the tool layer."""
    if mode == "top":
        return top_configs(results_path, n)
    if mode == "sweep":
        if not param:
            names = parameter_names(load_records(results_path))
            return f"mode='sweep' needs a param. Available: {', '.join(names) or '(none)'}."
        return param_sweep(results_path, param)
    if mode == "failures":
        return failure_breakdown(results_path)
    return f"Unknown mode '{mode}'. Use 'top', 'sweep', or 'failures'."
