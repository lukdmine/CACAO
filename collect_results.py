#!/usr/bin/env python3
"""Collect per-(problem, model) evaluation metrics into a CSV + printed table.

For each `problems/<base_problem>_<model_suffix>/` directory the script extracts:

    problem              base problem name (without model suffix)
    input_size           compact descriptor of the scalars in problem.yaml
    model                human model name (kimi / qwen / glm / ...)
    gpu                  GPU reported by KTT in the best iter's results.json
    best_time_us         best validated kernel time across the whole tree
    throughput           work_units / best_time_us  (Mpairs/s or Mvalues/s)
    metric_unit          unit string (Mpairs/s | Mvalues/s)
    iters_to_best_path   iterations consumed from root of the branch tree
                         to the iter that produced the best result
    wall_clock_min       elapsed minutes between first and last log entry
                         in run.log (handles single midnight wrap)
    prompt_tokens        prompt token count from token_usage.json
    completion_tokens    completion token count from token_usage.json
    tokens_total         prompt + completion tokens
    api_calls            number of LLM API calls
    tokens_per_call      tokens_total / api_calls
    correctness_rate     fraction of finished iters that produced at least
                         one validated kernel configuration
    strategies_explored  number of branches in the tree

Run from the repo root:

    python collect_results.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable, Optional

import yaml

SCRIPT_DIR = Path(__file__).parent
PROBLEMS_DIR = SCRIPT_DIR / "problems"
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
OUTPUT_CSV = SCRIPT_DIR / "eval_results.csv"

MODELS = {
    "kimi": "kimi-k2.6",
    "qwen": "qwen3.5",
    "glm": "glm-5",
}

TS_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]")

# (metric_unit, work_units(scalars))
METRICS: dict[str, tuple[str, Callable[[dict], float]]] = {
    "student_2022_galaxies": ("Mpairs/s", lambda s: s["N"] * (s["N"] - 1) / 2),
    "student_2024_savings": ("Mvalues/s", lambda s: s["CLIENTS"] * s["PERIODS"]),
    "student_2025_moving_average": ("Mvalues/s", lambda s: s["N"]),
}

INPUT_SIZE_FMT: dict[str, Callable[[dict], str]] = {
    "student_2022_galaxies": lambda s: f"{s['N']} stars",
    "student_2024_savings": lambda s: f"{s['CLIENTS']}x{s['PERIODS']}",
    "student_2025_moving_average": lambda s: f"{s['N']}x{s['R']}",
}


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def problem_scalars(base_problem_dir: Path) -> dict:
    config = load_yaml(base_problem_dir / "problem.yaml")
    return {s["name"]: s["value"] for s in config.get("scalars", [])}


# ---------------------------------------------------------------------------
# branch / iter scanning
# ---------------------------------------------------------------------------


def all_branches(output_dir: Path) -> list[Path]:
    """Every branch.json under output/branches (recursive)."""
    branches_root = output_dir / "branches"
    if not branches_root.exists():
        return []
    return sorted(branches_root.rglob("branch.json"))


def iter_dirs(branch_dir: Path) -> list[Path]:
    """Iter directories inside a branch (iter1, iter2, ...). Excludes nested branches."""
    return sorted(
        d for d in branch_dir.iterdir() if d.is_dir() and d.name.startswith("iter")
    )


def find_best_branch(output_dir: Path) -> tuple[Optional[Path], Optional[dict]]:
    """Return (branch_dir, branch_manifest) of the branch with the lowest best_time_us."""
    best_dir = None
    best_data = None
    best_t = float("inf")
    for bj in all_branches(output_dir):
        data = load_json(bj) or {}
        t = data.get("best_time_us")
        if t is None:
            continue
        if t < best_t:
            best_t = t
            best_dir = bj.parent
            best_data = data
    return best_dir, best_data


def best_iter_in_branch(branch_dir: Path) -> tuple[Optional[int], Optional[dict]]:
    """Iter index (1-based) and state.json of the iter with the lowest validated time."""
    best_idx = None
    best_state = None
    best_t = float("inf")
    for d in iter_dirs(branch_dir):
        state = load_json(d / "state.json") or {}
        rs = state.get("results_summary") or {}
        t = rs.get("best_time_us")
        if t is None:
            continue
        if t < best_t:
            best_t = t
            best_idx = state.get("iter_num") or int(d.name.replace("iter", ""))
            best_state = state
    return best_idx, best_state


def iters_to_best_path(output_dir: Path, best_branch_dir: Path, best_iter: int) -> int:
    """Total iterations from the root of the tree to the best iter.

    Uses ``path_iters_consumed`` on the branch manifest when present (newer runs).
    Falls back to walking up the parent chain (derived from directory nesting:
    sub-branches live at ``<parent>/branches/<name>``) and counting iter folders
    along the way for older runs that lacked the field.
    """
    manifest = load_json(best_branch_dir / "branch.json") or {}
    pic = manifest.get("path_iters_consumed")
    if pic is not None:
        return pic + best_iter

    # legacy fallback: walk the chain manually
    total = best_iter
    cur = best_branch_dir
    while True:
        parent_dir = cur.parent.parent
        if cur.parent.name != "branches" or not (parent_dir / "branch.json").exists():
            break
        # count fully-completed iters in the parent before we branched off.
        # without an explicit branch-point field, approximate as the number of
        # iter dirs that exist in the parent — this overcounts if the parent
        # kept running after branching, but it's the best we can do.
        total += len(iter_dirs(parent_dir))
        cur = parent_dir
    return total


def detect_gpu(
    best_branch_dir: Path, best_iter: int, output_dir: Path
) -> Optional[str]:
    """Authoritative: KTT's reported device in the best iter's results.json."""
    iter_dir = best_branch_dir / f"iter{best_iter}"
    rj = load_json(iter_dir / "results.json")
    if rj:
        device = (rj.get("Metadata") or {}).get("Device")
        if device:
            return device
    # fallback: any results.json in the run
    for rj_path in output_dir.rglob("results.json"):
        rj = load_json(rj_path)
        if rj:
            device = (rj.get("Metadata") or {}).get("Device")
            if device:
                return device
    # last resort: stale gpu_info from analysis time
    ctx = load_json(output_dir / "context.json")
    if ctx:
        return (ctx.get("gpu_info") or {}).get("model")
    return None


def wall_clock_minutes(output_dir: Path) -> Optional[float]:
    """Elapsed minutes between first and last [HH:MM:SS] entry in run.log."""
    log = output_dir / "run.log"
    if not log.exists():
        return None
    first = last = None
    with log.open() as f:
        for line in f:
            m = TS_RE.match(line)
            if not m:
                continue
            t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            if first is None:
                first = t
            last = t
    if first is None or last is None or first == last:
        return None
    delta = last - first
    if delta < 0:
        delta += 24 * 3600  # midnight wrap
    return delta / 60.0


def correctness_rate(output_dir: Path) -> tuple[float, int, int]:
    """Fraction of finished iters that produced at least one validated config."""
    finished = 0
    successful = 0
    for state_path in output_dir.rglob("state.json"):
        state = load_json(state_path) or {}
        # only count iters that reached a terminal-ish state (i.e. tuner ran)
        rs = state.get("results_summary") or {}
        if rs.get("num_total") is None:
            continue
        finished += 1
        if (rs.get("num_successful") or 0) > 0:
            successful += 1
    rate = successful / finished if finished else 0.0
    return rate, successful, finished


# ---------------------------------------------------------------------------
# main collection
# ---------------------------------------------------------------------------


def _collect_run(
    base_problem: str,
    output_dir: Path,
    experiment: Optional[str],
) -> Optional[dict]:
    """Extract one result row from a single problem run's output directory."""
    base_dir = PROBLEMS_DIR / base_problem
    scalars = problem_scalars(base_dir)
    metric_unit, work_fn = METRICS[base_problem]
    size_fn = INPUT_SIZE_FMT[base_problem]
    try:
        work_units = work_fn(scalars)
        input_size = size_fn(scalars)
    except KeyError:
        work_units = None
        input_size = "?"

    # infer model name from the run directory name
    run_dir_name = output_dir.parent.name  # e.g. "student_2024_savings_kimi"
    suffix = run_dir_name[len(base_problem) + 1 :]  # strip "<base_problem>_"
    model_name = MODELS.get(suffix, suffix)

    tokens = load_json(output_dir / "token_usage.json") or {}
    api_calls = tokens.get("api_calls") or 0
    prompt_tokens = tokens.get("prompt_tokens") or 0
    completion_tokens = tokens.get("completion_tokens") or 0
    total_tokens = tokens.get("total_tokens") or (prompt_tokens + completion_tokens)
    tokens_per_call = (total_tokens / api_calls) if api_calls else None
    wall_min = wall_clock_minutes(output_dir)

    best_branch_dir, best_branch = find_best_branch(output_dir)
    strategies_explored = len(all_branches(output_dir))
    corr_rate, n_ok, n_done = correctness_rate(output_dir)

    best_time = None
    throughput = None
    iters_path = None
    gpu = None

    if best_branch_dir is not None and best_branch is not None:
        best_iter, _ = best_iter_in_branch(best_branch_dir)
        if best_iter is None:
            best_iter = best_branch.get("current_iter") or 1
        best_time = best_branch.get("best_time_us")
        if best_time and work_units:
            throughput = work_units / best_time
        iters_path = iters_to_best_path(output_dir, best_branch_dir, best_iter)
        gpu = detect_gpu(best_branch_dir, best_iter, output_dir)

    if gpu is None:
        gpu = detect_gpu(output_dir, 1, output_dir)

    return {
        "experiment": experiment or "",
        "problem": base_problem,
        "input_size": input_size,
        "model": model_name,
        "gpu": gpu,
        "best_time_us": round(best_time, 2) if best_time else None,
        "throughput": round(throughput, 1) if throughput else None,
        "metric_unit": metric_unit,
        "iters_to_best_path": iters_path,
        "wall_clock_min": round(wall_min, 1) if wall_min else None,
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": completion_tokens or None,
        "tokens_total": total_tokens or None,
        "api_calls": api_calls or None,
        "tokens_per_call": round(tokens_per_call) if tokens_per_call else None,
        "correctness_rate": f"{n_ok}/{n_done}" if n_done else "0/0",
        "strategies_explored": strategies_explored,
    }


def _scan_search_root(search_root: Path, experiment: Optional[str]) -> list[dict]:
    """Collect rows from all matching problem dirs inside search_root."""
    rows = []
    if not search_root.exists():
        return rows
    for d in sorted(search_root.iterdir()):
        if not d.is_dir():
            continue
        for base_problem in METRICS:
            if d.name.startswith(f"{base_problem}_"):
                output_dir = d / "output"
                if output_dir.exists():
                    row = _collect_run(base_problem, output_dir, experiment)
                    if row is not None:
                        rows.append(row)
    return rows


def collect() -> list[dict]:
    rows: list[dict] = []

    # current problems/
    rows.extend(_scan_search_root(PROBLEMS_DIR, experiment=None))

    # archived experiments/
    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if exp_dir.is_dir():
                rows.extend(_scan_search_root(exp_dir, experiment=exp_dir.name))

    return rows


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def short_problem(name: str) -> str:
    return name.replace("student_", "").replace("_", " ")


def short_gpu(name: Optional[str]) -> str:
    if not name:
        return "-"
    return name.replace("NVIDIA GeForce ", "")


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results found.")
        return

    cols = [
        ("experiment", lambda r: r["experiment"] or "current"),
        ("problem", lambda r: short_problem(r["problem"])),
        ("input", lambda r: r["input_size"]),
        ("model", lambda r: r["model"]),
        ("gpu", lambda r: short_gpu(r["gpu"])),
        ("best_us", lambda r: r["best_time_us"] if r["best_time_us"] else "-"),
        (
            "throughput",
            lambda r: (
                f"{r['throughput']} {r['metric_unit']}" if r["throughput"] else "-"
            ),
        ),
        (
            "iters",
            lambda r: r["iters_to_best_path"] if r["iters_to_best_path"] else "-",
        ),
        ("wall_min", lambda r: r["wall_clock_min"] if r["wall_clock_min"] else "-"),
        ("prompt_tok", lambda r: r["prompt_tokens"] if r["prompt_tokens"] else "-"),
        (
            "compl_tok",
            lambda r: r["completion_tokens"] if r["completion_tokens"] else "-",
        ),
        ("calls", lambda r: r["api_calls"] if r["api_calls"] else "-"),
        ("tok/call", lambda r: r["tokens_per_call"] if r["tokens_per_call"] else "-"),
        ("correct", lambda r: r["correctness_rate"]),
        ("branches", lambda r: r["strategies_explored"]),
    ]

    header = [c[0] for c in cols]
    cells = [[str(fn(r)) for _, fn in cols] for r in rows]
    widths = [
        max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(header)
    ]

    def fmt(values):
        return "  ".join(v.rjust(w) for v, w in zip(values, widths, strict=True))

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print(fmt(row))


def write_csv(rows: list[dict]) -> None:
    fields = [
        "experiment",
        "problem",
        "input_size",
        "model",
        "gpu",
        "best_time_us",
        "throughput",
        "metric_unit",
        "iters_to_best_path",
        "wall_clock_min",
        "prompt_tokens",
        "completion_tokens",
        "tokens_total",
        "api_calls",
        "tokens_per_call",
        "correctness_rate",
        "strategies_explored",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {OUTPUT_CSV.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    rows = collect()
    print_table(rows)
    write_csv(rows)
