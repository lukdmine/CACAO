"""
Merge node - aggregates results from all branches.

This node runs after all parallel branches have completed.
It scans the output directory for branch results and selects the best one.
"""

from pathlib import Path
from typing import List, Optional

from config import get_output_dir, get_problem_dir
from utils.cases import primary_case_dir
from utils.files import save_json
from utils.log import log


def find_branch_results(branches_dir: Path) -> List[dict]:
    """Summarize every branch.json under ``branches_dir``, sub-branches included."""
    results = []

    if not branches_dir.exists():
        return results

    from state import load_branch_manifest

    from utils.results import (
        get_results_summary,
        calculate_speedup,
        load_reference_time,
    )

    ref_time = load_reference_time(get_output_dir())

    for branch_file in branches_dir.rglob("branch.json"):
        try:
            branch_path = branch_file.parent
            manifest = load_branch_manifest(branch_path)
            strategy = manifest.strategy or {}
            # Unconditional: the winning parameters exist only in an iteration's
            # results.json, so scanning only when the manifest lacks a time lost them.
            scanned_time = None
            best_config = None
            for iter_dir in sorted(branch_path.glob("iter*")):
                if not iter_dir.is_dir():
                    continue
                # Multi-case problems put each case's results in its own subdirectory,
                # so the iteration root holds no results.json at all. Anchor on the
                # primary case, as the tree API does.
                results_path = primary_case_dir(get_problem_dir(), iter_dir) / "results.json"
                summary = get_results_summary(results_path, ref_time)
                t_us = summary.get("best_time_us")
                if t_us is not None and (scanned_time is None or t_us < scanned_time):
                    scanned_time = t_us
                    best_config = summary.get("best_config")

            best_time_us = (
                manifest.best_time_us if manifest.best_time_us is not None else scanned_time
            )

            speedup = manifest.speedup
            if not speedup and best_time_us and ref_time:
                speedup = calculate_speedup(best_time_us, ref_time)

            results.append(
                {
                    "branch_name": strategy.name,
                    "branch_path": str(branch_path),
                    "status": manifest.status,
                    "best_time_us": best_time_us,
                    "best_config": best_config,
                    "speedup": speedup,
                    "iterations": manifest.current_iter,
                }
            )
        except Exception as e:
            log(f"Failed to load branch {branch_file}: {e}", "WARN")

    return results


def _best_finalist(finalize, branch_results: List[dict]) -> Optional[dict]:
    """The finalist whose reported variant scores best, as a branch-result-shaped dict.

    ``best_branch`` is the finalist's source branch name: the strategy name of the
    branch result whose path ends with the finalist's ``source_branch`` (a path relative
    to output/), else the last component of that path when the branch is not on disk.
    None when no finalist has a reported variant with a time.
    """
    from utils.branch_bests import summary_rank_key

    best = None
    for f in finalize.finalists:
        variant = f.finalized if f.reported == "finalized" else f.source
        summary = variant.summary if variant else None
        if not summary or summary.get("best_time_us") is None:
            continue
        if best is None or summary_rank_key(summary) < summary_rank_key(best["summary"]):
            best = {"finalist": f, "summary": summary}
    if best is None:
        return None
    f = best["finalist"]
    suffix = "/" + f.source_branch.strip("/")
    name = next(
        (
            r.get("branch_name")
            for r in branch_results
            if str(r.get("branch_path", "")).endswith(suffix)
        ),
        None,
    ) or f.source_branch.rstrip("/").rsplit("/", 1)[-1]
    return {
        "branch_name": name,
        "best_config": best["summary"].get("best_config"),
        "best_time_us": best["summary"].get("best_time_us"),
        "speedup": best["summary"].get("speedup"),
    }


def _finalize_section(finalize) -> dict:
    def speedup(variant):
        return ((variant.summary or {}).get("speedup")) if variant else None

    finalists = []
    for f in finalize.finalists:
        reported = f.finalized if f.reported == "finalized" else f.source
        finalists.append(
            {
                "name": f.name,
                "source_branch": f.source_branch,
                "source_iter": f.source_iter,
                "reported": f.reported,
                "reported_dir": f.reported_dir,
                "tags": reported.tags.model_dump() if reported and reported.tags else None,
                "source_speedup": speedup(f.source),
                "finalized_speedup": speedup(f.finalized),
            }
        )
    return {
        "status": finalize.status,
        "attempt": finalize.attempt,
        "finalists": finalists,
        "category_bests": finalize.category_bests,
        "overview": finalize.overview.markdown if finalize.overview else None,
    }


def build_final_summary(
    branch_results: List[dict], best: Optional[dict], finalize=None
) -> dict:
    """The contents of ``output/final_results.json``. One builder; there were two,
    with different shapes, and ``--best`` overwrote the richer with the poorer.

    With a finalize manifest the top-level best is the better of the best branch and the
    best finalist's reported variant (a tie keeps the branch), and the file gains
    ``best_source`` and a ``finalize`` section. Without one the shape is unchanged."""
    best_source = "branch"
    finalist = None
    if finalize is not None:
        from utils.branch_bests import summary_rank_key

        finalist = _best_finalist(finalize, branch_results)
        if finalist is not None and (
            best is None or summary_rank_key(finalist) < summary_rank_key(best)
        ):
            best, best_source = finalist, "finalize"
    best = best or {}

    summary = {
        "best_branch": best.get("branch_name"),
        "best_config": best.get("best_config"),
        "best_time_us": best.get("best_time_us"),
        "speedup": best.get("speedup"),
        "total_branches": len(branch_results),
        "all_branches": [
            {
                "name": r.get("branch_name"),
                "path": r.get("branch_path"),
                "status": r.get("status"),
                "best_time_us": r.get("best_time_us"),
                "speedup": r.get("speedup"),
                "iterations": r.get("iterations"),
            }
            for r in branch_results
        ],
    }
    if finalize is not None:
        summary["best_source"] = best_source
        summary["finalize"] = _finalize_section(finalize)
    return summary


def get_best_branch_result(results: List[dict]) -> Optional[dict]:
    """The fastest branch, or None when no branch produced a time."""
    # Include success and branching (parent branched but has valid results)
    candidates = [
        r
        for r in results
        if r.get("status") in ("success", "branching")
        and r.get("best_time_us") is not None
    ]

    if not candidates:
        return None

    return min(candidates, key=lambda r: r["best_time_us"])


def print_iteration_timeline(branches_dir: Path) -> None:
    """Print a per-branch iteration timeline with a normalized ASCII bar chart."""
    import json as _json
    from state import load_branch_manifest
    from utils.results import get_results_summary, load_reference_time

    ref_time = load_reference_time(get_output_dir())
    BAR_W = 32

    # First pass: collect all data so we can normalise the bar across all branches
    all_branches = []
    all_times_ms = []

    for branch_file in sorted(branches_dir.rglob("branch.json")):
        branch_path = branch_file.parent
        try:
            manifest = load_branch_manifest(branch_path)
        except Exception:
            continue
        name = manifest.strategy.name if manifest.strategy else branch_path.name

        iters = []
        for i in range(1, manifest.current_iter + 1):
            iter_dir = branch_path / f"iter{i}"
            results_path = primary_case_dir(get_problem_dir(), iter_dir) / "results.json"
            summary = get_results_summary(results_path, ref_time)
            ms = summary["best_time_us"] / 1000 if summary["best_time_us"] else None

            action = "?"
            state_path = iter_dir / "state.json"
            if state_path.exists():
                try:
                    s = _json.loads(state_path.read_text(encoding="utf-8"))
                    action = (s.get("decision") or {}).get("action", "?")
                except Exception:
                    pass

            iters.append((i, ms, action))
            if ms is not None:
                all_times_ms.append(ms)

        all_branches.append((name, iters))

    if not all_branches:
        return

    global_best = min(all_times_ms) if all_times_ms else 1

    print("\n" + "=" * 60)
    print("  ITERATION TIMELINE")
    print("=" * 60)
    for name, iters in all_branches:
        print(f"\n  [{name}]")
        branch_best = None
        for i, ms, action in iters:
            if ms is not None:
                new_best = branch_best is None or ms < branch_best
                if new_best:
                    branch_best = ms
                filled = min(BAR_W, int(BAR_W * global_best / ms))
                bar = "█" * filled + "░" * (BAR_W - filled)
                star = " ★" if new_best else ""
                print(f"    iter{i:>2}: {ms:6.2f} ms  [{bar}]  {action:8s}{star}")
            else:
                print(
                    f"    iter{i:>2}:    --      [{' ' * BAR_W}]  {action:8s}  (no result)"
                )


def merge_node() -> dict:
    """Print the end-of-run summary and write ``final_results.json``.

    The one owner of the success predicate: cli.py's normal path and its ``--best``
    bypass both come through here. They used to keep their own copy, which had drifted
    to calling a ``branching`` winner a failure and exiting 1 on a successful run.
    """
    print("\n" + "=" * 60)
    print("  NODE: Merge Results")
    print("=" * 60)

    from state import load_finalize_manifest

    branches_dir = get_output_dir() / "branches"

    # Find all branch results
    branch_results = find_branch_results(branches_dir)

    log(f"Found {len(branch_results)} branch result(s)")

    print_iteration_timeline(branches_dir)

    # Print summary of each branch (including nested sub-branches)
    for result in branch_results:
        branch_name = result.get("branch_name", "unknown")
        branch_path = result.get("branch_path", "")
        best_time = result.get("best_time_us")
        speedup = result.get("speedup")
        iterations = result.get("iterations", 0)

        # rglob returns the tree flat, so indent by nesting depth or a sub-branch reads
        # as a sibling of the strategy it was spawned from.
        depth = branch_path.count("/branches/") if branch_path else 0
        indent = "  " * depth

        status_icon = "✓" if best_time is not None else "✗"
        time_str = f"{best_time:.2f} µs" if best_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

        log(
            f"{indent}{status_icon} {branch_name}: {time_str} "
            f"(speedup: {speedup_str}, iters: {iterations})"
        )

    # Find best overall result
    best = get_best_branch_result(branch_results)
    finalize = load_finalize_manifest(get_output_dir())

    if finalize and finalize.overview and finalize.overview.markdown:
        print("\n" + finalize.overview.markdown)

    summary = build_final_summary(branch_results, best, finalize)

    if best or summary["best_time_us"] is not None:
        shown = best or {
            "branch_name": summary["best_branch"],
            "best_time_us": summary["best_time_us"],
            "speedup": summary["speedup"],
            "status": "success",
        }
        log(f"\nBest branch: {summary['best_branch']}", "SUCCESS")
        best_time = summary["best_time_us"]
        if best_time is not None:
            log(f"Best time: {best_time:.2f} µs")
        else:
            log("Best time: N/A (no successful runs)")
        if summary["speedup"]:
            log(f"Speedup: {summary['speedup']:.2f}x")

        save_json(get_output_dir() / "final_results.json", summary)

        return {
            "status": "success"
            if shown.get("status") in ("success", "branching")
            else "failed"
        }
    else:
        log("No results to merge", "WARN")
        return {"status": "failed"}
