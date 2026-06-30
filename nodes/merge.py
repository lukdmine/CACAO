"""
Merge node - aggregates results from all branches.

This node runs after all parallel branches have completed.
It scans the output directory for branch results and selects the best one.
"""

from pathlib import Path
from typing import List, Optional

from config import get_output_dir
from utils.files import save_json
from utils.log import log


def find_branch_results(branches_dir: Path) -> List[dict]:
    """
    Recursively find all branch.json files in branches and summarize results.

    Args:
        branches_dir: Path to branches directory

    Returns:
        List of summarized result dicts
    """
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
            best_time_us = manifest.best_time_us

            # If best_time not in manifest, scan iter dirs
            if best_time_us is None:
                for iter_dir in branch_path.glob("iter*"):
                    if iter_dir.is_dir():
                        results_path = iter_dir / "results.json"
                        summary = get_results_summary(results_path, ref_time)
                        t_us = summary.get("best_time_us")
                        if t_us and (best_time_us is None or t_us < best_time_us):
                            best_time_us = t_us

            speedup = manifest.speedup
            if not speedup and best_time_us and ref_time:
                speedup = calculate_speedup(best_time_us, ref_time)

            results.append(
                {
                    "branch_name": strategy.name,
                    "branch_path": str(branch_path),
                    "status": manifest.status,
                    "best_time_us": best_time_us,
                    "speedup": speedup,
                    "iterations": manifest.current_iter,
                }
            )
        except Exception as e:
            log(f"Failed to load branch {branch_file}: {e}", "WARN")

    return results


def get_best_branch_result(results: List[dict]) -> Optional[dict]:
    """
    Find the best result across all branches.

    Args:
        results: List of branch results

    Returns:
        Best result dict or None
    """
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
            summary = get_results_summary(iter_dir / "results.json", ref_time)
            ms = summary["best_time_us"] / 1000 if summary["best_time_us"] else None

            action = "?"
            state_path = iter_dir / "state.json"
            if state_path.exists():
                try:
                    s = _json.loads(state_path.read_text())
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
    """
    Merge results from all branches.

    Returns:
        Dict with final results
    """
    print("\n" + "=" * 60)
    print("  NODE: Merge Results")
    print("=" * 60)

    branches_dir = get_output_dir() / "branches"

    # Find all branch results
    branch_results = find_branch_results(branches_dir)

    log(f"Found {len(branch_results)} branch result(s)")

    print_iteration_timeline(branches_dir)

    # Print summary of each branch
    for result in branch_results:
        branch_name = result.get("branch_name", "unknown")
        best_time = result.get("best_time_us")
        speedup = result.get("speedup")

        status_icon = "✓" if best_time is not None else "✗"
        time_str = f"{best_time:.2f} µs" if best_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

        log(f"  {status_icon} {branch_name}: {time_str} (speedup: {speedup_str})")

    # Find best overall result
    best = get_best_branch_result(branch_results)

    if best:
        log(f"\nBest branch: {best.get('branch_name')}", "SUCCESS")
        best_time = best.get("best_time_us")
        if best_time is not None:
            log(f"Best time: {best_time:.2f} µs")
        else:
            log("Best time: N/A (no successful runs)")
        if best.get("speedup"):
            log(f"Speedup: {best.get('speedup'):.2f}x")

        # Save final summary
        final_summary = {
            "best_branch": best.get("branch_name"),
            "best_config": best.get("best_config"),
            "best_time_us": best.get("best_time_us"),
            "speedup": best.get("speedup"),
            "all_branches": [
                {
                    "name": r.get("branch_name"),
                    "status": r.get("status"),
                    "best_time_us": r.get("best_time_us"),
                    "speedup": r.get("speedup"),
                }
                for r in branch_results
            ],
        }
        save_json(get_output_dir() / "final_results.json", final_summary)

        return {
            "status": "success"
            if best.get("status") in ("success", "branching")
            else "failed"
        }
    else:
        log("No results to merge", "WARN")
        return {"status": "failed"}
