"""
Results parsing utilities.

Handles reading and analyzing KTT tuning results.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple  # noqa: F401 (Tuple used in check_results)

# ---------------------------------------------------------------------------
# Central reference time
# ---------------------------------------------------------------------------

_REFERENCE_TIME_FILE = "reference_time.json"


def parse_reference_time_from_output(tuner_output: str) -> Optional[float]:
    """
    Parse the CPU/CUDA reference computation time from raw KTT stdout.

    KTT logs lines like:
        [Info] Reference result for argument with id 4 was computed in 18688us

    Returns:
        Reference time in microseconds (first match) or None.
    """
    match = re.search(r"Reference result.*computed in (\d+)us", tuner_output)
    if match:
        return float(match.group(1))
    return None


def load_reference_time(output_dir: Path) -> Optional[float]:
    """
    Load the central reference time for a problem.

    Args:
        output_dir: The problem's output directory (e.g. problems/X/output/).

    Returns:
        Reference time in µs, or None if not yet measured.
    """
    path = output_dir / _REFERENCE_TIME_FILE
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("reference_time_us")
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to load reference time from %s: %s", path, e
        )
        return None


def save_reference_time(output_dir: Path, time_us: float) -> None:
    """
    Persist the reference time.  First-write-wins: if the file already
    exists it is NOT overwritten.  Uses O_CREAT|O_EXCL for atomic
    create-if-not-exists across concurrent workers.

    Args:
        output_dir: The problem's output directory.
        time_us:    Reference computation time in µs.
    """
    import os

    path = output_dir / _REFERENCE_TIME_FILE
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    try:
        os.write(fd, json.dumps({"reference_time_us": time_us}, indent=2).encode())
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------


def load_results(results_path: Path) -> Optional[dict]:
    """
    Load results.json file.

    Args:
        results_path: Path to results.json

    Returns:
        Parsed JSON dict or None if not found/invalid
    """
    if not results_path.exists():
        return None

    try:
        with open(results_path) as f:
            return json.load(f)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to load results from %s: %s", results_path, e
        )
        return None


def check_results(results_path: Path) -> Tuple[bool, int, int]:
    """
    Check results.json for successful configurations.

    Args:
        results_path: Path to results.json

    Returns:
        Tuple of (has_success, num_successful, num_total)
    """
    data = load_results(results_path)
    if data is None:
        return False, 0, 0

    results = data.get("Results", [])
    num_ok = sum(1 for r in results if r.get("Status") == "Ok")

    return num_ok > 0, num_ok, len(results)


def get_computation_duration(result: dict) -> float:
    """Extract the actual kernel computation duration from a KTT result dict."""
    comp_results = result.get("ComputationResults", [])
    if comp_results and isinstance(comp_results, list) and len(comp_results) > 0:
        duration = sum(cr.get("Duration", 0.0) for cr in comp_results)
        if duration > 0:
            return duration

    return result.get("TotalDuration", float("inf"))


def get_best_result(results_path: Path) -> Optional[dict]:
    """
    Get the best (fastest) successful configuration.

    Args:
        results_path: Path to results.json

    Returns:
        Best result dict or None if no successful results
    """
    data = load_results(results_path)
    if data is None:
        return None

    results = data.get("Results", [])
    successful = [r for r in results if r.get("Status") == "Ok"]

    if not successful:
        return None

    return min(successful, key=get_computation_duration)


def calculate_speedup(best_time: float, ref_time: float) -> float:
    """Calculate speedup ratio."""
    if best_time <= 0:
        return 0.0
    return float(ref_time) / float(best_time)


def get_results_summary(
    results_path: Path, reference_time_us: Optional[float] = None
) -> dict:
    """
    Generate a summary of tuning results.

    Args:
        results_path: Path to results.json
        reference_time_us: Central reference time (from load_reference_time)

    Returns:
        Summary dict with metrics
    """
    has_success, num_ok, num_total = check_results(results_path)

    summary = {
        "has_success": has_success,
        "num_successful": num_ok,
        "num_total": num_total,
        "best_config": None,
        "best_time_us": None,
        "reference_time_us": reference_time_us,
        "speedup": None,
    }

    if has_success:
        best = get_best_result(results_path)
        if best:
            summary["best_config"] = {
                p["Name"]: p["Value"] for p in best.get("Configuration", [])
            }
            summary["best_time_us"] = get_computation_duration(best)

    if reference_time_us and summary["best_time_us"]:
        summary["speedup"] = calculate_speedup(
            summary["best_time_us"],
            reference_time_us,
        )

    return summary


def format_results_summary(summary: dict) -> str:
    """
    Format results summary as human-readable string.

    Args:
        summary: Summary dict from get_results_summary

    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"Configurations tested: {summary['num_total']}")
    lines.append(f"Successful: {summary['num_successful']}")

    if summary["best_config"]:
        lines.append("\nBest Configuration:")
        for name, value in summary["best_config"].items():
            lines.append(f"  {name}: {value}")

    if summary["best_time_us"]:
        lines.append(f"\nKernel time:    {summary['best_time_us']:,.2f} µs")

    if summary["reference_time_us"]:
        lines.append(f"Reference time: {summary['reference_time_us']:,.2f} µs")

    if summary["speedup"]:
        status = "✓" if summary["speedup"] > 1 else "✗"
        lines.append(f"Speedup:        {summary['speedup']:.2f}x {status}")

    return "\n".join(lines)
