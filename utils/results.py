"""
Results parsing utilities.

Handles reading and analyzing KTT tuning results.
"""

import json
import re
from collections import OrderedDict
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

    With multiple validated buffers KTT logs one line per buffer, but each run
    executes the FULL reference (kernel or C function), so the first match is
    one complete reference execution — summing would double-count.

    Returns:
        Reference time in microseconds (first match) or None.
    """
    match = re.search(r"Reference result.*computed in (\d+)us", tuner_output)
    if match:
        return float(match.group(1))
    return None


# Heading prefix of the text summarize_failures() produces. Public because state.history
# keys the head-vs-tail excerpt off it — if these two drift apart, the summary is silently
# tailed and loses its header and most widespread diagnostic. Only the prefix is fixed; the
# rest of the line reports what was actually found.
FAILURE_SUMMARY_HEADING = "## Compile diagnostics:"

# KTT logs one of these per failed configuration, followed by the compiler's diagnostics.
_FAILURE_MARKER = "[Warning] Kernel run failed with reason:"

# A diagnostic, captured from the LOCATION onward wherever it sits in the line. Both
# compilers appear: NVRTC as `default_program(58): error: ...` (device) and g++ as
# `framework.cpp:12:25: error: ...` (host). The location anchor also strips KTT's
# "[Warning] Kernel run failed with reason: ... additional info: " prefix, which carries
# the block's first diagnostic inline.
_DIAGNOSTIC = re.compile(
    r"((?:default_program\(\d+\)|[\w./+-]+:\d+(?::\d+)?): (?:fatal error|error): .*?)\s*$"
)
# Lines the compiler emits UNDER a diagnostic: g++ prints the offending source and a caret
# ruler, plus "note:" candidates. NVRTC prints nothing. Kept so a host error arrives with
# the code it is complaining about.
_CONTEXT = re.compile(r"^\s*(?:\d+\s*\||\||~|\^|In file included|\s+from |.*\bnote:)")


def kernel_line_offset(results_path: Path) -> Optional[int]:
    """Lines KTT prepends to kernels.cu before handing it to NVRTC.

    Exact by construction, not inferred. KTT compiles ``GeneratePrefix() + GetSource()``
    (TunerCore.cpp:327), and GeneratePrefix appends one ``#define NAME value\\n`` per
    parameter pair and nothing else (KernelConfiguration.cpp:25-35). So NVRTC's line is the
    file's line plus the parameter count, and every device diagnostic is off by exactly
    that much from the file the LLM is editing.

    If those two ever prepend anything more, this goes silently wrong rather than loudly —
    the mapped line would simply point at innocent code. Cross-checked against four real
    kernels at parameter counts 8 and 7: every mapping landed on the offending line, and
    the 7-parameter ones would have been off by one under a hardcoded offset.
    """
    data = load_results(results_path)
    if not data:
        return None
    results = data.get("Results") or []
    if not results:
        return None
    config = results[0].get("Configuration")
    return len(config) if config else None


def summarize_failures(
    tuner_output: str, kernel_offset: Optional[int] = None
) -> Optional[str]:
    """Every distinct compiler diagnostic, once, with the context the compiler gave for it.

    ``kernel_offset`` (see kernel_line_offset) maps NVRTC's ``default_program(N)`` back to
    the kernels.cu line the LLM is actually editing. Without it the model has to guess the
    correspondence, and the numbers it reasons about are not the numbers in its file.

    KTT runs every configuration, so a kernel that does not compile fails all of them with
    the same diagnostics. One real run: 957 KB / 21,609 lines / 9,060 lines matching
    `error:` — carrying 46 distinct diagnostics. Excerpting that log hands the LLM one
    arbitrary configuration's block, cut mid-way: in that run the tail showed the wmma
    errors while the float2 errors, on 1,332 lines, never reached the retry at all.

    Deduplicating by diagnostic rather than by block is what makes the result complete: a
    per-block grouping reports the same error once per block it appears in, so a cause
    present in only one block is easy to miss among the repeats.

    Returns None when the output carries no diagnostics (a clean run, a crash), so callers
    fall back to an excerpt.
    """
    lines = tuner_output.split("\n")
    configs = max(tuner_output.count(_FAILURE_MARKER), 0)

    diags: "OrderedDict[str, dict]" = OrderedDict()
    for i, line in enumerate(lines):
        m = _DIAGNOSTIC.search(line)
        if not m:
            continue
        text = m.group(1).strip()
        entry = diags.get(text)
        if entry is None:
            entry = diags[text] = {"count": 0, "context": _context_below(lines, i)}
        entry["count"] += 1

    if not diags:
        return None

    header = f"{FAILURE_SUMMARY_HEADING} {len(diags)} distinct"
    note = (
        "Every distinct diagnostic, deduplicated; the full log is kept at "
        "iter_N/tuner_output.txt."
    )
    if configs:
        header += f", across {configs} failed configuration(s)"
        # Without this the counts read as severity, and the model chases the wrong error.
        note += (
            " `xN` is how many times the compiler emitted it: a kernel that fails to "
            "compile fails every configuration the same way, so a high count means "
            "widespread, not severe."
        )
    out = [header, "", note, "```"]
    for text, entry in sorted(diags.items(), key=lambda kv: -kv[1]["count"]):
        out.append(f"x{entry['count']:<4} {_map_device_line(text, kernel_offset)}")
        out.extend(f"      {c}" for c in entry["context"])
    out.append("```")
    return "\n".join(out)


def _map_device_line(text: str, offset: Optional[int]) -> str:
    """Rewrite `default_program(N)` to the kernels.cu line it corresponds to.

    The LLM edits kernels.cu, so give it kernels.cu numbers; NVRTC's own numbering is of no
    use to it and inviting it to do the arithmetic is inviting it to get it wrong. The
    untouched log stays at iter_N/tuner_output.txt if the mapping ever needs checking.
    """
    if not offset:
        return text

    def sub(m):
        dev = int(m.group(1))
        src = dev - offset
        return f"kernels.cu:{src}" if src > 0 else m.group(0)

    return re.sub(r"default_program\((\d+)\)", sub, text)


def _context_below(lines: list, i: int, limit: int = 4) -> list:
    """The compiler's own context under a diagnostic — g++'s source line and caret ruler.

    NVRTC emits none, so this returns empty for device errors and costs nothing.
    """
    out = []
    for line in lines[i + 1 : i + 1 + limit]:
        if not line.strip() or not _CONTEXT.match(line):
            break
        out.append(line.rstrip())
    return out


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
        with open(path, encoding="utf-8") as f:
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
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to load results from %s: %s", results_path, e
        )
        return None


def ensure_results_loadable(results_path: Path) -> bool:
    """Make a results.json written by an older KTT readable by the current one.

    KTT 2.3 added a ``Timestamp`` field to every serialised KernelResult and reads it
    back with ``j.at("Timestamp")``, which throws — it does not default — so a file
    written by 2.2 aborts ``Tuner::LoadResults`` with
    ``[json.exception.out_of_range.403] key 'Timestamp' not found``.

    Only the profile node loads a results.json back into KTT, and only ever the one
    from its own iteration. That file is written by the same driver moments earlier on
    a fresh run, so this is a no-op there; it matters when resuming a run tuned before
    the KTT upgrade, where the file on disk predates the field.

    An empty string is the right filler: the field is metadata KTT never interprets,
    and inventing a timestamp would date the result to the migration rather than the
    run. Returns True when the file was rewritten.
    """
    data = load_results(results_path)
    if data is None:
        return False

    results = data.get("Results")
    if not isinstance(results, list):
        return False

    patched = [r for r in results if isinstance(r, dict) and "Timestamp" not in r]
    if not patched:
        return False

    for result in patched:
        result["Timestamp"] = ""

    try:
        results_path.write_text(json.dumps(data), encoding="utf-8")
    except OSError as e:
        from utils.log import log

        log(f"Could not migrate {results_path.name} for the current KTT: {e}", "WARN")
        return False

    from utils.log import log

    log(
        f"Migrated {results_path.name} for KTT 2.3 "
        f"({len(patched)} result(s) had no Timestamp field)"
    )
    return True


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
