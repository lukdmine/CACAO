"""
Profile node - runs NCU profiling on the best configuration.

This node profiles the best kernel configuration using NVIDIA Nsight Compute
to gather hardware metrics for data-driven optimization decisions.
"""

import asyncio
import csv
import io
import json
import sys

import yaml

from config import SCRIPT_DIR, get_problem_dir
from utils.files import save_output, get_iter_dir
from utils.gpu_lock import acquire_gpu_lock
from utils.log import log
from utils.results import check_results, get_best_result, get_computation_duration
from state.types import WorkingState


async def check_ncu_available() -> bool:
    """Check if NCU is available on the system."""
    from utils.cuda_env import get_env

    env = get_env()
    if not env.ncu:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            str(env.ncu),
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False
        return proc.returncode == 0
    except FileNotFoundError:
        return False


def parse_ncu_csv(csv_output: str) -> dict:
    """
    Parse NCU CSV output (--page raw wide format) into a metrics dict.

    NCU --page raw outputs one row per kernel with all metrics as columns,
    plus a second header row containing units. Non-CSV preamble lines
    (==PROF==, plain text) are stripped before parsing.

    We pick the row whose "Kernel Name" is "kernel" (the agent kernel),
    falling back to the last row if no match is found.

    Returns:
        Dict of metric_name -> float (or str if not numeric). Values with
        locale-style comma separators (e.g. "1,234.56") are normalised.
    """
    metrics = {}

    try:
        # Strip non-CSV preamble lines (don't start with '"' or a digit)
        csv_lines = [
            line
            for line in csv_output.splitlines()
            if line.startswith('"') or (line and line[0].isdigit())
        ]
        if not csv_lines:
            return metrics

        # First line is the header, second line is the units row — skip units
        header_line = csv_lines[0]
        data_lines = csv_lines[2:]  # skip units row at index 1

        if not data_lines:
            return metrics

        reader = csv.DictReader(io.StringIO("\n".join([header_line] + data_lines)))
        rows = list(reader)
        if not rows:
            return metrics

        # Prefer the agent kernel row; fall back to the last row
        target = next(
            (r for r in rows if r.get("Kernel Name", "") == "kernel"), rows[-1]
        )

        for col, raw_value in target.items():
            if not col or not raw_value or raw_value in ("no data", ""):
                continue
            if col.startswith("device__attribute_"):
                continue
            # Normalise comma-separated numbers: "1,234,567.89" -> "1234567.89"
            cleaned = raw_value.replace(",", "")
            try:
                metrics[col] = float(cleaned)
            except ValueError:
                metrics[col] = raw_value

    except Exception as e:
        log(f"Failed to parse NCU CSV: {e}", "WARN")

    return metrics


async def profile_node(state: WorkingState) -> WorkingState:
    """
    Run NCU profiling on the best configuration if available.

    Args:
        state: Working state object with run_output

    Returns:
        Updated state object with ncu_metrics and updated status
    """
    iteration = state.iter_num
    strategy = state.strategy or {}
    branch_name = strategy.name if strategy else "default"

    print("\n" + "=" * 60)
    print(f"  NODE: NCU Profile [{branch_name}] (iter {iteration})")
    print("=" * 60)

    iter_dir = get_iter_dir(state)

    results_path = iter_dir / "results.json"

    # Check if we have successful results to profile
    has_success, num_ok, num_total = check_results(results_path)

    if not has_success:
        log(f"No successful configurations to profile ({num_ok}/{num_total})", "WARN")
        state.ncu_metrics = None
        state.status = "proposing"
        return state

    # Check if NCU is available
    if not await check_ncu_available():
        log("NCU not available, skipping profiling", "WARN")
        state.ncu_metrics = None
        state.status = "proposing"
        return state

    # Get best configuration
    best = get_best_result(results_path)
    if not best:
        log("Could not get best result", "WARN")
        state.ncu_metrics = None
        state.status = "proposing"
        return state

    log(
        f"Profiling best configuration (time: {get_computation_duration(best):,.2f} µs)"
    )

    # Create a single-config params.json for profiling
    # Must copy launch_config and constraints from original params so the tuner

    original_params_path = iter_dir / "params.json"
    try:
        with open(original_params_path, "r") as f:
            original_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log(f"params.json not readable ({e}), skipping profiling", "WARN")
        state.status = "proposing"
        return state

    config = best.get("Configuration", [])
    # Create a map of tuned values
    tuned_values = {p["Name"]: p["Value"] for p in config}

    # Reconstruct parameters, keeping fixed ones and overriding tuned ones
    profile_parameters = []
    for p in original_params.get("parameters", []):
        param_name = p.get("name")
        if param_name in tuned_values:
            profile_parameters.append(
                {"name": param_name, "values": [tuned_values[param_name]]}
            )
        else:
            profile_parameters.append(p)

    single_params = {
        "parameters": profile_parameters,
        "launch_config": original_params.get("launch_config", {}),
        "constraints": original_params.get("constraints", []),
    }

    params_profile_path = iter_dir / "params_profile.json"
    with open(params_profile_path, "w") as f:
        json.dump(single_params, f, indent=2)

    # Build NCU command
    # Key metrics for CUDA optimization
    metrics = [
        "gpu__time_duration.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
    ]

    # Always read gpu index from source problem.yaml so edits propagate
    problem_yaml_path = get_problem_dir() / "problem.yaml"
    gpu_index = 0
    try:
        with open(problem_yaml_path) as f:
            config_yaml = yaml.safe_load(f)
        gpu_index = config_yaml.get("gpu", {}).get("index", 0)
    except Exception as e:
        log(
            f"Failed to parse gpu_index from problem.yaml, defaulting to 0: {e}", "WARN"
        )

    # tuner.py requires a tuning duration. problem.yaml may omit it, so resolve
    # it the same way run_node does. The profile run only has 1 config so it
    # will finish well inside any reasonable budget.
    from nodes.run import _resolve_tuning_budget

    budget_s, _ = _resolve_tuning_budget(problem_yaml_path)

    tuner_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "tuner.py"),
        "--working-dir",
        str(iter_dir),
        "--problem",
        str(problem_yaml_path),
        "--params",
        str(iter_dir / "params_profile.json"),
        "--output",
        "results_profile",
        "--platform",
        "0",
        "--device",
        str(gpu_index),
        "--tuning-duration",
        str(budget_s),
    ]

    from utils.cuda_env import get_env

    env = get_env()
    ncu_cmd = [
        str(env.ncu),
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metrics),
        "--target-processes",
        "all",
    ]

    # Actually NCU doesn't have a reliable --device flag for selecting which process to profile
    # when the target process itself selects the device. It profiles the child process automatically.
    # The child tuner process will select the device correctly because of the --device flag added above.

    ncu_cmd = ncu_cmd + tuner_cmd

    log("Running NCU profiler...")

    ncu_metrics = None

    async with acquire_gpu_lock():
        proc = None
        try:
            from utils.cuda_env import get_subprocess_env

            proc = await asyncio.create_subprocess_exec(
                *ncu_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(SCRIPT_DIR),
                env=get_subprocess_env(),
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=300
                )
            except asyncio.TimeoutError:
                log("NCU profiling timed out", "ERROR")
                proc.kill()
                await proc.wait()
                stdout_b, stderr_b = b"", b""

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                ncu_metrics = parse_ncu_csv(stdout)
                save_output(iter_dir, stdout, "ncu_profile.csv")
                log(
                    f"Profiling complete, collected {len(ncu_metrics)} metrics",
                    "SUCCESS",
                )
                for key in [
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "sm__warps_active.avg.pct_of_peak_sustained_active",
                ]:
                    if key in ncu_metrics:
                        log(f"  {key}: {ncu_metrics[key]:.1f}%")
            elif proc.returncode is not None:
                # NCU emits most diagnostics to stdout (the CSV stream), not
                # stderr — save both so we can actually see what failed.
                combined = (
                    f"[NCU exit code: {proc.returncode}]\n\n"
                    f"===== STDERR =====\n{stderr}\n\n"
                    f"===== STDOUT =====\n{stdout}\n"
                )
                preview = (stderr.strip() or stdout.strip() or "(no output)")[:300]
                log(f"NCU failed (exit {proc.returncode}): {preview}", "ERROR")
                save_output(iter_dir, combined, "ncu_error.txt")

        except asyncio.CancelledError:
            log("Profile node cancelled — killing ncu", "WARN")
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise

        except Exception as e:
            log(f"NCU error: {e}", "ERROR")

    state.ncu_metrics = ncu_metrics
    state.status = "proposing"
    return state
