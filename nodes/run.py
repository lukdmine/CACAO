"""
Run node — compiles the framework driver, then runs it as a subprocess with GPU locking.

Framework-file mode: host-compiles framework.cpp -> driver (kernels stay
NVRTC-compiled inside KTT at tune time), then executes the driver. A host-compile
failure is routed to propose/fix. KTT enforces the wall-clock budget via
TuningDuration; a watchdog covers subprocess hangs.
"""

import asyncio
import os
import re
from typing import Dict, Optional
import signal
import time
import yaml
from pathlib import Path

import config as _cfg
from config import get_problem_dir
from utils.build import compile_framework, driver_command, reference_build_extras
from utils.files import get_iter_dir, save_output
from utils.framework import load_regions
from utils.gpu_lock import acquire_gpu_lock
from utils.log import log
from utils.problem import gpu_index as problem_gpu_index
from utils.results import (
    check_results,
    ensure_results_loadable,
    parse_precise_best_time_from_output,
    parse_reference_time_from_output,
    save_reference_time,
    get_results_summary,
    load_reference_time,
    summarize_failures,
    kernel_line_offset,
)
from state.types import WorkingState


# Extra slack beyond the tuning budget before the outer watchdog trips.
# KTT's TuningDuration stop is the primary mechanism; this only fires on hangs.
_WATCHDOG_MARGIN_S = 30.0
_WATCHDOG_MARGIN_FRAC = 0.2


def precise_runs() -> int:
    """Replays of the best config in the post-tune re-measure; 0 means off."""
    raw = os.environ.get("CACAO_PRECISE_RUNS")
    if raw is not None:
        try:
            return max(int(raw), 0)
        except ValueError:
            log(f"CACAO_PRECISE_RUNS={raw!r} is not an integer; using the default", "WARN")
    return max(int(_cfg.PRECISE_RUNS), 0)


async def _remeasure_best(cmd: list, cdir: Path) -> tuple:
    """Run a driver's remeasure command and return (precise_us | None, log_text).

    ``cmd`` comes from driver_command(mode="remeasure"): the driver loads results.json,
    replays its best config N times and prints CACAO_PRECISE_BEST_US=<min>. Never
    fatal: on any failure the iteration keeps its single-shot numbers.
    """
    results_json = Path(cdir) / "results.json"
    has_success, _, _ = check_results(results_json)
    if not has_success:
        return None, ""
    # As in profile mode: a results file from before KTT 2.3 has no Timestamp field
    # and would abort the driver in LoadResults.
    ensure_results_loadable(results_json)
    header = f"\n[RE-MEASURE] best config of {results_json.name}\n"
    process = None
    try:
        from utils.cuda_env import get_subprocess_env

        # Timing is as GPU-sensitive as the tune was, so it takes the same lock.
        async with acquire_gpu_lock():
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cdir),
                env=get_subprocess_env(),
            )
            try:
                out, _ = await asyncio.wait_for(
                    process.communicate(), timeout=_cfg.PRECISE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                log("Re-measure timed out; keeping the single-shot best time", "WARN")
                return None, header + "[re-measure timed out; killed]\n"
        text = out.decode("utf-8", errors="replace") if out else ""
        precise = parse_precise_best_time_from_output(text)
        if precise is None or process.returncode != 0:
            log(
                f"Re-measure produced no time (exit {process.returncode}); "
                "keeping the single-shot best time",
                "WARN",
            )
            return None, header + text
        log(f"Precise best time: {precise:.2f} µs")
        return precise, header + text
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        raise
    except Exception as e:
        log(f"Re-measure failed: {e}; keeping the single-shot best time", "WARN")
        return None, header + f"[re-measure failed: {e}]\n"


def _resolve_tuning_budget(
    meta: dict, case, override_s: Optional[float] = None
) -> tuple[float, str]:
    """Effective tuning budget for one case, as (seconds, source_label).

    Precedence: the finalize budget (``override_s``), then the user's runtime override, then the case's own ``duration_s``, then
    the problem's ``tuning.duration_s``, then the engine default. A tail case exists to
    catch a correctness bug rather than to find an optimum, so giving it a shorter
    budget than the primary is the normal configuration, not a degraded one.
    """
    from utils.cases import case_duration

    # Finalize passes its own per-finalist budget; a UI timeout meant for the branch
    # loop must not shrink the one run whose point is to out-tune it.
    if override_s is not None:
        return float(override_s), "finalize"
    if _cfg.TUNER_TIMEOUT_OVERRIDE is not None:
        return float(_cfg.TUNER_TIMEOUT_OVERRIDE), "user override"
    if case is not None and case.duration_s is not None:
        return float(case.duration_s), "case"
    try:
        value = case_duration(meta, case) if case is not None else None
        if value is not None:
            return float(value), "problem.yaml"
    except Exception as e:
        log(f"Failed to read tuning.duration_s from problem.yaml: {e}", "WARN")
    return float(_cfg.TUNER_TIMEOUT), "system default"


class TunerProgressTracker:
    """Tracker that parses tuner stdout for progress and timing (log-only).

    Accessed only from coroutines in a single event loop, so no locking needed.
    """

    def __init__(self):
        self.total_configs = 0
        self.launched = 0
        self.completed = 0
        self.config_durations: list[float] = []  # seconds per completed config
        self.last_launch_time: float | None = None

    def parse_line(self, line: str, wall_time: float):
        """Parse a single tuner output line and update state."""
        # "Total count of N configurations was generated"
        m = re.search(r"Total count of (\d+) configurations", line)
        if m:
            self.total_configs = int(m.group(1))

        # "Launching configuration N / M"
        m = re.search(r"Launching configuration (\d+) / (\d+)", line)
        if m:
            new_launched = int(m.group(1))
            self.total_configs = int(m.group(2))

            # If a previous config was in-flight, record its duration
            if self.last_launch_time is not None and new_launched > self.launched:
                duration = wall_time - self.last_launch_time
                self.config_durations.append(duration)
                self.completed = len(self.config_durations)

            self.launched = new_launched
            self.last_launch_time = wall_time

    def summary(self, elapsed: float) -> str:
        """Return a human-readable progress summary."""
        avg = (
            (sum(self.config_durations) / len(self.config_durations))
            if self.config_durations
            else 0
        )
        max_d = max(self.config_durations) if self.config_durations else 0
        return (
            f"Progress: {self.completed}/{self.total_configs} configs done, "
            f"avg {avg:.1f}s/config, max {max_d:.1f}s, elapsed {elapsed:.0f}s"
        )


async def _execute_driver(
    cmd: list,
    cwd: Path,
    budget_s: float,
    watchdog_s: float,
    tracker: "TunerProgressTracker",
    branch_name: str = "default",
) -> str:
    """Run one compiled driver under the GPU lock and return its combined output.

    Extracted from run_node so the case loop can call it once per case, and so the
    loop's ordering and fail-fast behaviour can be tested without a GPU. KTT's
    TuningDuration enforces the wall-clock budget inside the tuner; the watchdog here
    only fires on a genuinely hung subprocess.
    """
    output_lines = []
    watchdog_tripped = False

    def kill_tree(process) -> None:
        """Kill the driver *and* whatever it spawned.

        A python reference runs through std::system, so the driver has an `sh` and a
        torch process under it. Killing the driver alone reparents those, and they keep
        the CUDA context and its allocations while the next branch takes the GPU lock.
        """
        try:
            pgid = os.getpgid(process.pid)
            if pgid == os.getpgrp():
                raise OSError("driver shares our process group")  # never signal ourselves
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            # Already reaped, or it has no group of its own. The PID is still killable.
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass

    log("Acquiring GPU lock...")
    async with acquire_gpu_lock():
        log("GPU lock acquired. Running tuner...")

        process = None
        try:
            from utils.cuda_env import get_subprocess_env

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd),
                env=get_subprocess_env(),
                # Its own process group, so a kill reaches the reference subprocesses too.
                start_new_session=True,
            )

            start_time = time.time()

            async def collect_output():
                try:
                    while True:
                        line_bytes = await process.stdout.readline()
                        if not line_bytes:
                            break
                        line = line_bytes.decode("utf-8", errors="replace")
                        output_lines.append(line)
                        tracker.parse_line(line, time.time() - start_time)
                        if len(output_lines) % 100 == 0:
                            log(
                                f"[{branch_name}] {tracker.summary(time.time() - start_time)}",
                                "INFO",
                            )
                except Exception as e:
                    output_lines.append(f"\n[collector error: {e}]\n")

            collector_task = asyncio.create_task(collect_output())

            try:
                await asyncio.wait_for(process.wait(), timeout=watchdog_s)
            except asyncio.TimeoutError:
                watchdog_tripped = True
                elapsed = time.time() - start_time
                log(
                    f"Watchdog tripped at {elapsed:.0f}s (budget {budget_s:.0f}s + margin). "
                    f"KTT did not honor its stop condition — killing subprocess.",
                    "ERROR",
                )
                kill_tree(process)
                await process.wait()
                output_lines.append(
                    f"\n[WATCHDOG] Subprocess killed at {elapsed:.0f}s (budget {budget_s:.0f}s). "
                    f"This indicates a hang in KTT — not normal flow.\n"
                )

            # Wait for collector to drain remaining output
            try:
                await asyncio.wait_for(collector_task, timeout=2.0)
            except asyncio.TimeoutError:
                collector_task.cancel()

            elapsed = time.time() - start_time
            log(tracker.summary(elapsed), "INFO")

            if not watchdog_tripped:
                if process.returncode == 0:
                    log(f"Tuner completed in {elapsed:.0f}s", "SUCCESS")
                elif process.returncode is not None:
                    log(f"Tuner exited with code {process.returncode}", "ERROR")
                    output_lines.append(f"\n[EXIT CODE: {process.returncode}]")

        except asyncio.CancelledError:
            log("Run node cancelled — killing tuner", "WARN")
            if process is not None and process.returncode is None:
                kill_tree(process)
                await process.wait()
            output_lines.append("\n[INTERRUPTED] Tuner killed by cancellation")
            raise

        except Exception as e:
            output_lines.append(f"[ERROR] Failed to run tuner: {e}")
            log(f"Tuner error: {e}", "ERROR")

    log("GPU lock released")
    return "".join(output_lines)


async def run_node(
    state: WorkingState,
    *,
    budget_override_s: Optional[float] = None,
    mode: str = "random",
    seed_results_by_case: Optional[Dict[str, str]] = None,
) -> WorkingState:
    """
    Run the KTT tuner as a subprocess.

    The keywords are finalize's: a tuning budget, a searcher mode and, per case, the
    results base name that seeds the MCMC searcher. The branch worker passes none, and
    with the defaults the driver command is exactly the original.

    KTT's TuningDuration stop condition enforces the wall-clock budget inside
    the tuner itself. This function logs progress for visibility and provides
    a loose outer watchdog for truly hung subprocesses.
    """
    iteration = state.iter_num
    strategy = state.strategy or {}
    branch_name = strategy.name if strategy else "default"

    print("\n" + "=" * 60)
    print(f"  NODE: Run Tuner [{branch_name}] (iter {iteration})")
    print("=" * 60)

    iter_dir = get_iter_dir(state)

    log(f"Executing tuner in: {iter_dir}")

    # Read problem.yaml from source so edits propagate.
    problem_dir = get_problem_dir()
    problem_yaml_path = problem_dir / "problem.yaml"

    # Defaults for a file that will not parse. config_yaml belongs here too: a save from
    # the problem editor mid-run can be read half-written, and without a default the
    # "using defaults" path dies on a NameError that buries the YAML error.
    config_yaml = {}
    gpu_index = 0
    tolerance = 1e-4
    ref_file = "ref_kernel.cu"
    try:
        with open(problem_yaml_path, encoding="utf-8") as f:
            config_yaml = yaml.safe_load(f) or {}
        gpu_index = problem_gpu_index(config_yaml)
        tolerance = (config_yaml.get("validation") or {}).get("tolerance", tolerance)
        ref_file = (config_yaml.get("reference") or {}).get("file", ref_file)
    except Exception as e:
        log(f"Failed to parse problem.yaml, using defaults: {e}", "WARN")

    from config import get_output_dir
    from utils.cases import case_list, stage_case
    from utils.inputs import load_inputs_spec
    from utils.results import aggregate_case_summaries

    cases = case_list(config_yaml)
    primary = cases[0].name
    spec = load_inputs_spec(problem_dir / "inputs.yaml")
    regions = load_regions(iter_dir)
    if len(cases) > 1:
        log(f"{len(cases)} input cases: {', '.join(c.name for c in cases)}")

    # --- pass 1: stage and compile EVERY case ---
    # A build-level failure costs seconds here and a whole tuning budget if it is left
    # to be discovered in pass 2 instead.
    log("Compiling framework driver(s) (framework.cpp -> driver)...")
    built = []
    for case in cases:
        cdir = stage_case(problem_dir, iter_dir, spec, config_yaml, cases, case, regions)
        # cpu_c references add ref_cpu.c + this case's -D scalar macros to the build.
        extra_sources, extra_flags = reference_build_extras(problem_dir, case)
        # g++ takes a second or more and this coroutine shares its loop with three other
        # branches — including the tasks draining their drivers' stdout pipes.
        build_result = await asyncio.to_thread(
            compile_framework, cdir, extra_sources=extra_sources, extra_flags=extra_flags
        )
        if not build_result.ok:
            label = "" if len(cases) == 1 else f" [case {case.name}]"
            output = (
                f"[COMPILE ERROR]{label} Host compilation of framework.cpp failed.\n\n"
                f"{build_result.stderr}"
            )
            log(f"Framework compile failed{label} — routing error to propose", "ERROR")
            save_output(cdir, output, "tuner_output.txt")
            state.run_output = output
            state.results_summary = get_results_summary(cdir / "results.json", None)
            state.status = "proposing"
            return state
        built.append((case, cdir, build_result.binary))
    log(f"Framework driver compiled ({len(built)} case(s))", "SUCCESS")

    # --- pass 2: tune in declared order, fail fast ---
    per_case = {c.name: None for c in cases}
    combined_output = []
    for case, cdir, binary in built:
        case_key = None if case.name == primary else case.name
        budget_s, budget_source = _resolve_tuning_budget(
            config_yaml, case, budget_override_s
        )
        watchdog_s = budget_s + max(_WATCHDOG_MARGIN_S, budget_s * _WATCHDOG_MARGIN_FRAC)
        label = "" if len(cases) == 1 else f" [case {case.name}]"
        log(
            f"Tuning budget{label}: {budget_s:.0f}s ({budget_source}); "
            f"watchdog at {watchdog_s:.0f}s"
        )

        cmd = driver_command(
            binary,
            platform=0,
            device=gpu_index,
            duration=budget_s,
            tolerance=tolerance,
            output_base="results",
            kernel_file=cdir / "kernels.cu",
            ref_file=problem_dir / ref_file,
            mode=mode,
            seed_results=(seed_results_by_case or {}).get(case.name, ""),
        )
        tracker = TunerProgressTracker()
        output = await _execute_driver(
            cmd, cdir, budget_s, watchdog_s, tracker, branch_name
        )

        # Persist the central reference time (first-write-wins, per case).
        ref_time = parse_reference_time_from_output(output)
        if ref_time is not None:
            save_reference_time(get_output_dir(), ref_time, case_key)
            log(f"Reference time{label}: {ref_time:.0f} µs")

        # Post-tune re-measure: the same driver replays this case's best config for a
        # min-of-N time. It reads results.json, so it runs before anything summarises.
        precise_us, runs = None, precise_runs()
        if runs:
            precise_us, remeasure_log = await _remeasure_best(
                driver_command(
                    binary,
                    platform=0,
                    device=gpu_index,
                    duration=budget_s,
                    tolerance=tolerance,
                    output_base="results",
                    kernel_file=cdir / "kernels.cu",
                    ref_file=problem_dir / ref_file,
                    profile_results=str(cdir / "results"),
                    mode="remeasure",
                    precise_runs=runs,
                ),
                cdir,
            )
            output += remeasure_log

        # The full log is the artifact; keep it on disk.
        save_output(cdir, output, "tuner_output.txt")

        # State carries the summary, not the log. KTT prepends one #define per tuning
        # parameter before NVRTC sees the source, so device diagnostics cite line numbers
        # that do not exist in kernels.cu. The offset is the parameter count, which
        # results.json carries.
        failures = summarize_failures(
            output, kernel_offset=kernel_line_offset(cdir / "results.json")
        )
        if failures:
            log(
                f"Summarized {len(output) // 1024} KB of failures -> "
                f"{len(failures) // 1024} KB"
            )
        header = "" if len(cases) == 1 else f"### case {case.name}\n"
        combined_output.append(header + (failures or output))

        summary = get_results_summary(
            cdir / "results.json",
            load_reference_time(get_output_dir(), case_key),
            precise_best_time_us=precise_us,
        )
        per_case[case.name] = summary

        if not summary["has_success"]:
            # Any case failing fails the iteration. Tuning the rest would spend minutes
            # collecting errors the kernel author has to fix one at a time anyway.
            log(
                f"Case {case.name} produced no valid configuration — "
                "skipping the remaining cases",
                "ERROR",
            )
            break

    state.run_output = "\n\n".join(combined_output)

    # Save speedup/best_time eagerly so they persist even if later nodes fail.
    summary = aggregate_case_summaries(per_case, primary)
    if summary.get("best_time_us") is not None:
        if state.best_time_us is None or summary["best_time_us"] < state.best_time_us:
            state.best_time_us = summary["best_time_us"]
    if summary.get("speedup") is not None:
        if state.speedup is None or summary["speedup"] > state.speedup:
            state.speedup = summary["speedup"]
    state.results_summary = summary

    state.status = "profiling"
    return state
