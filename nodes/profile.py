"""
Profile node — runs NCU on the best framework configuration.

Instead of re-tuning, the compiled framework driver is invoked in *profile mode*
(argv[8] = results base name): it LoadResults() the tuning results, picks the
fastest valid config, and Run()s it exactly once. NCU wraps that process and
profiles each kernel launch separately. Validation is skipped in profile mode, so
only the agent kernel(s) are launched — the reference never appears in the metrics.
"""

import asyncio
import csv
import io

import yaml

from config import get_problem_dir
from utils.build import compile_framework, driver_command
from utils.files import save_output, get_iter_dir
from utils.gpu_lock import acquire_gpu_lock
from utils.log import log
from utils.results import check_results
from state.types import WorkingState

# Key metrics for CUDA optimization.
NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
    "l2__throughput.avg.pct_of_peak_sustained_elapsed",
]


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


def _to_number(raw: str):
    """Parse an NCU numeric cell, tolerant of US (1,234.56) and EU (1 234,56) locales."""
    cleaned = raw.replace("\xa0", "").replace(" ", "")
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")  # EU decimal comma
    else:
        cleaned = cleaned.replace(",", "")  # US thousands comma
    return float(cleaned)


def parse_ncu_csv(csv_output: str) -> dict:
    """
    Parse NCU CSV (--page raw) into a metrics dict.

    One row per kernel launch (header + a units row + data rows). For a single
    kernel the metrics are returned flat; for a multi-kernel pipeline each row's
    metrics are prefixed with the kernel name (duplicates get a #N suffix), so a
    pipeline yields isolated per-stage metrics.
    """
    try:
        csv_lines = [
            line
            for line in csv_output.splitlines()
            if line.startswith('"') or (line and line[0].isdigit())
        ]
        if len(csv_lines) < 3:
            return {}
        header_line, data_lines = csv_lines[0], csv_lines[2:]  # skip units row
        reader = csv.DictReader(io.StringIO("\n".join([header_line] + data_lines)))
        rows = list(reader)
        if not rows:
            return {}

        per_kernel = []
        for r in rows:
            name = r.get("Kernel Name", "") or "kernel"
            metrics = {}
            for col, raw_value in r.items():
                if not col or not raw_value or raw_value in ("no data", ""):
                    continue
                if col.startswith("device__attribute_"):
                    continue
                try:
                    metrics[col] = _to_number(raw_value)
                except ValueError:
                    metrics[col] = raw_value
            per_kernel.append((name, metrics))

        if len(per_kernel) == 1:
            return per_kernel[0][1]

        out, seen = {}, {}
        for name, metrics in per_kernel:
            seen[name] = seen.get(name, 0) + 1
            prefix = name if seen[name] == 1 else f"{name}#{seen[name]}"
            for k, v in metrics.items():
                out[f"{prefix}/{k}"] = v
        return out

    except Exception as e:
        log(f"Failed to parse NCU CSV: {e}", "WARN")
        return {}


async def profile_node(state: WorkingState) -> WorkingState:
    """Run NCU on the best framework configuration (single Run), if available."""
    iteration = state.iter_num
    strategy = state.strategy or {}
    branch_name = strategy.name if strategy else "default"

    print("\n" + "=" * 60)
    print(f"  NODE: NCU Profile [{branch_name}] (iter {iteration})")
    print("=" * 60)

    iter_dir = get_iter_dir(state)
    results_path = iter_dir / "results.json"

    has_success, num_ok, num_total = check_results(results_path)
    if not has_success:
        log(f"No successful configurations to profile ({num_ok}/{num_total})", "WARN")
        state.ncu_metrics = None
        state.status = "proposing"
        return state

    if not await check_ncu_available():
        log("NCU not available, skipping profiling", "WARN")
        state.ncu_metrics = None
        state.status = "proposing"
        return state

    # The driver from the run node should exist; rebuild if missing.
    driver = iter_dir / "driver"
    if not driver.exists():
        build_result = compile_framework(iter_dir)
        if not build_result.ok:
            log("Driver rebuild for profiling failed, skipping", "WARN")
            state.ncu_metrics = None
            state.status = "proposing"
            return state
        driver = build_result.binary

    # Read gpu index + reference file from source problem.yaml.
    problem_dir = get_problem_dir()
    gpu_index, ref_file = 0, "ref_kernel.cu"
    try:
        with open(problem_dir / "problem.yaml") as f:
            cfg_yaml = yaml.safe_load(f) or {}
        gpu_index = cfg_yaml.get("gpu", {}).get("index", 0)
        ref_file = (cfg_yaml.get("reference") or {}).get("file", ref_file)
    except Exception as e:
        log(f"Failed to parse problem.yaml, using defaults: {e}", "WARN")

    log(f"Profiling best configuration ({num_ok}/{num_total} valid configs)")

    # driver <plat> <dev> <dur> <tol> <out> <kernels> <ref> <profile_results_base>
    # duration/tolerance/out are unused in profile mode; the base name lets KTT
    # LoadResults() read "<base>.json" (it appends the extension).
    from utils.cuda_env import get_env

    cmd = driver_command(
        driver,
        platform=0,
        device=gpu_index,
        duration=1,
        tolerance=1.0,
        output_base="results_profile",
        kernel_file=iter_dir / "kernels.cu",
        ref_file=problem_dir / ref_file,
    ) + ["results"]

    ncu_cmd = [
        str(get_env().ncu),
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(NCU_METRICS),
        "--target-processes",
        "all",
    ] + cmd

    log("Running NCU profiler...")
    ncu_metrics = None

    async with acquire_gpu_lock():
        proc = None
        try:
            from utils.cuda_env import get_subprocess_env

            env = get_subprocess_env()
            env["LC_ALL"] = "C"  # stable numeric formatting in NCU CSV
            proc = await asyncio.create_subprocess_exec(
                *ncu_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(iter_dir),
                env=env,
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
                log(f"Profiling complete, collected {len(ncu_metrics)} metrics", "SUCCESS")
                for key in (
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                ):
                    if key in ncu_metrics:
                        log(f"  {key}: {ncu_metrics[key]:.1f}%")
            elif proc.returncode is not None:
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
