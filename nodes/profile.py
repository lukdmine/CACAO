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
from utils.build import compile_framework, driver_command, reference_build_extras
from utils.files import save_output, get_iter_dir
from utils.gpu_lock import acquire_gpu_lock
from utils.log import log
from utils.results import check_results, ensure_results_loadable
from state.types import WorkingState

# Key metrics for CUDA optimization. Names verified against NCU's metric databases
# for every supported chip (ncu --list-chips: Turing -> Blackwell, incl. Jetson/DGX
# Spark SoCs). A name unknown to the local GPU/NCU is skipped with an stderr warning
# and exit 0, so generation-specific spellings can be listed freely - coverage was
# audited as: dram bytes spellings differ per generation, SoCs (ga10b/gb10b) expose
# DRAM only via mcc__ (gb20b/c expose no DRAM counter at all), and l2__throughput
# exists on no chip this NCU knows - but Pascal/Volta databases are no longer
# shipped to audit, so it stays as fallback insurance for CUDA-12-era toolchains.
NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # SoC DRAM throughput (read/write are separate on Jetson/DGX-Spark).
    "mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed",
    "mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Issue-slot utilization + FMA/tensor pipe saturation: separates issue-bound,
    # compute-bound and latency-bound kernels.
    "smsp__issue_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    # Achieved occupancy.
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # L2/L1 throughput and hit rates (reuse from tiling/staging). l2__ is the
    # fallback spelling for toolchains whose chips predate the audit (see header).
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "l2__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    # Register-spill traffic into local memory (~0 = register pressure is contained)
    # and branch uniformity (100% = no divergence from boundary conditions).
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "sm__sass_average_branch_targets_threads_uniform.pct",
    # Actual DRAM traffic -> real arithmetic intensity.
    "dram__bytes_op_read.sum",
    "dram__bytes_read.sum",
    "dram__bytes_op_write.sum",
    "dram__bytes_write.sum",
    # Tensor-pipe utilization (0% = WMMA/WGMMA not engaging) and shared-memory bank
    # conflicts. Stable names Turing->Blackwell; Pascal lacks both pipes (metric is
    # absent -> skipped), which is itself the correct signal there.
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Global-load bandwidth at L1 and access coalescing (4 sectors/request = perfect
    # fp32, 32 = fully scattered).
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
    # Top stall reasons: memory latency vs. dependency vs. sync bound.
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warp_latency_issue_stalled_wait.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warp_latency_issue_stalled_barrier.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
]

# Columns worth keeping besides NCU_METRICS. --page raw always emits ~300 columns
# (device attributes, nvlink/numa/c2clink, profiler bookkeeping) that no flag
# removes, so the parser whitelists instead. Only launch facts the LLM cannot
# already know are kept: registers/thread is compiler-decided, the occupancy limits
# name the binding constraint, waves captures grid-vs-SM fit. Grid/block sizes are
# the LLM's own launch config (also in results_summary) and are dropped.
# "Kernel Name" is not optimization signal either, but the parser needs it to
# attribute rows to stages of a multi-kernel pipeline — it is never emitted.
_NCU_KEEP_COLUMNS = {
    # Theoretical occupancy ceiling: paired with achieved (sm__warps_active), the gap
    # separates a register/smem cap from launch-tail loss.
    "sm__maximum_warps_per_active_cycle_pct",
    "launch__registers_per_thread",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__waves_per_multiprocessor",
} | set(NCU_METRICS)


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

    Only columns in _NCU_KEEP_COLUMNS survive — the raw page emits hundreds of
    device-attribute/bookkeeping columns that would otherwise land verbatim in the
    LLM prompt. One row per kernel launch (header + a units row + data rows). For a
    single kernel the metrics are returned flat; for a multi-kernel pipeline each
    row's metrics are prefixed with the kernel name (duplicates get a #N suffix), so
    a pipeline yields isolated per-stage metrics.
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
                if col not in _NCU_KEEP_COLUMNS:
                    continue
                if not raw_value or raw_value in ("no data", ""):
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
        extra_sources, extra_flags = reference_build_extras(get_problem_dir())
        build_result = compile_framework(
            iter_dir, extra_sources=extra_sources, extra_flags=extra_flags
        )
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
        with open(problem_dir / "problem.yaml", encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f) or {}
        gpu_index = cfg_yaml.get("gpu", {}).get("index", 0)
        ref_file = (cfg_yaml.get("reference") or {}).get("file", ref_file)
    except Exception as e:
        log(f"Failed to parse problem.yaml, using defaults: {e}", "WARN")

    # Profile mode hands results.json back to KTT via LoadResults. A file written by a
    # pre-2.3 KTT has no Timestamp field and aborts the driver on read, which only
    # happens when resuming a run tuned before the upgrade.
    ensure_results_loadable(results_path)

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
        # Skip the default section set; collect launch/occupancy stats explicitly.
        "--set",
        "none",
        "--section",
        "LaunchStats",
        "--section",
        "Occupancy",
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
