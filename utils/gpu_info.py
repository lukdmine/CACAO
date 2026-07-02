"""
Utility to auto-detect GPU hardware specifications using NVIDIA Nsight Compute (NCU).

This module compiles and profiles a minimal dummy CUDA kernel to extract precise
device attributes reported by the driver/hardware.
"""

import csv
import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from utils.cuda_env import get_env, get_ncu_tmpdir
from utils.log import log

DUMMY_KERNEL_CODE = """
#include <cstdio>
__global__ void dummy_kernel(float *out) {
    out[threadIdx.x] = 1.0f;
}
int main() {
    float *d_out;
    cudaMalloc(&d_out, 32 * sizeof(float));
    dummy_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
"""


def _query_nvidia_smi_gpus() -> list[dict]:
    """Query basic GPU info from nvidia-smi when available."""
    try:
        cuda_env = get_env()
        nvidia_smi = str(cuda_env.nvidia_smi) if cuda_env.nvidia_smi else "nvidia-smi"
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        gpus = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue

            try:
                gpus.append(
                    {
                        "index": int(parts[0].replace(" ", "")),
                        "name": parts[1],
                        "total_memory_mb": int(parts[2].replace(" ", "")),
                        "free_memory_mb": int(parts[3].replace(" ", "")),
                    }
                )
            except ValueError:
                continue

        return gpus
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("GPU detection failed: %s", e)
        return []


def _parse_ncu_csv(csv_output: str) -> dict:
    """
    Parse NCU CSV output (--page raw wide format) into a flat dict of metrics.
    Only takes the specific device attribute metrics we requested.
    """
    metrics = {}

    try:
        # Strip bare text preamble lines that don't look like CSV
        csv_lines = [
            line
            for line in csv_output.splitlines()
            if line.startswith('"') or (line and line[0].isdigit())
        ]

        if not csv_lines or len(csv_lines) < 3:
            return metrics

        header_line = csv_lines[0]
        # skip units row at index 1
        data_lines = csv_lines[2:]

        reader = csv.DictReader(io.StringIO("\n".join([header_line] + data_lines)))
        rows = list(reader)

        if not rows:
            return metrics

        # The dummy kernel rows
        target = next(
            (r for r in rows if r.get("Kernel Name", "") == "dummy_kernel"), rows[-1]
        )

        for col, raw_value in target.items():
            if not col or not raw_value or raw_value in ("no data", ""):
                continue

            cleaned = raw_value.replace(",", "").replace(" ", "")
            try:
                metrics[col] = float(cleaned) if "." in cleaned else int(cleaned)
            except ValueError:
                metrics[col] = raw_value

    except Exception as e:
        log(f"Failed to parse NCU CSV: {e}", "WARN")

    return metrics


def get_gpu_details(device_index: int = 0) -> Optional[Dict[str, Any]]:
    """
    Get full GPU specifications by compiling and profiling a dummy kernel.

    Args:
        device_index: Which CUDA device to query (default 0)

    Returns:
        Dictionary matching the problem.yaml `gpu` section schema,
        or None if detection failed.
    """
    metrics_to_query = [
        "device__attribute_display_name",
        "device__attribute_compute_capability_major",
        "device__attribute_compute_capability_minor",
        "device__attribute_multiprocessor_count",
        "device__attribute_max_threads_per_block",
        "device__attribute_max_shared_memory_per_block",
        "device__attribute_max_registers_per_block",
        "device__attribute_memory_bandwidth",  # Some cards lack this, we'll try to calculate fallback
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cu_file = tmpdir_path / "dummy.cu"
        exe_file = tmpdir_path / "dummy"

        cu_file.write_text(DUMMY_KERNEL_CODE)

        # 1. Compile dummy kernel
        cuda_env = get_env()
        if not cuda_env.nvcc:
            log("nvcc not found. Cannot auto-detect GPU using dummy kernel.", "ERROR")
            return None
        try:
            subprocess.run(
                [str(cuda_env.nvcc), "-o", str(exe_file), str(cu_file)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            log(
                f"Failed to compile dummy kernel for GPU detection: {e.stderr}", "ERROR"
            )
            return None
        except FileNotFoundError:
            log("nvcc not found. Cannot auto-detect GPU using dummy kernel.", "ERROR")
            return None

        # 2. Run NCU on dummy kernel
        ncu_bin = str(cuda_env.ncu) if cuda_env.ncu else "ncu"
        ncu_cmd = [
            ncu_bin,
            "--csv",
            "--page",
            "raw",
            "--set",
            "none",  # Don't run default sets, we only want specific metrics
            "--metrics",
            ",".join(metrics_to_query),
            "--target-processes",
            "all",
            str(exe_file),
        ]

        # Make the executable run on the selected device
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)
        # Per-user TMPDIR so NCU's /tmp/nsight-compute-lock doesn't clash between
        # users on a shared machine (see cuda_env.get_ncu_tmpdir).
        env["TMPDIR"] = str(get_ncu_tmpdir())

        try:
            # Note: NCU might require sudo for tracing depending on system config,
            # but querying device attributes often works even if counter collection fails.
            result = subprocess.run(
                ncu_cmd,
                check=False,  # We handle errors manually
                capture_output=True,
                text=True,
                env=env,
                timeout=30,  # Should be very fast
            )

            # 3. Parse output
            ncu_metrics = _parse_ncu_csv(result.stdout)

            if (
                not ncu_metrics
                or "device__attribute_compute_capability_major" not in ncu_metrics
            ):
                if result.returncode != 0:
                    log(
                        f"NCU failed during GPU detection (exit {result.returncode})",
                        "WARN",
                    )
                    # NCU writes its ==ERROR== diagnostics to stdout, not stderr.
                    detail = result.stderr.strip() or result.stdout.strip()
                    log(f"ncu output: {detail}", "WARN")
                else:
                    log(
                        "NCU completed but did not return expected device metrics.",
                        "WARN",
                    )
                return None

            # Build the config dictionary
            major = ncu_metrics.get("device__attribute_compute_capability_major", 0)
            minor = ncu_metrics.get("device__attribute_compute_capability_minor", 0)

            gpu_config = {
                "model": str(
                    ncu_metrics.get(
                        "device__attribute_display_name", f"Unknown GPU {device_index}"
                    )
                ),
                "compute_capability": f"{major}.{minor}",
                "sm_count": int(
                    ncu_metrics.get("device__attribute_multiprocessor_count", 0)
                ),
                "max_threads_per_block": int(
                    ncu_metrics.get("device__attribute_max_threads_per_block", 1024)
                ),
                "shared_memory_per_block": int(
                    ncu_metrics.get(
                        "device__attribute_max_shared_memory_per_block", 49152
                    )
                ),
                "registers_per_block": int(
                    ncu_metrics.get("device__attribute_max_registers_per_block", 65536)
                ),
            }

            # Handle bandwidth (not consistently reported via device__attribute_memory_bandwidth)
            # If we don't get a valid number, we fall back to a reasonable default based on CC
            bw = ncu_metrics.get("device__attribute_memory_bandwidth")
            if bw is not None and isinstance(bw, (int, float)) and bw > 0:
                gpu_config["memory_bandwidth_gb"] = int(bw / (1000 * 1000 * 1000))
            else:
                # Fallback approximations for peak memory bandwidth (GB/s) based on compute capability
                cc = float(f"{major}.{minor}")
                if cc >= 9.0:
                    gpu_config["memory_bandwidth_gb"] = (
                        2000  # Hopper typically 2-3 TB/s
                    )
                elif cc >= 8.0:
                    gpu_config["memory_bandwidth_gb"] = (
                        936  # Ampere typically ~900 GB/s (A100/3090)
                    )
                elif cc >= 7.0:
                    gpu_config["memory_bandwidth_gb"] = (
                        616  # Volta/Turing typically ~600 GB/s
                    )
                else:
                    gpu_config["memory_bandwidth_gb"] = 320  # Pascal/older

            return gpu_config

        except subprocess.TimeoutExpired:
            log("NCU profiling timed out during GPU detection", "ERROR")
            return None
        except Exception as e:
            log(f"Error during GPU detection: {e}", "ERROR")
            return None


def detect_gpus() -> list[dict]:
    """
    Detect all available GPUs and their basic specs.
    Currently just returns the primary GPU (index 0) using get_gpu_details,
    or empty list if detection fails.
    """
    # NCU dummy kernel approach is slow to do for EVERY possible device index.
    # For now, we'll just query device 0, unless CUDA_VISIBLE_DEVICES is set.
    # If the user needs another device, they explicitly pass --device N.
    smi_gpus = _query_nvidia_smi_gpus()
    if smi_gpus:
        devices = []
        for smi_gpu in smi_gpus:
            idx = smi_gpu["index"]
            details = get_gpu_details(idx) or {}
            merged = {
                **details,
                **smi_gpu,
            }
            if "model" in merged and "name" not in merged:
                merged["name"] = merged["model"]
            elif "name" in merged and "model" not in merged:
                merged["model"] = merged["name"]
            devices.append(merged)
        return devices

    gpu = get_gpu_details(0)
    if gpu:
        gpu["index"] = 0
        gpu["name"] = gpu.get("model", f"GPU {gpu['index']}")
        gpu["total_memory_mb"] = gpu.get("total_memory_mb")
        gpu["free_memory_mb"] = gpu.get("free_memory_mb")
        return [gpu]
    return []


def format_gpu_table(gpus: list[dict]) -> str:
    """Format GPUs as a numbered table for CLI display."""
    if not gpus:
        return "No GPUs detected"

    lines = ["Detected GPUs:"]
    for i, gpu in enumerate(gpus):
        idx = gpu.get("index", i)
        name = gpu.get("model", "Unknown")
        cc = gpu.get("compute_capability", "?")
        sms = gpu.get("sm_count", "?")
        bw = gpu.get("memory_bandwidth_gb", "?")
        lines.append(f"  [{idx}] {name} (CC {cc}, {sms} SMs, {bw} GB/s bandwidth)")

    return "\n".join(lines)


if __name__ == "__main__":
    gpus = detect_gpus()
    print(format_gpu_table(gpus))

    if gpus:
        print("\nDetailed config for GPU 0:")
        import yaml

        print(yaml.dump(get_gpu_details(0)))
