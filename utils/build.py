"""Compile a generated framework.cpp into a runnable KTT driver binary.

Phase 2 of framework-file autotuning. Kept separate from run.py so it can be
tested independently and reused once configure.py emits framework.cpp per
iteration. Compile/link flags were locked in Phase 0 (see FRAMEWORK_FILE_SPEC §14):
the driver is host-compiled and linked against libktt.so; libcuda/libnvrtc come
transitively, and no CUDA host includes are needed (NVRTC uses them at runtime).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# Repo root = parent of utils/ (libktt.so symlink + KTT/Source live here).
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BuildResult:
    ok: bool
    binary: Path | None
    stderr: str
    cmd: list[str]


def compile_command(
    framework_cpp: Path, out_binary: Path, repo_root: Path = REPO_ROOT
) -> list[str]:
    """g++ command to build a framework driver against libktt.so."""
    return [
        "g++",
        "-std=c++17",
        "-m64",
        "-O3",
        f"-I{repo_root / 'KTT' / 'Source'}",
        str(framework_cpp),
        str(repo_root / "libktt.so"),
        f"-Wl,-rpath,{repo_root}",
        "-o",
        str(out_binary),
    ]


def compile_framework(
    iter_dir, repo_root: Path = REPO_ROOT, timeout: float = 180.0
) -> BuildResult:
    """Compile ``<iter_dir>/framework.cpp`` -> ``<iter_dir>/driver``.

    ``inputs.hpp`` is resolved relative to framework.cpp (same directory), so it
    must sit alongside. On failure, ``stderr`` holds the g++ diagnostics for the
    LLM feedback loop; ``ok`` is False and ``binary`` is None.
    """
    iter_dir = Path(iter_dir)
    framework_cpp = iter_dir / "framework.cpp"
    out_binary = iter_dir / "driver"
    if not framework_cpp.exists():
        return BuildResult(False, None, f"framework.cpp not found in {iter_dir}", [])

    cmd = compile_command(framework_cpp, out_binary, repo_root)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return BuildResult(False, None, f"compile timed out after {timeout:.0f}s", cmd)

    ok = proc.returncode == 0 and out_binary.exists()
    return BuildResult(ok, out_binary if ok else None, proc.stderr, cmd)


def driver_command(
    binary,
    platform: int,
    device: int,
    duration: float,
    tolerance: float,
    output_base: str,
    kernel_file,
    ref_file,
) -> list[str]:
    """argv for a compiled framework driver (see FRAMEWORK_FILE_SPEC §14).

    KTT writes ``<output_base>.json``.
    """
    return [
        str(binary),
        str(platform),
        str(device),
        str(duration),
        str(tolerance),
        str(output_base),
        str(kernel_file),
        str(ref_file),
    ]
