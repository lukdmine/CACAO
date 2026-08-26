"""Compile a generated framework.cpp into a runnable KTT driver binary.

Phase 2 of framework-file autotuning. Kept separate from run.py so it can be
tested independently and reused once configure.py emits framework.cpp per
iteration. Compile/link flags were locked in Phase 0:
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


def object_command(
    source: Path,
    out_obj: Path,
    repo_root: Path = REPO_ROOT,
    extra_flags: list = (),
) -> list[str]:
    """g++ command to compile one extra source (ref_cpu.c) to an object file.

    The CPU-reference -D scalar macros are applied HERE, in a separate translation
    unit — they must never reach framework.cpp, where a macro named STUDENTS would
    clobber inputs.hpp's `inline constexpr int STUDENTS` of the same name. g++
    builds .c as C++; the contract requires ``extern "C"``.
    """
    return [
        "g++",
        "-std=c++17",
        "-m64",
        "-O3",
        f"-I{repo_root / 'KTT' / 'Source'}",
        *[str(f) for f in extra_flags],
        "-c",
        str(source),
        "-o",
        str(out_obj),
    ]


def compile_command(
    framework_cpp: Path,
    out_binary: Path,
    repo_root: Path = REPO_ROOT,
    extra_objects: list = (),
) -> list[str]:
    """g++ command to build a framework driver against libktt.so.

    ``extra_objects`` are pre-compiled CPU-reference objects (see object_command)
    linked into the driver.
    """
    return [
        "g++",
        "-std=c++17",
        "-m64",
        "-O3",
        f"-I{repo_root / 'KTT' / 'Source'}",
        str(framework_cpp),
        *[str(o) for o in extra_objects],
        str(repo_root / "libktt.so"),
        f"-Wl,-rpath,{repo_root}",
        "-o",
        str(out_binary),
    ]


def reference_build_extras(problem_dir) -> tuple[list, list]:
    """(extra_sources, extra_flags) the problem's reference adds to the driver build.

    Empty for cuda references. For cpu_c: ref_cpu.c plus -DNAME=value for every
    problem scalar (the C contract: pointer args only, scalars as macros). Shared by
    run and profile so both build the same driver.
    """
    import yaml

    problem_dir = Path(problem_dir)
    try:
        cfg = yaml.safe_load((problem_dir / "problem.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return [], []
    ref = cfg.get("reference") or {}
    if str(ref.get("type", "cuda")).lower() != "cpu_c":
        return [], []

    from utils.inputs import load_inputs_spec, scalar_define_flags

    sources = [problem_dir / ref.get("file", "ref_cpu.c")]
    try:
        flags = scalar_define_flags(load_inputs_spec(problem_dir / "inputs.yaml"))
    except Exception as e:
        from utils.log import log

        log(
            f"cpu_c reference: could not derive -D flags from inputs.yaml ({e}) — "
            "ref_cpu.c will compile without scalar macros",
            "WARN",
        )
        flags = []
    return sources, flags


def compile_dump_inputs(
    dump_cpp: Path,
    out_binary: Path,
    problem_dir: Path,
    repo_root: Path = REPO_ROOT,
    timeout: float = 60.0,
) -> BuildResult:
    """Compile the standalone input-dump tool (utils/inputs.generate_dump_inputs_cpp).

    Includes inputs.hpp (from ``problem_dir``) so the ``cacao_ref::h_*`` statics
    build the real input data at static-init. No ktt::Tuner is instantiated and
    no KTT symbols are called, but libktt.so is linked to resolve anything Ktt.h
    pulls in transitively. Run the binary with cwd = the desired output dir.
    """
    cmd = [
        "g++",
        "-std=c++17",
        "-m64",
        "-O3",
        f"-I{repo_root / 'KTT' / 'Source'}",
        f"-I{problem_dir}",
        str(dump_cpp),
        str(repo_root / "libktt.so"),
        f"-Wl,-rpath,{repo_root}",
        "-o",
        str(out_binary),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return BuildResult(False, None, f"compile timed out after {timeout:.0f}s", cmd)
    ok = proc.returncode == 0 and out_binary.exists()
    return BuildResult(ok, out_binary if ok else None, proc.stderr, cmd)


def compile_framework(
    iter_dir,
    repo_root: Path = REPO_ROOT,
    timeout: float = 180.0,
    extra_sources: list = (),
    extra_flags: list = (),
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

    # Extra sources (ref_cpu.c) compile to objects first so extra_flags (-D scalar
    # macros) stay out of the framework.cpp translation unit — see object_command.
    extra_objects = []
    for src in extra_sources:
        src = Path(src)
        obj = iter_dir / f"{src.stem}.o"
        obj_cmd = object_command(src, obj, repo_root, extra_flags)
        try:
            proc = subprocess.run(obj_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return BuildResult(False, None, f"compile timed out after {timeout:.0f}s", obj_cmd)
        if proc.returncode != 0 or not obj.exists():
            return BuildResult(False, None, proc.stderr, obj_cmd)
        extra_objects.append(obj)

    cmd = compile_command(framework_cpp, out_binary, repo_root, extra_objects)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
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
    """argv for a compiled framework driver.

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
