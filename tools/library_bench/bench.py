"""bench.py — time a vendor-library op against a CACAO problem's real inputs and
seed the result as the problem's baseline (``output/reference_time.json``).

One command per (problem, GPU):

    python -m tools.library_bench.bench problems/<name> --op <op>

Pipeline (scratch in ``<problem>/output/_bench/``):

1. Regenerate ``inputs.hpp`` (the engine's canonical boundary) and dump the
   problem's real input buffers to ``cacao_in_<name>.bin`` — via a generated
   dump tool of the same shape the engine already compiles for the
   python-reference timer, but driven by the ``gen_*()`` generators directly,
   which exist for every reference type.
2. Generate ``bench_spec.h`` (buffer/scalar tables, ref launch geometry) and
   ``bench_ref_launch.cu`` (marshals the boundary args into the problem's
   ``ref_kernel.cu``) — both derived from the SAME ``InputsSpec`` the engine
   consumes, so the tool cannot drift from the engine's launch contract.
3. nvcc: fixed ``harness.cu`` (timing core) + ``ops/<op>.cu`` (the one file of
   library-specific code) + the generated launcher. The harness runs the
   reference kernel AND the op in one GPU session, dumps both outputs, then
   times the op: warmup + per-rep CUDA events, emit ``CACAO_REF_TIME_US=<min>``.
4. Verification gate: compare op vs reference-kernel output in numpy within the
   problem's ``validation.tolerance`` (worst index/value diagnostics on
   mismatch). A mismatch refuses to seed — a layout bug in an op would
   otherwise silently benchmark the wrong math.
5. Seed ``output/reference_time.json`` (first-write-wins, like the engine's
   ggml/python baselines) + a provenance sidecar ``reference_bench.json`` in
   the problem dir root (GPU, op, timing params, hash of scalars + input
   blobs). ``--check`` re-verifies the sidecar against the current problem.

v1 limits: the verification gate supports ``reference.type: cuda`` problems
(the trusted oracle runs as a device kernel), and exactly one validated buffer
— true for every comparison problem this tool exists for. Extend when a
problem needs it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# _literal/_size_expr are the engine's own scalar formatting + size-expression
# widening helpers. Private names, reused deliberately: buffer sizes and grid
# expressions must evaluate with exactly the engine's 64-bit widening rules,
# and duplicating that logic is how the tool would drift.
from utils.inputs import (  # noqa: E402
    _literal,
    _size_expr,
    ensure_inputs_hpp,
    load_inputs_spec,
)
from utils.build import compile_dump_inputs  # noqa: E402
from utils.cuda_env import get_env, get_subprocess_env  # noqa: E402
from utils.results import (  # noqa: E402
    _REFERENCE_TIME_FILE,
    load_reference_time,
    save_reference_time,
)

TOOL_DIR = Path(__file__).resolve().parent
OPS_DIR = TOOL_DIR / "ops"
BENCH_SUBDIR = "_bench"
SIDECAR_FILE = "reference_bench.json"
TIME_MARKER = "CACAO_REF_TIME_US"

_ACCESS = {"read": 0, "write": 1, "readwrite": 2}


class BenchError(RuntimeError):
    """User-facing failure; main() prints the message without a traceback."""


# ---------------------------------------------------------------------------
# Problem loading + generated code (the drift-proofing lives here)
# ---------------------------------------------------------------------------


def _load_problem(problem_dir: Path):
    problem_dir = problem_dir.resolve()
    for name in ("problem.yaml", "inputs.yaml"):
        if not (problem_dir / name).is_file():
            raise BenchError(f"{problem_dir / name} not found — not a framework problem")
    cfg = yaml.safe_load((problem_dir / "problem.yaml").read_text(encoding="utf-8")) or {}
    spec = load_inputs_spec(problem_dir / "inputs.yaml")
    ref = cfg.get("reference") or {}
    if str(ref.get("type", "cuda")).lower() != "cuda":
        raise BenchError(
            "bench currently supports reference.type: cuda problems only "
            f"(this one is {ref.get('type')!r}) — the verification gate runs the "
            "reference as a device kernel"
        )
    validated = spec.validated
    if len(validated) != 1:
        raise BenchError(
            f"bench v1 requires exactly one validated buffer, got "
            f"{[b.name for b in validated]}"
        )
    return problem_dir, cfg, spec


def generate_bench_spec_h(spec, cfg) -> str:
    """buffer/scalar tables + ref launch geometry, evaluated host-side with the
    engine's own expression widening. Included by harness.cu only.

    define-placed scalars arrive as -D macros on the compile line (the ref
    kernel's contract), so this header must NOT also declare them as consts —
    the preprocessor would rewrite the declaration itself. Non-define scalars
    are declared here for size/grid expressions to reference."""
    scalars = "\n".join(
        f"static const {s.dtype} {s.name} = {_literal(s.value, s.dtype)};"
        for s in spec.scalars
        if "define" not in s.placements
    )
    buf_rows = []
    validated_index = None
    for i, b in enumerate(spec.buffers):
        if b.validate_output:
            validated_index = i
        buf_rows.append(
            f'    {{"{b.name}", "{b.dtype}", static_cast<size_t>({_size_expr(spec, b.size)}),'
            f" {_ACCESS[b.access]}, {'true' if b.validate_output else 'false'}}},"
        )
    scalar_rows = ",\n".join(
        f'    {{"{s.name}", {1 if s.dtype == "float" else 0}, {float(s.value)}}}'
        for s in spec.scalars
    )
    grid = cfg["grid"]
    block = (cfg.get("reference") or {}).get("block") or {"x": 1}
    gst = 1 if str(cfg.get("global_size_type", "cuda")).lower() == "opencl" else 0
    grid_dims = ", ".join(
        f"static_cast<size_t>({_size_expr(spec, str(grid[d]))})" if d in grid else "1"
        for d in ("x", "y", "z")
    )
    block_dims = ", ".join(str(block.get(d, 1)) for d in ("x", "y", "z"))
    headers = "\n".join(f"#include {h}" for h in spec.headers)

    return f"""// bench_spec.h — GENERATED by tools/library_bench/bench.py. Do not edit by hand.
#pragma once
#include <cstddef>
{headers}
namespace cacao_bench {{
{scalars}

struct BufferInfo {{ const char* name; const char* dtype; size_t elems; int access; bool validated; }};
static const BufferInfo kBuffers[] = {{
{os.linesep.join(buf_rows)}
}};
static const int kNumBuffers = {len(buf_rows)};
static const int kValidatedIndex = {validated_index};

struct ScalarInfo {{ const char* name; int is_float; double value; }};
static const ScalarInfo kScalars[] = {{
{scalar_rows}
}};
static const int kNumScalars = {len(spec.scalars)};

static const size_t kRefGrid[3] = {{ {grid_dims} }};
static const unsigned kRefBlock[3] = {{ {block_dims} }};
static const int kGlobalSizeIsOpenCL = {gst};
}}  // namespace cacao_bench
"""


def generate_ref_launch_cu(spec, cfg) -> str:
    """Marshals the boundary (+ device buffers) into the problem's ref_kernel.cu.

    The launch geometry arrives computed from the harness; the ARG LIST comes
    from InputsSpec.boundary — the same ordering rule the engine's KTT driver
    uses for SetArguments(refDef, in.boundary). ref_kernel.cu is #included
    textually so the launch sees the definition in its own TU (no -rdc)."""
    ref = cfg["reference"]
    buf_index = {b.name: i for i, b in enumerate(spec.buffers)}
    args = []
    for a in spec.boundary:
        if a.kind == "scalar":
            args.append(_literal(a.value, a.dtype))
        else:
            args.append(f"({a.dtype}*)d[{buf_index[a.name]}]")
    return f"""// bench_ref_launch.cu — GENERATED by tools/library_bench/bench.py. Do not edit by hand.
#include <cuda_runtime.h>
#include "{ref['file']}"

extern "C" void cacao_launch_ref(void* const* d, unsigned gx, unsigned gy, unsigned gz,
                                 unsigned bx, unsigned by, unsigned bz) {{
    {ref['function']}<<<dim3(gx, gy, gz), dim3(bx, by, bz)>>>({", ".join(args)});
}}
"""


def generate_dump_inputs_cpp(spec) -> str:
    """Like utils.inputs.generate_dump_inputs_cpp but driven by gen_*()
    directly — the engine's variant reads the cacao_ref::h_* statics, which
    only exist for cpu_c/python references. gen_*() exists for every problem
    and IS the data the tuner driver registers for cuda references."""
    lines = [
        "// dump_inputs.cpp — GENERATED by tools/library_bench/bench.py. Do not edit by hand.",
        "#include <fstream>",
        '#include "inputs.hpp"',
        "",
        "int main() {",
    ]
    for b in spec.buffers:
        if b.access not in ("read", "readwrite"):
            continue
        lines.append(
            f'    {{ const auto v = gen_{b.name}();'
            f' std::ofstream f("cacao_in_{b.name}.bin", std::ios::binary);'
            f" f.write(reinterpret_cast<const char*>(v.data()),"
            f" static_cast<std::streamsize>(v.size() * sizeof({b.dtype}))); }}"
        )
    lines += ["    return 0;", "}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Compile + run helpers
# ---------------------------------------------------------------------------


def _nvcc_gencode(env, gpu_index: int) -> list:
    """Device-targeted gencode; PTX-only fallback so a broken/absent driver at
    COMPILE time (nvcc -arch=native would fail outright) still builds a binary
    that JITs on whatever GPU it later lands on."""
    if env.nvidia_smi:
        try:
            out = subprocess.run(
                [str(env.nvidia_smi), f"--id={gpu_index}",
                 "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            cc = out.stdout.strip().splitlines()[0].replace(".", "")
            if out.returncode == 0 and cc.isdigit():
                return [f"-gencode=arch=compute_{cc},code=sm_{cc}"]
        except Exception:
            pass
    return ["-gencode=arch=compute_75,code=compute_75"]  # PTX JITs anywhere >= 7.5


def _op_flags(op: str) -> list:
    """Extra nvcc/link flags for an op (e.g. ``-lcusparse`` or a cudnn_frontend
    include dir), whitespace-separated, # comments allowed, ``$VARS`` expanded
    from the environment (an unset var is an error naming the op, not a silent
    empty flag). The op's code is the .cu; its build-specific bits are this
    sidecar — no other file learns about the vendor library."""
    path = OPS_DIR / f"{op}.flags.txt"
    if not path.is_file():
        return []
    flags = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split():
            token = os.path.expandvars(token)
            if "$" in token:
                raise BenchError(
                    f"{path.name}: unset environment variable in flag {token!r} — "
                    "set the variables documented at the top of the flags file"
                )
            flags.append(token)
    return flags


def _run(cmd: list, cwd: Path, timeout: float, what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, cwd=str(cwd), env=get_subprocess_env(),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise BenchError(f"{what} failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")
    return proc


def _gpu_name(env, gpu_index: int) -> str:
    if env.nvidia_smi:
        try:
            out = subprocess.run(
                [str(env.nvidia_smi), f"--id={gpu_index}",
                 "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except Exception:
            pass
    return "unknown"


# ---------------------------------------------------------------------------
# Verification gate (in numpy, where mismatch diagnostics are actually useful)
# ---------------------------------------------------------------------------


def _compare_outputs(ref_path: Path, op_path: Path, dtype: str,
                     tolerance: float, elems: int) -> tuple[float, str | None]:
    """(max |diff|, diagnostic or None on match). The measured max diff is
    returned even on match — tolerance calibration needs it (the problem.yaml
    runbook: bench with --tolerance 1.0, read the max diff, set 10x above)."""
    np_dtype = np.float32 if dtype == "float" else np.int32
    ref = np.fromfile(ref_path, dtype=np_dtype)
    op = np.fromfile(op_path, dtype=np_dtype)
    if ref.size != elems or op.size != elems:
        return float("inf"), (
            f"output size mismatch: ref has {ref.size} elements, op has {op.size}, "
            f"the problem's validated buffer holds {elems} — the op wrote the wrong "
            "amount of data"
        )
    diff = np.abs(ref.astype(np.float64) - op.astype(np.float64))
    worst = int(np.argmax(diff))
    n_bad = int((diff > tolerance).sum())
    if n_bad == 0:
        return float(diff[worst]), None
    return float(diff[worst]), (
        f"{n_bad}/{elems} elements outside tolerance {tolerance:g} — "
        f"worst at index {worst}: ref={ref[worst]!r} op={op[worst]!r} "
        f"(|diff|={diff[worst]:.6g}). Refusing to seed: the op is not computing "
        "what the problem's reference kernel computes."
    )


def _validated_elems(spec) -> int:
    """Element count of the validated buffer from its size expression, with
    C++ integer-division semantics over the scalar values. The validated
    buffer is typically write-only, so unlike the inputs there is no dumped
    file whose byte count we could read instead."""
    b = spec.validated[0]
    env = {s.name: s.value for s in spec.scalars}
    expr = b.size.replace("/", "//")
    try:
        value = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 — scalars only
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        raise BenchError(
            f"cannot evaluate validated buffer size expression {b.size!r}: {e}"
        ) from e
    return int(value)


# ---------------------------------------------------------------------------
# Sidecar (provenance) + staleness check
# ---------------------------------------------------------------------------


def _inputs_sha256(problem_dir: Path, spec) -> str:
    """Fingerprint of everything that determines the input data: the scalar
    values (sizes, shapes) and the init=file blobs. The built-in generators
    derive deterministically from buffer names, so changed data always
    changes this hash."""
    h = hashlib.sha256()
    scalars = [(s.name, s.dtype, s.value) for s in spec.scalars]
    h.update(json.dumps(scalars).encode())
    for b in spec.buffers:
        if b.init == "file":
            blob = problem_dir / "inputs" / b.file_name
            h.update(b.file_name.encode())
            h.update(hashlib.sha256(blob.read_bytes()).digest())
    return h.hexdigest()


def _write_sidecar(problem_dir: Path, info: dict) -> Path:
    path = problem_dir / SIDECAR_FILE
    tmp = problem_dir / (SIDECAR_FILE + ".tmp")
    tmp.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def check(problem_dir: Path) -> int:
    """Compare the sidecar's provenance against the current problem state."""
    problem_dir = problem_dir.resolve()
    sidecar_path = problem_dir / SIDECAR_FILE
    if not sidecar_path.is_file():
        print(f"no baseline sidecar at {sidecar_path} — nothing has been benched here")
        return 2
    try:
        info = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"corrupt sidecar {sidecar_path}: {e} — re-seed with --force")
        return 2
    if not isinstance(info, dict) or info.get("time_us") is None or not info.get("op"):
        print(f"sidecar {sidecar_path} is missing fields — re-seed with --force")
        return 2
    _, _, spec = _load_problem(problem_dir)
    stale = []
    if info.get("inputs_sha256") != _inputs_sha256(problem_dir, spec):
        stale.append("input scalars/blob contents changed")
    gpu_now = _gpu_name(get_env(), int(info.get("gpu", {}).get("index", 0)))
    gpu_then = info.get("gpu", {}).get("name")
    # "unknown" on either side means "could not probe the GPU" (e.g. a broken
    # nvidia-smi at bench time) — it must not read as a GPU change.
    if (gpu_then and gpu_then != "unknown" and gpu_now != "unknown"
            and gpu_now != gpu_then):
        stale.append(f"GPU changed: benched on {gpu_then!r}, now {gpu_now!r}")
    if stale:
        print(f"STALE baseline ({sidecar_path.name}):")
        for s in stale:
            print(f"  - {s}")
        print("re-run: python -m tools.library_bench.bench "
              f"{problem_dir} --op {info.get('op')} --force")
        return 1
    print(f"baseline OK: {info.get('time_us'):.2f} us via op {info.get('op')!r} "
          f"on {gpu_then}, {info.get('date_utc')}")
    return 0


# ---------------------------------------------------------------------------
# Main bench flow
# ---------------------------------------------------------------------------


def bench(problem_dir: Path, op: str, reps: int, warmup: int,
          force: bool, tolerance: float | None, device_override: int | None) -> int:
    problem_dir, cfg, spec = _load_problem(problem_dir)
    op_cu = OPS_DIR / f"{op}.cu"
    if not op_cu.is_file():
        known = sorted(p.stem for p in OPS_DIR.glob("*.cu"))
        raise BenchError(f"op {op!r} not found at {op_cu} — available: {known}")

    gpu_index = device_override if device_override is not None else int(
        (cfg.get("gpu") or {}).get("index", 0))
    scratch = problem_dir / "output" / BENCH_SUBDIR
    scratch.mkdir(parents=True, exist_ok=True)
    env = get_env()
    if env.nvcc is None:
        raise BenchError("nvcc not found — set up the CUDA toolkit (see cuda_env.yaml)")

    # Cross-process serialization via two advisory flocks, acquired in a fixed
    # order (GPU first, problem second) to avoid deadlock: the GPU lock keeps
    # timings uncontended, the problem lock keeps two concurrent benches of the
    # same problem (possible on different GPUs) from colliding on the shared
    # _bench scratch dir. The engine's own lock (utils/gpu_lock) is in-process
    # asyncio and cannot interoperate.
    problem_key = hashlib.sha1(str(problem_dir).encode()).hexdigest()[:12]
    tmp = Path(tempfile.gettempdir())
    with contextlib.ExitStack() as stack:
        for name in (f"cacao_bench_gpu{gpu_index}.lock", f"cacao_bench_{problem_key}.lock"):
            lock_file = stack.enter_context(open(tmp / name, "w", encoding="utf-8"))
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _bench_locked(problem_dir, cfg, spec, op, op_cu, reps, warmup,
                             force, tolerance, gpu_index, scratch, env)


def _prepare(problem_dir, cfg, spec, op, op_cu, gpu_index, scratch, env) -> Path:
    """Steps 1-3: fresh boundary, dumped inputs, generated spec/launcher, nvcc
    build. Everything up to (and excluding) touching the GPU — tests exercise
    exactly this much on a device-less machine."""
    # 1. fresh boundary + dumped inputs
    ensure_inputs_hpp(problem_dir)
    dump_cpp = scratch / "dump_inputs.cpp"
    dump_cpp.write_text(generate_dump_inputs_cpp(spec), encoding="utf-8")
    build = compile_dump_inputs(dump_cpp, scratch / "dump_inputs", problem_dir)
    if not build.ok:
        raise BenchError(f"dump tool compile failed:\n{build.stderr[-4000:]}")
    _run([str(scratch / "dump_inputs")], scratch, 300, "input dump")

    # 2. generated spec + ref launcher
    (scratch / "bench_spec.h").write_text(generate_bench_spec_h(spec, cfg), encoding="utf-8")
    (scratch / "bench_ref_launch.cu").write_text(generate_ref_launch_cu(spec, cfg), encoding="utf-8")

    # 3. nvcc: harness + op + launcher (one invocation links all three)
    define_flags = [
        f"-D{s.name}={_literal(s.value, s.dtype)}"
        for s in spec.scalars
        if "define" in s.placements
    ]  # ONLY define-placed scalars: macro-defining a runtime scalar would
    # rewrite the ref kernel's parameter names before compilation.
    bench_bin = scratch / "bench"
    cmd = [
        str(env.nvcc), "-std=c++17", "-O3", "-Wno-deprecated-gpu-targets",
        *_nvcc_gencode(env, gpu_index),
        f"-I{TOOL_DIR}", f"-I{scratch}", f"-I{problem_dir}",
        *define_flags, *_op_flags(op),
        str(TOOL_DIR / "harness.cu"), str(op_cu), str(scratch / "bench_ref_launch.cu"),
        "-o", str(bench_bin),
    ]
    _run(cmd, scratch, 600, "bench compile")
    return bench_bin


def _bench_locked(problem_dir, cfg, spec, op, op_cu, reps, warmup,
                  force, tolerance, gpu_index, scratch, env) -> int:
    print(f"[bench] problem={problem_dir.name} op={op} gpu={gpu_index} scratch={scratch}")
    bench_bin = _prepare(problem_dir, cfg, spec, op, op_cu, gpu_index, scratch, env)

    # 4. run: ref kernel gate outputs + op outputs + timed reps, one session
    proc = _run(
        [str(bench_bin), str(gpu_index), str(scratch), str(warmup), str(reps), str(scratch)],
        scratch, 1800, "bench run")
    for line in proc.stderr.splitlines():
        print(f"[harness] {line}", file=sys.stderr)
    match = re.search(rf"{TIME_MARKER}=([\d.]+)", proc.stdout)
    if not match:
        raise BenchError(f"bench produced no {TIME_MARKER} marker:\n{proc.stdout[-2000:]}")
    time_us = float(match.group(1))

    # 5. verification gate (worst-index diagnostics, refuses to seed on miss)
    vb = spec.validated[0]
    tol = float(tolerance) if tolerance is not None else float(
        (cfg.get("validation") or {}).get("tolerance", 1e-4))
    max_diff, diag = _compare_outputs(
        scratch / f"bench_ref_{vb.name}.bin", scratch / f"bench_op_{vb.name}.bin",
        vb.dtype, tol, _validated_elems(spec),
    )
    if diag:
        raise BenchError(f"verification gate FAILED: {diag}")
    print(f"[bench] gate passed: op == ref kernel within tolerance {tol:g} "
          f"(measured max |diff| = {max_diff:.6g})")
    sidecar_max_diff = max_diff

    # 6. seed (first-write-wins unless --force) + sidecar
    out_dir = problem_dir / "output"
    existing = load_reference_time(out_dir)
    if existing is not None and not force:
        print(f"[bench] baseline already seeded at {existing:.2f} us "
              f"(measured now: {time_us:.2f} us) — keeping existing; use --force to overwrite")
        return 0
    if existing is not None:
        tmp = out_dir / (_REFERENCE_TIME_FILE + ".tmp")
        tmp.write_text(json.dumps({"reference_time_us": time_us}, indent=2),
                       encoding="utf-8")
        os.replace(tmp, out_dir / _REFERENCE_TIME_FILE)
    else:
        save_reference_time(out_dir, time_us)
    sidecar = _write_sidecar(problem_dir, {
        "op": op,
        "time_us": time_us,
        "warmup": warmup,
        "reps": reps,
        "tolerance": tol,
        "measured_max_diff": sidecar_max_diff,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": {"index": gpu_index, "name": _gpu_name(env, gpu_index)},
        "validated_buffer": vb.name,
        "inputs_sha256": _inputs_sha256(problem_dir, spec),
    })
    print(f"[bench] seeded {out_dir / _REFERENCE_TIME_FILE} = {time_us:.2f} us")
    print(f"[bench] wrote {sidecar}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="library_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("problem", type=Path, help="problem directory (problem.yaml + inputs.yaml)")
    p.add_argument("--op", help="op name: tools/library_bench/ops/<op>.cu")
    p.add_argument("--reps", type=int, default=100,
                   help="timed repetitions, min is taken (default 100)")
    p.add_argument("--warmup", type=int, default=20, help="untimed warmup calls (default 20)")
    p.add_argument("--force", action="store_true", help="overwrite an existing seeded baseline")
    p.add_argument("--tolerance", type=float, default=None,
                   help="override validation.tolerance for the verification gate (calibration runs)")
    p.add_argument("--device", type=int, default=None, help="override problem.yaml gpu.index")
    p.add_argument("--check", action="store_true",
                   help="do not bench: report whether the seeded baseline is stale")
    args = p.parse_args(argv)

    try:
        if args.check:
            return check(args.problem)
        if not args.op:
            raise BenchError("--op is required when benching (see tools/library_bench/ops/)")
        return bench(args.problem, args.op, args.reps, args.warmup,
                     args.force, args.tolerance, args.device)
    except BenchError as e:
        print(f"bench: error: {e}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("bench: error: subprocess timed out", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
