"""Running a ``reference.type: python`` ref.py from the engine, out of process.

Both halves are host-side plumbing — codegen, compile, subprocess, regex — not branch
orchestration, which is why they are not in engine/master.py. They stay together
because they are the two places the engine shells out to a python reference, and the
loader snippet below has to mirror utils.python_ref_runner.load_ref_module exactly.
"""

import asyncio
import re
from pathlib import Path

from utils.log import log


async def _time_reference_for_case(problem_dir, output_dir, cfg, ref, case, case_key) -> None:
    """Time a python reference's GPU op once, up-front, on the real inputs.

    For ``reference.type: python`` only: compiles+runs a standalone dump tool
    (which includes inputs.hpp, so the ``cacao_ref::h_*`` statics build the real,
    deterministic input data) to materialize ``cacao_in_<name>.bin``, then runs
    ``utils.torch_ref_timer`` — it loads ref.py, calls ``prepare_input`` (H2D,
    not timed), and times the reference function via ``torch.cuda.Event`` (min
    over a few single-call runs). The result (µs) is persisted to
    reference_time.json (first-write-wins), so every iteration's speedup uses
    this precise GPU-only value instead of KTT's coarse wall-clock time.

    Silent no-op for non-python references and on any failure (no torch/CUDA, no
    ``prepare_input``, compile error): KTT's coarse post-tune time remains the
    fallback, written by run_node's existing parse path. Skipped on resume when
    reference_time.json already exists.
    """
    import sys
    import yaml as _yaml

    from utils.inputs import (
        generate_dump_inputs_cpp,
        generate_inputs_hpp,
        load_inputs_spec,
    )
    from utils.build import compile_dump_inputs
    from utils.results import reference_time_recorded, save_reference_time
    from utils.cuda_env import get_subprocess_env
    from utils.gpu_lock import acquire_gpu_lock

    if reference_time_recorded(output_dir, case_key):
        # A measured value, a committed seed, or an explicit null meaning "this shape has
        # no incumbent measurement". All three mean: do not time the reference for it.
        return

    problem_dir = Path(problem_dir).resolve()
    output_dir = Path(output_dir).resolve()
    ref_file = ref.get("file", "ref.py")
    ref_function = ref.get("function", "")
    label = "" if case_key is None else f" [{case.name}]"

    # Per case, because the dump tool #includes this case's header and the .bin files it
    # writes therefore hold this case's shapes.
    timing_dir = output_dir / "_ref_timing" / case.name
    timing_dir.mkdir(parents=True, exist_ok=True)
    dump_cpp = timing_dir / "dump_inputs.cpp"
    dump_bin = timing_dir / "dump_inputs"

    try:
        spec = load_inputs_spec(problem_dir / "inputs.yaml").for_case(case)
        (timing_dir / "inputs.hpp").write_text(
            generate_inputs_hpp(spec, ref, problem_dir), encoding="utf-8"
        )
        (timing_dir / "inputs.yaml").write_text(
            _yaml.safe_dump(
                spec.model_dump(by_alias=True, exclude_none=True), sort_keys=False
            ),
            encoding="utf-8",
        )
        dump_cpp.write_text(generate_dump_inputs_cpp(spec), encoding="utf-8")
    except Exception as e:
        log(f"reference timing{label}: could not generate dump_inputs.cpp: {e}", "WARN")
        return

    # The include dir is the timing dir, not the problem dir: this case's header lives
    # beside the generated .cpp, and the problem dir only ever holds the primary's.
    build = compile_dump_inputs(dump_cpp, dump_bin, timing_dir)
    if not build.ok:
        log(
            f"reference timing{label}: dump_inputs compile failed — "
            f"falling back to KTT coarse time: {build.stderr}",
            "WARN",
        )
        return

    async with acquire_gpu_lock():
        # Dump the real inputs (cwd = timing_dir so the .bin land there).
        try:
            dump_proc = await asyncio.create_subprocess_exec(
                str(dump_bin),
                cwd=str(timing_dir),
                env=get_subprocess_env(),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr_b = await asyncio.wait_for(
                    dump_proc.communicate(), timeout=300
                )
            except asyncio.TimeoutError:
                log("reference timing: dump_inputs timed out", "WARN")
                dump_proc.kill()
                await dump_proc.wait()
                return
            if dump_proc.returncode != 0:
                log(
                    f"reference timing: dump_inputs run failed — "
                    f"falling back to KTT coarse time: {stderr_b.decode(errors='replace')}",
                    "WARN",
                )
                return
        except Exception as e:
            log(f"reference timing: dump_inputs run error: {e}", "WARN")
            return

        timer_cmd = [
            sys.executable,
            "-m",
            "utils.torch_ref_timer",
            "--inputs",
            str(timing_dir / "inputs.yaml"),
            "--ref",
            str(problem_dir / ref_file),
            "--function",
            ref_function,
            "--inputs-dir",
            str(timing_dir),
        ]
        try:
            timer_proc = await asyncio.create_subprocess_exec(
                *timer_cmd,
                cwd=str(problem_dir),
                env=get_subprocess_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    timer_proc.communicate(), timeout=300
                )
            except asyncio.TimeoutError:
                log("reference timing: torch_ref_timer timed out", "WARN")
                timer_proc.kill()
                await timer_proc.wait()
                return
        except Exception as e:
            log(f"reference timing: torch_ref_timer error: {e}", "WARN")
            return

        if timer_proc.returncode != 0:
            log(
                f"reference timing skipped (torch_ref_timer exit "
                f"{timer_proc.returncode}) — falling back to KTT coarse time: "
                f"{stderr_b.decode(errors='replace').strip()}",
                "WARN",
            )
            return

    # Parse outside the GPU lock: matching + persistence don't touch the GPU.
    out = stdout_b.decode(errors="replace")
    m = re.search(r"^CACAO_REF_TIME_US=([0-9.]+)$", out, re.MULTILINE)
    if not m:
        log(f"reference timing: unparseable timer output {out[:200]!r}", "WARN")
        return
    us = float(m.group(1))

    save_reference_time(output_dir, us, case_key)
    log(f"Reference time{label} (precise, GPU-only): {us:.2f} µs", "SUCCESS")


async def time_python_reference(problem_dir: Path, output_dir: Path) -> None:
    """Time a python reference's GPU op once per input case, up-front, on the real inputs.

    Silent no-op for non-python references. A case that cannot be timed is skipped rather
    than failing the run: KTT's coarse post-tune number remains the fallback, and a case
    with no baseline at all reports no speedup, which utils.results.aggregate_case_summaries
    turns into a null geomean rather than a flattering average over a subset.
    """
    import yaml as _y

    from utils.cases import case_list

    problem_dir = Path(problem_dir).resolve()
    output_dir = Path(output_dir).resolve()
    try:
        cfg = _y.safe_load((problem_dir / "problem.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return
    ref = cfg.get("reference") or {}
    if str(ref.get("type", "cuda")).lower() != "python":
        return

    cases = case_list(cfg)
    primary = cases[0].name
    for case in cases:
        await _time_reference_for_case(
            problem_dir,
            output_dir,
            cfg,
            ref,
            case,
            None if case.name == primary else case.name,
        )


# Module-level loader executed by ``python3 -c`` in preflight_python_reference.
# Mirrors utils.python_ref_runner.load_ref_module so module-level import errors
# (e.g. missing torch) surface exactly as they will during tuning.
_REF_LOADER_SNIPPET = (
    "import importlib.util,sys;"
    "spec=importlib.util.spec_from_file_location('ref',sys.argv[1]);"
    "mod=importlib.util.module_from_spec(spec);"
    "spec.loader.exec_module(mod)"
)


async def preflight_python_reference(problem_yaml: str) -> bool:
    """Fail fast when a ``reference.type: python`` ref.py cannot be imported.

    The C++ driver computes the reference by invoking
    ``python3 -m utils.python_ref_runner`` once per evaluated config, so a
    missing ref.py dependency (typically the optional ``torch``) would
    otherwise surface only as every tuning config failing validation — after
    the LLM iterations were already spent. This loads ref.py module-level in a
    ``python3`` subprocess (the same interpreter the driver lambda will use)
    before any LLM work. Returns False (abort) on failure, True otherwise.
    Silent no-op for non-python references.
    """
    import yaml as _yaml

    from utils.cuda_env import get_subprocess_env

    try:
        cfg = _yaml.safe_load(problem_yaml) or {}
    except Exception:
        return True
    ref = cfg.get("reference") or {}
    if str(ref.get("type", "cuda")).lower() != "python":
        return True

    problem_dir = _cfg.get_problem_dir()
    ref_path = problem_dir / ref.get("file", "ref.py")
    if not ref_path.exists():
        log(f"Python reference not found at {ref_path}", "ERROR")
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            _REF_LOADER_SNIPPET,
            str(ref_path),
            cwd=str(problem_dir),
            env=get_subprocess_env(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log(f"Python reference import check timed out on {ref_path}", "ERROR")
            return False
    except FileNotFoundError:
        log(
            "python3 not found on PATH — a python reference cannot run "
            "(the driver invokes python3 -m utils.python_ref_runner)",
            "ERROR",
        )
        return False

    if proc.returncode != 0:
        log(
            f"Python reference {ref_path} failed to import:\n"
            f"{stderr_b.decode(errors='replace').strip()}",
            "ERROR",
        )
        return False
    log(f"Python reference {ref_path.name} imports cleanly", "SUCCESS")
    return True
