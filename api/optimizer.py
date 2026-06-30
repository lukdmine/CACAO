"""Run / Stop / Resume endpoints."""

import multiprocessing
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.helpers import (
    get_problem_dir,
    _running_problems,
    _running_lock,
    RunEntry,
    load_problem_inputs,
    _is_running_unlocked,
    _check_gpu_orphans,
    is_problem_running,
    terminate_run,
    load_yaml,
    write_pid_file,
    remove_pid_file,
)
from api.schemas import RunConfig
from utils.log import TeeWriter

router = APIRouter()


def _get_gpu_index(problem_dir) -> int:
    """Read the GPU index from problem.yaml."""
    py = problem_dir / "problem.yaml"
    if py.exists():
        config = load_yaml(py)
        return config.get("gpu", {}).get("index", 0)
    return 0


# ── Top-level process targets (must be picklable) ────────────────────────────


def _run_target(
    problem_dir: Path,
    max_iter: int,
    max_depth: int,
    path_budget: int,
    timeout: Optional[int],
    model: str = None,
    provider: str = None,
):
    """Process target for a fresh optimization run."""
    try:
        import config as cfg
        from config import OptimizerConfig, init_from_config
        from engine.master import run_optimization_engine
        from utils.files import clean_output_dir

        try:
            os.setsid()
        except OSError:
            pass

        init_from_config(
            OptimizerConfig(
                output_dir=problem_dir / "output",
                problem_dir=problem_dir,
                max_iterations=max_iter,
                max_branch_depth=max_depth,
                path_budget=path_budget,
                tuner_timeout=timeout,
                model=model,
                provider=provider,
            )
        )
        cfg.check_api_key()

        problem_yaml, ref_kernel = load_problem_inputs(problem_dir)
        clean_output_dir()

        # Write run metadata so the server can report the correct model/provider
        import json as _json

        meta_dir = problem_dir / "output"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "run_meta.json").write_text(
            _json.dumps(
                {
                    "model": cfg._current_model,
                    "provider": cfg.get_provider(),
                }
            )
        )

        # Re-write PID file after clean_output_dir
        write_pid_file(problem_dir, os.getpid())

        log_path = problem_dir / "output" / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("w") as log_file:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = TeeWriter(old_stdout, log_file)
            sys.stderr = TeeWriter(old_stderr, log_file)

            try:
                import asyncio as _asyncio

                _asyncio.run(run_optimization_engine(problem_yaml, ref_kernel))
            except Exception as e:
                print(f"\n[ERROR] {e}", flush=True)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    finally:
        remove_pid_file(problem_dir)


def _resume_target(
    problem_dir: Path,
    max_iter: int,
    max_depth: int,
    path_budget: int,
    timeout: Optional[int],
    model: str = None,
    provider: str = None,
):
    """Process target for resuming an interrupted optimization."""
    try:
        log_path = problem_dir / "output" / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.setsid()
        except OSError:
            pass

        write_pid_file(problem_dir, os.getpid())

        with log_path.open("a") as log_file:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = TeeWriter(old_stdout, log_file)
            sys.stderr = TeeWriter(old_stderr, log_file)

            try:
                import config as cfg
                from config import OptimizerConfig, init_from_config
                from engine.master import run_optimization_engine
                from utils.resume import get_resume_states

                init_from_config(
                    OptimizerConfig(
                        output_dir=problem_dir / "output",
                        problem_dir=problem_dir,
                        max_iterations=max_iter,
                        max_branch_depth=max_depth,
                        path_budget=path_budget,
                        tuner_timeout=timeout,
                        model=model,
                        provider=provider,
                    )
                )
                cfg.check_api_key()

                # Write run metadata so the server can report the correct model/provider
                import json as _json

                (problem_dir / "output" / "run_meta.json").write_text(
                    _json.dumps(
                        {
                            "model": cfg._current_model,
                            "provider": cfg.get_provider(),
                        }
                    )
                )

                problem_yaml, ref_kernel = load_problem_inputs(problem_dir)

                resume_states = get_resume_states()
                if resume_states:
                    import asyncio as _asyncio

                    _asyncio.run(
                        run_optimization_engine(problem_yaml, ref_kernel, resume_states)
                    )
            except Exception as e:
                print(f"\n[ERROR] {e}", flush=True)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    finally:
        remove_pid_file(problem_dir)


# ── Shared logic (callable from other modules) ───────────────────────────────


def _check_gpu_unlocked(gpu_index: int):
    """Raise 409 if the in-memory dict shows the GPU is in use. Caller must hold _running_lock."""
    for name, entry in list(_running_problems.items()):
        if not entry.process.is_alive():
            del _running_problems[name]
            continue
        if entry.gpu_index == gpu_index:
            raise HTTPException(
                status_code=409,
                detail=f"GPU {gpu_index} is in use by '{name}'. Stop it or use a different GPU.",
            )


def _spawn_optimizer(
    target, problem_dir: Path, config: RunConfig, label: str
) -> multiprocessing.Process:
    proc = multiprocessing.Process(
        target=target,
        args=(
            problem_dir,
            config.max_iter,
            config.max_depth,
            config.path_budget,
            config.timeout,
            config.model,
            config.provider,
        ),
        daemon=True,
        name=label,
    )
    proc.start()
    return proc


def _reserve_and_spawn(
    target, name: str, problem_dir: Path, config: RunConfig, gpu_index: int, label: str
):
    """Atomically verify name/GPU are free, spawn the process, and record the entry.

    Holds _running_lock through the check+spawn+insert so concurrent requests
    can't both pass the preflight and race into a double-spawn.
    """
    _check_gpu_orphans(gpu_index, skip_name=name)
    with _running_lock:
        if _is_running_unlocked(name):
            raise HTTPException(
                status_code=409, detail=f"Optimization already running for '{name}'"
            )
        _check_gpu_unlocked(gpu_index)
        proc = _spawn_optimizer(target, problem_dir, config, label)
        _running_problems[name] = RunEntry(process=proc, gpu_index=gpu_index)


def start_run(name: str, problem_dir: Path, config: RunConfig, gpu_index: int):
    """Start a fresh optimization run. Callers must NOT pre-check under _running_lock."""
    _reserve_and_spawn(
        _run_target, name, problem_dir, config, gpu_index, f"optimizer-{name}"
    )


def start_resume(name: str, problem_dir: Path, config: RunConfig, gpu_index: int):
    """Resume an interrupted optimization. Callers must NOT pre-check under _running_lock."""
    _reserve_and_spawn(
        _resume_target, name, problem_dir, config, gpu_index, f"optimizer-resume-{name}"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/api/problems/{name}/run")
def run_problem(name: str, config: RunConfig = RunConfig()):
    """Start optimization for a problem."""
    problem_dir = get_problem_dir(name)
    gpu_index = _get_gpu_index(problem_dir)
    start_run(name, problem_dir, config, gpu_index)
    return {"status": "started", "problem": name, "gpu_index": gpu_index}


@router.post("/api/problems/{name}/stop")
def stop_problem(name: str):
    """Stop a running optimization."""
    if not is_problem_running(name):
        raise HTTPException(status_code=404, detail="No running optimization found")

    problem_dir = get_problem_dir(name)
    terminate_run(name, problem_dir)

    log_path = problem_dir / "output" / "run.log"
    if log_path.exists():
        with log_path.open("a") as f:
            f.write("\n\n[STOPPED] Optimization stopped by user.\n")

    return {"status": "stopped", "problem": name}


@router.post("/api/problems/{name}/resume")
def resume_problem(name: str, config: RunConfig = RunConfig()):
    """Resume interrupted optimization."""
    problem_dir = get_problem_dir(name)
    gpu_index = _get_gpu_index(problem_dir)
    start_resume(name, problem_dir, config, gpu_index)
    return {"status": "resumed", "problem": name, "gpu_index": gpu_index}
