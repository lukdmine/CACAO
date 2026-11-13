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
    write_pid_file,
    remove_pid_file,
    _last_exit_codes,
    _reap_unlocked,
    load_json,
)
from api.schemas import RunConfig
from utils.log import TeeWriter
from utils.problem import gpu_index as problem_gpu_index, load_problem_yaml

router = APIRouter()


def _get_gpu_index(problem_dir) -> int:
    """The GPU ordinal from this problem's problem.yaml, 0 when unset."""
    return problem_gpu_index(load_problem_yaml(problem_dir))


# ── Top-level process target (must be picklable) ─────────────────────────────


def _write_startup_error(problem_dir: Path, exc: Exception):
    """Record a failure that happened before the log redirect was installed.

    run.log is the only place the UI ever looks, so a config or archive failure that
    only reached the server's own stderr showed up as a run that simply stopped.
    """
    import traceback

    try:
        log_path = problem_dir / "output" / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] {exc}\n{traceback.format_exc()}\n")
    except OSError:
        pass


def _optimizer_target(
    problem_dir: Path,
    max_iter: int,
    max_depth: int,
    path_budget: int,
    timeout: Optional[int],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    resume: bool = False,
    finalize_only: bool = False,
    finalize: Optional[bool] = None,
    finalize_timeout: Optional[int] = None,
):
    """Process target for a run, or for resuming an interrupted one.

    ``finalize_only`` runs just the finalize pass over the existing output/: no archive,
    no run_meta.json rewrite (the recorded one still describes the run), run.log appended.

    The preflight runs *inside* the log redirect: a missing API key (ValueError from
    check_api_key) and a missing inputs.yaml (FileNotFoundError from load_problem_inputs)
    are the two startup failures users actually hit, and on the fresh-run path they used
    to be raised before the TeeWriter existed — the traceback went to the server's stderr
    and the UI showed the previous run's log.

    Ordering is load-bearing: archive_output_dir renames output/ wholesale, so the log
    cannot be opened before it, and the PID file — which also lives in output/ — has to
    be written again on the other side of it.
    """
    died: Optional[BaseException] = None
    try:
        import config as cfg
        from config import OptimizerConfig, init_from_config

        try:
            os.setsid()
        except OSError:
            pass

        write_pid_file(problem_dir, os.getpid())

        try:
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
                    finalize=finalize,
                    finalize_timeout=finalize_timeout,
                )
            )
            if not resume and not finalize_only:
                from utils.files import archive_output_dir

                archive_output_dir()
                write_pid_file(problem_dir, os.getpid())  # output/ was just renamed away
        except Exception as e:
            _write_startup_error(problem_dir, e)
            # Non-zero, or the registry records this as "finished normally" and
            # last_exit_code cannot tell a dead run from a completed one.
            raise SystemExit(1) from e

        log_path = problem_dir / "output" / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        log_mode = "a" if resume or finalize_only else "w"
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = TeeWriter(old_stdout, log_file)
            sys.stderr = TeeWriter(old_stderr, log_file)

            try:
                import asyncio as _asyncio
                import json as _json

                from engine.master import run_optimization_engine

                cfg.check_api_key()

                if finalize_only:
                    from engine.finalize import run_manual_finalize
                    from engine.master import _write_final_results
                    from utils.inputs import ensure_inputs_hpp

                    # Finalize's staging raises without inputs.hpp and would reuse a
                    # stale one after an inputs.yaml edit.
                    ensure_inputs_hpp(problem_dir)
                    _asyncio.run(run_manual_finalize())
                    _write_final_results()
                    return

                # Run metadata: the server reports the model/provider from it, and a
                # resume repeats all of it (_as_recorded).
                (problem_dir / "output" / "run_meta.json").write_text(
                    _json.dumps(
                        {
                            "model": cfg.get_model(),
                            "provider": cfg.get_provider(),
                            "max_iter": max_iter,
                            "max_depth": max_depth,
                            "path_budget": path_budget,
                            "timeout": timeout,
                        }
                    ),
                    encoding="utf-8",
                )

                problem_yaml, ref_kernel = load_problem_inputs(problem_dir)

                resume_states = None
                if resume:
                    from engine.resume import get_resume_states

                    resume_states = get_resume_states()
                    if not resume_states:
                        return  # get_resume_states has already said why

                _asyncio.run(
                    run_optimization_engine(problem_yaml, ref_kernel, resume_states)
                )
            except Exception as e:
                import traceback as _traceback

                print(f"\n[ERROR] {e}\n{_traceback.format_exc()}", flush=True)
                died = e
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        if died is not None:
            raise SystemExit(1) from died
    finally:
        remove_pid_file(problem_dir)


# ── Shared logic (callable from other modules) ───────────────────────────────


def _check_gpu_unlocked(gpu_index: int):
    """Raise 409 if the in-memory dict shows the GPU is in use. Caller must hold _running_lock."""
    for name, entry in list(_running_problems.items()):
        if not entry.process.is_alive():
            _reap_unlocked(name)
            continue
        if entry.gpu_index == gpu_index:
            raise HTTPException(
                status_code=409,
                detail=f"GPU {gpu_index} is in use by '{name}'. Stop it or use a different GPU.",
            )


def _spawn_optimizer(
    problem_dir: Path,
    config: RunConfig,
    label: str,
    resume: bool,
    finalize_only: bool = False,
) -> multiprocessing.Process:
    proc = multiprocessing.Process(
        target=_optimizer_target,
        args=(
            problem_dir,
            config.max_iter,
            config.max_depth,
            config.path_budget,
            config.timeout,
            config.model,
            config.provider,
            resume,
            finalize_only,
            config.finalize,
            config.finalize_timeout,
        ),
        daemon=True,
        name=label,
    )
    proc.start()
    return proc


def _reserve_and_spawn(
    name: str,
    problem_dir: Path,
    config: RunConfig,
    gpu_index: int,
    label: str,
    resume: bool,
    finalize_only: bool = False,
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
        # finalize_only is passed only when set, so a spawn without it keeps its old call.
        extra = {"finalize_only": True} if finalize_only else {}
        proc = _spawn_optimizer(problem_dir, config, label, resume, **extra)
        _running_problems[name] = RunEntry(process=proc, gpu_index=gpu_index)
        # The new run's outcome is the only one worth reporting.
        _last_exit_codes.pop(name, None)


def start_run(name: str, problem_dir: Path, config: RunConfig, gpu_index: int):
    """Start a fresh optimization run. Callers must NOT pre-check under _running_lock."""
    _reserve_and_spawn(
        name, problem_dir, config, gpu_index, f"optimizer-{name}", resume=False
    )


_RECORDED_BUDGET = ("max_iter", "max_depth", "path_budget", "timeout")


def _as_recorded(problem_dir: Path, config: RunConfig) -> RunConfig:
    """``config`` with every setting it leaves unset taken from the run being resumed.

    A resume handed no model auto-detects one, and with several API keys set that can
    be another provider: change-decision resumed a Claude run on cerit's default model.
    Model and provider are taken as a pair, and only when the request names neither, so
    an explicit provider never gets the other provider's model.
    """
    meta = load_json(problem_dir / "output" / "run_meta.json")
    given = config.model_fields_set
    recorded = {k: meta[k] for k in _RECORDED_BUDGET if k in meta and k not in given}
    if not given & {"model", "provider"}:
        recorded.update({k: meta[k] for k in ("model", "provider") if meta.get(k)})
    return config.model_copy(update=recorded)


def start_resume(name: str, problem_dir: Path, config: RunConfig, gpu_index: int):
    """Resume an interrupted optimization. Callers must NOT pre-check under _running_lock."""
    _reserve_and_spawn(
        name,
        problem_dir,
        _as_recorded(problem_dir, config),
        gpu_index,
        f"optimizer-resume-{name}",
        resume=True,
    )


def start_finalize(name: str, problem_dir: Path, config: RunConfig, gpu_index: int):
    """Run only the finalize pass over an existing output/. Callers must NOT pre-check."""
    _reserve_and_spawn(
        name,
        problem_dir,
        _as_recorded(problem_dir, config),
        gpu_index,
        f"optimizer-finalize-{name}",
        resume=True,
        finalize_only=True,
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
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n\n[STOPPED] Optimization stopped by user.\n")

    return {"status": "stopped", "problem": name}


@router.post("/api/problems/{name}/resume")
def resume_problem(name: str, config: RunConfig = RunConfig()):
    """Resume interrupted optimization."""
    problem_dir = get_problem_dir(name)
    gpu_index = _get_gpu_index(problem_dir)
    start_resume(name, problem_dir, config, gpu_index)
    return {"status": "resumed", "problem": name, "gpu_index": gpu_index}
