"""
Master Engine — orchestrates parallel branch workers.

Runs the initial analysis and strategy-generation passes, then populates
an ``asyncio.Queue`` with one entry per strategy and dispatches a small
pool of worker coroutines (``engine/worker.py``) to consume the queue.

Writes ``output/context.json`` once and creates a ``branch.json`` manifest
per branch.
"""

import asyncio
import re
from pathlib import Path
from typing import List, Optional

import config as _cfg
from config import get_output_dir, global_tracker
from engine.worker import run_branch_loop
from nodes.analyze import analyze_node
from nodes.strategize import strategize_node

from utils.files import create_branch_dir
from utils.log import log
from state import (
    MainState,
    BranchConfig,
    BranchManifest,
    Context,
    save_context,
    save_branch_config,
    load_branch_config,
    save_branch_manifest,
    load_branch_manifest,
    read_requeue,
)


async def _time_python_reference_once(problem_dir: Path, output_dir: Path) -> None:
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

    from utils.inputs import generate_dump_inputs_cpp, load_inputs_spec
    from utils.build import compile_dump_inputs
    from utils.results import load_reference_time, save_reference_time
    from utils.cuda_env import get_subprocess_env
    from utils.gpu_lock import acquire_gpu_lock

    if load_reference_time(output_dir) is not None:
        return  # resume: already timed (first-write-wins persists it)

    problem_dir = Path(problem_dir).resolve()
    output_dir = Path(output_dir).resolve()

    try:
        cfg = _yaml.safe_load((problem_dir / "problem.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return
    ref = cfg.get("reference") or {}
    if str(ref.get("type", "cuda")).lower() != "python":
        return
    ref_file = ref.get("file", "ref.py")
    ref_function = ref.get("function", "")

    timing_dir = output_dir / "_ref_timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    dump_cpp = timing_dir / "dump_inputs.cpp"
    dump_bin = timing_dir / "dump_inputs"

    try:
        spec = load_inputs_spec(problem_dir / "inputs.yaml")
        dump_cpp.write_text(generate_dump_inputs_cpp(spec), encoding="utf-8")
    except Exception as e:
        log(f"reference timing: could not generate dump_inputs.cpp: {e}", "WARN")
        return

    build = compile_dump_inputs(dump_cpp, dump_bin, problem_dir)
    if not build.ok:
        log(
            f"reference timing: dump_inputs compile failed — "
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
            str(problem_dir / "inputs.yaml"),
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

    save_reference_time(output_dir, us)
    log(f"Reference time (precise, GPU-only): {us:.2f} µs", "SUCCESS")


def _init_branch(
    strategy: dict,
    parent_branch: str = None,
    current_depth: int = _cfg.MAX_BRANCH_DEPTH,
    path_iters_consumed: int = 0,
    inherited_max_iter: Optional[int] = None,
    path_budget_total: Optional[int] = None,
) -> Path:
    """
    Creates the directory for a new branch and initializes its branch.json.
    Returns the path to the newly created branch directory.

    ``inherited_max_iter`` and ``path_budget_total`` carry the parent's *effective*
    budget down. A user who raises a branch's max_iter in the UI is saying that
    subtree needs more room, so its children have to start from the raised number —
    reading config.MAX_ITERATIONS / config.PATH_BUDGET here meant the grant stopped
    at the branch it was made on. Both are None for a root branch, which is the only
    place config is still the source.
    """
    if parent_branch:
        parent_path = Path(parent_branch)
        branches_dir = parent_path / "branches"
    else:
        branches_dir = get_output_dir() / "branches"

    branch_name = (
        strategy.get("name", "default") if isinstance(strategy, dict) else strategy.name
    )

    # Create the physical directory
    branch_path = create_branch_dir(
        parent=branches_dir,
        name=branch_name,
        strategy=strategy,
        depth=current_depth,
    )

    # Compute max_iter based on mode; depth always tracked so both constraints can apply
    if _cfg.PATH_BUDGET > 0:
        budget_total = path_budget_total or _cfg.PATH_BUDGET
        # At least one iteration: a branch the master decided to spawn but that cannot
        # run at all is worse than one that runs once and stops on its own budget.
        max_iter = max(budget_total - path_iters_consumed, 1)
    else:
        budget_total = 0
        max_iter = inherited_max_iter or _cfg.MAX_ITERATIONS
    depth = current_depth

    # Build the initial BranchManifest using Pydantic model
    manifest = BranchManifest(
        strategy=strategy,
        branch_depth=depth,
        path_iters_consumed=path_iters_consumed,
        path_budget_total=budget_total,
        current_iter=1,
        status="initialized",
    )

    # Save to disk so the worker can pick it up. The budget goes in its own
    # file: from here on the frontend owns it and the worker only reads it.
    save_branch_config(branch_path, BranchConfig(max_iter=max_iter))
    save_branch_manifest(branch_path, manifest)

    log(f"Initialized branch at {branch_path} (max_iter={max_iter})", "SUCCESS")
    return branch_path


def _spawn_sub_branches(branch_path: Path, sub_strategies: list) -> List[Path]:
    """Create the child branches a branch asked for, and return their paths.

    Sizing them is the whole job. Depth is a simple decrement, but the iteration
    budget has to come from the parent's *effective* max_iter — the value in
    branch_config.json, which the UI can raise and which grant_one_more_iteration
    raises when an exhausted branch is revived — not from config. Reading config here
    meant a grant died on the branch it was made on: a branch given 12 iterations
    spawned children with MAX_ITERATIONS.

    Returns an empty list, with a reason logged, when depth or budget is exhausted.
    """
    manifest = load_branch_manifest(branch_path)
    new_depth = manifest.branch_depth - 1
    consumed = manifest.path_iters_consumed + manifest.current_iter
    effective_max_iter = load_branch_config(branch_path).max_iter

    if new_depth <= 0:
        log(
            "Branch requested sub-strategies but MAX_DEPTH reached. Discarding.",
            "WARN",
        )
        return []

    if _cfg.PATH_BUDGET > 0:
        budget_total = manifest.path_budget_total or _cfg.PATH_BUDGET
        # Whatever the parent was granted beyond its allocated share extends the path
        # budget rather than being taken out of its children's. The point of raising a
        # branch's budget is to buy more work on that path, not to move work off the
        # branches below it.
        allocated = max(budget_total - manifest.path_iters_consumed, 1)
        granted = max(effective_max_iter - allocated, 0)
        child_budget_total = budget_total + granted
        remaining = child_budget_total - consumed
        if remaining <= 0:
            log(
                "Branch requested sub-strategies but path budget exhausted. Discarding.",
                "WARN",
            )
            return []
        budget_note = f", path budget: {remaining} iters remaining"
    else:
        child_budget_total = None
        budget_note = f", max_iter: {effective_max_iter}"

    log(
        f"Master received {len(sub_strategies)} sub-strategies "
        f"(depth left: {new_depth}{budget_note}).",
        "INFO",
    )

    return [
        _init_branch(
            strategy=sub_strat,
            parent_branch=str(branch_path),
            current_depth=new_depth,
            path_iters_consumed=consumed,
            inherited_max_iter=effective_max_iter,
            path_budget_total=child_budget_total,
        )
        for sub_strat in sub_strategies
    ]


# Module-level loader executed by ``python3 -c`` in _preflight_python_reference.
# Mirrors utils.python_ref_runner.load_ref_module so module-level import errors
# (e.g. missing torch) surface exactly as they will during tuning.
_REF_LOADER_SNIPPET = (
    "import importlib.util,sys;"
    "spec=importlib.util.spec_from_file_location('ref',sys.argv[1]);"
    "mod=importlib.util.module_from_spec(spec);"
    "spec.loader.exec_module(mod)"
)


async def _preflight_python_reference(problem_yaml: str) -> bool:
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


async def run_optimization_engine(
    problem_yaml: str, ref_kernel: str, resume_states: list = None
):
    """
    Main entry point for optimization execution.
    """
    if not await _preflight_python_reference(problem_yaml):
        log("Aborting run: fix the reference issue above and retry.", "ERROR")
        return

    print("\n" + "=" * 60)
    print("  PHASE 1: Analysis & Strategy")
    print("=" * 60)

    mode = "path-budget" if _cfg.PATH_BUDGET > 0 else "depth"
    log(
        f"Effective config: mode={mode} "
        f"PATH_BUDGET={_cfg.PATH_BUDGET} MAX_ITERATIONS={_cfg.MAX_ITERATIONS} "
        f"MAX_BRANCH_DEPTH={_cfg.MAX_BRANCH_DEPTH}"
    )

    # Auto-detect GPU specs (cached in context.json for LLM prompts)
    import yaml
    from utils.gpu_info import get_gpu_details

    gpu_info = None
    try:
        config = yaml.safe_load(problem_yaml)
        gpu_index = config.get("gpu", {}).get("index", 0)
        gpu_info = get_gpu_details(gpu_index)
        if gpu_info:
            config["gpu"].update(gpu_info)
            problem_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
            log(f"Detected GPU {gpu_index}: {gpu_info.get('model')}", "INFO")
    except Exception as e:
        log(f"Failed to detect GPU info: {e}", "WARN")

    main_state = MainState(
        problem_yaml=problem_yaml,
        ref_kernel=ref_kernel,
        analysis="",
        strategies=[],
    )

    strategy_queue: asyncio.Queue = asyncio.Queue()

    if resume_states:
        global_tracker.load()
        log(f"Resuming {len(resume_states)} branches directly into the queue!", "INFO")
        for branch_path in resume_states:
            strategy_queue.put_nowait(Path(branch_path))
    else:
        main_state = await analyze_node(main_state)
        global_tracker.save()

        if not main_state.analysis:
            log("Analysis failed. Aborting.", "ERROR")
            return

        main_state = await strategize_node(main_state)
        global_tracker.save()
        strategies = main_state.strategies

        if not strategies:
            log("No strategies generated. Aborting.", "ERROR")
            return

        save_context(
            get_output_dir(),
            Context(
                analysis=main_state.analysis,
                gpu_info=gpu_info,
            ),
        )

        for strat in strategies:
            branch_path = _init_branch(strat, current_depth=_cfg.MAX_BRANCH_DEPTH)
            strategy_queue.put_nowait(branch_path)

    print("\n" + "=" * 60)
    print("  PHASE 2: Parallel Branch Execution")
    print("=" * 60)

    # Precise GPU-only reference timing for python refs: once, up-front, on the
    # real inputs. Non-python refs and any failure fall back to KTT's coarse
    # post-tune time. No-op on resume (reference_time.json already exists).
    from config import get_problem_dir

    await _time_python_reference_once(get_problem_dir(), get_output_dir())

    # No lock guards `busy` or the exit check: every read/write below sits
    # between awaits, which is atomic under asyncio's cooperative scheduler.
    busy = 0
    shutdown_event = asyncio.Event()

    async def worker():
        nonlocal busy
        while not shutdown_event.is_set():
            branch_path = None
            try:
                branch_path = strategy_queue.get_nowait()
                busy += 1
            except asyncio.QueueEmpty:
                pass

            if branch_path is None:
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                continue

            new_sub_strategies = None
            try:
                new_sub_strategies = await run_branch_loop(branch_path)
            finally:
                if new_sub_strategies:
                    for child_path in _spawn_sub_branches(
                        branch_path, new_sub_strategies
                    ):
                        strategy_queue.put_nowait(child_path)

                busy -= 1

    async def monitor():
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            leftovers = read_requeue(get_output_dir())
            for bp in leftovers:
                strategy_queue.put_nowait(bp)
                log(f"Re-queued branch from disk: {bp.name}", "INFO")
            if busy == 0 and strategy_queue.empty():
                shutdown_event.set()

    max_workers = 4
    worker_tasks = [asyncio.create_task(worker()) for _ in range(max_workers)]
    monitor_task = asyncio.create_task(monitor())

    await asyncio.gather(monitor_task, *worker_tasks)

    log("All branches and sub-branches have completed!", "SUCCESS")

    global_tracker.save()

    from config import get_tracker_stats

    stats = get_tracker_stats()
    log("\n" + "=" * 50)
    log(" LLM USAGE STATISTICS")
    log("=" * 50)
    log(f" API Calls: {stats['api_calls']}")
    log(f" Prompt Tokens: {stats['prompt_tokens']}")
    log(f" Completion Tokens: {stats['completion_tokens']}")
    log(f" Total Tokens: {stats['total_tokens']}")
    log("=" * 50 + "\n")
