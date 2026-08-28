"""
Master Engine — orchestrates parallel branch workers.

Runs the initial analysis and strategy-generation passes, then populates
an ``asyncio.Queue`` with one entry per strategy and dispatches a small
pool of worker coroutines (``engine/worker.py``) to consume the queue.

Writes ``output/context.json`` once and creates a ``branch.json`` manifest
per branch.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import config as _cfg
from config import get_output_dir, global_tracker
from engine.worker import run_branch_loop
from nodes.analyze import analyze_node
from nodes.strategize import strategize_node

from utils.files import create_branch_dir
from utils.log import log
from utils.ref_timing import preflight_python_reference, time_python_reference
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

    ``inherited_max_iter`` and ``path_budget_total`` carry the parent's effective
    budget down; both are None for a root branch, the only place config still applies.
    """
    if parent_branch:
        parent_path = Path(parent_branch)
        branches_dir = parent_path / "branches"
    else:
        branches_dir = get_output_dir() / "branches"

    branch_name = (
        strategy.get("name", "default") if isinstance(strategy, dict) else strategy.name
    )

    # The directory name wins: create_branch_dir suffixes a name that is already taken,
    # and a manifest naming the branch something the directory is not called cannot be
    # resolved by anything that looks a branch up by name.
    branch_path = create_branch_dir(branches_dir, branch_name)
    strategy = {
        **(strategy if isinstance(strategy, dict) else strategy.model_dump()),
        "name": branch_path.name,
    }

    # Compute max_iter based on mode; depth always tracked so both constraints can apply
    if _cfg.PATH_BUDGET > 0:
        budget_total = path_budget_total or _cfg.PATH_BUDGET
        # At least one: a spawned branch that cannot run at all is worse than one that
        # runs once and stops.
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


def _write_final_results() -> None:
    """Write ``output/final_results.json`` at the end of a run.

    Here rather than cli.py because the API's run target calls only this engine, so
    UI-started runs never produced the file. Failure is logged: the run has already
    succeeded and its per-branch results are on disk.
    """
    from nodes.merge import build_final_summary, find_branch_results, get_best_branch_result
    from utils.files import save_json

    try:
        output_dir = get_output_dir()
        branch_results = find_branch_results(output_dir / "branches")
        best = get_best_branch_result(branch_results)
        if not best:
            log("No branch results to summarise; skipping final_results.json", "WARN")
            return
        save_json(
            output_dir / "final_results.json",
            build_final_summary(branch_results, best),
        )
        log(f"Wrote {output_dir / 'final_results.json'}", "SUCCESS")
    except Exception as e:
        log(f"Could not write final_results.json: {e}", "WARN")


def _spawn_sub_branches(branch_path: Path, sub_strategies: list) -> List[Path]:
    """Create the child branches a branch asked for, and return their paths.

    The budget comes from the parent's effective max_iter (branch_config.json, which
    the UI can raise), not from config — otherwise a grant dies on the branch it was
    made on. Empty list, with a reason logged, when depth or budget is exhausted.
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
        # A grant extends the path budget rather than coming out of the children's.
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


async def run_optimization_engine(
    problem_yaml: str, ref_kernel: str, resume_states: list = None
):
    """
    Main entry point for optimization execution.
    """
    if not await preflight_python_reference(problem_yaml):
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
        global_tracker.load(get_output_dir())
        log(f"Resuming {len(resume_states)} branches directly into the queue!", "INFO")
        for branch_path in resume_states:
            strategy_queue.put_nowait(Path(branch_path))
    else:
        main_state = await analyze_node(main_state)
        global_tracker.save(get_output_dir())

        if not main_state.analysis:
            log("Analysis failed. Aborting.", "ERROR")
            return

        main_state = await strategize_node(main_state)
        global_tracker.save(get_output_dir())
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

    await time_python_reference(get_problem_dir(), get_output_dir())

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

    _write_final_results()

    global_tracker.save(get_output_dir())

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
