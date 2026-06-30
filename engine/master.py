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

import config as _cfg
from config import get_output_dir, global_tracker
from engine.worker import run_branch_loop
from nodes.analyze import analyze_node
from nodes.strategize import strategize_node

from utils.files import create_branch_dir
from utils.log import log
from state import (
    MainState,
    BranchManifest,
    Context,
    save_context,
    save_branch_manifest,
    load_branch_manifest,
    read_requeue,
)


def _init_branch(
    strategy: dict,
    parent_branch: str = None,
    current_depth: int = _cfg.MAX_BRANCH_DEPTH,
    path_iters_consumed: int = 0,
) -> Path:
    """
    Creates the directory for a new branch and initializes its branch.json.
    Returns the path to the newly created branch directory.
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
        max_iter = _cfg.PATH_BUDGET - path_iters_consumed
    else:
        max_iter = _cfg.MAX_ITERATIONS
    depth = current_depth

    # Build the initial BranchManifest using Pydantic model
    manifest = BranchManifest(
        strategy=strategy,
        branch_depth=depth,
        path_iters_consumed=path_iters_consumed,
        current_iter=1,
        max_iter=max_iter,
        status="initialized",
    )

    # Save to disk so the worker can pick it up
    save_branch_manifest(branch_path, manifest)

    log(f"Initialized branch at {branch_path} (max_iter={max_iter})", "SUCCESS")
    return branch_path


async def run_optimization_engine(
    problem_yaml: str, ref_kernel: str, resume_states: list = None
):
    """
    Main entry point for optimization execution.
    """
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
                    manifest = load_branch_manifest(branch_path)
                    new_depth = manifest.branch_depth - 1
                    consumed = manifest.path_iters_consumed + manifest.current_iter

                    depth_ok = new_depth > 0
                    if _cfg.PATH_BUDGET > 0:
                        remaining = _cfg.PATH_BUDGET - consumed
                        budget_ok = remaining > 0
                    else:
                        remaining = None
                        budget_ok = True

                    if depth_ok and budget_ok:
                        budget_note = (
                            f", path budget: {remaining} iters remaining"
                            if remaining is not None
                            else ""
                        )
                        log(
                            f"Master received {len(new_sub_strategies)} sub-strategies (depth left: {new_depth}{budget_note}).",
                            "INFO",
                        )
                        for sub_strat in new_sub_strategies:
                            child_path = _init_branch(
                                strategy=sub_strat,
                                parent_branch=str(branch_path),
                                current_depth=new_depth,
                                path_iters_consumed=consumed,
                            )
                            strategy_queue.put_nowait(child_path)
                    elif not depth_ok:
                        log(
                            "Branch requested sub-strategies but MAX_DEPTH reached. Discarding.",
                            "WARN",
                        )
                    else:
                        log(
                            "Branch requested sub-strategies but path budget exhausted. Discarding.",
                            "WARN",
                        )

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
