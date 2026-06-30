"""Worker Engine — executes a single strategy branch in isolation."""

import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

from nodes.plan import plan_node
from nodes.implement import implement_node
from nodes.configure import configure_node
from nodes.run import run_node
from nodes.profile import profile_node
from nodes.propose import propose_node
from nodes.decide import decide_node

from config import get_problem_dir
from utils.log import log
from state import (
    load_branch_manifest,
    save_branch_manifest,
    load_iter_state_if_exists,
    save_iter_state,
    create_initial_iter_state,
    load_context_for_branch,
    BranchManifest,
    IterState,
    Context,
    WorkingState,
    read_control_signal,
    revert_branch_on_disk,
    continue_branch_on_disk,
    validate_transition,
)


# -------------------------------------------------------------------------
# Composed view helpers
# -------------------------------------------------------------------------


def _read_fresh_problem_yaml(gpu_info: dict | None) -> str:
    """Re-read problem.yaml from source and apply cached GPU enrichment."""
    problem_dir = get_problem_dir()
    with open(problem_dir / "problem.yaml") as f:
        raw = f.read()
    if gpu_info:
        config = yaml.safe_load(raw)
        config.setdefault("gpu", {}).update(gpu_info)
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    return raw


def _read_fresh_ref_kernel() -> str:
    """Re-read the reference kernel source from the problem directory."""
    problem_dir = get_problem_dir()
    with open(problem_dir / "problem.yaml") as f:
        config = yaml.safe_load(f)
    ref_file = config.get("reference", {}).get("file", "")
    if ref_file:
        return (problem_dir / ref_file).read_text()
    return ""


def _compose_working_state(
    context: Context, manifest: BranchManifest, iter_state: IterState, branch_path: Path
) -> WorkingState:
    """Assemble a unified model view for nodes from the three Pydantic models.

    Source files (problem.yaml, reference kernel) are always read fresh
    from disk so that user edits propagate immediately.
    """
    working_dict = {}
    working_dict.update(context.model_dump())
    working_dict.update(manifest.model_dump())
    working_dict.update(iter_state.model_dump())
    working_dict["branch_path"] = str(branch_path)
    working_dict["problem_yaml"] = _read_fresh_problem_yaml(context.gpu_info)
    working_dict["ref_kernel"] = _read_fresh_ref_kernel()

    return WorkingState.model_validate(working_dict)


def _decompose_to_iter_state(working: WorkingState) -> IterState:
    """Extract iteration fields from working state back into the IterState Pydantic model."""
    # This automatically ignores fields not in IterState and handles validation
    return IterState.model_validate(working.model_dump())


def _update_manifest_from_working(manifest: BranchManifest, working: WorkingState):
    """Update manifest with fields that nodes may have mutated."""
    manifest.best_time_us = working.best_time_us
    manifest.speedup = working.speedup
    manifest.status = working.status
    manifest.current_iter = working.current_iter
    if working.sub_strategies:
        manifest.sub_strategies_cache = working.sub_strategies


# -------------------------------------------------------------------------
# Signal handling
# -------------------------------------------------------------------------


def _handle_signal(
    signal: dict, manifest: BranchManifest, iter_state: IterState, branch_path: Path
):
    """Process a control signal, mutating manifest and iter_state in place."""
    action = signal["action"]

    if signal.get("content"):
        iter_state.user_messages.append(
            {
                "content": signal["content"],
                "timestamp": datetime.now().isoformat(),
            }
        )

    try:
        if action == "stop":
            manifest.pre_stop_status = manifest.status
            manifest.status = "stopped"
            save_branch_manifest(branch_path, manifest)
            save_iter_state(branch_path, iter_state.iter_num, iter_state)

        elif action == "revert":
            target_iter = signal["target_iter"]
            if target_iter < manifest.current_iter:
                reverted_dict = revert_branch_on_disk(
                    branch_path, target_iter, signal.get("content")
                )
            else:
                reverted_dict = continue_branch_on_disk(
                    branch_path, signal.get("content")
                )
            new_manifest = BranchManifest.model_validate(reverted_dict)
            for field in new_manifest.model_fields.keys():
                setattr(manifest, field, getattr(new_manifest, field))

        elif action in ("resume", "message"):
            if manifest.pre_stop_status and manifest.status == "stopped":
                manifest.status = manifest.pre_stop_status
                manifest.pre_stop_status = None
            save_branch_manifest(branch_path, manifest)
            save_iter_state(branch_path, iter_state.iter_num, iter_state)

    except Exception as e:
        log(f"Failed to apply signal '{action}' on {branch_path.name}: {e}", "ERROR")
        manifest.status = "failed"
        try:
            save_branch_manifest(branch_path, manifest)
        except Exception:
            pass


async def _wait_for_resume(
    manifest: BranchManifest, iter_state: IterState, branch_path: Path
):
    """Spin-wait for a resume/message/revert signal."""
    strategy_name = manifest.strategy.name
    log(f"Branch {strategy_name} stopped. Waiting for resume...", "INFO")

    while True:
        await asyncio.sleep(2)

        if not branch_path.exists():
            log(
                f"Branch {strategy_name} directory deleted (parent reverted). Exiting.",
                "WARN",
            )
            manifest.status = "failed"
            return

        signal = read_control_signal(branch_path)
        if not signal:
            continue

        _handle_signal(signal, manifest, iter_state, branch_path)
        if manifest.status != "stopped":
            log(f"Branch {strategy_name} resumed (status: {manifest.status})", "INFO")
            return


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------


async def run_branch_loop(branch_path: Path) -> List[dict]:
    """
    Execute the optimization loop for a specific branch.

    Loads state from disk, runs nodes, saves state back.
    Uses the composed view pattern: nodes see a unified dict.

    Returns:
        List of SubStrategy dicts for master to spawn (if branching).
    """
    branch_path = Path(branch_path)

    try:
        manifest = load_branch_manifest(branch_path)
        context = load_context_for_branch(branch_path)
    except Exception as e:
        log(f"Worker failed to load state from {branch_path}: {e}", "ERROR")
        return []

    strategy_name = manifest.strategy.name
    log(f"Worker picked up branch: {strategy_name} (Status: {manifest.status})", "INFO")

    # Set branch to running
    if manifest.status == "initialized":
        manifest.status = "running"
        save_branch_manifest(branch_path, manifest)

    while manifest.status not in ["success", "failed", "branching"]:
        try:
            iter_num = manifest.current_iter

            # Load or create iteration state
            iter_state = load_iter_state_if_exists(branch_path, iter_num)
            if iter_state is None:
                prev_state = None
                if iter_num > 1:
                    prev_state = load_iter_state_if_exists(branch_path, iter_num - 1)
                iter_state = create_initial_iter_state(iter_num, prev_state)
                save_iter_state(branch_path, iter_num, iter_state)

            # Re-read config fields from disk (API writes them directly)
            try:
                disk_manifest = load_branch_manifest(branch_path)
                manifest.max_iter = disk_manifest.max_iter
            except Exception as e:
                log(f"Failed to reload manifest for {strategy_name}: {e}", "WARN")

            # Check for control signals before each node
            signal = read_control_signal(branch_path)
            if signal:
                _handle_signal(signal, manifest, iter_state, branch_path)
                if manifest.status == "stopped":
                    await _wait_for_resume(manifest, iter_state, branch_path)
                continue
            current_status = iter_state.status

            # Compose working state for the node
            working = _compose_working_state(context, manifest, iter_state, branch_path)

            if current_status == "planning":
                working = await plan_node(working)
            elif current_status == "implementing":
                working = await implement_node(working)
            elif current_status == "configuring":
                working = await configure_node(working)
            elif current_status == "running":
                working = await run_node(working)
            elif current_status == "profiling":
                working = await profile_node(working)
            elif current_status == "proposing":
                working = await propose_node(working)
            elif current_status == "deciding":
                working = await decide_node(working)
            else:
                log(
                    f"Unknown iter status '{current_status}' in branch {strategy_name}",
                    "ERROR",
                )
                manifest.status = "failed"
                break

            # Validate the status transition the node made
            validate_transition(current_status, working.status)

            # Decompose back
            iter_state = _decompose_to_iter_state(working)
            _update_manifest_from_working(manifest, working)

            # Save iteration state and intermediate manifest after every node
            save_iter_state(branch_path, iter_num, iter_state)

            # Flush token stats to disk after every node
            from config import global_tracker

            global_tracker.save()

            # Save manifest strictly if we are NOT finished with the iteration yet
            # (If it's deciding, the logic below will save the manifest instead)
            if current_status != "deciding":
                save_branch_manifest(branch_path, manifest)

            # On iteration complete: advance or finish
            if current_status == "deciding":
                if iter_state.next_status in ("success", "failed", "branching"):
                    manifest.status = iter_state.next_status
                else:
                    # Start new iteration
                    manifest.current_iter += 1

                save_branch_manifest(branch_path, manifest)

        except Exception as e:
            log(f"Unhandled exception in branch {strategy_name}: {e}", "ERROR")
            traceback.print_exc()
            iter_state.status = "decided"
            iter_state.run_output = f"CRITICAL WORKER CRASH: {e}"
            save_iter_state(branch_path, iter_num, iter_state)
            manifest.status = "failed"
            save_branch_manifest(branch_path, manifest)

    if manifest.status == "success":
        log(
            f"Branch {strategy_name} completed successfully at iteration {manifest.current_iter}.",
            "SUCCESS",
        )

    elif manifest.status == "failed":
        log(
            f"Branch {strategy_name} failed at iteration {manifest.current_iter}.",
            "ERROR",
        )

    elif manifest.status == "branching":
        subs = manifest.sub_strategies_cache or []
        log(
            f"Branch {strategy_name} actively spawning {len(subs)} sub-strategies!",
            "INFO",
        )
        manifest.sub_strategies_cache = None
        save_branch_manifest(branch_path, manifest)
        return subs

    return []
