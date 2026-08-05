"""Branch control endpoints."""

import shutil

from fastapi import APIRouter, Header, HTTPException, Response

from api.helpers import (
    get_problem_dir,
    load_json,
    resolve_branch_path,
    is_branch_live,
    get_branch_lock,
    is_problem_running,
)
from api.schemas import BranchMessageRequest, ChangeDecisionRequest, BranchConfigRequest

router = APIRouter()


@router.post("/api/problems/{name}/branches/{branch_id:path}/stop")
def stop_branch(name: str, branch_id: str):
    """Stop a specific branch (cooperative -- takes effect between nodes)."""
    from state import write_control_signal

    branch_path = resolve_branch_path(name, branch_id)
    write_control_signal(branch_path, {"action": "stop"})
    return {"status": "stopped", "branch": branch_id}


@router.post("/api/problems/{name}/branches/{branch_id:path}/resume")
def resume_branch(name: str, branch_id: str):
    """Resume a stopped branch."""
    from state import write_control_signal

    branch_path = resolve_branch_path(name, branch_id)
    write_control_signal(branch_path, {"action": "resume"})
    return {"status": "resumed", "branch": branch_id}


@router.post("/api/problems/{name}/branches/{branch_id:path}/message")
def message_branch(name: str, branch_id: str, req: BranchMessageRequest):
    """Send a message to a branch (resumes if stopped)."""
    from state import write_control_signal

    branch_path = resolve_branch_path(name, branch_id)
    write_control_signal(branch_path, {"action": "message", "content": req.content})
    return {"status": "sent", "branch": branch_id}


@router.post("/api/problems/{name}/branches/{branch_id:path}/change-decision")
def change_decision(name: str, branch_id: str, req: ChangeDecisionRequest):
    """
    Override the LLM's decision at target_iter.

    Handles live workers (signal), dead branches with running optimizer (disk surgery + requeue),
    and dead branches with stopped optimizer (disk surgery + auto-resume).
    """
    from state import (
        write_control_signal,
        revert_branch_on_disk,
        continue_branch_on_disk,
        write_requeue,
        load_branch_config,
        load_branch_manifest,
    )

    branch_path = resolve_branch_path(name, branch_id)
    output_dir = get_problem_dir(name) / "output"

    iter_state_path = branch_path / f"iter{req.target_iter}" / "state.json"
    if not iter_state_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Iteration {req.target_iter} state not found"
        )

    lock = get_branch_lock(branch_path)
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="Another operation is in progress on this branch"
        )
    try:
        live = is_branch_live(name, branch_path)

        if live:
            write_control_signal(
                branch_path,
                {
                    "action": "revert",
                    "target_iter": req.target_iter,
                    "content": req.content,
                },
            )
            return {
                "status": "signaled",
                "branch": branch_id,
                "to_iter": req.target_iter,
            }

        # Dead branch — server-side disk surgery
        manifest = load_branch_manifest(branch_path)
        current_iter = manifest.current_iter

        if req.target_iter < current_iter:
            revert_branch_on_disk(branch_path, req.target_iter, req.content)
        else:
            continue_branch_on_disk(branch_path, req.content)

        (branch_path / "control.json").unlink(missing_ok=True)

        optimizer_alive = is_problem_running(name)
        if optimizer_alive:
            write_requeue(output_dir, branch_path)
            return {
                "status": "requeued",
                "branch": branch_id,
                "to_iter": req.target_iter,
            }

        # Optimizer is stopped: auto-resume
        from api.optimizer import start_resume, _get_gpu_index
        from api.schemas import RunConfig

        gpu_index = _get_gpu_index(get_problem_dir(name))
        # Re-read the (possibly lifted) budget so the resumed run inherits it
        start_resume(
            name,
            get_problem_dir(name),
            RunConfig(max_iter=load_branch_config(branch_path).max_iter),
            gpu_index,
        )
        return {
            "status": "resumed_optimizer",
            "branch": branch_id,
            "to_iter": req.target_iter,
        }
    finally:
        lock.release()


@router.post("/api/problems/{name}/branches/{branch_id:path}/config")
def configure_branch(name: str, branch_id: str, req: BranchConfigRequest):
    """Update branch configuration (e.g. max_iter).

    Writes branch_config.json, which nothing but this endpoint ever writes. The
    worker reads it fresh when it composes the state for each node, so the new
    budget takes effect from the next node onwards with no signal needed.

    This deliberately does not touch branch.json: the worker holds that manifest
    in memory for the duration of a node and rewrites it wholesale afterwards,
    so a setting written there was reverted a few minutes later, every time.
    """
    from state import load_branch_config, save_branch_config

    branch_path = resolve_branch_path(name, branch_id)

    config = load_branch_config(branch_path)
    if req.max_iter is not None:
        config.max_iter = req.max_iter
    save_branch_config(branch_path, config)

    return {"status": "configured", "branch": branch_id}


@router.get("/api/problems/{name}/branches/{branch_id:path}/iterations/{iter_num}")
def get_iteration(
    name: str,
    branch_id: str,
    iter_num: int,
    response: Response,
    if_none_match: str | None = Header(default=None),
):
    """Full state for one iteration — the work products the tree leaves out.

    The panel fetches this when an iteration is expanded. A finished iteration
    never changes, so its ETag is stable and re-opening it costs a 304.
    """
    branch_path = resolve_branch_path(name, branch_id)
    state_file = branch_path / f"iter{iter_num}" / "state.json"
    try:
        info = state_file.stat()
    except OSError:
        raise HTTPException(
            status_code=404, detail=f"Iteration {iter_num} not found in '{branch_id}'"
        ) from None

    etag = f'W/"{info.st_mtime_ns:x}-{info.st_size:x}"'
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})
    response.headers["ETag"] = etag

    return load_json(state_file)


@router.delete("/api/problems/{name}/branches/{branch_id:path}")
def delete_branch(name: str, branch_id: str):
    """Delete a leaf branch (no children allowed). Only when optimizer is stopped."""
    if is_problem_running(name):
        raise HTTPException(
            status_code=409, detail="Stop the optimizer before deleting a branch."
        )

    branch_path = resolve_branch_path(name, branch_id)

    children_dir = branch_path / "branches"
    if children_dir.is_dir() and any(children_dir.iterdir()):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a branch that has children. Delete children first.",
        )

    lock = get_branch_lock(branch_path)
    with lock:
        shutil.rmtree(branch_path)
    return {"status": "deleted", "branch": branch_id}
