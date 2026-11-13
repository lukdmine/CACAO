"""Finalize endpoints: start the post-run pass and report where it stands."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.helpers import finalize_phase, get_problem_dir, is_problem_running
from api.optimizer import _get_gpu_index, start_finalize
from api.schemas import RunConfig

router = APIRouter()

_TERMINAL = ("done", "failed")


def finalize_state(problem_dir: Path, running: bool) -> str:
    """not_run, finalizing, stopped, failed, stale or up_to_date.

    "stopped" is never written (the process that would write it is the one that died),
    so it is read off a non-terminal manifest with no live process.
    """
    from state import load_finalize_manifest
    from utils.branch_bests import collect_branch_bests, source_fingerprint

    output_dir = problem_dir / "output"
    manifest = load_finalize_manifest(output_dir)
    if manifest is None:
        return "not_run"
    if manifest.status not in _TERMINAL:
        return "finalizing" if finalize_phase(problem_dir, running) else "stopped"
    if manifest.status == "failed":
        return "failed"
    current = source_fingerprint(collect_branch_bests(output_dir, problem_dir))
    return "up_to_date" if manifest.source_fingerprint == current else "stale"


@router.get("/api/problems/{name}/finalize")
def get_finalize(name: str):
    from state import load_finalize_manifest

    problem_dir = get_problem_dir(name)
    manifest = load_finalize_manifest(problem_dir / "output")
    overview: Optional[str] = None
    if manifest is not None and manifest.overview is not None:
        overview = manifest.overview.markdown or None
    return {
        "state": finalize_state(problem_dir, is_problem_running(name)),
        "manifest": manifest.model_dump() if manifest is not None else None,
        "overview_markdown": overview,
    }


@router.post("/api/problems/{name}/finalize")
def post_finalize(name: str, config: RunConfig = RunConfig()):
    """Run finalize on its own over the existing branches."""
    from utils.branch_bests import collect_branch_bests

    problem_dir = get_problem_dir(name)
    if is_problem_running(name):
        raise HTTPException(
            status_code=409, detail=f"Optimization already running for '{name}'"
        )
    if not collect_branch_bests(problem_dir / "output", problem_dir):
        raise HTTPException(status_code=400, detail="No branch has a valid result yet.")
    start_finalize(name, problem_dir, config, _get_gpu_index(problem_dir))
    return {"status": "finalizing", "problem": name}
