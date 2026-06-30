"""Problem CRUD endpoints."""

import re
import shutil

from fastapi import APIRouter, HTTPException
import yaml

from api.helpers import (
    PROBLEMS_DIR,
    get_problem_dir,
    load_yaml,
    terminate_run,
    is_problem_running,
)
from api.schemas import CreateProblemRequest, CloneProblemRequest

router = APIRouter()


def _build_problem_data(req: CreateProblemRequest, gpu_index: int):
    """Validate reference config and build problem.yaml dict."""
    ref_type = req.reference_type
    if ref_type == "cuda" and (not req.ref_function or not req.ref_kernel_code.strip()):
        raise HTTPException(
            status_code=400,
            detail="CUDA reference requires ref_function and ref_kernel_code",
        )
    if ref_type == "cpu_c" and (not req.ref_function or not req.ref_cpu_code.strip()):
        raise HTTPException(
            status_code=400,
            detail="CPU C reference requires ref_function and ref_cpu_code",
        )

    reference = {"type": ref_type, "function": req.ref_function}
    if ref_type == "cuda":
        reference.update(
            {
                "file": "ref_kernel.cu",
                "block_x": req.ref_block_x,
                "block_y": req.ref_block_y,
                "block_z": req.ref_block_z,
            }
        )
    else:
        reference.update({"file": "ref_cpu.c"})

    data = {
        "name": req.name,
        "description": req.description,
        "gpu": {"index": gpu_index},
        "kernel": {"file": "kernel.cu", "function": "kernel"},
        "reference": reference,
        "scalars": [s.model_dump() for s in req.scalars],
        "grid": {"x": req.grid_x, "y": req.grid_y, "z": req.grid_z},
        "vectors": [
            v.model_dump(by_alias=True, exclude_none=True) for v in req.vectors
        ],
        "validation": {"tolerance": req.tolerance},
    }
    if req.tuning is not None:
        data["tuning"] = req.tuning.model_dump()
    return data, ref_type


def _write_problem_files(problem_dir, problem_data, req, ref_type):
    """Write problem.yaml and reference source files."""
    with (problem_dir / "problem.yaml").open("w") as f:
        yaml.dump(problem_data, f, default_flow_style=False, sort_keys=False)
    if ref_type == "cuda":
        (problem_dir / "ref_kernel.cu").write_text(req.ref_kernel_code)
        (problem_dir / "ref_cpu.c").unlink(missing_ok=True)
    else:
        (problem_dir / "ref_cpu.c").write_text(req.ref_cpu_code)
        (problem_dir / "ref_kernel.cu").unlink(missing_ok=True)


@router.get("/api/problems")
def list_problems():
    """List all problem directories with their status."""
    problems = []
    if not PROBLEMS_DIR.is_dir():
        return {"problems": []}

    for entry in sorted(PROBLEMS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        problem_yaml = entry / "problem.yaml"
        if not problem_yaml.exists():
            continue

        config = load_yaml(problem_yaml)
        name = entry.name

        status = "idle"
        if is_problem_running(name):
            status = "running"
        elif (entry / "output" / "branches").is_dir():
            status = "completed"

        problems.append(
            {
                "name": name,
                "status": status,
                "description": config.get("description", config.get("name", name)),
            }
        )

    return {
        "problems": problems,
    }


@router.post("/api/problems")
def create_problem(req: CreateProblemRequest):
    """Create a new problem directory with problem.yaml and reference files."""
    if not re.match(r"^[a-z0-9_]+$", req.slug):
        raise HTTPException(
            status_code=400,
            detail="Slug must be lowercase alphanumeric with underscores",
        )

    problem_dir = PROBLEMS_DIR / req.slug
    if problem_dir.exists():
        raise HTTPException(
            status_code=409, detail=f"Problem '{req.slug}' already exists"
        )

    problem_data, ref_type = _build_problem_data(req, req.gpu.index if req.gpu else 0)
    problem_dir.mkdir(parents=True)
    _write_problem_files(problem_dir, problem_data, req, ref_type)
    return {"status": "created", "name": req.slug, "path": str(problem_dir)}


@router.post("/api/problems/{name}/clone")
def clone_problem(name: str, req: CloneProblemRequest):
    """Clone a problem's definition files into a new directory."""
    if not re.match(r"^[a-z0-9_]+$", req.new_name):
        raise HTTPException(
            status_code=400,
            detail="Name must be lowercase alphanumeric with underscores",
        )

    source_dir = get_problem_dir(name)
    target_dir = PROBLEMS_DIR / req.new_name
    if target_dir.exists():
        raise HTTPException(
            status_code=409, detail=f"Problem '{req.new_name}' already exists"
        )

    # Copy entire problem directory, excluding runtime artifacts.
    # symlinks=True copies links as links so stale/broken symlinks in old
    # output dirs don't crash the copy.
    shutil.copytree(
        source_dir,
        target_dir,
        symlinks=True,
        ignore=shutil.ignore_patterns("run.pid", "run.log", "requeue"),
    )

    return {"status": "cloned", "name": req.new_name, "source": name}


@router.put("/api/problems/{name}")
def update_problem(name: str, req: CreateProblemRequest):
    """Update an existing problem definition."""
    if is_problem_running(name):
        raise HTTPException(
            status_code=400, detail=f"Cannot update '{name}' while running"
        )

    problem_dir = get_problem_dir(name)
    if req.slug != name:
        raise HTTPException(
            status_code=400, detail="Renaming slug via update is not supported"
        )

    problem_data, ref_type = _build_problem_data(req, req.gpu.index if req.gpu else 0)
    _write_problem_files(problem_dir, problem_data, req, ref_type)
    return {"status": "updated", "name": name, "path": str(problem_dir)}


@router.delete("/api/problems/{name}")
def delete_problem(name: str):
    """Delete a problem entirely from the file system."""
    problem_dir = get_problem_dir(name)
    terminate_run(name, problem_dir)
    try:
        shutil.rmtree(problem_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}") from e
    return {"status": "deleted", "name": name}


@router.get("/api/problems/{name}/logs")
def get_logs(name: str, tail: int = 200):
    """Get the last N lines of the optimization log."""
    problem_dir = get_problem_dir(name)
    log_path = problem_dir / "output" / "run.log"
    if not log_path.exists():
        return {"log": "", "lines": 0}

    lines = log_path.read_text(errors="replace").splitlines()
    total = len(lines)
    tail_lines = lines[-tail:] if tail < total else lines
    return {"log": "\n".join(tail_lines), "lines": total, "truncated": total > tail}


@router.get("/api/problems/{name}/detail")
def get_problem(name: str):
    """Get detailed problem configuration."""
    problem_dir = get_problem_dir(name)
    problem_yaml = problem_dir / "problem.yaml"

    if not problem_yaml.exists():
        raise HTTPException(status_code=404, detail="problem.yaml not found")

    config = load_yaml(problem_yaml)

    ref_kernel = ""
    ref_kernel_path = problem_dir / "ref_kernel.cu"
    if ref_kernel_path.exists():
        ref_kernel = ref_kernel_path.read_text()

    ref_cpu = ""
    ref_cpu_path = problem_dir / "ref_cpu.c"
    if ref_cpu_path.exists():
        ref_cpu = ref_cpu_path.read_text()

    return {
        "name": name,
        "config": config,
        "ref_kernel": ref_kernel,
        "ref_cpu": ref_cpu,
        "has_output": (problem_dir / "output").is_dir(),
    }
