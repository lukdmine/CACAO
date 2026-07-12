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
from api.schemas import (
    CreateProblemRequest,
    CloneProblemRequest,
    PreviewInputsRequest,
)
from utils.inputs import generate_inputs_hpp, load_inputs_spec, write_inputs

router = APIRouter()


_REF_SIGNATURE = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^)]*)\)', re.S
)


def _signature_warnings(req: CreateProblemRequest) -> list[str]:
    """Compare the declared boundary against the reference kernel's signature.

    A boundary in the wrong order is the nastiest failure in this design: KTT binds
    SetArguments by position, so the reference silently computes from shuffled inputs and
    validation compares against garbage. Nothing errors.

    Reported, never blocking (design I6): this regex cannot parse every legal declaration,
    and a false rejection would be worse than a banner.
    """
    if req.reference_type != "cuda":
        # Persisted so the problem round-trips, but the engine skeleton only wires
        # SetReferenceKernel. KTT does offer SetReferenceComputation (a host callable)
        # for this; until the skeleton uses it, such a problem cannot run.
        return [
            "Framework mode validates only against a CUDA reference kernel. This "
            "problem's C reference is saved, but it cannot be run until CPU-reference "
            "support (KTT SetReferenceComputation) lands."
        ]

    match = _REF_SIGNATURE.search(req.ref_kernel_code or "")
    if not match:
        return []

    func, params = match.group(1), match.group(2)
    if func != req.ref_function:
        return [
            f"problem.yaml names reference function '{req.ref_function}', but "
            f"ref_kernel.cu declares '{func}'."
        ]

    declared = [p for p in (p.strip() for p in params.split(",")) if p]
    boundary = req.inputs.boundary
    if len(declared) != len(boundary):
        return [
            f"Reference kernel '{func}' takes {len(declared)} arguments, but the boundary "
            f"passes {len(boundary)} ({', '.join(a.name for a in boundary) or 'none'}). "
            "They are bound by position, so a mismatch validates against garbage."
        ]

    mismatched = [
        f"#{i + 1}: boundary '{arg.name}' vs reference '{decl}'"
        for i, (arg, decl) in enumerate(zip(boundary, declared))
        if not re.search(rf"\b{re.escape(arg.name)}\b", decl)
    ]
    if mismatched:
        return [
            f"Boundary order may not match reference kernel '{func}' — "
            + "; ".join(mismatched)
            + ". Arguments bind by position, not by name."
        ]
    return []


def _build_problem_data(req: CreateProblemRequest, gpu_index: int):
    """Build the pure-metadata problem.yaml dict (the I/O boundary lives in inputs.yaml)."""
    is_cuda = req.reference_type == "cuda"
    code = req.ref_kernel_code if is_cuda else req.ref_cpu_code
    if not req.ref_function or not code.strip():
        raise HTTPException(
            status_code=400,
            detail=f"A {req.reference_type} reference requires ref_function and its source code",
        )

    reference = {
        "type": req.reference_type,
        "function": req.ref_function,
        "file": "ref_kernel.cu" if is_cuda else "ref_cpu.c",
    }
    if is_cuda:
        # Nested, not flat block_x/block_y: utils/framework.py reads ref["block"] and
        # falls back to a 1x1 block when it is absent.
        reference["block"] = {
            "x": req.ref_block_x,
            "y": req.ref_block_y,
            "z": req.ref_block_z,
        }

    data = {
        "name": req.name,
        "description": req.description,
        "gpu": {"index": gpu_index},
        "global_size_type": req.global_size_type,
        "grid": {"x": req.grid_x, "y": req.grid_y, "z": req.grid_z},
        "reference": reference,
        "validation": {"tolerance": req.tolerance},
    }
    if req.tuning is not None:
        data["tuning"] = req.tuning.model_dump()
    return data


def _write_problem_files(problem_dir, problem_data, req: CreateProblemRequest):
    """Write problem.yaml, the reference source, and the inputs.yaml/inputs.hpp pair."""
    with (problem_dir / "problem.yaml").open("w") as f:
        yaml.dump(problem_data, f, default_flow_style=False, sort_keys=False)
    if req.reference_type == "cuda":
        (problem_dir / "ref_kernel.cu").write_text(req.ref_kernel_code)
    else:
        (problem_dir / "ref_cpu.c").write_text(req.ref_cpu_code)
    write_inputs(problem_dir, req.inputs)


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

    problem_data = _build_problem_data(req, req.gpu.index if req.gpu else 0)
    problem_dir.mkdir(parents=True)
    _write_problem_files(problem_dir, problem_data, req)
    return {
        "status": "created",
        "name": req.slug,
        "path": str(problem_dir),
        "warnings": _signature_warnings(req),
    }


@router.post("/api/problems/preview-inputs")
def preview_inputs(req: PreviewInputsRequest):
    """Render inputs.hpp for a spec without saving.

    Lets the form show the generated C++ live while keeping a single generator — the
    frontend never builds the header itself.
    """
    return {"inputs_hpp": generate_inputs_hpp(req.inputs)}


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

    problem_data = _build_problem_data(req, req.gpu.index if req.gpu else 0)
    _write_problem_files(problem_dir, problem_data, req)
    return {
        "status": "updated",
        "name": name,
        "path": str(problem_dir),
        "warnings": _signature_warnings(req),
    }


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

    # The structured boundary, not the generated C++. Reloading the spec is what makes
    # Edit lossless: the form never has to reconstruct its state from inputs.hpp.
    inputs = None
    inputs_yaml = problem_dir / "inputs.yaml"
    if inputs_yaml.exists():
        inputs = load_inputs_spec(inputs_yaml).model_dump(by_alias=True)

    return {
        "name": name,
        "config": config,
        "ref_kernel": ref_kernel,
        "ref_cpu": ref_cpu,
        "inputs": inputs,
        "has_output": (problem_dir / "output").is_dir(),
    }
