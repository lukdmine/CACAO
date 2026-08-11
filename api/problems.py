"""Problem CRUD endpoints."""

import re
import shutil

from fastapi import APIRouter, HTTPException, Request
import numpy as np
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
from utils.inputs import (
    INPUTS_SUBDIR,
    generate_inputs_hpp,
    load_inputs_spec,
    write_inputs,
)
from utils.python_ref_runner import DTYPE_MAP, eval_size

router = APIRouter()


_REF_SIGNATURE = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^)]*)\)', re.S
)
# A C reference is plain host code: no __global__, and `extern "C"` is optional in the
# source the user pastes (inputs.hpp re-declares it with extern "C" itself).
_REF_CPU_SIGNATURE = re.compile(
    r'(?:extern\s+"C"\s+)?void\s+(\w+)\s*\(([^)]*)\)\s*\{', re.S
)
# A python reference is a top-level def. Its parameters are (scalars, buffers) dicts, so
# only the name matters — there is no positional binding to get wrong.
_REF_PY_SIGNATURE = re.compile(r"^def\s+(\w+)\s*\(", re.M)


def _signature_warnings(req: CreateProblemRequest) -> list[str]:
    """Compare the declared boundary against the reference kernel's signature.

    A boundary in the wrong order is the nastiest failure in this design: KTT binds
    SetArguments by position, so the reference silently computes from shuffled inputs and
    validation compares against garbage. Nothing errors.

    Reported, never blocking (design I6): this regex cannot parse every legal declaration,
    and a false rejection would be worse than a banner.
    """
    # A python reference is called as f(scalars, buffers) with dicts keyed by NAME
    # (utils/python_ref_runner), so none of the positional reasoning below applies to it.
    # Only the name has to match — the runner looks the function up.
    if req.reference_type == "python":
        found = _REF_PY_SIGNATURE.findall(req.ref_python_code or "")
        if not found:
            return []
        if req.ref_function not in found:
            declared = ", ".join(f"'{f}'" for f in found)
            return [
                f"problem.yaml names reference function '{req.ref_function}', but ref.py "
                f"defines {declared}. The runner looks it up by name."
            ]
        return []

    is_cuda = req.reference_type == "cuda"
    pattern = _REF_SIGNATURE if is_cuda else _REF_CPU_SIGNATURE
    found = list(pattern.finditer((req.ref_kernel_code if is_cuda else req.ref_cpu_code) or ""))
    if not found:
        return []

    # Match by name, not position: a C reference file may define helpers alongside the
    # reference, and the first declaration is not necessarily the one problem.yaml names.
    match = next((m for m in found if m.group(1) == req.ref_function), None)
    if match is None:
        declared = ", ".join(f"'{m.group(1)}'" for m in found)
        return [
            f"problem.yaml names reference function '{req.ref_function}', but the "
            f"reference source declares {declared}. The driver looks it up by name."
        ]

    func, params = match.group(1), match.group(2)

    # The two references take different things. A CUDA kernel receives the boundary
    # (buffers + runtime scalars). A C reference receives every buffer as a pointer,
    # and reads scalars as -D macros instead (utils/inputs.py: scalar_define_flags).
    expected = req.inputs.boundary if is_cuda else req.inputs.buffers
    kind = "boundary" if is_cuda else "buffer list"

    declared = [p for p in (p.strip() for p in params.split(",")) if p and p != "void"]
    if len(declared) != len(expected):
        return [
            f"Reference '{func}' takes {len(declared)} arguments, but the {kind} passes "
            f"{len(expected)} ({', '.join(a.name for a in expected) or 'none'}). "
            "They bind by position, so a mismatch validates against garbage."
        ]

    mismatched = [
        f"#{i + 1}: {kind} '{arg.name}' vs reference '{decl}'"
        for i, (arg, decl) in enumerate(zip(expected, declared))
        if not re.search(rf"\b{re.escape(arg.name)}\b", decl)
    ]
    if mismatched:
        return [
            f"Argument order may not match reference '{func}' — "
            + "; ".join(mismatched)
            + ". Arguments bind by position, not by name."
        ]
    return []


# Filename each reference kind is written to, and read back from. The engine reads
# reference.file, so these are the names nodes/configure.py and utils/build.py expect.
_REFERENCE_FILE = {
    "cuda": "ref_kernel.cu",
    "cpu_c": "ref_cpu.c",
    "python": "ref.py",
}


def _reference_source(req: CreateProblemRequest) -> str:
    """The request field carrying this reference kind's code."""
    return {
        "cuda": req.ref_kernel_code,
        "cpu_c": req.ref_cpu_code,
        "python": req.ref_python_code,
    }[req.reference_type]


def _build_problem_data(req: CreateProblemRequest, gpu_index: int):
    """Build the pure-metadata problem.yaml dict (the I/O boundary lives in inputs.yaml)."""
    is_cuda = req.reference_type == "cuda"
    code = _reference_source(req)
    if not req.ref_function or not code.strip():
        raise HTTPException(
            status_code=400,
            detail=f"A {req.reference_type} reference requires ref_function and its source code",
        )

    reference = {
        "type": req.reference_type,
        "function": req.ref_function,
        "file": _REFERENCE_FILE[req.reference_type],
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


def _write_problem_files(problem_dir, problem_data, req: CreateProblemRequest) -> list[str]:
    """Write problem.yaml, the reference source, and the inputs.yaml/inputs.hpp pair.

    Returns warnings. A missing init=file binary does not block the save — the UI
    uploads the file right after saving (and re-uploads on demand in Edit).
    ensure_inputs_hpp still hard-fails at run start.
    """
    with (problem_dir / "problem.yaml").open("w") as f:
        yaml.dump(problem_data, f, default_flow_style=False, sort_keys=False)
    (problem_dir / _REFERENCE_FILE[req.reference_type]).write_text(_reference_source(req))
    # The reference drives codegen: cpu_c needs its extern "C" declaration, host input
    # copies, and a SetReferenceComputation per validated buffer; python needs a lambda
    # that dumps the buffers and shells out to the runner.
    return write_inputs(problem_dir, req.inputs, problem_data["reference"])


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
    warnings = _write_problem_files(problem_dir, problem_data, req)
    return {
        "status": "created",
        "name": req.slug,
        "path": str(problem_dir),
        "warnings": warnings + _signature_warnings(req),
    }


@router.post("/api/problems/preview-inputs")
def preview_inputs(req: PreviewInputsRequest):
    """Render inputs.hpp for a spec without saving.

    Lets the form show the generated C++ live while keeping a single generator — the
    frontend never builds the header itself.
    """
    reference = {
        "type": req.reference_type,
        "function": req.ref_function,
        "file": "ref_kernel.cu" if req.reference_type == "cuda" else "ref_cpu.c",
    }
    return {"inputs_hpp": generate_inputs_hpp(req.inputs, reference)}


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
    warnings = _write_problem_files(problem_dir, problem_data, req)
    return {
        "status": "updated",
        "name": name,
        "path": str(problem_dir),
        "warnings": warnings + _signature_warnings(req),
    }


@router.post("/api/problems/{name}/inputs/{buffer_name}")
async def upload_input_file(name: str, buffer_name: str, request: Request):
    """Upload the binary for an init=file buffer.

    Raw request body (application/octet-stream), streamed to disk in chunks — inputs
    can be larger than RAM-friendly sizes, and python-multipart is not a dependency.
    ``file_name`` comes from the saved spec, never from the client, so the path cannot
    be steered outside the problem's inputs/ directory. The byte count must equal
    size * sizeof(dtype) — the same contract the generated driver enforces at start.
    """
    problem_dir = get_problem_dir(name)
    if is_problem_running(name):
        raise HTTPException(
            status_code=400, detail=f"Cannot upload inputs for '{name}' while running"
        )

    spec = load_inputs_spec(problem_dir / "inputs.yaml")
    buf = next((b for b in spec.buffers if b.name == buffer_name), None)
    if buf is None:
        raise HTTPException(
            status_code=404, detail=f"Buffer '{buffer_name}' not found in inputs.yaml"
        )
    if buf.init != "file":
        raise HTTPException(
            status_code=400,
            detail=f"Buffer '{buffer_name}' has init={buf.init} — uploads need init=file",
        )

    # Belt and braces: the model already rejects absolute paths and '..', but the
    # resolved path staying under inputs/ is what makes this endpoint safe.
    inputs_dir = problem_dir / INPUTS_SUBDIR
    path = (inputs_dir / buf.file_name).resolve()
    if not path.is_relative_to(inputs_dir.resolve()):
        raise HTTPException(
            status_code=400, detail="file_name escapes the inputs/ directory"
        )

    scalars = {s.name: s.value for s in spec.scalars}
    try:
        elems = eval_size(buf.size, scalars)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Buffer '{buffer_name}': cannot evaluate size '{buf.size}': {e}",
        )
    want = elems * np.dtype(DTYPE_MAP[buf.dtype]).itemsize

    inputs_dir.mkdir(exist_ok=True)
    got = 0
    too_big = False
    with path.open("wb") as f:
        async for chunk in request.stream():
            got += len(chunk)
            if got > want:
                too_big = True
                break
            f.write(chunk)
    if got != want:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Buffer '{buffer_name}' expects {want} bytes ({elems} {buf.dtype}"
            f" elements) but the upload {'exceeds that' if too_big else f'has {got} bytes'}",
        )

    return {
        "status": "uploaded",
        "buffer": buffer_name,
        "file_name": buf.file_name,
        "bytes": want,
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

    # Without this, opening Edit on a python problem and saving would write an empty
    # ref.py — the same way the boundary used to be wiped.
    ref_python = ""
    ref_python_path = problem_dir / _REFERENCE_FILE["python"]
    if ref_python_path.exists():
        ref_python = ref_python_path.read_text()

    # The structured boundary, not the generated C++. Reloading the spec is what makes
    # Edit lossless: the form never has to reconstruct its state from inputs.hpp.
    inputs = None
    input_files = {}
    inputs_yaml = problem_dir / "inputs.yaml"
    if inputs_yaml.exists():
        spec = load_inputs_spec(inputs_yaml)
        inputs = spec.model_dump(by_alias=True)
        for b in spec.buffers:
            if b.init == "file":
                p = problem_dir / INPUTS_SUBDIR / b.file_name
                input_files[b.name] = {
                    "file_name": b.file_name,
                    "exists": p.is_file(),
                    "bytes": p.stat().st_size if p.is_file() else None,
                }

    return {
        "name": name,
        "config": config,
        "ref_kernel": ref_kernel,
        "ref_cpu": ref_cpu,
        "ref_python": ref_python,
        "inputs": inputs,
        # Upload status per init=file buffer; the Edit dialog shows/replaces from this.
        "input_files": input_files,
        "has_output": (problem_dir / "output").is_dir(),
    }
