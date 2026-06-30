#!/usr/bin/env python3
"""
KTT Python Tuner for CUDA Kernels

Reads problem definition from problem.yaml and tuning parameters from params.json
Uses pyktt bindings for direct KTT integration.
"""

import argparse
import ctypes
import sys
import json
import subprocess
import numpy as np
import yaml
from pathlib import Path

# Add script directory to path for pyktt.so
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import pyktt as ktt


_DTYPE_DEFS = [
    ("char", ["int8", "char"], np.int8, ctypes.c_int8),
    ("short", ["int16", "short"], np.int16, ctypes.c_int16),
    ("int", ["int32", "int"], np.int32, ctypes.c_int32),
    ("long", ["int64", "long"], np.int64, ctypes.c_int64),
    ("float", ["float32", "float"], np.float32, ctypes.c_float),
    ("double", ["float64", "double"], np.float64, ctypes.c_double),
]
DTYPE_INFO = {
    name: {
        "numpy": npt,
        "ctype": ct,
        "vector_method": f"AddArgumentVector{name.capitalize()}",
    }
    for name, _, npt, ct in _DTYPE_DEFS
}
DTYPE_ALIASES = {
    alias: name for name, aliases, _, _ in _DTYPE_DEFS for alias in aliases
}


def normalize_dtype(dtype: str) -> str:
    normalized = DTYPE_ALIASES.get(str(dtype).lower())
    if normalized is None:
        supported = ", ".join(sorted(DTYPE_ALIASES.keys()))
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes: {supported}")
    return normalized


def get_dtype_info(dtype: str) -> dict:
    return DTYPE_INFO[normalize_dtype(dtype)]


def build_scalar_defines(scalar_defs: list) -> str:
    """Build ``-DNAME=VALUE`` compiler flags from scalar definitions."""
    flags = []
    for s in scalar_defs:
        name, dtype, value = s["name"], s["dtype"], s["value"]
        if dtype == "float":
            formatted = f"{float(value):.10g}"
            if "." not in formatted and "e" not in formatted.lower():
                formatted += ".0"
            formatted += "f"
        elif dtype == "double":
            formatted = f"{float(value):.10g}"
            if "." not in formatted and "e" not in formatted.lower():
                formatted += ".0"
        else:
            formatted = str(int(value))
        flags.append(f"-D{name}={formatted}")
    return " ".join(flags)


def add_vector_argument(tuner, dtype: str, data, access):
    dtype_info = get_dtype_info(dtype)
    method = getattr(tuner, dtype_info["vector_method"])
    return method(data, access)


def create_vector_data(
    rng: np.random.Generator,
    size: int,
    dtype: str,
    init_mode: str,
    init_min=None,
    init_max=None,
):
    numpy_dtype = np.dtype(get_dtype_info(dtype)["numpy"])
    if init_mode == "random":
        if np.issubdtype(numpy_dtype, np.floating):
            lo = float(init_min) if init_min is not None else -2.0
            hi = float(init_max) if init_max is not None else 2.0
            return rng.uniform(lo, hi, size).astype(numpy_dtype)
        lo = int(init_min) if init_min is not None else -2
        hi = (int(init_max) + 1) if init_max is not None else 3
        return rng.integers(lo, hi, size=size, dtype=numpy_dtype)
    return np.zeros(size, dtype=numpy_dtype)


def evaluate_size_expr(expr, scalars: dict) -> int:
    """Evaluate integer size expression using problem scalar values."""
    if isinstance(expr, (int, np.integer)):
        size = int(expr)
    elif isinstance(expr, (float, np.floating)):
        if not float(expr).is_integer():
            raise ValueError(f"Size expression must evaluate to an integer, got {expr}")
        size = int(expr)
    else:
        try:
            result = eval(str(expr), {"__builtins__": {}}, dict(scalars))
        except Exception as e:
            raise ValueError(f"Failed to evaluate size expression '{expr}': {e}") from e

        if isinstance(result, bool):
            raise ValueError(
                f"Size expression must evaluate to an integer, got boolean: {expr}"
            )
        if isinstance(result, (int, np.integer)):
            size = int(result)
        elif isinstance(result, (float, np.floating)) and float(result).is_integer():
            size = int(result)
        else:
            raise ValueError(
                f"Size expression must evaluate to an integer, got {result!r} from '{expr}'"
            )

    if size < 0:
        raise ValueError(
            f"Size expression must be non-negative, got {size} from '{expr}'"
        )
    return size


def create_constraint_fn(expr: str, param_names: list, scalar_values: dict = None):
    """Build a KTT constraint function that can reference both tuning params and problem scalars."""
    _scalars = dict(scalar_values) if scalar_values else {}

    def constraint_fn(values):
        # Start with problem scalars, then overlay tuning param values
        namespace = dict(_scalars)
        namespace.update(
            {name: val for name, val in zip(param_names, values, strict=True)}
        )
        try:
            result = eval(expr, {"__builtins__": {}}, namespace)
            return bool(result)
        except ZeroDivisionError:
            return False
        except Exception as e:
            print(f"[Warning] Constraint eval failed: {expr} with {namespace} -> {e}")
            return False

    return constraint_fn


def safe_eval(expr: str, context: dict) -> int:
    """Evaluate an integer formula against a name->value context, with no builtins."""
    try:
        result = eval(expr, {"__builtins__": {}}, context)
        return int(result)
    except ZeroDivisionError:
        print(f"[Warning] Division by zero in formula: {expr}")
        return 0
    except Exception as e:
        print(f"[Warning] Formula eval failed: {expr} with context -> {e}")
        return 0


def _compile_cpu_reference(
    source_path: Path, build_dir: Path, scalar_defines: str = ""
) -> Path:
    """Compile CPU reference C/C++ source into a shared object."""
    from utils.cuda_env import get_env

    env = get_env()
    ext = source_path.suffix.lower()
    if ext in (".cc", ".cpp", ".cxx"):
        compiler_candidates = [str(env.gpp)] if env.gpp else ["g++"]
    else:
        candidates = []
        if env.gcc:
            candidates.append(str(env.gcc))
        if env.gpp:
            candidates.append(str(env.gpp))
        compiler_candidates = candidates or ["gcc", "g++"]

    output_so = build_dir / f"{source_path.stem}.so"
    # -D flags must come before the source file for preprocessor visibility
    define_flags = scalar_defines.split() if scalar_defines else []
    compile_cmd_base = [
        "-O3",
        "-shared",
        "-fPIC",
        *define_flags,
        str(source_path),
        "-o",
        str(output_so),
    ]

    errors = []
    for compiler in compiler_candidates:
        cmd = [compiler, *compile_cmd_base]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_so.exists():
            print(f"Compiled CPU reference: {output_so} (via {compiler})")
            return output_so
        errors.append(f"{compiler}: {result.stderr.strip()}")

    raise RuntimeError("Failed to compile CPU reference source. " + " | ".join(errors))


def _build_cpu_reference_callback(
    source_path: Path,
    function_name: str,
    vectors: list,
    vector_data: list,
    scalar_defs: list,
    validated_vector_index: int,
    build_dir: Path,
):
    """
    Build callback for KTT SetReferenceComputation from a compiled CPU function.

    Scalars are injected as -D compiler defines. The function signature
    contains only vector pointer arguments in problem.yaml order.
    """
    if validated_vector_index < 0:
        raise RuntimeError(
            "No validate=true vector found for CPU reference computation"
        )

    scalar_defines = build_scalar_defines(scalar_defs)
    shared_obj = _compile_cpu_reference(source_path, build_dir, scalar_defines)
    reference_lib = ctypes.CDLL(str(shared_obj))

    if not hasattr(reference_lib, function_name):
        raise RuntimeError(
            f"CPU reference function '{function_name}' not found in {shared_obj}. "
            'Use extern "C" for C++ sources to avoid name mangling.'
        )

    cpu_fn = getattr(reference_lib, function_name)
    vector_argtypes = [
        ctypes.POINTER(get_dtype_info(v["dtype"])["ctype"]) for v in vectors
    ]
    cpu_fn.argtypes = vector_argtypes
    cpu_fn.restype = None

    validated_vector = vectors[validated_vector_index]
    validate_size = int(validated_vector["size"])
    validate_dtype = np.dtype(get_dtype_info(validated_vector["dtype"])["numpy"])
    reference_size = validate_size * validate_dtype.itemsize

    def callback(buffer_view):
        output_array = np.frombuffer(
            buffer_view, dtype=validate_dtype, count=validate_size
        )
        vector_args = []
        temporary_outputs = []

        for vector_index, vector in enumerate(vectors):
            pointer_type = ctypes.POINTER(get_dtype_info(vector["dtype"])["ctype"])

            if vector_index == validated_vector_index:
                vector_args.append(output_array.ctypes.data_as(pointer_type))
            elif vector["access"] == "read":
                vector_args.append(
                    vector_data[vector_index].ctypes.data_as(pointer_type)
                )
            else:
                temp_output = np.zeros(
                    vector["size"], dtype=get_dtype_info(vector["dtype"])["numpy"]
                )
                temporary_outputs.append(temp_output)
                vector_args.append(temp_output.ctypes.data_as(pointer_type))

        cpu_fn(*vector_args)

    return callback, reference_size


def _load_problem(working_dir: Path, problem_file: str) -> dict:
    """Load and parse problem.yaml into a problem spec dict."""
    problem_file = working_dir / problem_file
    print(f"Loading problem definition from: {problem_file}")
    with open(problem_file, "r") as f:
        problem = yaml.safe_load(f)

    problem_path = Path(problem_file).resolve()
    kernel_file = str(working_dir / problem["kernel"]["file"])
    kernel_func = problem["kernel"]["function"]
    reference = problem.get("reference", {}) or {}
    reference_type = reference.get("type", "cuda")
    ref_file = reference.get("file")
    ref_func = reference.get("function")
    if not ref_file or not ref_func:
        raise ValueError(
            "problem.yaml reference section must include 'file' and 'function'"
        )

    reference_source_path = Path(ref_file)
    if not reference_source_path.is_absolute():
        reference_source_path = (problem_path.parent / reference_source_path).resolve()

    print(f"Problem: {problem['name']}")
    print(f"Kernel: {kernel_func} from {kernel_file}")
    print(f"Reference ({reference_type}): {ref_func} from {reference_source_path}")

    # Parse scalars
    scalar_values = {}
    scalar_defs = []
    for s in problem["scalars"]:
        name, dtype = s["name"], normalize_dtype(s.get("dtype", "int"))
        scalar_values[name] = s["value"]
        scalar_defs.append({"name": name, "dtype": dtype, "value": s["value"]})
        print(f"Scalar: {name} = {s['value']}")

    # Parse vectors
    vectors = []
    for v in problem["vectors"]:
        vd = {
            "name": v["name"],
            "dtype": normalize_dtype(v["dtype"]),
            "size": evaluate_size_expr(v["size"], scalar_values),
            "access": v["access"],
            "init": v.get("init", "zeros"),
            "validate": v.get("validate", False),
            "init_min": v.get("init_min"),
            "init_max": v.get("init_max"),
        }
        vectors.append(vd)
        print(
            f"Vector: {vd['name']} [{vd['dtype']}, size={vd['size']}, {vd['access']}, init={vd['init']}]"
        )

    # Grid dimensions
    base_grid_x = evaluate_size_expr(problem["grid"]["x"], scalar_values)
    base_grid_y = evaluate_size_expr(problem["grid"].get("y", "1"), scalar_values)
    base_grid_z = evaluate_size_expr(problem["grid"].get("z", "1"), scalar_values)
    print(f"Base grid: {base_grid_x} x {base_grid_y} x {base_grid_z}")

    return {
        "kernel_file": kernel_file,
        "kernel_func": kernel_func,
        "reference": reference,
        "reference_type": reference_type,
        "reference_source_path": reference_source_path,
        "ref_func": ref_func,
        "scalar_values": scalar_values,
        "scalar_defs": scalar_defs,
        "vectors": vectors,
        "base_grid_x": base_grid_x,
        "base_grid_y": base_grid_y,
        "base_grid_z": base_grid_z,
        "tolerance": problem["validation"]["tolerance"],
        "tuning_duration_s": (problem.get("tuning") or {}).get("duration_s"),
    }


def _init_data(vectors: list) -> list:
    """Create randomized test data for all vectors."""
    rng = np.random.default_rng()
    return [
        create_vector_data(
            rng,
            vd["size"],
            vd["dtype"],
            vd["init"],
            vd.get("init_min"),
            vd.get("init_max"),
        )
        for vd in vectors
    ]


def _setup_kernels(tuner, prob: dict):
    """Load agent and reference kernels. Returns (agent_def, agent_kernel, ref_def, ref_kernel, has_cuda_ref)."""
    agent_grid = ktt.DimensionVector(
        prob["base_grid_x"], prob["base_grid_y"], prob["base_grid_z"]
    )
    agent_def = tuner.AddKernelDefinitionFromFile(
        prob["kernel_func"],
        prob["kernel_file"],
        agent_grid,
        ktt.DimensionVector(1, 1, 1),
    )
    agent_kernel = tuner.CreateSimpleKernel("AgentKernel", agent_def)

    # NVRTC compiler options: CUDA headers + scalar defines
    from utils.cuda_env import get_env

    env = get_env()
    opts_parts = []
    if env.cuda_include:
        opts_parts.append(f"-I{env.cuda_include}")
        print(f"NVRTC include path: {env.cuda_include}")
    else:
        print(
            "[Warning] Could not find CUDA headers. Tensor core kernels may fail to compile."
        )
    scalar_defines = build_scalar_defines(prob["scalar_defs"])
    if scalar_defines:
        opts_parts.append(scalar_defines)
        print(f"Scalar defines: {scalar_defines}")
    if opts_parts:
        tuner.SetCompilerOptions(" ".join(opts_parts))

    ref_def = ref_kernel = None
    has_cuda_ref = False
    reference = prob["reference"]

    if prob["reference_type"] == "cuda":
        bx = reference.get("block_x", 8)
        by = reference.get("block_y", 8)
        bz = reference.get("block_z", 1)
        if bx <= 0 or by <= 0 or bz <= 0:
            raise ValueError(
                f"Reference block dimensions must be positive, got ({bx}, {by}, {bz})"
            )
        gx = max(1, (prob["base_grid_x"] + bx - 1) // bx)
        gy = max(1, (prob["base_grid_y"] + by - 1) // by)
        gz = max(1, (prob["base_grid_z"] + bz - 1) // bz)
        ref_def = tuner.AddKernelDefinitionFromFile(
            prob["ref_func"],
            str(prob["reference_source_path"]),
            ktt.DimensionVector(gx, gy, gz),
            ktt.DimensionVector(bx, by, bz),
        )
        ref_kernel = tuner.CreateSimpleKernel("ReferenceKernel", ref_def)
        has_cuda_ref = True
    elif prob["reference_type"] == "cpu_c":
        print(
            "CPU C reference selected: validation will use SetReferenceComputation callback"
        )
    else:
        print(
            f"[Warning] Unknown reference type '{prob['reference_type']}'. Validation disabled."
        )

    return agent_def, agent_kernel, ref_def, ref_kernel, has_cuda_ref


def _setup_arguments(tuner, prob: dict, vector_data: list, ref_def, has_cuda_ref: bool):
    """Add vector arguments. Returns (arg_ids, validated_vectors).

    Scalars are injected as compiler defines, not as kernel arguments.
    """
    arg_ids = []
    validated_vectors = []
    for i, vd in enumerate(prob["vectors"]):
        access = (
            ktt.ArgumentAccessType.ReadOnly
            if vd["access"] == "read"
            else ktt.ArgumentAccessType.WriteOnly
        )
        aid = add_vector_argument(tuner, vd["dtype"], vector_data[i], access)
        arg_ids.append(aid)
        if vd["validate"]:
            validated_vectors.append((aid, i))

    if has_cuda_ref and ref_def is not None:
        tuner.SetArguments(ref_def, arg_ids)

    return arg_ids, validated_vectors


def _setup_launcher(
    tuner, agent_def, agent_kernel, params: dict, prob: dict, arg_ids: list
):
    """Configure custom launcher from params.json. Returns error dict or None."""
    launch_config = params.get("launch_config", {})

    if not launch_config:
        print(
            "[Error] params.json is missing 'launch_config'. All kernels must use launch_config."
        )
        tuner.SetArguments(agent_def, arg_ids)
        return {
            "success": False,
            "message": "params.json missing required 'launch_config' field",
        }

    print("\n--- Using Custom Launcher ---")
    base_context = dict(prob["scalar_values"])
    tuner.SetArguments(agent_def, arg_ids)
    base_gx, base_gy = prob["base_grid_x"], prob["base_grid_y"]

    def generic_launcher(compute_interface):
        context = dict(base_context)
        config = compute_interface.GetCurrentConfiguration()
        for pair in config.GetPairs():
            context[pair.GetName()] = pair.GetValueUint()

        gx = safe_eval(launch_config.get("grid_x", str(base_gx)), context)
        gy = safe_eval(launch_config.get("grid_y", str(base_gy)), context)
        gz = safe_eval(launch_config.get("grid_z", "1"), context)
        bx = safe_eval(launch_config.get("block_x", "1"), context)
        by = safe_eval(launch_config.get("block_y", "1"), context)
        bz = safe_eval(launch_config.get("block_z", "1"), context)

        try:
            compute_interface.RunKernel(
                agent_def,
                ktt.DimensionVector(gx, gy, gz),
                ktt.DimensionVector(bx, by, bz),
            )
        except Exception as e:
            print(f"[Warning] Kernel launch failed: {e}")

    tuner.SetLauncher(agent_kernel, generic_launcher)
    print(
        f"  grid: ({launch_config.get('grid_x', 'default')}, {launch_config.get('grid_y', 'default')}, {launch_config.get('grid_z', '1')})"
    )
    print(
        f"  block: ({launch_config.get('block_x', '1')}, {launch_config.get('block_y', '1')}, {launch_config.get('block_z', '1')})"
    )
    return None


def _setup_constraints(tuner, agent_kernel, params: dict, scalar_values: dict):
    """Add tuning parameter constraints from params.json."""
    scalar_name_set = set(scalar_values.keys())
    for c in params.get("constraints", []):
        raw_param_names = c.get("params", [])
        expr = c.get("expr")

        if not expr:
            print(f"[Warning] Constraint missing 'expr': {c}")
            continue
        if not raw_param_names:
            print(f"[Warning] Constraint missing 'params': {c}")
            continue

        tuning_param_names = [n for n in raw_param_names if n not in scalar_name_set]
        removed = [n for n in raw_param_names if n in scalar_name_set]
        if removed:
            print(
                f"[Info] Constraint '{expr}': stripped scalar name(s) {removed} from params (auto-injected)"
            )

        if not tuning_param_names:
            try:
                result = eval(expr, {"__builtins__": {}}, dict(scalar_values))
                if not result:
                    raise ValueError(
                        f"Static constraint always false: {expr} — no valid configurations possible"
                    )
                else:
                    print(f"Constraint (static, always true): {expr}")
            except Exception as e:
                print(f"[Warning] Static constraint eval error: {expr} -> {e}")
            continue

        constraint_fn = create_constraint_fn(expr, tuning_param_names, scalar_values)
        tuner.AddConstraint(agent_kernel, tuning_param_names, constraint_fn)
        print(f"Constraint: {expr}")


def _setup_validation(
    tuner,
    prob: dict,
    validated_vectors,
    has_cuda_ref,
    ref_kernel,
    vector_data,
    working_dir,
):
    """Configure validation reference. Returns error dict or None."""
    if not validated_vectors:
        print(
            "[Warning] No vector with validate=true. Validation reference is disabled."
        )
        return None

    if has_cuda_ref and ref_kernel is not None:
        tuner.SetValidationMethod(
            ktt.ValidationMethod.SideBySideComparison, prob["tolerance"]
        )
        for validate_arg_id, _ in validated_vectors:
            tuner.SetReferenceKernel(
                validate_arg_id, ref_kernel, ktt.KernelConfiguration()
            )
        print(
            f"Reference kernel registered for {len(validated_vectors)} validated vector(s)"
        )
    elif prob["reference_type"] == "cpu_c":
        try:
            tuner.SetValidationMethod(
                ktt.ValidationMethod.SideBySideComparison, prob["tolerance"]
            )
            for validate_arg_id, validate_vector_index in validated_vectors:
                callback, reference_size = _build_cpu_reference_callback(
                    source_path=prob["reference_source_path"],
                    function_name=prob["ref_func"],
                    vectors=prob["vectors"],
                    vector_data=vector_data,
                    scalar_defs=prob["scalar_defs"],
                    validated_vector_index=validate_vector_index,
                    build_dir=working_dir,
                )
                tuner.SetReferenceComputation(validate_arg_id, reference_size, callback)
            print(
                f"CPU reference callback registered for {len(validated_vectors)} validated vector(s)"
            )
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to initialize CPU reference computation: {e}",
            }
    else:
        print(
            "[Warning] Validation reference disabled (unknown or missing reference configuration)"
        )
    return None


def run_tuner(
    problem_file: str = "./problem.yaml",
    params_file: str = "./params.json",
    output_file: str = None,
    working_dir: str = None,
    platform_index: int = 0,
    device_index: int = 0,
    tuning_duration_s: float = None,
) -> dict:
    """Run the KTT tuner. Returns dict with 'success' and 'results_file' or 'message'.

    tuning_duration_s is the wall-clock budget passed to KTT's TuningDuration stop
    condition. Falls back to problem.yaml tuning.duration_s. Must be set somewhere —
    this module intentionally has no hard-coded default (the agent wrapper resolves
    against its own system default).
    """
    working_dir = Path(working_dir) if working_dir else Path.cwd()
    if working_dir != Path.cwd():
        print(f"Working directory: {working_dir}")

    try:
        prob = _load_problem(working_dir, problem_file)
    except ValueError as e:
        return {"success": False, "message": str(e)}

    if tuning_duration_s is None:
        tuning_duration_s = prob.get("tuning_duration_s")
    if tuning_duration_s is None:
        return {
            "success": False,
            "message": (
                "No tuning duration set. Pass --tuning-duration or add "
                "`tuning.duration_s:` to problem.yaml."
            ),
        }
    tuning_duration_s = float(tuning_duration_s)

    # Load params.json
    pf = working_dir / params_file
    print(f"\nLoading agent parameters from: {pf}")
    with open(pf, "r") as f:
        params = json.load(f)

    vector_data = _init_data(prob["vectors"])

    # Create tuner and set up components
    tuner = ktt.Tuner(platform_index, device_index, ktt.ComputeApi.CUDA)
    tuner.SetGlobalSizeType(ktt.GlobalSizeType.CUDA)
    tuner.SetTimeUnit(ktt.TimeUnit.Microseconds)

    agent_def, agent_kernel, ref_def, ref_kernel, has_cuda_ref = _setup_kernels(
        tuner, prob
    )
    arg_ids, validated_vectors = _setup_arguments(
        tuner, prob, vector_data, ref_def, has_cuda_ref
    )

    for p in params["parameters"]:
        tuner.AddParameter(agent_kernel, p["name"], p["values"])
        print(f"Parameter: {p['name']} = {p['values']}")

    err = _setup_launcher(tuner, agent_def, agent_kernel, params, prob, arg_ids)
    if err:
        return err

    _setup_constraints(tuner, agent_kernel, params, prob["scalar_values"])

    err = _setup_validation(
        tuner,
        prob,
        validated_vectors,
        has_cuda_ref,
        ref_kernel,
        vector_data,
        working_dir,
    )
    if err:
        return err

    # Run tuning
    if output_file is None:
        output_file = str(working_dir / "results")
    elif not Path(output_file).is_absolute():
        output_file = str(working_dir / output_file)

    # Random sampling instead of exhaustive enumeration — combined with
    # TuningDuration, this lets KTT use the whole budget on diverse configs.
    tuner.SetSearcher(agent_kernel, ktt.RandomSearcher())

    stop = ktt.TuningDuration(tuning_duration_s)
    print(
        f"\n===== Starting Tuning (budget: {tuning_duration_s:.0f}s, searcher: Random) =====\n"
    )
    results = tuner.Tune(agent_kernel, stop)
    tuner.SaveResults(results, output_file, ktt.OutputFormat.JSON)
    print(f"\nResults saved to: {output_file}")

    return {
        "success": True,
        "results_file": output_file,
        "num_configurations": len(results),
    }


def parse_args():
    p = argparse.ArgumentParser(description="KTT Python Tuner for CUDA Kernels")
    p.add_argument("--platform", "-p", type=int, default=0, help="CUDA platform index")
    p.add_argument("--device", "-d", type=int, default=0, help="CUDA device index")
    p.add_argument("--problem", default="./problem.yaml", help="Path to problem.yaml")
    p.add_argument("--params", default="./params.json", help="Path to params.json")
    p.add_argument("--output", "-o", default=None, help="Output file path")
    p.add_argument("--working-dir", "-w", default=None, help="Working directory")
    p.add_argument(
        "--tuning-duration",
        type=float,
        default=None,
        help="Wall-clock tuning budget in seconds (overrides problem.yaml tuning.duration_s)",
    )
    return p.parse_args()


def main():
    """Main entry point for command-line usage."""
    args = parse_args()

    try:
        result = run_tuner(
            problem_file=args.problem,
            params_file=args.params,
            output_file=args.output,
            working_dir=args.working_dir,
            platform_index=args.platform,
            device_index=args.device,
            tuning_duration_s=args.tuning_duration,
        )
        if result["success"]:
            print(
                f"\nTuning completed successfully. Tested {result['num_configurations']} configurations."
            )
            return 0
        else:
            print(f"\nTuning failed: {result.get('message', 'Unknown error')}")
            return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
