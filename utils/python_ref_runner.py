"""Invoked via system() from the C++ driver's SetReferenceComputation lambda.

Reads inputs.yaml for metadata (dtype, size, scalar values) and the binary
cacao_in_<name>.bin files that the C++ lambda wrote from its host copies, calls
the reference function from ref.py, and writes the target buffer to
cacao_ref_<target>.bin for the C++ lambda to read back.

Usage:
    python3 -m utils.python_ref_runner \
        --inputs inputs.yaml --ref ref.py \
        --function gemm_reference \
        --target mat_c --output cacao_ref_mat_c.bin
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import yaml

DTYPE_MAP = {"float": np.float32, "int": np.int32}


def load_ref_module(ref_path: Path):
    """Load ref.py as a module and return it (executes module-level code)."""
    spec = importlib.util.spec_from_file_location("ref", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_ref_function(
    mod, ref_path: Path, function_name: str | None = None
) -> callable:
    """Resolve the reference callable from an already-loaded module.

    If ``function_name`` is given, look it up by name and verify it is callable
    and defined in ref.py. Otherwise fall back to the first non-dunder callable
    that is not imported from another module, preserving backwards-compatible
    direct CLI use.
    """
    if function_name:
        obj = getattr(mod, function_name, None)
        if callable(obj) and getattr(obj, "__module__", None) == "ref":
            return obj
        raise RuntimeError(
            f"Reference function '{function_name}' not found in {ref_path}. "
            f"Ensure ref.py defines a top-level callable named '{function_name}'."
        )

    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if callable(obj) and getattr(obj, "__module__", None) == "ref":
            return obj
    raise RuntimeError(
        f"No public function found in {ref_path}. "
        "ref.py must export a callable, e.g. 'def matmul(scalars, buffers) -> np.ndarray:'."
    )


def _load_ref_function(ref_path: Path, function_name: str | None = None) -> callable:
    """Load the python reference file and return the requested callable."""
    return _resolve_ref_function(load_ref_module(ref_path), ref_path, function_name)


def eval_size(size_expr: str, scalars: dict) -> int:
    """Evaluate a size expression like 'kSizeM * kSizeK' using scalar values."""
    # Only allow simple arithmetic and known scalar names
    ns = {**scalars}
    return int(eval(size_expr, {"__builtins__": {}}, ns))


def load_inputs(inputs_yaml: Path, bin_dir: Path) -> tuple[dict, dict, dict]:
    """Load (scalars, buffers, meta) from inputs.yaml + cacao_in_<name>.bin files.

    The .bin files are written beforehand under ``bin_dir`` — by the C++ driver
    during tuning (its cwd; per-iteration validation path) or by the standalone
    dump tool (reference-timing path, see utils.inputs.generate_dump_inputs_cpp).
    read/readwrite buffers hold the real dumped data, write buffers are zeros;
    meta maps buffer name -> {"dtype", "size"} so callers can validate results.
    """
    spec = yaml.safe_load(Path(inputs_yaml).read_text()) or {}
    args = spec.get("args", [])
    scalars = {a["name"]: a["value"] for a in args if a["kind"] == "scalar"}
    buffers = {}
    meta = {}
    for a in args:
        if a["kind"] != "buffer":
            continue
        name = a["name"]
        dtype = DTYPE_MAP[a["dtype"]]
        size = eval_size(a["size"], scalars)
        if a.get("access", "read") in ("read", "readwrite"):
            in_file = bin_dir / f"cacao_in_{name}.bin"
            if not in_file.exists():
                raise FileNotFoundError(
                    f"{in_file} not found for read-buffer '{name}'. It is written "
                    "beforehand by the C++ driver or the standalone dump tool."
                )
            raw = in_file.read_bytes()
            arr = np.frombuffer(raw, dtype=dtype).copy()
            if len(arr) != size:
                raise RuntimeError(
                    f"Buffer '{name}': expected {size} elements of {dtype.__name__} "
                    f"({size * np.dtype(dtype).itemsize} bytes) but "
                    f"{in_file} is {len(raw)} bytes ({len(arr)} elements)."
                )
            buffers[name] = arr
        else:
            buffers[name] = np.zeros(size, dtype=dtype)
        meta[name] = {"dtype": dtype, "size": size}
    return scalars, buffers, meta


def run(
    inputs_yaml: Path, ref_py: Path, target: str, output: Path, function_name: str | None = None
) -> None:
    # The C++ lambda dumps the bins into the driver's cwd, which is our cwd.
    scalars, buffers, meta = load_inputs(inputs_yaml, Path("."))
    if target not in meta:
        raise ValueError(f"Target buffer '{target}' not found in inputs.yaml args.")
    target_entry = {"name": target, **meta[target]}

    mod = load_ref_module(ref_py)
    ref_func = _resolve_ref_function(mod, ref_py, function_name)
    prepare = getattr(mod, "prepare_input", None)

    if prepare is not None:
        prepared = prepare(scalars, buffers)
        result = ref_func(prepared)
        if hasattr(result, "cpu"):  # device tensor -> host numpy
            result = result.cpu().numpy()
    else:
        result = ref_func(scalars, buffers)

    if not isinstance(result, np.ndarray):
        result = np.asarray(result, dtype=target_entry["dtype"])

    expected_size = target_entry["size"]
    if result.size != expected_size:
        raise RuntimeError(
            f"ref.py returned {result.size} elements for buffer '{target}' "
            f"but inputs.yaml says {expected_size}."
        )

    result.astype(target_entry["dtype"]).tofile(output)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Python reference runner (invoked from C++ driver)"
    )
    p.add_argument("--inputs", required=True, help="path to inputs.yaml")
    p.add_argument("--ref", required=True, help="path to ref.py")
    p.add_argument(
        "--function", default=None, help="name of the reference function in ref.py"
    )
    p.add_argument("--target", required=True, help="buffer name to validate")
    p.add_argument("--output", required=True, help="output binary file path")
    args = p.parse_args()

    run(
        inputs_yaml=Path(args.inputs),
        ref_py=Path(args.ref),
        target=args.target,
        output=Path(args.output),
        function_name=args.function,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
