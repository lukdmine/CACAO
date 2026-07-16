"""Invoked via system() from the C++ driver's SetReferenceComputation lambda.

Reads inputs.yaml for metadata (dtype, size, scalar values) and the binary
cacao_in_<name>.bin files that the C++ lambda wrote from its host copies, calls
the reference function from ref.py, and writes the target buffer to
cacao_ref_<target>.bin for the C++ lambda to read back.

Usage:
    python3 -m utils.python_ref_runner \
        --inputs inputs.yaml --ref ref.py \
        --target mat_c --output cacao_ref_mat_c.bin
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import yaml

_DTYPE_MAP = {"float": np.float32, "int": np.int32}


def _load_ref_function(ref_path: Path) -> callable:
    """Load the python reference file and return its solitary public callable.

    Convention: ref.py exposes ONE top-level function (the reference).
    It may also expose type stubs or helpers; we pick the first callable that is
    not dunder and not imported from another module."""

    spec = importlib.util.spec_from_file_location("ref", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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


def _eval_size(size_expr: str, scalars: dict) -> int:
    """Evaluate a size expression like 'kSizeM * kSizeK' using scalar values."""
    # Only allow simple arithmetic and known scalar names
    ns = {**scalars}
    return int(eval(size_expr, {"__builtins__": {}}, ns))


def run(inputs_yaml: Path, ref_py: Path, target: str, output: Path) -> None:
    spec = yaml.safe_load(inputs_yaml.read_text())
    args = spec.get("args", [])

    scalars = {}
    buffers = {}
    sizes = {}
    dtypes = {}
    accesses = {}
    target_entry = None

    for a in args:
        if a["kind"] == "scalar":
            scalars[a["name"]] = a["value"]
        elif a["kind"] == "buffer":
            sizes[a["name"]] = a["size"]
            dtypes[a["name"]] = a["dtype"]
            accesses[a["name"]] = a["access"]

    # Load input buffers from binary files written by the C++ lambda
    for name, access in accesses.items():
        dtype = _DTYPE_MAP[dtypes[name]]
        size = _eval_size(sizes[name], scalars)
        if access in ("read", "readwrite"):
            in_file = Path(f"cacao_in_{name}.bin")
            if in_file.exists():
                raw = in_file.read_bytes()
                arr = np.frombuffer(raw, dtype=dtype).copy()
                if len(arr) != size:
                    raise RuntimeError(
                        f"Buffer '{name}': expected {size} elements of {dtype.__name__} "
                        f"({size * np.dtype(dtype).itemsize} bytes) but "
                        f"cacao_in_{name}.bin is {len(raw)} bytes ({len(arr)} elements)."
                    )
                buffers[name] = arr
            else:
                raise FileNotFoundError(
                    f"cacao_in_{name}.bin not found for read-buffer '{name}'. "
                    "The C++ driver should write this file before invoking the runner."
                )
        else:
            buffers[name] = np.zeros(size, dtype=dtype)

        if name == target:
            target_entry = {"name": name, "dtype": dtype, "size": size}

    if target_entry is None:
        raise ValueError(f"Target buffer '{target}' not found in inputs.yaml args.")

    # Check if the ref uses a different function name via the problem.yaml reference
    ref_func = _load_ref_function(ref_py)

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
    p.add_argument("--target", required=True, help="buffer name to validate")
    p.add_argument("--output", required=True, help="output binary file path")
    args = p.parse_args()

    run(
        inputs_yaml=Path(args.inputs),
        ref_py=Path(args.ref),
        target=args.target,
        output=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
