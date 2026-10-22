"""Integration test for tools/library_bench.

Proves the pipeline end to end with no vendor library: the passthrough op
(device memcpy) is benched against a toy problem whose reference kernel
computes the same elementwise copy. GPU-independent stages (codegen, input
dump compile+run, nvcc build of harness+op+launcher) run anywhere nvcc
exists; the run/gate/seed stages require a CUDA device and skip cleanly
when there is none.
"""

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.library_bench import bench as lb
from utils.cuda_env import get_env

_KVEC_N = 131072  # divisible by the 128-thread ref block

_PROBLEM_YAML = """\
name: toy vector copy
gpu:
  index: 0
global_size_type: opencl
grid:
  x: kVecN
  y: '1'
  z: '1'
reference:
  type: cuda
  function: {function}
  file: ref_kernel.cu
  block:
    x: 128
    y: 1
    z: 1
validation:
  tolerance: 0.0
"""

_INPUTS_YAML = f"""\
headers: []
shared_setup: ''
args:
- kind: scalar
  name: kVecN
  dtype: int
  value: {_KVEC_N}
  placements: [host, runtime]
- kind: buffer
  name: x
  dtype: float
  size: kVecN
  access: read
  init: random
  min: -1.0
  max: 1.0
  validate: false
- kind: buffer
  name: y
  dtype: float
  size: kVecN
  access: write
  init: zeros
  validate: true
"""

_REF_KERNEL = """\
extern "C" __global__ void {name}(const int n, const float* x, float* y) {{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] = {expr};
}}
"""


def _toy(tmp_path: Path, function: str, expr: str) -> Path:
    d = tmp_path / "toyvec"
    d.mkdir(parents=True, exist_ok=True)
    (d / "problem.yaml").write_text(_PROBLEM_YAML.format(function=function))
    (d / "inputs.yaml").write_text(_INPUTS_YAML)
    (d / "ref_kernel.cu").write_text(_REF_KERNEL.format(name=function, expr=expr))
    return d


@pytest.fixture()
def toy(tmp_path):
    return _toy(tmp_path, "vec_copy", "x[i]")


@pytest.fixture()
def toy_times_two(tmp_path):
    return _toy(tmp_path, "vec_times_two", "2.0f * x[i]")


def _cuda_device_count() -> int:
    candidates = []
    env = get_env()
    if env.cuda_path:
        candidates.append(str(env.cuda_path / "lib64" / "libcudart.so"))
    candidates += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    for c in candidates:
        try:
            lib = ctypes.CDLL(c)
        except OSError:
            continue
        n = ctypes.c_int(-1)
        try:
            lib.cudaGetDeviceCount(ctypes.byref(n))
        except Exception:
            return -1
        return n.value
    return -1


_NO_CUDA = _cuda_device_count() <= 0
_NO_NVCC = get_env().nvcc is None
needs_cuda = pytest.mark.skipif(_NO_CUDA, reason="no usable CUDA device")
needs_nvcc = pytest.mark.skipif(_NO_NVCC, reason="nvcc not found")


def test_codegen_matches_problem_contract(toy):
    _, cfg, spec = lb._load_problem(toy)

    spec_h = lb.generate_bench_spec_h(spec, cfg)
    assert '"x", "float"' in spec_h and '"y", "float"' in spec_h
    assert "kValidatedIndex = 1" in spec_h  # y, not x
    assert "kGlobalSizeIsOpenCL = 1" in spec_h

    launcher = lb.generate_ref_launch_cu(spec, cfg)
    # boundary order: runtime scalar first, then buffers in inputs.yaml order
    assert ("vec_copy<<<dim3(gx, gy, gz), dim3(bx, by, bz)>>>"
            f"({_KVEC_N}, (float*)d[0], (float*)d[1]);") in launcher

    dump = lb.generate_dump_inputs_cpp(spec)
    assert "gen_x()" in dump
    assert "gen_y()" not in dump  # write-only buffers need no host data


def test_gate_compare_diagnostics(tmp_path):
    ref = np.arange(8, dtype=np.float32)
    r = tmp_path / "r.bin"
    o = tmp_path / "o.bin"
    r.write_bytes(ref.tobytes())
    o.write_bytes(ref.tobytes())
    max_diff, diag = lb._compare_outputs(r, o, "float", 0.0, 8)
    assert diag is None and max_diff == 0.0

    wrong = ref.copy()
    wrong[3] += 1.0
    o.write_bytes(wrong.tobytes())
    max_diff, diag = lb._compare_outputs(r, o, "float", 0.0, 8)
    assert "1/8" in diag and "index 3" in diag and "Refusing to seed" in diag
    assert max_diff == 1.0
    max_diff, diag = lb._compare_outputs(r, o, "float", 2.0, 8)
    assert diag is None and max_diff == 1.0  # tolerance admits it; diff reported

    max_diff, diag = lb._compare_outputs(r, o, "float", 0.0, 16)
    assert "size mismatch" in diag and max_diff == float("inf")


@needs_nvcc
def test_dump_and_compile_without_gpu(toy):
    """Everything up to touching the GPU: inputs.hpp, input dump, nvcc build of
    harness + op + generated launcher."""
    problem_dir, cfg, spec = lb._load_problem(toy)
    scratch = toy / "output" / "_bench"
    scratch.mkdir(parents=True)
    bench_bin = lb._prepare(problem_dir, cfg, spec, "passthrough",
                            lb.OPS_DIR / "passthrough.cu", 0, scratch, get_env())
    assert bench_bin.is_file()
    assert (scratch / "cacao_in_x.bin").stat().st_size == _KVEC_N * 4
    assert not (scratch / "cacao_in_y.bin").exists()  # write-only: not dumped


@needs_cuda
def test_end_to_end_seed_first_write_wins_and_check(toy):
    rc = lb.bench(toy, "passthrough", reps=5, warmup=2, force=False,
                  tolerance=None, device_override=None)
    assert rc == 0
    ref_json = toy / "output" / "reference_time.json"
    t1 = json.loads(ref_json.read_text())["reference_time_us"]
    assert t1 > 0

    sidecar = json.loads((toy / "reference_bench.json").read_text())
    assert sidecar["op"] == "passthrough"
    assert sidecar["reps"] == 5
    assert sidecar["validated_buffer"] == "y"

    # first-write-wins: a second bench must not overwrite the seeded value
    assert lb.bench(toy, "passthrough", 5, 2, False, None, None) == 0
    assert json.loads(ref_json.read_text())["reference_time_us"] == t1

    # --force re-seeds and refreshes the sidecar
    assert lb.bench(toy, "passthrough", 6, 2, True, None, None) == 0
    assert json.loads((toy / "reference_bench.json").read_text())["reps"] == 6

    # --check: OK now, STALE after the input scalar changes
    assert lb.check(toy) == 0
    inputs = yaml.safe_load((toy / "inputs.yaml").read_text())
    inputs["args"][0]["value"] = _KVEC_N // 2
    (toy / "inputs.yaml").write_text(yaml.safe_dump(inputs))
    assert lb.check(toy) == 1


@needs_cuda
def test_gate_refuses_to_seed_when_op_computes_wrong_math(toy_times_two):
    """passthrough copies x; the ref kernel computes 2*x. The gate must fail
    and nothing may be seeded."""
    with pytest.raises(lb.BenchError, match="verification gate FAILED"):
        lb.bench(toy_times_two, "passthrough", 5, 2, False, None, None)
    assert not (toy_times_two / "output" / "reference_time.json").exists()
