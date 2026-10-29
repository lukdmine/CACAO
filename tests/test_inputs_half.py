"""The `half` dtype on the I/O boundary.

cacao's buffers were fp32/int only, which made any problem whose *contract* is
half-precision inexpressible. KTT was never the blocker: it has
``ArgumentDataType::Half``, ``Source/Tuner.inl`` maps ``half_float::half`` onto it, and
``ResultValidator`` already validates it. The gap was entirely in this repo's codegen.

Two spellings, one buffer. Host-side (inputs.hpp, compiled by g++) a half is
``half_float::half``, the software type KTT vendors and ``Ktt.h`` includes. Device-side
the kernel author writes ``__half`` and includes ``<cuda_fp16.h>``. They are the same two
bytes, and cacao only ever emits the host one -- the kernel signature is not generated.

The compile test at the bottom is the one that matters: everything above it asserts on
generated *text*, and text that looks right can still fail g++.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from models.inputs import InputsSpec
from utils.inputs import generate_inputs_hpp

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBKTT = REPO_ROOT / "libktt.so"
KTT_INCLUDE = REPO_ROOT / "KTT" / "Source"


def _spec(*buffers, scalars=None):
    args = list(scalars or [{"kind": "scalar", "name": "N", "dtype": "int",
                             "value": 256, "placements": ["host", "runtime"]}])
    args += list(buffers)
    return InputsSpec.model_validate({"headers": [], "shared_setup": "", "args": args})


def _buf(name, init, **kw):
    base = {"kind": "buffer", "name": name, "dtype": "half", "size": "N",
            "access": "read", "init": init, "validate": False}
    base.update(kw)
    return base


CUDA_REF = {"type": "cuda", "function": "ref", "file": "ref_kernel.cu"}


def _hpp(*buffers, reference=CUDA_REF, **kw):
    out = _buf("out", "zeros", access="write", validate=True)
    return generate_inputs_hpp(_spec(*buffers, out, **kw), reference, None)


# --- the spec accepts half for buffers, and only for buffers -------------------

def test_buffer_accepts_half():
    spec = _spec(_buf("a", "random"), _buf("out", "zeros", access="write", validate=True))
    assert [b.dtype for b in spec.buffers] == ["half", "half"]


def test_scalar_rejects_half():
    """A half -D macro has no sensible meaning, so the narrowing is deliberate."""
    with pytest.raises(ValueError):
        _spec(_buf("out", "zeros", access="write", validate=True),
              scalars=[{"kind": "scalar", "name": "S", "dtype": "half", "value": 1.0,
                        "placements": ["host"]}])


# --- codegen: every init path spells the host type ----------------------------

def test_half_buffers_use_the_host_type_not_the_device_one():
    text = _hpp(_buf("a", "random"))
    assert "std::vector<half_float::half>" in text
    assert "__half" not in text, "inputs.hpp is host code; __half is the kernel's spelling"


@pytest.mark.parametrize(
    "reference",
    [CUDA_REF,
     {"type": "python", "function": "ref", "file": "ref.py"},
     {"type": "cpu_c", "function": "ref", "file": "ref_cpu.c"}],
    ids=["cuda", "python", "cpu_c"],
)
def test_no_code_path_emits_a_bare_half_as_a_type(reference):
    """The regression that motivated this test.

    Asserting that `half_float::half` is PRESENT is too weak: the generators emit it, so
    a check for it passes even while the reference host copies (`inline const
    std::vector<half> h_a = ...`) and the reference-call scratch buffers still emit the
    bare spec spelling, which names no type on the host and fails g++. Scan for the bad
    token directly, across every reference kind, since each pulls in different emitters.
    """
    text = _hpp(_buf("a", "random"),
                _buf("rw", "zeros", access="readwrite"),
                reference=reference)
    for bad in ("std::vector<half>", "sizeof(half)", "static_cast<half*>",
                "const half*", "<half> ", " half*"):
        assert bad not in text, f"{bad!r} names no host type; expected half_float::half"


def test_zeros_emits_a_half_literal():
    """A bare `0` compiled only via an accidental int->float->half conversion chain."""
    text = _hpp(_buf("c", "zeros"))
    assert "half_float::half(0.0f)" in text


def test_random_draws_in_float_and_narrows():
    """std::uniform_real_distribution requires a floating-point type, and
    half_float::half is a class -- instantiating the distribution on it is undefined."""
    text = _hpp(_buf("a", "random", min=-2, max=2))
    assert "std::uniform_real_distribution<float> d(-2.0f, 2.0f);" in text
    assert "static_cast<half_float::half>(d(rng))" in text
    assert "uniform_real_distribution<half_float::half>" not in text


def test_random_half_defaults_to_the_float_range():
    text = _hpp(_buf("a", "random"))
    assert "d(-1.0f, 1.0f)" in text


def test_file_sizes_by_the_host_type():
    text = _hpp(_buf("d", "file", file_name="d.bin"))
    assert "sizeof(half_float::half)" in text
    # the human-facing message keeps the spec spelling, not the C++ one
    assert "N half elements" in text


def test_int_and_float_buffers_are_unchanged_alongside_half():
    text = _hpp(_buf("a", "random"),
                _buf("i", "random", dtype="int", min=0, max=7),
                _buf("f", "random", dtype="float"))
    assert "std::uniform_int_distribution<int> d(0, 7);" in text
    assert "std::vector<float> gen_f()" in text
    assert "std::uniform_real_distribution<float> d(-1.0f, 1.0f);" in text


# --- the one combination that cannot work -------------------------------------

def test_cpu_c_reference_takes_the_host_type():
    """utils/build.object_command compiles ref_cpu.c with g++ (as C++, per its own
    docstring) and passes -I KTT/Source, so the C++ half type is reachable there. The
    extern "C" declaration therefore names half_float::half, and the reference source
    must include <Utility/External/half.hpp>. Verified by linking one, below."""
    text = _hpp(_buf("a", "random"),
                reference={"type": "cpu_c", "function": "ref", "file": "ref_cpu.c"})
    assert 'extern "C" void ref(const half_float::half* a, half_float::half* out);' in text


def test_python_reference_accepts_half():
    text = _hpp(_buf("a", "random"),
                reference={"type": "python", "function": "ref", "file": "ref.py"})
    assert "std::vector<half_float::half>" in text


def test_dtype_map_covers_every_declared_buffer_dtype():
    """The enum and the numpy map must not drift apart.

    `DTYPE_MAP` sizes the upload byte check (api/problems.py) and marshals the python
    reference. A dtype in the Literal but missing here is a KeyError inside a request
    handler — a 500 where the contract promises a 400.
    """
    import typing

    from models.inputs import BufferSpec
    from utils.python_ref_runner import DTYPE_MAP

    declared = set(typing.get_args(BufferSpec.model_fields["dtype"].annotation))
    assert declared <= set(DTYPE_MAP), f"no numpy dtype for {declared - set(DTYPE_MAP)}"


def test_half_validation_is_bit_exact_not_tolerance_based():
    """Pinned because it is surprising, undocumented in KTT, and shapes problem design.

    KTT's ResultValidator dispatches on std::is_floating_point<T>. That is FALSE for
    half_float::half (it is a class), so a Half argument takes the integral overload,
    `result[i] != reference[i]`, and never reaches ValidateSideBySide. Measured on this
    repo's libktt.so: a candidate one ulp from the reference is rejected at tolerance
    10.0. If a future KTT makes half tolerance-aware, this assertion is where to notice.
    """
    validator = (REPO_ROOT / "KTT" / "Source" / "KernelRunner" / "ResultValidator.inl")
    if not validator.exists():
        pytest.skip("KTT sources not present")
    text = validator.read_text(encoding="utf-8", errors="replace")
    assert "std::is_floating_point<T>()" in text, (
        "ResultValidator no longer dispatches on is_floating_point — re-check whether "
        "half is still compared exactly, and update docs/INPUTS_YAML_GUIDE.md"
    )


def test_numpy_mapping_is_float16():
    """The python reference, the init=file byte check and the upload endpoint all size
    a half through this one map."""
    import numpy as np

    from utils.python_ref_runner import DTYPE_MAP

    assert DTYPE_MAP["half"] is np.float16
    assert np.dtype(DTYPE_MAP["half"]).itemsize == 2


# --- and the part that text assertions cannot prove ---------------------------

def _toolchain_or_skip():
    """Guards match tests/test_integration_compile.py, so the fast suite stays green on
    a bare clone."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not present")
    if not LIBKTT.exists():
        pytest.skip("libktt.so not present")
    if not (KTT_INCLUDE / "Ktt.h").exists():
        pytest.skip("KTT headers not present")


@pytest.mark.integration
@pytest.mark.parametrize(
    "reference",
    [CUDA_REF,
     {"type": "python", "function": "ref", "file": "ref.py"},
     {"type": "cpu_c", "function": "ref", "file": "ref_cpu.c"}],
    ids=["cuda", "python", "cpu_c"],
)
def test_every_reference_kind_generates_compilable_half_code(reference):
    """The text assertions above are necessary but demonstrably not sufficient.

    Each reference kind pulls in a different set of emitters -- host copies, reference-call
    scratch buffers, the input dump, the output read-back -- and two of them shipped a bare
    `half` past a check that only asserted `half_float::half` was present somewhere. Only
    g++ settles it.
    """
    _toolchain_or_skip()
    tmp = Path(tempfile.mkdtemp())
    try:
        text = _hpp(
            _buf("a", "random", min=-2, max=2),
            _buf("rw", "zeros", access="readwrite"),
            _buf("i", "random", dtype="int", min=0, max=7),
            reference=reference,
        )
        (tmp / "inputs.hpp").write_text(text, encoding="utf-8")
        (tmp / "main.cpp").write_text(
            '#include "inputs.hpp"\nint main() { return static_cast<int>(gen_a().size()); }\n',
            encoding="utf-8",
        )
        cmd = ["g++", "-std=c++17", "-m64", "-O0", f"-I{KTT_INCLUDE}",
               "-fsyntax-only", str(tmp / "main.cpp")]
        build = subprocess.run(cmd, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr[-4000:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.integration
def test_cpu_c_reference_links_against_a_half_buffer():
    """The cpu_c contract end to end: g++ builds ref_cpu.c AS C++ with -I KTT/Source, so
    the reference can name half_float::half if it includes the vendored header."""
    _toolchain_or_skip()
    tmp = Path(tempfile.mkdtemp())
    try:
        text = _hpp(_buf("a", "random"),
                    reference={"type": "cpu_c", "function": "ref", "file": "ref_cpu.c"})
        (tmp / "inputs.hpp").write_text(text, encoding="utf-8")
        (tmp / "ref_cpu.c").write_text(
            "#include <Utility/External/half.hpp>\n"
            'extern "C" void ref(const half_float::half* a, half_float::half* out)\n'
            "{ out[0] = half_float::half(static_cast<float>(a[0]) * 2.0f); }\n",
            encoding="utf-8",
        )
        (tmp / "main.cpp").write_text(
            '#include "inputs.hpp"\n'
            "int main() { auto a = gen_a(); auto o = gen_out(); ref(a.data(), o.data());\n"
            "             return static_cast<float>(o[0]) == 0.0f ? 0 : 0; }\n",
            encoding="utf-8",
        )
        obj = subprocess.run(
            ["g++", "-std=c++17", "-m64", "-O0", f"-I{KTT_INCLUDE}", "-c",
             str(tmp / "ref_cpu.c"), "-o", str(tmp / "ref_cpu.o")],
            capture_output=True, text=True)
        assert obj.returncode == 0, obj.stderr[-4000:]
        link = subprocess.run(
            ["g++", "-std=c++17", "-m64", "-O0", f"-I{KTT_INCLUDE}", str(tmp / "main.cpp"),
             str(tmp / "ref_cpu.o"), str(LIBKTT), f"-Wl,-rpath,{REPO_ROOT}",
             "-o", str(tmp / "a.out")],
            capture_output=True, text=True)
        assert link.returncode == 0, link.stderr[-4000:]
        assert subprocess.run([str(tmp / "a.out")]).returncode == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.integration
def test_generated_half_header_actually_compiles():
    """Generate a header touching every half init path, compile it, and run it."""
    _toolchain_or_skip()

    tmp = Path(tempfile.mkdtemp())
    try:
        text = _hpp(
            _buf("a", "random", min=-2, max=2),
            _buf("b", "random"),
            _buf("c", "zeros"),
            _buf("e", "custom",
                 body="return std::vector<half_float::half>("
                      "static_cast<size_t>(N), half_float::half(1.5f));"),
            _buf("i", "random", dtype="int", min=0, max=7),
            _buf("f", "random", dtype="float"),
        )
        (tmp / "inputs.hpp").write_text(text, encoding="utf-8")
        # Touch every generator so none is elided, and print one value per path.
        (tmp / "main.cpp").write_text(
            '#include <cstdio>\n'
            '#include "inputs.hpp"\n'
            "int main() {\n"
            "    auto a = gen_a(); auto c = gen_c(); auto e = gen_e();\n"
            "    auto i = gen_i(); auto f = gen_f(); auto o = gen_out();\n"
            "    std::printf(\"%zu %zu %f %f %f\\n\", sizeof(half_float::half), a.size(),\n"
            "                static_cast<float>(c[0]), static_cast<float>(e[0]),\n"
            "                static_cast<float>(o[0]));\n"
            "    (void)i; (void)f;\n"
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        build = subprocess.run(
            ["g++", "-std=c++17", "-m64", "-O2", f"-I{KTT_INCLUDE}",
             str(tmp / "main.cpp"), str(LIBKTT), f"-Wl,-rpath,{REPO_ROOT}",
             "-o", str(tmp / "a.out")],
            capture_output=True, text=True,
        )
        assert build.returncode == 0, build.stderr[-4000:]

        run = subprocess.run([str(tmp / "a.out")], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]
        size, count, zero, custom, out = run.stdout.split()
        assert size == "2", "half must be two bytes or every init=file byte count is wrong"
        assert count == "256"
        assert float(zero) == 0.0
        assert float(custom) == 1.5
        assert float(out) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
