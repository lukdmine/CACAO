"""Generate a problem's inputs.hpp from its structured spec (models/inputs.py).

Mirrors utils/framework.py: structured data -> C++ text, no templating engine. The
spec in inputs.yaml is canonical and inputs.hpp is always regenerated from it, so the
create/edit form never has to parse C++ back into form state.

Target: reproduce problems/mmul/inputs.hpp, a file proven to compile, tune, and
validate 106/106 on the GPU.
"""

from __future__ import annotations

import re
import shlex
import sys
import textwrap
import zlib
from pathlib import Path, PurePosixPath

import yaml

from models.inputs import BufferSpec, InputsSpec, ScalarSpec

# Repo root (parent of utils/) — baked into the generated python-reference command so
# the driver can import utils from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Subdirectory of the problem dir holding binary inputs for init=file buffers.
INPUTS_SUBDIR = "inputs"

_ACCESS = {
    "read": "ReadOnly",
    "write": "WriteOnly",
    "readwrite": "ReadWrite",
}


def _seed(name: str) -> int:
    """Deterministic per-buffer RNG seed.

    crc32, not hash(): Python's hash() is salted per process, which would make a
    problem's input data differ between runs and break reproducibility of timings.
    """
    return zlib.crc32(name.encode()) & 0xFFFFFFFF


def _literal(value, dtype: str) -> str:
    """C++ literal for a spec value. Floats need the f suffix to stay single-precision."""
    if dtype == "float":
        return f"{float(value)}f"
    return str(int(value))


def _scalar_ids(spec: InputsSpec) -> dict:
    """ArgumentId member name per runtime scalar.

    Trailing underscore so the id reads distinctly from the host constexpr of the same
    name (`in.kSizeM_` vs `kSizeM`), matching problems/mmul/inputs.hpp.
    """
    return {s.name: f"{s.name}_" for s in spec.scalars if "runtime" in s.placements}


def _member(arg, scalar_ids: dict) -> str:
    return scalar_ids[arg.name] if arg.kind == "scalar" else arg.name


def _size_expr(spec: InputsSpec, expr: str) -> str:
    """Widen every scalar in a size expression to size_t.

    Casting the whole expression -- static_cast<size_t>(kSizeM * kSizeK) -- widens only
    the RESULT: the multiply still happens in int and can overflow first. Widening each
    operand instead makes the arithmetic itself 64-bit.
    """
    names = sorted(
        (s.name for s in spec.scalars if "host" in s.placements), key=len, reverse=True
    )
    for name in names:
        expr = re.sub(rf"\b{re.escape(name)}\b", f"static_cast<size_t>({name})", expr)
    return expr


def _cpp_escape(text: str) -> str:
    """Escape a string for inclusion in a C++ string literal (path baking)."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _file_path(problem_dir, buf: BufferSpec) -> str:
    """Path baked into the generator. Absolute so the driver (cwd = the iteration
    directory) finds it; inputs.hpp is regenerated at the start of every run, so the
    bake stays fresh. problem_dir=None happens only in the preview endpoint, where the
    header is display-only and a relative path is fine."""
    rel = PurePosixPath(buf.file_name)
    if problem_dir is None:
        return str(PurePosixPath(INPUTS_SUBDIR) / rel)
    return str(Path(problem_dir).resolve() / INPUTS_SUBDIR / rel)


def _generator(spec: InputsSpec, buf: BufferSpec, problem_dir=None) -> str:
    ctype = buf.dtype
    size = _size_expr(spec, buf.size)

    if buf.init == "custom":
        body = textwrap.indent(textwrap.dedent(buf.body).strip(), "    ")
        return f"inline std::vector<{ctype}> gen_{buf.name}() {{\n{body}\n}}"

    if buf.init == "zeros":
        zero = _literal(0, ctype)
        return (
            f"inline std::vector<{ctype}> gen_{buf.name}() {{\n"
            f"    return std::vector<{ctype}>({size}, {zero});\n"
            f"}}"
        )

    if buf.init == "file":
        fpath = _cpp_escape(_file_path(problem_dir, buf))
        # One byte-count check up front covers both element-count mismatch and a file
        # whose size is not a multiple of the dtype — before anything is read, so the
        # read can never overflow the vector.
        return (
            f"inline std::vector<{ctype}> gen_{buf.name}() {{\n"
            f'    std::ifstream f("{fpath}", std::ios::binary | std::ios::ate);\n'
            f"    if (!f)\n"
            f'        throw std::runtime_error("init=file: cannot open {fpath}");\n'
            f"    const std::streamsize nbytes = f.tellg();\n"
            f"    const std::streamsize want = static_cast<std::streamsize>({size})"
            f" * static_cast<std::streamsize>(sizeof({ctype}));\n"
            f"    if (nbytes != want)\n"
            f'        throw std::runtime_error("init=file: {fpath} holds " +'
            f' std::to_string(nbytes) + " bytes, expected " + std::to_string(want) +'
            f' " ({_cpp_escape(buf.size)} {ctype} elements)");\n'
            f"    std::vector<{ctype}> v(static_cast<size_t>(nbytes) / sizeof({ctype}));\n"
            f"    f.seekg(0);\n"
            f"    f.read(reinterpret_cast<char*>(v.data()), nbytes);\n"
            f"    if (f.gcount() != nbytes)\n"
            f'        throw std::runtime_error("init=file: short read on {fpath}");\n'
            f"    return v;\n"
            f"}}"
        )

    lo = _literal(
        buf.min if buf.min is not None else (-1 if ctype == "float" else 0), ctype
    )
    hi = _literal(buf.max if buf.max is not None else 1, ctype)
    dist = (
        "uniform_real_distribution" if ctype == "float" else "uniform_int_distribution"
    )
    return (
        f"inline std::vector<{ctype}> gen_{buf.name}() {{\n"
        f"    std::vector<{ctype}> v({size});\n"
        f"    std::mt19937 rng({_seed(buf.name)}u);\n"
        f"    std::{dist}<{ctype}> d({lo}, {hi});\n"
        f"    for (auto& x : v) x = d(rng);\n"
        f"    return v;\n"
        f"}}"
    )


def scalar_define_flags(spec: InputsSpec) -> list:
    """-DNAME=value for EVERY scalar — the CPU-reference contract: ref_cpu.c takes
    only pointer args and reads scalars as macros, so the host compile of ref_cpu.c
    must receive them all (placements only govern the kernel/NVRTC side)."""
    return [f"-D{s.name}={_literal(s.value, s.dtype)}" for s in spec.scalars]


def _is_cpu_reference(reference) -> bool:
    return bool(reference) and str(reference.get("type", "cuda")).lower() == "cpu_c"


def _is_python_reference(reference) -> bool:
    return bool(reference) and str(reference.get("type", "cuda")).lower() == "python"


def _ref_param(buf: BufferSpec) -> str:
    const = "const " if buf.access == "read" else ""
    return f"{const}{buf.dtype}* {buf.name}"


def _cpu_reference_call(spec: InputsSpec, target: BufferSpec, func: str) -> list:
    """One SetReferenceComputation lambda: KTT hands the validated buffer; inputs come
    from the kept host copies; every other output is scratch."""
    lines = [f"    t.SetReferenceComputation(in.{target.name}, [](void* buffer) {{"]
    args = []
    for b in spec.buffers:
        if b.name == target.name:
            if b.access == "readwrite":
                lines.append(
                    f"        std::memcpy(buffer, cacao_ref::h_{b.name}.data(), "
                    f"cacao_ref::h_{b.name}.size() * sizeof({b.dtype}));"
                )
            args.append(f"static_cast<{b.dtype}*>(buffer)")
        elif b.access == "read":
            args.append(f"cacao_ref::h_{b.name}.data()")
        elif b.access == "readwrite":
            lines.append(
                f"        std::vector<{b.dtype}> c_{b.name} = cacao_ref::h_{b.name};"
            )
            args.append(f"c_{b.name}.data()")
        else:  # write-only, not validated by this lambda: scratch
            lines.append(
                f"        std::vector<{b.dtype}> s_{b.name}({_size_expr(spec, b.size)});"
            )
            args.append(f"s_{b.name}.data()")
    lines.append(f"        {func}({', '.join(args)});")
    lines.append("    });")
    return lines


def _dump_buffer_lines(spec: InputsSpec, indent: str) -> list:
    """C++ lines writing each read/readwrite buffer to ``cacao_in_<name>.bin`` in cwd.

    The buffers come from the already-constructed ``cacao_ref::h_<name>`` statics,
    so this writes the real, deterministic input data — the same buffers the
    tuner registers via AddArgumentVector. Shared by the per-iteration validation
    lambda and the standalone dump tool (utils/inputs.generate_dump_inputs_cpp).
    """
    lines = []
    for b in spec.buffers:
        if b.access in ("read", "readwrite"):
            lines.append(
                f'{indent}{{ std::ofstream f("cacao_in_{b.name}.bin", std::ios::binary);'
                f" f.write(reinterpret_cast<const char*>(cacao_ref::h_{b.name}.data()),"
                f" static_cast<std::streamsize>(cacao_ref::h_{b.name}.size() * sizeof({b.dtype}))); }}"
            )
    return lines


def _python_reference_call(spec: InputsSpec, target: BufferSpec, function_name: str) -> list:
    """SetReferenceComputation lambda: dump input buffers to binary files, invoke
    utils.python_ref_runner, and read the result back into the validated buffer.

    Called once per Tune (cached by KTT), so the subprocess overhead is negligible.
    The runner reads inputs.yaml for metadata (sizes, types, scalar values) and the
    cacao_in_<name>.bin files for actual input data, then writes cacao_ref_<target>.bin.

    All three failure modes (runner exit code, missing output file, short read)
    throw so KTT aborts instead of validating against garbage.
    """
    lines = [f"    t.SetReferenceComputation(in.{target.name}, [](void* buffer) {{"]
    lines += _dump_buffer_lines(spec, indent="        ")
    # Interpreter and repo baked in at codegen time. The driver runs with cwd set to the
    # iteration directory, so a bare `python3 -m utils.python_ref_runner` resolves to
    # whatever python is first on PATH (the base conda env, not the project's) and cannot
    # import utils at all. inputs.hpp is regenerated at the start of every run, so pinning
    # these costs nothing.
    out = f"cacao_ref_{target.name}.bin"
    cmd = (
        f"PYTHONPATH={shlex.quote(str(_REPO_ROOT))} {shlex.quote(sys.executable)} "
        f"-m utils.python_ref_runner --inputs inputs.yaml --ref ref.py "
        f"--function {function_name} "
        f"--target {target.name} --output {out}"
    )
    # Every step is checked. A reference that silently fails does not look like an error:
    # the buffer keeps the zeros it came in with, KTT reports the reference as computed,
    # and then every configuration "differs" from it — so a correct kernel is reported
    # broken and the LLM is sent to debug a bug that does not exist.
    target_size = _size_expr(spec, target.size)
    nbytes = f"static_cast<std::streamsize>({target_size} * sizeof({target.dtype}))"
    lines += [
        f'        const int rc = std::system("{cmd}");',
        "        if (rc != 0)",
        f'            throw std::runtime_error("python reference failed (exit " +'
        f' std::to_string(rc) + "): {target.name}");',
        f'        std::ifstream f("{out}", std::ios::binary);',
        "        if (!f)",
        f'            throw std::runtime_error("python reference wrote no {out}");',
        f"        f.read(static_cast<char*>(buffer), {nbytes});",
        f"        if (f.gcount() != {nbytes})",
        f'            throw std::runtime_error("python reference produced a short {out}");',
        "    });",
    ]
    return lines


def generate_dump_inputs_cpp(spec: InputsSpec) -> str:
    """Emit a standalone ``dump_inputs.cpp`` that writes every input buffer to
    ``cacao_in_<name>.bin`` in the process cwd.

    ``#include``s inputs.hpp, so the ``cacao_ref::h_*`` statics (built by
    ``gen_*()`` at static-init) hold the real, deterministic input data — the
    same buffers the tuner driver registers via AddArgumentVector. Writing them
    out lets the reference timer (utils.torch_ref_timer) run on the real inputs
    without instantiating a ktt::Tuner. No KTT symbols are referenced, so the
    dump tool is a plain g++ build (it links libktt.so only to resolve anything
    Ktt.h pulls in).
    """
    lines = [
        "// dump_inputs.cpp — GENERATED by utils/inputs.py. Do not edit by hand.",
        "// Standalone: writes each input buffer to cacao_in_<name>.bin in cwd.",
        "#include <fstream>",
        '#include "inputs.hpp"',
        "",
        "int main() {",
    ]
    lines += _dump_buffer_lines(spec, indent="    ")
    lines += [
        "    return 0;",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _defines(spec: InputsSpec) -> str:
    """-D macros for `define`-placement scalars, as one compiler-options fragment.

    Carries its OWN leading space when non-empty, and is empty otherwise, so the skeleton
    can concatenate it directly onto the include flag. If the separator lived in the
    skeleton instead, a problem with no `define` scalars (e.g. mmul, whose scalars are
    host+runtime) would leave a trailing space that NVRTC tokenizes into an empty option
    and rejects — failing every configuration.

    Emitted into Inputs.defines rather than passed to SetCompilerOptions from here:
    KTT's SetCompilerOptions REPLACES the option string (Tuner.h:908) and the engine
    skeleton already calls it with the CUDA include path, so a call from inside
    DefineInputs would wipe that include and break every NVRTC compile. The skeleton
    concatenates this string instead.
    """
    parts = [
        f"-D{s.name}={_literal(s.value, s.dtype)}"
        for s in spec.scalars
        if "define" in s.placements
    ]
    return "".join(f" {p}" for p in parts)


def scalar_contract_text(spec: InputsSpec) -> str:
    """How this problem's scalars reach the kernel — as prompt markdown.

    NVRTC compiles kernels.cu standalone: it never sees inputs.hpp, so a scalar is
    visible to a kernel ONLY as a -D macro or as an argument. Which one depends on the
    problem's placements, and getting it wrong is an undefined-identifier compile error.
    A generic prompt cannot state this; it has to be derived per problem.
    """
    macros = [s for s in spec.scalars if "define" in s.placements]
    runtime = [s for s in spec.scalars if "runtime" in s.placements]

    lines = ["## Problem scalars — how THIS problem's kernels must access them"]
    if macros:
        lines.append(
            "- **Compile-time `-D` macros** (KTT passes them to NVRTC; use the name "
            "directly, keep it OUT of the signature): "
            + ", ".join(f"`{s.name}` = {_literal(s.value, s.dtype)}" for s in macros)
        )
    if runtime:
        lines.append(
            "- **Runtime arguments** (NOT macros — each MUST be a kernel parameter, "
            "bound positionally in the next step): "
            + ", ".join(f"`{s.dtype} {s.name}`" for s in runtime)
        )
    host_only = [
        s for s in spec.scalars if "define" not in s.placements and "runtime" not in s.placements
    ]
    if host_only:
        lines.append(
            "- **Host-only** (size the input buffers; NOT reachable from a kernel at all): "
            + ", ".join(f"`{s.name}`" for s in host_only)
        )
    if not spec.scalars:
        lines.append("- This problem declares no scalars.")

    lines.append(
        "\nA `constexpr` in inputs.hpp is compiled into the host driver, not the kernel — "
        "referencing one from a kernel is an undefined identifier."
    )
    return "\n".join(lines)


def generate_inputs_hpp(spec: InputsSpec, reference: dict = None, problem_dir=None) -> str:
    """Render the spec as the inputs.hpp the engine compiles.

    ``reference`` is the problem.yaml reference mapping ({type, function, file}).
    For type cpu_c the generated DefineInputs keeps host copies of the input
    buffers and registers a KTT SetReferenceComputation per validated buffer that
    calls the C function (linked into the driver by utils/build.py). For type
    python the host copies are dumped to binary files and a SetReferenceComputation
    lambda invokes utils.python_ref_runner to compute the reference. The engine
    skeleton stays argument-agnostic — it never learns the reference signature.

    ``problem_dir`` locates the inputs/ directory for init=file buffers; their
    generators bake the absolute path (the driver's cwd is the iteration dir). It
    is None only for the preview endpoint, whose header is display-only.
    """
    scalar_ids = _scalar_ids(spec)
    cpu = _is_cpu_reference(reference)
    python = _is_python_reference(reference)
    ref = cpu or python
    # init=file generators read the problem's inputs/ dir and throw on open/size/read
    # failure — independent of the reference type (a python reference already pulls
    # both includes, so they are skipped for it).
    file_init = any(b.init == "file" for b in spec.buffers)

    out: list[str] = [
        "// inputs.hpp — GENERATED by utils/inputs.py from inputs.yaml. Do not edit by hand.",
        "// Owns the problem's ENTIRE I/O boundary: scalars, data generators, KTT argument",
        "// registration, and which buffers are validated. The engine skeleton is",
        "// argument-agnostic and consumes only the Inputs struct returned by DefineInputs().",
        "#pragma once",
        "#include <string>",
        "#include <vector>",
        "#include <random>",
        "#include <Ktt.h>",
    ]
    if cpu:
        out.append("#include <cstring>")
    if python:
        out.append("#include <cstdlib>")
        out.append("#include <fstream>")
        out.append("#include <cstring>")
        out.append("#include <stdexcept>")  # the reference lambda throws on failure
    if file_init and not python:
        out.append("#include <fstream>")
        out.append("#include <stdexcept>")
    out += [f"#include {h}" for h in spec.headers]

    host = [s for s in spec.scalars if "host" in s.placements]
    if host:
        out.append("")
        out.append(
            "// --- scalars (host consts) ---------------------------------------------------"
        )
        out += [
            f"inline constexpr {s.dtype} {s.name} = {_literal(s.value, s.dtype)};"
            for s in host
        ]

    if spec.shared_setup.strip():
        out.append("")
        out.append(
            "// --- shared setup (coupled/derived data) -------------------------------------"
        )
        out.append(textwrap.dedent(spec.shared_setup).strip())

    if spec.buffers:
        out.append("")
        out.append(
            "// --- data generators (one per buffer) ----------------------------------------"
        )
        out += [_generator(spec, b, problem_dir) for b in spec.buffers]

    if ref:
        out.append("")
        if cpu:
            params = ", ".join(_ref_param(b) for b in spec.buffers)
            out.append(
                "// --- CPU reference (ref_cpu.c, linked into the driver; scalars are -D flags) --"
            )
            out.append(f'extern "C" void {reference["function"]}({params});')
        else:
            out.append(
                "// --- Python reference (ref.py, invoked via utils.python_ref_runner) -------------"
            )
        # The reference lambdas run during Tune, long after DefineInputs returns, so
        # input host copies must outlive it — hence static.
        out.append("namespace cacao_ref {")
        for b in spec.buffers:
            if b.access in ("read", "readwrite"):
                out.append(
                    f"inline const std::vector<{b.dtype}> h_{b.name} = gen_{b.name}();"
                )
        out.append("}")

    # --- Inputs struct ---
    out.append("")
    out.append(
        "// --- boundary exposed to the engine skeleton + the LLM regions ---------------"
    )
    out.append("struct Inputs {")
    for arg in spec.args:
        if arg.kind == "scalar" and arg.name not in scalar_ids:
            continue  # host-const / define-only: no ArgumentId exists
        out.append(f"    ktt::ArgumentId {_member(arg, scalar_ids)};")
    out.append(
        "    std::vector<ktt::ArgumentId> validated; // buffers checked against the reference"
    )
    out.append(
        "    std::vector<ktt::ArgumentId> boundary;  // args in reference-signature order"
    )
    out.append(
        "    std::string defines;                    // -D macros; the skeleton appends these"
    )
    out.append("};")

    # --- DefineInputs ---
    out.append("")
    out.append("inline Inputs DefineInputs(ktt::Tuner& t) {")
    out.append("    Inputs in;")
    for arg in spec.args:
        if arg.kind == "scalar":
            if arg.name in scalar_ids:
                out.append(
                    f"    in.{scalar_ids[arg.name]} = t.AddArgumentScalar({arg.name});"
                )
        elif ref and arg.access in ("read", "readwrite"):
            # Register from the kept host copy so reference lambdas see the same data.
            out.append(
                f"    in.{arg.name} = t.AddArgumentVector(cacao_ref::h_{arg.name}, "
                f"ktt::ArgumentAccessType::{_ACCESS[arg.access]});"
            )
        else:
            out.append(
                f"    in.{arg.name} = t.AddArgumentVector(gen_{arg.name}(), "
                f"ktt::ArgumentAccessType::{_ACCESS[arg.access]});"
            )
    validated = ", ".join(f"in.{b.name}" for b in spec.validated)
    out.append(f"    in.validated = {{{validated}}};")
    boundary = ", ".join(f"in.{_member(a, scalar_ids)}" for a in spec.boundary)
    out.append(f"    in.boundary  = {{{boundary}}};")
    out.append(f'    in.defines   = "{_defines(spec)}";')
    if cpu:
        out.append("")
        out.append(
            "    // CPU reference: one computation per validated buffer (engine skeleton"
        )
        out.append(
            "    // skips SetReferenceKernel when the problem's reference is cpu_c)."
        )
        for target in spec.validated:
            out += _cpu_reference_call(spec, target, reference["function"])
    elif python:
        out.append("")
        out.append(
            "    // Python reference: one computation per validated buffer (engine skeleton"
        )
        out.append(
            "    // skips SetReferenceKernel when the problem's reference is python)."
        )
        for target in spec.validated:
            out += _python_reference_call(spec, target, reference["function"])
    out.append("    return in;")
    out.append("}")

    return "\n".join(out) + "\n"


def load_inputs_spec(path: Path) -> InputsSpec:
    return InputsSpec.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def check_input_files(problem_dir, spec: InputsSpec) -> list:
    """List the init=file buffers whose binary is missing under inputs/.

    Callers decide the severity: the save flow reports these as warnings (the UI
    uploads right after saving); the run-start flow must treat any as fatal —
    better than the driver aborting at static-init minutes later with an
    std::runtime_error dug out of a log.
    """
    problem_dir = Path(problem_dir)
    missing = []
    for b in spec.buffers:
        if b.init == "file" and not (problem_dir / INPUTS_SUBDIR / b.file_name).is_file():
            missing.append(
                f"buffer '{b.name}' (init=file): "
                f"{problem_dir / INPUTS_SUBDIR / b.file_name} does not exist — put or "
                f"upload the binary ({b.size} {b.dtype} elements, raw "
                f"little-endian) in the problem's {INPUTS_SUBDIR}/ directory"
            )
    return missing


def ensure_inputs_hpp(problem_dir) -> Path:
    """Regenerate ``inputs.hpp`` from the canonical ``inputs.yaml``. Call once per run.

    inputs.yaml is the source of truth; inputs.hpp is a build artifact. Regenerating
    unconditionally makes drift impossible — a hand-edited inputs.yaml would otherwise be
    silently ignored, and the run would compile stale input data yet still pass validation
    (the reference computes from the same stale buffers).

    Nothing is lost by overwriting: everything a user authors — generator bodies
    (``init: custom``), ``shared_setup``, ``headers`` — lives in inputs.yaml. The rest of
    the file is the engine's contract (the Inputs struct, KTT registration, reference
    wiring) and must match the skeleton.

    Raises FileNotFoundError if inputs.yaml is missing: the boundary cannot be invented,
    and the LLM cannot author it, so there is nothing downstream can do about it.
    """
    problem_dir = Path(problem_dir)
    inputs_yaml = problem_dir / "inputs.yaml"
    if not inputs_yaml.exists():
        raise FileNotFoundError(
            f"{inputs_yaml} not found — the problem has no I/O boundary definition. "
            "Framework mode generates inputs.hpp from inputs.yaml; define the problem's "
            "inputs (in the UI, or by writing inputs.yaml) before running."
        )

    import yaml as _yaml

    reference = None
    problem_yaml = problem_dir / "problem.yaml"
    if problem_yaml.exists():
        reference = (_yaml.safe_load(problem_yaml.read_text()) or {}).get("reference")

    spec = load_inputs_spec(inputs_yaml)
    missing = check_input_files(problem_dir, spec)
    if missing:
        raise FileNotFoundError("Missing input files: " + "; ".join(missing))
    out = problem_dir / "inputs.hpp"
    out.write_text(generate_inputs_hpp(spec, reference, problem_dir))
    return out


def write_inputs(problem_dir: Path, spec: InputsSpec, reference: dict = None) -> list:
    """Persist the canonical spec and the generated header side by side.

    ``reference`` (the problem.yaml reference mapping) is required for cpu_c
    problems — it shapes the generated header; omitting it yields a CUDA-reference
    header. Callers that regenerate inputs.hpp must pass the problem's reference.

    Returns warnings for init=file buffers whose binary is missing — reported,
    never blocking: the UI uploads binaries right after saving, and
    ensure_inputs_hpp hard-fails at run start if they never arrived.
    """
    problem_dir = Path(problem_dir)
    problem_dir.joinpath("inputs.yaml").write_text(
        yaml.safe_dump(
            spec.model_dump(by_alias=True, exclude_none=True), sort_keys=False
        )
    )
    problem_dir.joinpath("inputs.hpp").write_text(
        generate_inputs_hpp(spec, reference, problem_dir)
    )
    return check_input_files(problem_dir, spec)
