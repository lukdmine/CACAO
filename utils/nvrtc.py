"""Standalone NVRTC compile check for LLM-authored kernels.

The kernel is never host-compiled: KTT NVRTC-compiles it at tune time, so a check
built on ``nvcc`` would answer a different question than the one that matters. This
module binds ``libnvrtc.so`` through ctypes and reproduces what KTT actually does:

* ``KernelConfiguration::GeneratePrefix()`` prepends ``#define NAME VALUE\\n`` per
  tuning parameter (KernelConfiguration.cpp:25, ParameterPair::GetString).
* The driver calls ``SetCompilerOptions("-I<cuda_include>" + in.defines)``. KTT
  appends its own ``--gpu-architecture=compute_XX`` to that (CudaEngine.cpp:741,
  ``overrideDefault`` defaults to false), then splits the whole string on spaces
  (CudaProgram.cpp:48).

No CUDA context is created and no device is touched, so this needs neither the GPU
lock nor a free GPU.

Parameter *constraints* are not evaluated: ``AddConstraint`` lives in the params
region as C++ lambdas. A sampled configuration may therefore be one KTT would never
run, so every result carries the exact defines it used and callers must report which
configuration failed rather than a bare pass/fail.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_NVRTC_SUCCESS = 0

# Candidate sonames, most specific first. ctypes.util.find_library("nvrtc") misses the
# versioned-only installs that ship with a system CUDA (libnvrtc.so.12 with no
# unversioned symlink outside the -dev package).
_SONAMES = ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so.13")

_lib = None
_load_error: Optional[str] = None


def _load() -> Optional[ctypes.CDLL]:
    """Load libnvrtc once. Returns None (never raises) when NVRTC is unavailable."""
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib

    found = ctypes.util.find_library("nvrtc")
    candidates = ([found] if found else []) + list(_SONAMES)
    errors = []
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except OSError as e:
            errors.append(f"{name}: {e}")
            continue
        # Stub libraries (…/stubs/libnvrtc.so) resolve but have no usable entry points.
        if not hasattr(lib, "nvrtcCreateProgram"):
            errors.append(f"{name}: no nvrtcCreateProgram (stub?)")
            continue
        _bind(lib)
        _lib = lib
        return _lib

    _load_error = "; ".join(errors) or "no libnvrtc candidate found"
    return None


def _bind(lib: ctypes.CDLL) -> None:
    lib.nvrtcCreateProgram.restype = ctypes.c_int
    lib.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcCompileProgram.restype = ctypes.c_int
    lib.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
    lib.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.nvrtcGetProgramLog.restype = ctypes.c_int
    lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcDestroyProgram.restype = ctypes.c_int
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]


def available() -> bool:
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    return _load_error or ""


# -------------------------------------------------------------------------
# Parsing the params region
# -------------------------------------------------------------------------

# tuner.AddParameter(kernel, "NAME", std::vector<uint64_t>{2, 4, 8}, "syrk");
# The value list may be uint64_t / int64_t / double. Bool and string parameter types
# exist in KTT but the framework spec does not use them, and a value we cannot turn
# into a #define token is skipped rather than guessed at.
_ADD_PARAM = re.compile(
    r"AddParameter\s*\(\s*[^,]+,\s*\"(?P<name>[A-Za-z_]\w*)\"\s*,\s*"
    r"std::vector\s*<\s*(?P<type>[\w:]+)\s*>\s*\{(?P<values>[^}]*)\}",
    re.MULTILINE,
)


def parse_parameters(params_src: str) -> Dict[str, List[str]]:
    """Extract ``{name: [value, ...]}`` from a CACAO:PARAMS region body.

    Values stay strings: they are pasted into ``#define NAME VALUE`` verbatim, exactly
    as KTT's ParameterPair::GetValueString does, so no numeric round-trip can alter
    them.
    """
    params: Dict[str, List[str]] = {}
    for m in _ADD_PARAM.finditer(params_src):
        raw = m.group("values")
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if values:
            params[m.group("name")] = values
    return params


def has_constraints(params_src: str) -> bool:
    return "AddConstraint" in params_src


def parse_defines_from_inputs_hpp(inputs_hpp: str) -> List[str]:
    """The problem's -D scalar macros, as the driver passes them to NVRTC.

    inputs.hpp assigns them as a single string literal: ``in.defines = " -DA=1 -DB=2";``
    Reading the generated artifact keeps this honest — deriving them again from
    inputs.yaml would be a second implementation that can disagree with the first.
    """
    m = re.search(r"\.defines\s*=\s*\"([^\"]*)\"", inputs_hpp)
    if not m:
        return []
    return [tok for tok in m.group(1).split(" ") if tok]


def arch_option(compute_capability: Optional[str]) -> str:
    """``--gpu-architecture=compute_XY`` from a "8.6"-style capability string.

    Mirrors CudaEngine::GetDefaultCompilerOptions, which concatenates major and minor
    with no separator. Falls back to compute_52 (NVRTC's own default) when the
    capability is unknown, which is the conservative choice: it under-reports feature
    availability rather than claiming a kernel compiles for hardware it does not.
    """
    if not compute_capability:
        return "--gpu-architecture=compute_52"
    digits = re.findall(r"\d+", str(compute_capability))
    if len(digits) >= 2:
        return f"--gpu-architecture=compute_{digits[0]}{digits[1]}"
    if len(digits) == 1:
        return f"--gpu-architecture=compute_{digits[0]}0"
    return "--gpu-architecture=compute_52"


# -------------------------------------------------------------------------
# Compiling
# -------------------------------------------------------------------------


@dataclass
class NvrtcCheck:
    """One compile attempt at one concrete configuration."""

    label: str
    config: Dict[str, str]
    ok: bool
    log: str
    options: List[str] = field(default_factory=list)

    @property
    def defines(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.config.items()) or "(no parameters)"


def _generate_prefix(config: Dict[str, str]) -> str:
    """KernelConfiguration::GeneratePrefix — one ``#define NAME VALUE`` per pair."""
    return "".join(f"#define {name} {value}\n" for name, value in config.items())


_LINE_REF = re.compile(r"\((\d+)\)")


def _shift_line_numbers(log: str, offset: int) -> str:
    """Map diagnostics back onto kernels.cu.

    The prefix shifts every line by the parameter count, so NVRTC reports positions
    that do not exist in the file the model is editing. Same correction as
    utils.results.kernel_line_offset applies to KTT's own output.
    """
    if offset <= 0:
        return log

    def sub(m: re.Match) -> str:
        n = int(m.group(1)) - offset
        return f"({n})" if n > 0 else "(prefix)"

    return _LINE_REF.sub(sub, log)


def compile_source(
    kernel_src: str,
    config: Dict[str, str],
    options: List[str],
    label: str = "config",
) -> NvrtcCheck:
    """Compile one configuration. Never raises — an unusable NVRTC is a skipped check."""
    lib = _load()
    if lib is None:
        return NvrtcCheck(
            label=label,
            config=config,
            ok=True,
            log=f"NVRTC unavailable ({unavailable_reason()}) — kernel compile check skipped.",
            options=options,
        )

    prefix = _generate_prefix(config)
    source = prefix + kernel_src
    offset = len(config)

    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(
        ctypes.byref(prog),
        source.encode("utf-8"),
        b"kernels.cu",
        0,
        None,
        None,
    )
    if rc != _NVRTC_SUCCESS:
        return NvrtcCheck(
            label=label,
            config=config,
            ok=False,
            log=f"nvrtcCreateProgram failed: {lib.nvrtcGetErrorString(rc).decode()}",
            options=options,
        )

    try:
        encoded = [o.encode("utf-8") for o in options]
        arr = (ctypes.c_char_p * len(encoded))(*encoded) if encoded else None
        rc = lib.nvrtcCompileProgram(prog, len(encoded), arr)

        size = ctypes.c_size_t()
        log = ""
        if lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size)) == _NVRTC_SUCCESS and size.value > 1:
            buf = ctypes.create_string_buffer(size.value)
            if lib.nvrtcGetProgramLog(prog, buf) == _NVRTC_SUCCESS:
                log = buf.value.decode("utf-8", errors="replace")

        return NvrtcCheck(
            label=label,
            config=config,
            ok=(rc == _NVRTC_SUCCESS),
            log=_shift_line_numbers(log.strip(), offset),
            options=options,
        )
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))


def sample_configurations(params: Dict[str, List[str]]) -> List[tuple]:
    """(label, config) pairs to compile: the extremes of each parameter's range.

    Two points, not the full cross product. The minimum configuration is the one most
    likely to be constraint-valid; the maximum is where shared-memory and register
    sizing derived from parameters actually overflows. A middle sample adds compile
    time without covering a distinct failure mode.
    """
    if not params:
        return [("default", {})]
    lo = {name: values[0] for name, values in params.items()}
    hi = {name: values[-1] for name, values in params.items()}
    if lo == hi:
        return [("only configuration", lo)]
    return [("minimum values", lo), ("maximum values", hi)]


def check_kernel(
    kernel_src: str,
    params_src: str,
    *,
    cuda_include: str,
    scalar_defines: Optional[List[str]] = None,
    compute_capability: Optional[str] = None,
) -> List[NvrtcCheck]:
    """NVRTC-compile the kernel at sampled configurations, as KTT would.

    Returns one NvrtcCheck per sampled configuration. An empty params region is not an
    error — a kernel with no tuning parameters compiles with no prefix.
    """
    params = parse_parameters(params_src)
    options = [f"-I{cuda_include}"] + list(scalar_defines or []) + [arch_option(compute_capability)]
    return [
        compile_source(kernel_src, config, options, label)
        for label, config in sample_configurations(params)
    ]


def format_checks(checks: List[NvrtcCheck], constrained: bool = False) -> str:
    """Render results for a tool response. Failures lead; passes are one line."""
    if not checks:
        return "No configuration was compiled (no parameters parsed and no default)."

    failed = [c for c in checks if not c.ok]
    lines: List[str] = []

    if not failed:
        lines.append(f"NVRTC: PASS at all {len(checks)} sampled configuration(s).")
        for c in checks:
            lines.append(f"- {c.label}: {c.defines}")
        # Warnings still matter (unused variable, implicit conversion) even on success.
        for c in checks:
            if c.log:
                lines.append(f"\nWarnings ({c.label}):\n{c.log}")
        return "\n".join(lines)

    lines.append(f"NVRTC: FAIL at {len(failed)} of {len(checks)} sampled configuration(s).")
    for c in failed:
        lines.append(f"\n--- FAILED: {c.label} ---")
        lines.append(f"Defines: {c.defines}")
        lines.append(c.log or "(no diagnostics returned)")
    for c in checks:
        if c.ok:
            lines.append(f"\nPASSED: {c.label} ({c.defines})")

    if constrained:
        lines.append(
            "\nNote: this params region declares AddConstraint. Sampled configurations "
            "are parameter-range extremes and are not checked against those "
            "constraints, so a failure at a configuration your constraints exclude is "
            "not a real defect — confirm the failing configuration is one KTT would "
            "actually run before changing the kernel."
        )
    return "\n".join(lines)
