"""Tag a kernel's winning configuration by what its active source actually uses.

The tags describe the variant that was benchmarked, not the kernel file: a
`#if USE_TC` block that the winning config turns off must not make the winner
look like a tensor-core kernel. `active_lines` therefore replays the
preprocessor for the toggle conditions it can evaluate and scans what survives.
"""
import re
from typing import Dict, List, Optional, Sequence

from models.finalize import ArchReport, KernelTags

# Finalists are forced per value of these (design D7); min_arch joins them only
# when picking a category's best.
CATEGORY_TAGS = ("tensor_cores", "precision")
CATEGORY_BEST_TAGS = ("tensor_cores", "precision", "min_arch")

_TENSOR_PIPE = "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active"

_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
_TOGGLE = r"(USE_[A-Za-z0-9_]+)"
_COND_FORMS = (
    (re.compile(rf"^{_TOGGLE}$"), lambda v, d: bool(v)),
    (re.compile(rf"^!\s*{_TOGGLE}$"), lambda v, d: not v),
    (re.compile(rf"^defined\s*\(\s*{_TOGGLE}\s*\)$"), lambda v, d: d),
    (re.compile(rf"^!\s*defined\s*\(\s*{_TOGGLE}\s*\)$"), lambda v, d: not d),
)
_COMPARE = re.compile(rf"^{_TOGGLE}\s*(==|!=)\s*([01])$")

_TENSOR = re.compile(r"wmma::|mma\.sync|ldmatrix")
_ASYNC = re.compile(r"cp\.async|__pipeline_memcpy_async|cuda::memcpy_async|memcpy_async")
_TF32 = re.compile(r"precision::tf32")
_BF16 = re.compile(r"__nv_bfloat16")
_FP16 = re.compile(r"\b(__half\w*|half2)\b")
_FP64 = re.compile(r"\bdouble\b")


_LITERAL = re.compile(r"^[01]$")
_TOKEN = re.compile(
    r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'""", re.S
)


def _strip_comments(text: str, blank_strings: bool) -> str:
    """Drop `//` and `/* */` comments; optionally empty string literals.

    Comments never count (`// double buffering` is not fp64). String contents
    are kept for the tensor/async scans because inline PTX lives in strings
    (`asm("cp.async...")`), and blanked for precision and sm_scaled where a
    word in a string is not a type.
    """
    def repl(m: "re.Match") -> str:
        tok = m.group(0)
        if tok.startswith("/"):
            return "\n" * tok.count("\n") if tok.startswith("/*") else ""
        return tok[0] + tok[-1] if blank_strings else tok
    return _TOKEN.sub(repl, text)


def _toggle_value(config: Dict[str, str], name: str) -> int:
    try:
        return int(str(config.get(name, 0)).strip())
    except ValueError:
        return 0


def _evaluate(kind: str, expr: str, config: Dict[str, str]) -> Optional[bool]:
    """True/False for a condition on a toggle, None when the scanner cannot know.

    `config` must be the full winning configuration: KTT defines every tuning
    parameter in every configuration (utils/nvrtc._generate_prefix), so a key
    missing from `config` is read as "not defined" and its value as 0.
    """
    expr = re.sub(r"(//|/\*).*$", "", expr).strip()
    if _LITERAL.match(expr) and kind in ("if", "elif"):
        return expr == "1"
    if kind == "ifdef" or kind == "ifndef":
        name = expr.split()[0] if expr.split() else ""
        if not name.startswith("USE_"):
            return None
        defined = name in config
        return defined if kind == "ifdef" else not defined
    for pattern, fn in _COND_FORMS:
        m = pattern.match(expr)
        if m:
            name = m.group(1)
            return fn(_toggle_value(config, name), name in config)
    m = _COMPARE.match(expr)
    if m:
        equal = _toggle_value(config, m.group(1)) == int(m.group(3))
        return equal if m.group(2) == "==" else not equal
    return None


def active_lines(kernel_src: str, config: Dict[str, str]) -> List[str]:
    """Source lines live under `config`, with preprocessor directives dropped.

    Only `USE_X` toggle conditions are evaluated. Anything else (`#if TILE > 16`)
    keeps every branch (`#if 0` / `#if 1` are literal and evaluated), because over-reporting a feature is recoverable from the
    NCU reading and under-reporting is not.
    """
    out: List[str] = []
    # frame: [parent_active, branch_active, a_true_branch_was_taken]
    stack: List[List[bool]] = []
    active = True
    for line in kernel_src.splitlines():
        m = _DIRECTIVE.match(line)
        if not m:
            if active:
                out.append(line)
            continue
        kind, rest = m.group(1), m.group(2)
        if kind in ("if", "ifdef", "ifndef"):
            verdict = _evaluate(kind, rest, config)
            branch = True if verdict is None else verdict
            stack.append([active, branch, verdict is True])
            active = active and branch
        elif kind == "elif" and stack:
            parent, _, taken = stack[-1]
            verdict = _evaluate("if", rest, config)
            branch = (not taken) if verdict is None else (verdict and not taken)
            stack[-1] = [parent, branch, taken or verdict is True]
            active = parent and branch
        elif kind == "else" and stack:
            parent, _, taken = stack[-1]
            stack[-1][1] = not taken
            active = parent and not taken
        elif kind == "endif" and stack:
            active = stack.pop()[0]
    return out


def static_tags(
    kernel_src: str,
    *,
    config: Dict[str, str],
    launcher_src: str = "",
    params_src: str = "",
    buffer_dtypes: Sequence[str] = (),
) -> KernelTags:
    active = "\n".join(active_lines(kernel_src, config))
    text = _strip_comments(active, blank_strings=False)
    code = _strip_comments(active, blank_strings=True)
    if _TF32.search(code):
        precision = "tf32"
    elif _BF16.search(code):
        precision = "bf16"
    elif _FP16.search(code) or "half" in buffer_dtypes:
        precision = "fp16"
    elif _FP64.search(code):
        precision = "fp64"
    else:
        precision = "fp32"
    return KernelTags(
        tensor_cores=bool(_TENSOR.search(text)),
        async_copy=bool(_ASYNC.search(text)),
        precision=precision,
        sm_scaled="GetMaxComputeUnits" in _strip_comments(launcher_src + "\n" + params_src, True),
    )


def full_tags(
    static: KernelTags,
    *,
    smem_bytes: Optional[int],
    arch: Optional[ArchReport],
    ncu_metrics: Optional[dict],
    note: str = "",
) -> KernelTags:
    """Merge the measured facts into the static tags (returns a copy)."""
    tags = static.model_copy(update={
        "smem_bytes": smem_bytes,
        "min_arch": arch.winner_min_arch if arch is not None else None,
        "note": note or static.note,
    })
    # Code can mention wmma and never issue it (a dead path the scanner kept).
    if tags.tensor_cores and ncu_metrics and ncu_metrics.get(_TENSOR_PIPE) == 0:
        tags.tensor_cores = False
        idle = "tensor-core code present, tensor pipe idle"
        tags.note = f"{tags.note}; {idle}" if tags.note else idle
    return tags


def category_keys(tags: KernelTags, keys: Sequence[str]) -> List[str]:
    """`tensor_cores=true`, `precision=fp16`, `min_arch=80`; None values skipped."""
    out: List[str] = []
    for key in keys:
        value = getattr(tags, key)
        if value is None:
            continue
        out.append(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
    return out
