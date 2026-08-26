"""File I/O must not depend on the locale.

A real run died on this: launched from tmux without LANG, the interpreter's preferred
encoding was ASCII, and `inputs.hpp` opens with "// inputs.hpp — GENERATED" — an em
dash at byte 14. Every bare `read_text()` in the pipeline raised UnicodeDecodeError,
including the one on the single-shot fallback path, so all four branches died at
iteration 1.

The fix is that no read or write in the engine relies on the locale. These tests fail
if that regresses.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are not the engine: vendored code, the frontend, and the tests
# themselves (which construct their own fixtures and control their own encoding).
SKIP_PARTS = ("KTT", "frontend", "node_modules", ".git", "__pycache__", "tests", "problems")

# Text-mode I/O whose codec comes from the locale unless `encoding` is passed.
TEXT_IO = {"read_text", "write_text", "open"}


def engine_sources():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if any(part in SKIP_PARTS for part in path.relative_to(REPO_ROOT).parts):
            continue
        yield path


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _mode_arg(node: ast.Call):
    """The mode argument of an ``open`` call, or None if there isn't one.

    Its positional index depends on the call form: the builtin is ``open(path, mode)``
    but ``Path.open(mode)`` carries the path as the receiver, so the mode is first.
    ``read_text``/``write_text`` take no mode at all and are never binary.
    """
    if _call_name(node) != "open":
        return None
    index = 1 if isinstance(node.func, ast.Name) else 0
    if len(node.args) > index:
        return node.args[index]
    for kw in node.keywords:
        if kw.arg == "mode":
            return kw.value
    return None


def _is_binary_mode(node: ast.Call) -> bool:
    """A binary-mode open takes no encoding, and must not be flagged for lacking one.

    Scanning from ``args[1]`` regardless of call form read ``Path.open("wb")`` as text
    and demanded an ``encoding`` kwarg that binary mode rejects at runtime.
    """
    mode = _mode_arg(node)
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and "b" in mode.value
    )


def test_no_engine_file_io_depends_on_the_locale():
    """AST-based, so a call whose encoding kwarg sits on a later line still passes and
    a genuinely bare call on any line still fails."""
    offenders = []
    for path in engine_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            kwargs = {kw.arg for kw in node.keywords}

            if name in TEXT_IO:
                # os.open returns a raw fd and takes no encoding — not text I/O.
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "os":
                        continue
                if "encoding" in kwargs or _is_binary_mode(node):
                    continue
                offenders.append(f"{rel}:{node.lineno}: {name}() without encoding=")

            # subprocess(..., text=True) decodes with the locale codec too.
            if any(
                kw.arg == "text"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            ) and "encoding" not in kwargs:
                offenders.append(f"{rel}:{node.lineno}: text=True without encoding=")

    assert not offenders, (
        'locale-dependent I/O found — pass encoding="utf-8" explicitly:\n'
        + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source,binary",
    [
        ('p.open("wb")', True),
        ('p.open("rb")', True),
        ('p.open(mode="wb")', True),
        ('p.open("w", encoding="utf-8")', False),
        ('open(p, "wb")', True),
        ('open(p, mode="rb")', True),
        ('open(p, "w", encoding="utf-8")', False),
        # The path is args[0] for the builtin; a "b" in it is not a mode.
        ('open("blob.txt")', False),
    ],
)
def test_binary_mode_detected_in_both_call_forms(source, binary):
    """Path.open puts the mode where the builtin puts the path, and vice versa."""
    call = ast.parse(source, mode="eval").body
    assert _is_binary_mode(call) is binary


def test_generated_inputs_hpp_really_does_contain_non_ascii(cov_problem):
    """Guards the premise. If the generator stopped emitting non-ASCII this whole
    module would quietly become a test of nothing."""
    from utils.inputs import generate_inputs_hpp, load_inputs_spec

    spec = load_inputs_spec(cov_problem / "inputs.yaml")
    header = generate_inputs_hpp(spec, {"type": "cuda", "function": "f", "file": "r.cu"})
    assert not header.isascii(), "expected the generated header banner to be non-ASCII"
    assert "—" in header


def test_reads_work_under_an_ascii_locale(tmp_path):
    """Reproduce the run's environment and prove a read of the header survives it.

    LC_ALL=C alone is not enough: PEP 538 coerces the C locale to UTF-8, which is why
    the original failure needed a real non-UTF-8 locale rather than a bare `C`.
    PYTHONCOERCECLOCALE=0 disables that coercion, and PYTHONUTF8=0 blocks UTF-8 mode,
    leaving the ASCII default the run actually hit.

    The snippet itself stays pure ASCII — argv is encoded with the filesystem codec,
    which is also ASCII here, so an em dash in the source would fail before the
    interpreter ever started.
    """
    target = tmp_path / "inputs.hpp"
    target.write_text("// inputs.hpp — GENERATED by utils/inputs.py\n", encoding="utf-8")

    snippet = (
        "import locale, sys\n"
        "enc = locale.getpreferredencoding(False)\n"
        "assert 'ascii' in enc.lower() or 'ANSI_X3.4' in enc, enc\n"
        "from pathlib import Path\n"
        "text = Path(sys.argv[1]).read_text(encoding='utf-8')\n"
        "assert '\\u2014' in text\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONUTF8": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-c", snippet, str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


def test_a_bare_read_would_still_fail_under_that_locale(tmp_path):
    """Proves the previous test is testing something.

    If the locale reproduction were wrong, the explicit-encoding read would pass for
    the wrong reason. A bare read must still fail there.
    """
    target = tmp_path / "inputs.hpp"
    target.write_text("// inputs.hpp — GENERATED\n", encoding="utf-8")

    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONUTF8": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-c", "import sys;from pathlib import Path;Path(sys.argv[1]).read_text()", str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.returncode != 0
    assert "UnicodeDecodeError" in proc.stderr


def test_compile_framework_survives_non_ascii_compiler_output(tmp_path):
    """g++ quotes the offending source line back; a kernel comment with µ or an em dash
    then arrives as non-ASCII bytes on stderr. Decoding that with the locale codec is
    what made check_compilation raise instead of reporting a compile error."""
    from utils.build import compile_framework

    (tmp_path / "framework.cpp").write_text(
        '// tile — 305 µs baseline\nthis is not valid c++ — at all\n', encoding="utf-8"
    )
    result = compile_framework(tmp_path, timeout=60.0)
    assert result.ok is False
    assert isinstance(result.stderr, str)  # decoded, not raised


def test_check_compilation_reports_failure_rather_than_raising(filled_workspace, tmp_path):
    """The end-to-end shape of the bug: a tool that raises leaves the model with
    nothing actionable, and the step dead-ends without ever producing a kernel."""
    from agentic.tools import Toolbox

    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "inputs.hpp").write_text("// inputs.hpp — GENERATED\n", encoding="utf-8")
    (problem / "problem.yaml").write_text("grid: {x: 1}\n", encoding="utf-8")

    box = Toolbox(
        filled_workspace,
        branch_path=tmp_path,
        output_dir=tmp_path,
        problem_dir=problem,
        meta={"grid": {"x": 1}, "reference": {"type": "cpu_c", "file": "ref_cpu.c"}},
    )
    output = box.dispatch("check_compilation", {})
    assert "codec" not in output, f"tool raised on an encoding issue: {output}"
    assert box.internal_errors == 0, output
