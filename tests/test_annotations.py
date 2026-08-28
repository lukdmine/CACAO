"""No parameter may be annotated with a non-Optional type and default to None.

PEP 484 originally made this legal — a `None` default implicitly widened the
annotation to Optional — and mypy shipped `--implicit-optional` on by default until
0.990. It is not legal any more, and it was always ambiguous to a reader: `foo: str =
None` gives no way to tell a deliberate "unset" state from a mistake.

Python enforces none of this at runtime, and this repo configures no type checker, so
without a test the style simply returns.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {
    "KTT",
    "node_modules",
    "dist",
    ".git",
    "__pycache__",
    "problems",
    "docs",
    ".claude",
    "archive",
    "output",
}


def _sources():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        yield rel, path


def _is_optional(annotation) -> bool:
    """Whether an annotation admits None.

    Covers ``Optional[X]``, ``Union[X, None]``, the ``X | None`` form, a bare ``None``
    return-style annotation, and ``Any``.
    """
    if annotation is None:
        return True
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return True
    if isinstance(annotation, ast.Name) and annotation.id == "Any":
        return True
    # `X | None`
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_optional(annotation.left) or _is_optional(annotation.right)
    if isinstance(annotation, ast.Subscript):
        base = annotation.value
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name == "Optional":
            return True
        if name == "Union":
            elts = getattr(annotation.slice, "elts", [])
            return any(_is_optional(e) for e in elts)
    # A string annotation, e.g. "Optional[str]" under postponed evaluation.
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return "Optional" in annotation.value or "None" in annotation.value
    return False


def implicit_optionals():
    """(location, parameter) for every non-Optional annotation defaulting to None."""
    for rel, path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - not first-party code
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            # Positional/keyword args pair with defaults right-aligned; kw-only pair 1:1.
            positional = args.posonlyargs + args.args
            paired = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
            paired += [
                (a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
            ]
            for arg, default in paired:
                if not isinstance(default, ast.Constant) or default.value is not None:
                    continue
                if arg.annotation is None or _is_optional(arg.annotation):
                    continue
                yield f"{rel}:{arg.lineno}", f"{node.name}({arg.arg}=None)"


def test_no_implicit_optional_annotations():
    found = [f"{where}: {what}" for where, what in implicit_optionals()]
    assert not found, (
        "annotate these Optional[...] — a None default does not widen the type any "
        "more:\n  " + "\n  ".join(found)
    )
