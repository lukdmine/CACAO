"""Reading problem.yaml.

The reference stanza was parsed at a dozen sites with a hardcoded default filename
each, and they disagreed: "ref_kernel.cu" regardless of type in the CLI and API, and
"" in the worker — which silently handed every prompt an empty reference kernel
whenever reference.file was omitted.
"""

from pathlib import Path
from typing import NamedTuple

import yaml

_REF_DEFAULTS = {"cuda": "ref_kernel.cu", "cpu_c": "ref_cpu.c", "python": "ref.py"}


class Reference(NamedTuple):
    type: str
    file: str
    function: str


def load_problem_yaml(problem_dir) -> dict:
    """problem.yaml as a dict; ``{}`` when it is missing or unparseable."""
    try:
        text = (Path(problem_dir) / "problem.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError):
        return {}


def reference_spec(meta: dict) -> Reference:
    """The reference stanza of a parsed problem.yaml, with per-type defaults."""
    ref = (meta or {}).get("reference") or {}
    ref_type = str(ref.get("type", "cuda")).lower()
    return Reference(
        type=ref_type,
        file=ref.get("file") or _REF_DEFAULTS.get(ref_type, "ref_kernel.cu"),
        function=ref.get("function", ""),
    )


def reference_source(problem_dir) -> str:
    """The reference implementation's source text, or "" if it cannot be read."""
    ref = reference_spec(load_problem_yaml(problem_dir))
    try:
        return (Path(problem_dir) / ref.file).read_text(encoding="utf-8")
    except OSError:
        return ""
