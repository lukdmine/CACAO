"""Staged file workspace for an agentic authoring step.

The step is atomic from the worker's point of view: the LLM writes into
``<iter_dir>/.staging/`` and nothing lands in the iteration directory until the step
ends successfully. The worker persists state after each node, so a crash mid-step must
not leave a half-written kernels.cu behind that the next entry would treat as the
previous attempt's output.

Only the four LLM-owned files live here. The framework skeleton, inputs.hpp and the
reference sources are engine-owned and are placed in the iteration directory directly,
which is what makes skeleton corruption structurally impossible rather than merely
discouraged.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional


class ToolError(Exception):
    """A precondition breach, surfaced to the model as a tool result.

    Not an exception the loop handles: the whole point is that the model reads the
    message and corrects on the next turn. Removing the tool instead would leave it
    guessing at a shape it cannot observe.
    """


KERNEL_FILE = "kernels.cu"
REGION_FILES = ("region_kernels.cpp", "region_params.cpp", "region_launcher.cpp")
ALL_FILES = (KERNEL_FILE,) + REGION_FILES

# Region file -> key expected by utils.framework.assemble_framework_cpp.
REGION_KEYS = {
    "region_kernels.cpp": "kernels",
    "region_params.cpp": "params",
    "region_launcher.cpp": "launcher",
}

_STAGING_DIRNAME = ".staging"


class Workspace:
    """The four LLM-owned files, staged until the step commits."""

    def __init__(self, iter_dir):
        self.iter_dir = Path(iter_dir)
        self.staging = self.iter_dir / _STAGING_DIRNAME

    # -- lifecycle ---------------------------------------------------------

    def _ensure(self) -> None:
        """Staging must exist before a write. Cheap, and it removes an ordering trap:
        a caller that seeds or writes without reset() first would otherwise fail on
        the filesystem rather than on anything meaningful."""
        self.staging.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Start clean. Stale staging is another attempt's abandoned work."""
        if self.staging.exists():
            shutil.rmtree(self.staging, ignore_errors=True)
        self._ensure()

    def seed(self, contents: Dict[str, str]) -> List[str]:
        """Pre-populate files so the step can edit rather than rewrite.

        Used when an iteration continues from a working kernel: rewriting 400 lines to
        change a tile size is how unrelated parts silently regress.
        """
        self._ensure()
        seeded = []
        for name, text in contents.items():
            if name in ALL_FILES and text:
                (self.staging / name).write_text(text, encoding="utf-8")
                seeded.append(name)
        return seeded

    def commit(self) -> List[str]:
        """Move staged files into the iteration directory. Returns what landed."""
        written = []
        for name in ALL_FILES:
            src = self.staging / name
            if src.exists():
                shutil.copyfile(src, self.iter_dir / name)
                written.append(name)
        shutil.rmtree(self.staging, ignore_errors=True)
        return written

    # -- queries -----------------------------------------------------------

    def path(self, name: str) -> Path:
        return self.staging / name

    def exists(self, name: str) -> bool:
        return (self.staging / name).exists()

    def missing(self) -> List[str]:
        return [n for n in ALL_FILES if not self.exists(n)]

    def complete(self) -> bool:
        return not self.missing()

    def snapshot(self) -> Dict[str, str]:
        return {n: self.read(n) for n in ALL_FILES if self.exists(n)}

    def regions(self) -> Dict[str, str]:
        """Region bodies keyed as assemble_framework_cpp expects."""
        return {
            key: self.read(name)
            for name, key in REGION_KEYS.items()
            if self.exists(name)
        }

    # -- operations --------------------------------------------------------

    def _check_name(self, name: str) -> None:
        if name not in ALL_FILES:
            raise ToolError(
                f"'{name}' is not a file this step owns. "
                f"Writable files: {', '.join(ALL_FILES)}."
            )

    def read(self, name: str) -> str:
        self._check_name(name)
        p = self.staging / name
        if not p.exists():
            raise ToolError(f"{name} has not been written yet.")
        return p.read_text(encoding="utf-8")

    def write(self, name: str, content: str) -> str:
        self._check_name(name)
        if not content or not content.strip():
            raise ToolError(f"Refusing to write empty content to {name}.")
        self._ensure()
        existed = self.exists(name)
        (self.staging / name).write_text(content, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        lines = content.count("\n") + 1
        return f"{verb} {name} ({len(content):,} bytes, {lines} lines)."

    def edit(self, name: str, old: str, new: str) -> str:
        """Exact-match replacement.

        Exact rather than fuzzy, and unique rather than first-match, so an edit either
        applies where the model intended or fails loudly. A half-applied diff to a
        kernel is far more expensive to diagnose than a rejected tool call.
        """
        self._check_name(name)
        if not self.exists(name):
            raise ToolError(
                f"{name} does not exist yet — use write_file to create it first."
            )
        if old == new:
            raise ToolError("old_text and new_text are identical; nothing to do.")
        if not old:
            raise ToolError("old_text must not be empty; use write_file to replace a file.")

        content = self.read(name)
        count = content.count(old)
        if count == 0:
            raise ToolError(
                f"old_text was not found in {name}. Read the file and copy the exact "
                "text, including indentation."
            )
        if count > 1:
            raise ToolError(
                f"old_text occurs {count} times in {name}; it must be unique. "
                "Include surrounding lines to disambiguate."
            )

        (self.staging / name).write_text(content.replace(old, new), encoding="utf-8")
        delta = len(new) - len(old)
        return f"Edited {name} ({delta:+,} bytes)."
