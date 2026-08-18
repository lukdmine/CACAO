"""Per-problem rules constraining what a kernel is allowed to do.

Nothing in the pipeline stops a branch from winning the wrong way. A kernel that
drops to fp16 accumulation, or swaps a real reduction for tensor cores at a precision
the problem never permitted, is faster and passes validation whenever the tolerance is
loose enough — and the run reports a speedup that does not mean what it appears to
mean. The reference implementation defines correctness, not intent, so intent has to
be stated somewhere.

Rules live in ``problem.yaml`` under ``rules:``, beside the problem they constrain,
and are re-read every iteration like the rest of that file.

Two kinds, deliberately:

* ``text`` — stated in every prompt that writes or judges a kernel. Guidance the model
  can reason about and apply in cases nobody enumerated.
* ``forbid`` — regular expressions checked against the kernel at compile time. A rule
  that is only asked for is a rule that gets dropped when it competes with a speedup,
  so anything worth enforcing gets teeth here.

Example::

    rules:
      text:
        - "Accumulate in fp32. Reduced-precision accumulation is not a valid optimization."
      forbid:
        - pattern: "wmma::|mma\\\\.sync"
          reason: "tensor cores change the numerics this problem is measuring"
        - pattern: "\\\\b__half\\\\b|nv_bfloat16"
          reason: "fp16/bf16 storage is not permitted"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from utils.log import log


@dataclass
class ForbiddenPattern:
    pattern: str
    reason: str = ""
    _compiled: Optional[re.Pattern] = field(default=None, repr=False)

    def compiled(self) -> Optional[re.Pattern]:
        if self._compiled is None:
            try:
                self._compiled = re.compile(self.pattern)
            except re.error as e:
                log(f"rules.forbid: invalid pattern {self.pattern!r} ({e}) — ignored", "WARN")
                return None
        return self._compiled


@dataclass
class Rules:
    text: List[str] = field(default_factory=list)
    forbid: List[ForbiddenPattern] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text or self.forbid)

    def prompt_block(self) -> str:
        """The rules as they appear in a prompt. Empty when no rules are set.

        Forbidden patterns are shown alongside the prose: a model that knows a check
        exists writes conforming code the first time instead of discovering the
        constraint through a failed compile.
        """
        if not self:
            return ""
        lines = ["## Hard Rules (non-negotiable — these override any optimization)"]
        for item in self.text:
            lines.append(f"- {item}")
        for pattern in self.forbid:
            reason = f" — {pattern.reason}" if pattern.reason else ""
            lines.append(f"- Must not match /{pattern.pattern}/{reason}")
        if self.forbid:
            lines.append(
                "\nThe patterns above are checked automatically; a kernel that matches "
                "one fails its compilation check regardless of correctness or speed."
            )
        return "\n".join(lines)

    def violations(self, kernel_src: str) -> List[str]:
        """Forbidden patterns present in the kernel, as reportable strings."""
        found = []
        for pattern in self.forbid:
            rx = pattern.compiled()
            if rx is None:
                continue
            match = rx.search(kernel_src)
            if match:
                line = kernel_src[: match.start()].count("\n") + 1
                reason = pattern.reason or "forbidden by this problem's rules"
                found.append(
                    f"line {line}: matched /{pattern.pattern}/ ({match.group(0)!r}) — {reason}"
                )
        return found


def parse_rules(meta: Optional[dict]) -> Rules:
    """Read the ``rules:`` block of a parsed problem.yaml.

    A bare list is accepted as shorthand for ``text``, since prose rules are the common
    case and making the simple form require nesting invites it being skipped.
    """
    raw = (meta or {}).get("rules")
    if not raw:
        return Rules()

    if isinstance(raw, str):
        return Rules(text=[raw])
    if isinstance(raw, list):
        return Rules(text=[str(item) for item in raw if str(item).strip()])
    if not isinstance(raw, dict):
        log(f"rules: expected a list or mapping, got {type(raw).__name__} — ignored", "WARN")
        return Rules()

    text_raw = raw.get("text") or []
    if isinstance(text_raw, str):
        text_raw = [text_raw]
    text = [str(item) for item in text_raw if str(item).strip()]

    forbid: List[ForbiddenPattern] = []
    for entry in raw.get("forbid") or []:
        if isinstance(entry, str):
            forbid.append(ForbiddenPattern(pattern=entry))
        elif isinstance(entry, dict) and entry.get("pattern"):
            forbid.append(
                ForbiddenPattern(
                    pattern=str(entry["pattern"]), reason=str(entry.get("reason", ""))
                )
            )
        else:
            log(f"rules.forbid: skipping malformed entry {entry!r}", "WARN")

    return Rules(text=text, forbid=forbid)


def load_rules(problem_yaml: Optional[str] = None, problem_dir=None) -> Rules:
    """Rules for the current problem, from YAML text or the problem directory.

    Never raises: a malformed rules block must not take a run down, and a warning in
    the log is more useful than a crashed branch.
    """
    try:
        if problem_yaml:
            meta = yaml.safe_load(problem_yaml) or {}
        elif problem_dir:
            path = Path(problem_dir) / "problem.yaml"
            if not path.exists():
                return Rules()
            meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            return Rules()
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as e:
        log(f"Could not read rules from problem.yaml: {e}", "WARN")
        return Rules()
    return parse_rules(meta)
