"""Per-problem rules constraining what a kernel is allowed to do.

Declared under ``rules:`` in problem.yaml and re-read every iteration. ``text`` rules
are stated in every prompt that writes or judges a kernel; ``forbid`` regexes are
checked against the kernel before it compiles. forbid exists because a rule that is
only asked for is dropped the moment it competes with a speedup — a kernel that
quietly accumulates in fp16 is faster and still validates whenever the tolerance is
loose enough. See docs/PROBLEM_YAML_GUIDE.md for the schema and an example.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils.log import log


@dataclass
class ForbiddenPattern:
    pattern: str
    reason: str = ""
    _compiled: Optional[re.Pattern] = field(default=None, repr=False)
    error: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Compile at construction, and remember a failure.

        Compiling lazily meant a bad pattern was first noticed deep inside the
        authoring loop, and — because the failure was not cached — re-logged on every
        check. Doing it here reports it once, at problem load.
        """
        if self._compiled is not None:
            return
        try:
            self._compiled = re.compile(self.pattern)
        except re.error as e:
            self.error = str(e)
            log(
                f"rules.forbid: {self.pattern!r} is not a valid regular expression "
                f"({e}) — every kernel check will fail until problem.yaml is fixed",
                "ERROR",
            )

    def compiled(self) -> Optional[re.Pattern]:
        """The compiled pattern, or None when it does not compile."""
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
        """Forbidden patterns present in the kernel, as reportable strings.

        A pattern that does not compile is reported as a violation rather than skipped.
        Skipping disarms the rule while ``prompt_block`` still tells the model it is
        machine-checked, so a typo'd regex becomes a silent failure of enforcement —
        the one outcome this module exists to prevent. Failing every check instead
        makes it a loud failure of the problem, which is fixable.
        """
        found = []
        for pattern in self.forbid:
            rx = pattern.compiled()
            if rx is None:
                found.append(
                    f"problem.yaml declares the forbid pattern /{pattern.pattern}/, "
                    f"which is not a valid regular expression ({pattern.error}). "
                    "Nothing can be checked against it until the rule is fixed."
                )
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
