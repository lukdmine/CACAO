"""A named instantiation of a problem's inputs — the ``cases:`` block in problem.yaml.

Deliberately NOT part of inputs.yaml: ``duration_s`` never appears in inputs.hpp, and
models/inputs.py holds that file to being "the canonical form of inputs.hpp".
problem.yaml already names inputs.yaml scalars in its ``grid:`` expression, so the
cross-file reference is an established pattern rather than a new one.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator

# The name becomes a directory (``case_<name>``), so it is restricted to characters
# that are safe on every filesystem the engine runs on.
_CASE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class CaseSpec(BaseModel):
    """One case: a partial override of the declared inputs, plus its tuning budget."""

    name: str = "default"
    # Scalar value overrides. Keys must name a scalar declared in inputs.yaml; the
    # value is coerced to that scalar's dtype by InputsSpec.for_case.
    scalars: Dict[str, Union[int, float]] = Field(default_factory=dict)
    # file_name overrides for init=file buffers, relative to the problem's inputs/ dir.
    files: Dict[str, str] = Field(default_factory=dict)
    # Falls back to problem.yaml's tuning.duration_s. A tail case exists to catch a
    # correctness bug, not to find an optimum, and does not need the full budget.
    duration_s: Optional[float] = None

    @field_validator("name")
    @classmethod
    def valid_directory_name(cls, v: str) -> str:
        if not _CASE_NAME.match(v):
            raise ValueError(
                f"Case name '{v}' is not a usable directory name — use only letters, "
                "digits, underscore and hyphen"
            )
        return v
