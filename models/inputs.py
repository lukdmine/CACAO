"""Structured spec for a problem's I/O boundary — the canonical form of inputs.hpp.

`inputs.yaml` holds this spec; `utils.inputs.generate_inputs_hpp()` turns it into the
C++ the engine actually reads. The form round-trips through the spec, never through the
generated C++. See docs/superpowers/specs/2026-07-12-inputs-authoring-design.md.
"""

import re
from pathlib import PurePosixPath
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# host    -> inline constexpr in inputs.hpp (sizes the generators; visible to the driver)
# define  -> -D macro handed to NVRTC (compile-time constant inside the kernels)
# runtime -> AddArgumentScalar + an entry in `boundary` (kernel takes it as a parameter)
Placement = Literal["host", "define", "runtime"]


class ScalarSpec(BaseModel):
    kind: Literal["scalar"] = "scalar"
    name: str
    dtype: Literal["int", "float"] = "int"
    value: Union[int, float]
    placements: List[Placement] = Field(default_factory=lambda: ["host", "define"])

    @field_validator("name")
    @classmethod
    def valid_identifier(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"Scalar name '{v}' is not a valid C++ identifier")
        return v

    @model_validator(mode="after")
    def check(self):
        if not self.placements:
            raise ValueError(
                f"Scalar '{self.name}' has no placements — it would be declared nowhere"
            )
        # A -D macro is textually substituted into every NVRTC translation unit, so a
        # lowercase name can collide with a CUDA built-in header identifier. Host consts
        # and runtime args are ordinary C++ and carry no such risk.
        if "define" in self.placements and self.name != self.name.upper():
            raise ValueError(
                f"Scalar '{self.name}' uses the 'define' placement, so its name must be "
                f"UPPERCASE (e.g. '{self.name.upper()}') — lowercase -D macros collide "
                "with NVRTC built-in headers. Drop 'define' to use any identifier."
            )
        return self


class BufferSpec(BaseModel):
    kind: Literal["buffer"] = "buffer"
    name: str
    dtype: Literal["int", "float"] = "float"
    size: str  # C++ expression over host-const scalars, e.g. "kSizeM * kSizeK"
    access: Literal["read", "write", "readwrite"] = "read"
    init: Literal["random", "zeros", "custom", "file"] = "random"
    min: Optional[Union[int, float]] = None  # random only
    max: Optional[Union[int, float]] = None  # random only
    body: Optional[str] = None  # custom only: verbatim C++, must return std::vector<dtype>
    # file only: path relative to problems/<slug>/inputs/. Raw little-endian elements
    # of dtype; byte count must equal size * sizeof(dtype) (checked at driver start).
    file_name: Optional[str] = None
    validate_output: bool = Field(False, alias="validate")

    model_config = {"populate_by_name": True}

    @field_validator("name")
    @classmethod
    def valid_identifier(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"Buffer name '{v}' is not a valid C++ identifier")
        return v

    @model_validator(mode="after")
    def check(self):
        if self.init == "custom" and not (self.body or "").strip():
            raise ValueError(f"Buffer '{self.name}' has init=custom but an empty body")
        if self.init == "file":
            if not (self.file_name or "").strip():
                raise ValueError(
                    f"Buffer '{self.name}' has init=file but an empty file_name"
                )
            parts = PurePosixPath(self.file_name).parts
            if (
                self.file_name.startswith("/")
                or ".." in parts
            ):
                raise ValueError(
                    f"Buffer '{self.name}': file_name '{self.file_name}' must be a "
                    "relative path inside the problem's inputs/ directory"
                )
        if self.init == "random" and self.min is not None and self.max is not None:
            if self.min > self.max:
                raise ValueError(f"Buffer '{self.name}': random min > max")
        if not self.size.strip():
            raise ValueError(f"Buffer '{self.name}' has an empty size expression")
        return self


ArgSpec = Annotated[Union[ScalarSpec, BufferSpec], Field(discriminator="kind")]


class InputsSpec(BaseModel):
    """The whole I/O boundary. `args` is ORDERED: its order is the reference signature."""

    headers: List[str] = Field(default_factory=list)
    shared_setup: str = ""
    args: List[ArgSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def check(self):
        names = [a.name for a in self.args]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate argument names: {sorted(dupes)}")

        validated = [a for a in self.buffers if a.validate_output]
        if not validated:
            raise ValueError(
                "At least one buffer must set validate=true — the buffer(s) compared "
                "against the reference."
            )
        return self

    @property
    def scalars(self) -> List[ScalarSpec]:
        return [a for a in self.args if a.kind == "scalar"]

    @property
    def buffers(self) -> List[BufferSpec]:
        return [a for a in self.args if a.kind == "buffer"]

    @property
    def validated(self) -> List[BufferSpec]:
        """All buffers checked against the reference (at least one)."""
        return [b for b in self.buffers if b.validate_output]

    @property
    def boundary(self) -> List[ArgSpec]:
        """Args passed to the reference kernel, in declaration order.

        Host-const-only scalars are declared but never passed, so they can sit anywhere
        in `args` without disturbing the reference signature.
        """
        return [
            a
            for a in self.args
            if a.kind == "buffer" or "runtime" in a.placements
        ]

    def for_case(self, case) -> "InputsSpec":
        """This boundary with a case's overrides applied. Pure — no I/O, no mutation.

        An override that names nothing declared raises rather than being ignored. A
        typo'd `kt: 127` would otherwise tune every case at the same shape while every
        signal downstream — validation, timings, the geomean — looked perfectly healthy.
        """
        known_scalars = {s.name for s in self.scalars}
        unknown = sorted(set(case.scalars) - known_scalars)
        if unknown:
            raise ValueError(
                f"Case '{case.name}' overrides unknown scalar(s) {unknown}; "
                f"inputs.yaml declares {sorted(known_scalars)}"
            )

        file_buffers = {b.name for b in self.buffers if b.init == "file"}
        not_files = sorted(set(case.files) - file_buffers)
        if not_files:
            raise ValueError(
                f"Case '{case.name}' overrides file_name for {not_files}, which "
                f"is not an init=file buffer; init=file buffers are {sorted(file_buffers)}"
            )

        resolved = self.model_copy(deep=True)
        for arg in resolved.args:
            if arg.kind == "scalar" and arg.name in case.scalars:
                value = case.scalars[arg.name]
                arg.value = int(value) if arg.dtype == "int" else float(value)
            elif arg.kind == "buffer" and arg.name in case.files:
                arg.file_name = case.files[arg.name]
        return resolved

