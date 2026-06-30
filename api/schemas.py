"""Pydantic request/response schemas for the API."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ScalarArg(BaseModel):
    name: str
    dtype: str = "int"
    value: int | float

    @field_validator("name")
    @classmethod
    def name_must_be_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(
                f"Scalar name '{v}' must be UPPERCASE (e.g. '{v.upper()}'). "
                "Lowercase names conflict with NVRTC built-in headers."
            )
        return v


class VectorArg(BaseModel):
    model_config = {"populate_by_name": True}

    name: str
    dtype: str = "float"
    size: str  # expression like "M * K"
    access: str = "read"
    init: str = "random"
    init_min: Optional[int | float] = None
    init_max: Optional[int | float] = None
    validate_output: bool = Field(False, alias="validate")


class GpuConfig(BaseModel):
    index: int = 0


class TuningConfig(BaseModel):
    # Wall-clock budget in seconds for one tuner run. Includes KTT's reference
    # computation at start. Prefilled into the Run dialog as the default.
    duration_s: int = 100


class CreateProblemRequest(BaseModel):
    slug: str
    name: str
    description: str
    gpu: Optional[GpuConfig] = None
    tuning: Optional[TuningConfig] = None
    reference_type: Literal["cuda", "cpu_c"] = "cuda"
    ref_function: str = "reference"
    ref_block_x: int = 256
    ref_block_y: int = 1
    ref_block_z: int = 1
    ref_kernel_code: str = ""
    ref_cpu_code: str = ""
    scalars: list[ScalarArg] = []
    vectors: list[VectorArg] = []
    grid_x: str = "N"
    grid_y: str = "1"
    grid_z: str = "1"
    tolerance: float = 0.05


class RunConfig(BaseModel):
    max_iter: int = 5
    max_depth: int = 2
    path_budget: int = 20
    # None = let problem.yaml tuning.duration_s drive the tuner budget (falling back
    # to config.TUNER_TIMEOUT). Set a value to override for this run only.
    timeout: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class BranchMessageRequest(BaseModel):
    content: str


class ChangeDecisionRequest(BaseModel):
    target_iter: int
    content: Optional[str] = None


class BranchConfigRequest(BaseModel):
    max_iter: Optional[int] = None


class CloneProblemRequest(BaseModel):
    new_name: str
