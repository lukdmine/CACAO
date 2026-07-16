"""Pydantic request/response schemas for the API."""

from typing import Optional, Literal
from pydantic import BaseModel, Field

from models.inputs import InputsSpec


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
    # Both kinds run (spec D6), but they take different arguments: a cuda reference is a
    # kernel over the boundary (buffers + runtime scalars); a cpu_c reference is a C
    # function linked into the driver, taking every buffer as a pointer with scalars as
    # -D macros. block_* applies to cuda only.
    reference_type: Literal["cuda", "cpu_c"] = "cuda"
    ref_function: str = "reference"
    ref_block_x: int = 256
    ref_block_y: int = 1
    ref_block_z: int = 1
    ref_kernel_code: str = ""
    ref_cpu_code: str = ""
    # The I/O boundary. Canonical: persisted to inputs.yaml, and inputs.hpp is generated
    # from it. Never parsed back out of the generated C++.
    inputs: InputsSpec = Field(default_factory=InputsSpec)
    # OpenCL: grid is total work-items (KTT divides by the per-config local size).
    # CUDA:   grid is the number of blocks. Getting this wrong launches a GEMM with
    #         2048x2048 blocks instead of work-items, so it must be explicit.
    global_size_type: Literal["cuda", "opencl"] = "cuda"
    grid_x: str = "N"
    grid_y: str = "1"
    grid_z: str = "1"
    tolerance: float = 0.05


class PreviewInputsRequest(BaseModel):
    inputs: InputsSpec
    # The reference shapes the generated header: a cpu_c problem gets an extern "C"
    # declaration and a SetReferenceComputation per validated buffer. Without these the
    # preview would show CUDA-shaped output for a C-reference problem.
    reference_type: Literal["cuda", "cpu_c"] = "cuda"
    ref_function: str = "reference"


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
