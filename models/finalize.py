"""Models of the finalize step that utils/ code returns.

They live here, not in state/types.py, because nothing under utils/ imports the
``state`` package and the portability probes (utils/ code) produce these types.
``state.types.Variant`` imports them.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class KernelTags(BaseModel):
    tensor_cores: bool = False
    async_copy: bool = False
    precision: str = "fp32"  # fp32 | fp16 | bf16 | tf32 | fp64
    min_arch: Optional[int] = None  # e.g. 80
    smem_bytes: Optional[int] = None
    sm_scaled: bool = False
    note: str = ""


class ArchReport(BaseModel):
    archs: List[int] = Field(default_factory=list)
    base_min_arch: Optional[int] = None
    winner_min_arch: Optional[int] = None
    per_toggle: Dict[str, Optional[int]] = Field(default_factory=dict)
    skipped: str = ""


class FinalistPick(BaseModel):
    branch: str  # BranchBest.name
    reason: str


class FinalistPicks(BaseModel):
    picks: List[FinalistPick] = Field(default_factory=list)


class OverviewDraft(BaseModel):
    """What the model writes for the run overview. The limits (5 findings, 3 dead ends,
    3 caveats) are applied by code, keeping the first items, so a long reply is trimmed
    rather than failing validation."""

    headline: str
    findings: List[str] = Field(default_factory=list)
    dead_ends: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
