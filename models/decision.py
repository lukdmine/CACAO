"""
Pydantic models for optimization decisions.

Used by the decide_node to structure LLM output.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from config import MAX_STRATEGIES


class ErrorAnalysis(BaseModel):
    error_type: Literal["compilation", "runtime", "validation", "timeout", "none"] = (
        Field(..., description="Category of error encountered")
    )
    root_cause: str = Field(..., description="Identified root cause of the error")
    suggested_fix: str = Field(
        ..., description="Specific fix to apply in the next iteration"
    )


class SubStrategy(BaseModel):
    name: str = Field(
        ..., description="Short identifier for the sub-strategy (e.g., 'larger_tiles')"
    )
    description: str = Field(
        ..., description="Description of what this sub-strategy explores"
    )
    hypothesis: str = Field(
        ..., description="Hypothesis about why this approach might improve performance"
    )
    key_parameters: List[str] = Field(
        default_factory=list,
        description="Key tuning parameters this sub-strategy focuses on",
    )


class OptimizationDecision(BaseModel):
    action: Literal["continue", "retry", "stop", "branch"] = Field(
        ...,
        description=(
            "'continue' - try to improve further; "
            "'retry' - fix errors and try again; "
            "'stop' - optimization complete; "
            "'branch' - split into sub-strategies"
        ),
    )
    reasoning: str = Field(
        ..., description="Why this action was chosen (1-2 sentences)"
    )
    feedback: str = Field(
        ..., description="Actionable instructions for the next iteration"
    )
    error_analysis: Optional[ErrorAnalysis] = Field(
        None, description="Error analysis (required if action is 'retry')"
    )
    sub_strategies: Optional[List[SubStrategy]] = Field(
        None, description="Sub-strategies to explore (required if action is 'branch')"
    )
    skip_implement: bool = Field(
        False,
        description="If true, skip kernel rewrite and only reconfigure tuning parameters",
    )
    iteration_summary: str = Field(
        "",
        description=(
            "One-line summary of what was attempted this iteration and the outcome. "
            "Format: '<what was tried> → <quantitative result> (<brief diagnosis>)'. "
            "Examples: "
            "'prefix sum scan in shared memory → 16/16 validation fail, ~1500µs (scan only covered BLOCK_X elements)' "
            "'4-way ILP accumulator unrolling → 12,410µs, no gain over best 12,411µs' "
            "'fix Hillis-Steele off-by-one → compile error (runtime shared mem size)'"
        ),
    )

    @model_validator(mode="after")
    def enforce_max_sub_strategies(self):
        if (
            self.sub_strategies is not None
            and len(self.sub_strategies) > MAX_STRATEGIES
        ):
            raise ValueError(
                f"Too many sub-strategies: got {len(self.sub_strategies)}, maximum is {MAX_STRATEGIES}"
            )
        return self
