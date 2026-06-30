"""
Pydantic models for strategy extraction.

Used by the strategize_node to structure LLM output.
"""

from typing import List

from pydantic import BaseModel, Field, model_validator

from config import MAX_STRATEGIES


class Strategy(BaseModel):
    """
    A discrete optimization strategy to explore.

    Each strategy represents a fundamentally different approach
    that warrants its own optimization branch.
    """

    name: str = Field(
        ...,
        description="Short identifier for the strategy (e.g., 'shared_mem_tiling', 'tensor_core')",
        min_length=1,
        max_length=50,
    )
    description: str = Field(
        ..., description="Detailed explanation of what this optimization strategy does"
    )
    hypothesis: str = Field(
        ..., description="Why this strategy might improve performance"
    )
    key_parameters: List[str] = Field(
        ...,
        description="Main tuning parameters this strategy introduces (e.g., ['TILE_M', 'TILE_N'])",
    )


class StrategizeOutput(BaseModel):
    """
    Output of the strategize node.

    Contains a list of strategies extracted from the optimization plan,
    along with metadata about how to execute them.
    """

    strategies: List[Strategy] = Field(
        ...,
        description="List of discrete optimization strategies to explore",
        min_length=1,
    )
    reasoning: str = Field(
        ..., description="Explanation of why these specific strategies were chosen"
    )

    @model_validator(mode="after")
    def enforce_max_strategies(self):
        if len(self.strategies) > MAX_STRATEGIES:
            raise ValueError(
                f"Too many strategies: got {len(self.strategies)}, maximum is {MAX_STRATEGIES}"
            )
        return self
