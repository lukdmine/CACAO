"""
Pydantic models for KTT tuning parameter configuration.

Used by configure_node to structure LLM output, ensuring valid params.json.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TuningParameter(BaseModel):
    """A single tuning parameter with its possible values."""

    name: str = Field(
        ...,
        description="Parameter name used in kernel code (e.g., 'TILE_M', 'THREADS_X')",
    )
    values: List[int] = Field(
        ...,
        description="List of possible integer values to try (e.g., [32, 64, 128])",
        min_length=1,
    )


class LaunchConfig(BaseModel):
    """
    Formula-based kernel launch configuration.

    All fields are Python expressions that can reference parameter names
    and problem scalars (M, N, K). Evaluated at runtime per configuration.
    """

    grid_x: Optional[str] = Field(
        None, description="Grid X dimension formula (e.g., 'M // TILE_M')"
    )
    grid_y: Optional[str] = Field(
        None, description="Grid Y dimension formula (e.g., 'N // TILE_N')"
    )
    grid_z: Optional[str] = Field(
        "1", description="Grid Z dimension formula (default: '1')"
    )
    block_x: Optional[str] = Field(
        None, description="Block X dimension formula (e.g., 'THREADS_X')"
    )
    block_y: Optional[str] = Field(
        None, description="Block Y dimension formula (e.g., 'THREADS_Y')"
    )
    block_z: Optional[str] = Field(
        "1", description="Block Z dimension formula (default: '1')"
    )


class Constraint(BaseModel):
    """A constraint that filters invalid parameter combinations."""

    params: List[str] = Field(
        ...,
        description="List of TUNING parameter names used in the expression. DO NOT include problem scalar names (e.g. M, N, students) — they are automatically available in the expression.",
    )
    expr: str = Field(
        ...,
        description="Python boolean expression (e.g., 'TILE_M == THREADS_X * TM'). Can use both tuning parameter names and problem scalar names.",
    )


class TuningConfig(BaseModel):
    """
    Complete KTT tuning parameter configuration.

    Defines the parameter search space, launch configuration, and constraints
    for the KTT auto-tuner.
    """

    parameters: List[TuningParameter] = Field(
        ...,
        description="List of tunable parameters with their possible values",
        min_length=1,
    )
    launch_config: Optional[LaunchConfig] = Field(
        None,
        description="Formula-based launch configuration for grid/block/shared memory",
    )
    constraints: Optional[List[Constraint]] = Field(
        default_factory=list,
        description="Constraints to filter invalid parameter combinations",
    )
