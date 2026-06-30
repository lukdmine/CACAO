"""
Pydantic models for structured LLM output.

These models define the schema for:
- Strategy extraction (StrategizeOutput)
- Decision making (OptimizationDecision)
"""

from .strategy import Strategy, StrategizeOutput
from .decision import ErrorAnalysis, OptimizationDecision

__all__ = [
    "Strategy",
    "StrategizeOutput",
    "ErrorAnalysis",
    "OptimizationDecision",
]
