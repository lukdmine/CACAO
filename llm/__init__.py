"""LLM runtime: retry policy, token accounting, and the provider clients.

Split out of config.py, which had grown to 867 lines by holding the whole LLM
stack alongside the handful of settings that are actually configuration.
"""

from llm.providers import (
    LLM_PROVIDER,
    MODELS,
    MODEL_CONFIGS,
    PROVIDERS,
    check_api_key,
    get_default_model,
    get_llm_creative,
    get_llm_precise,
    get_model,
    get_provider,
    set_model,
)
from llm.tokens import TokenTracker, get_tracker_stats, global_tracker
from llm.tracked import TrackedLLM, TrackedStructuredLLM

__all__ = [
    "LLM_PROVIDER",
    "MODELS",
    "MODEL_CONFIGS",
    "PROVIDERS",
    "TokenTracker",
    "TrackedLLM",
    "TrackedStructuredLLM",
    "check_api_key",
    "get_default_model",
    "get_llm_creative",
    "get_llm_precise",
    "get_model",
    "get_provider",
    "get_tracker_stats",
    "global_tracker",
    "set_model",
]
