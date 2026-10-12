"""The model roster, provider resolution, and the configured LLM instances."""

import os
from typing import Literal, Optional

from utils.log import log

from llm.tracked import TrackedLLM

LLM_PROVIDER: Optional[Literal["openai", "anthropic", "gemini", "cerit"]] = None

# ===== Model defaults for each provider =====
MODELS = {
    "openai": {
        "default": "gpt-5.4",
        "available": [
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.2-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex",
            "gpt-5-codex",
            "gpt-5.1-codex-mini",
            "gpt-4.1-nano",
        ],
    },
    "anthropic": {
        "default": "claude-opus-4-6",
        "available": ["claude-opus-4-6", "claude-haiku-4-5-20251001"],
    },
    "gemini": {
        "default": "gemini-3-pro",
        "available": ["gemini-3-pro", "gemini-3-flash"],
    },
    # Chat models served by cerit's endpoint, as listed by GET /v1/models. Version-pinned
    # names only: the endpoint also serves bare aliases (glm, kimi, coder, thinker, ...)
    # whose target moves without notice, plus embedding/reranker/whisper models that
    # cannot answer a chat completion at all.
    "cerit": {
        "default": "glm-5.2",
        "available": [
            "glm-5.2",
            "glm-5.3",
            "kimi-k3",
            "qwen3.5",
            "qwen3.5-122b",
            "qwen3.5-int4",
            "qwen3.8-27b",
            "deepseek-v4-flash",
            "deepseek-v4-flash-thinking",
            "deepseek-thinking",
            "command-a",
            "mistral-medium-3.5",
            "gpt-oss-120b",
            "gemma4",
        ],
    },
}

# ===== Model-Specific Configurations =====
MODEL_CONFIGS = {
    "default": {
        "creative_temperature": 0.3,
        "precise_temperature": 0.0,
    },
    "qwen": {
        "creative_temperature": 1.0,
        "precise_temperature": 0.7,
    },
    "deepseek": {
        "creative_temperature": 0.4,
        "precise_temperature": 0.0,
    },
    "claude": {
        "creative_temperature": 0.3,
        "precise_temperature": 0.0,
    },
    "kimi": {
        "creative_temperature": 1.0,
        "precise_temperature": 0.6,
    },
    "glm": {
        "creative_temperature": 1.0,
        "precise_temperature": 0.7,
    },
}


def _get_model_config(model: str) -> dict:
    """Get the specific configuration parameters for a given model."""
    model_lower = model.lower()
    for prefix, config in MODEL_CONFIGS.items():
        if prefix != "default" and prefix in model_lower:
            return config
    return MODEL_CONFIGS["default"]


# ===== LLM Instances =====
# Lazy initialization to avoid import errors when API key not set
_llm_creative: Optional[object] = None
_llm_precise: Optional[object] = None
_current_provider: Optional[str] = None
_current_model: Optional[str] = None


PROVIDERS = ("openai", "anthropic", "gemini", "cerit")

# Names this project's own docs used; ignoring them fell through to key detection.
_PROVIDER_ALIASES = {"claude": "anthropic", "google": "gemini"}


def _normalise_provider(value) -> Optional[str]:
    """Map a user-supplied provider name to a canonical one, or None if unknown."""
    name = str(value or "").strip().lower()
    name = _PROVIDER_ALIASES.get(name, name)
    return name if name in PROVIDERS else None


def _detect_provider() -> str:
    """Auto-detect provider based on available API keys."""
    # Check module-level LLM_PROVIDER setting first
    if LLM_PROVIDER:
        resolved = _normalise_provider(LLM_PROVIDER)
        if resolved:
            return resolved

    # Then check environment variable
    explicit_provider = os.getenv("LLM_PROVIDER", "").lower()
    if explicit_provider:
        resolved = _normalise_provider(explicit_provider)
        if resolved:
            return resolved
        # Silent fallback is how a run configured for one provider executes on another.
        log(
            f"LLM_PROVIDER={explicit_provider!r} is not a known provider "
            f"({', '.join(PROVIDERS)}) — falling back to API-key detection.",
            "WARN",
        )

    # Auto-detect based on API keys (priority: cerit > anthropic > openai > gemini)
    if os.getenv("CERIT_API_KEY") or os.getenv("CERIT_API_BASE"):
        return "cerit"
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini"

    raise ValueError(
        "No LLM provider detected. Please set one of:\n"
        "  - CERIT_API_KEY or CERIT_API_BASE (for CERIT)\n"
        "  - ANTHROPIC_API_KEY or CLAUDE_API_KEY (for Claude)\n"
        "  - OPENAI_API_KEY (for OpenAI)\n"
        "  - GOOGLE_API_KEY or GEMINI_API_KEY (for Gemini)\n"
        "Or set LLM_PROVIDER=openai|anthropic|gemini|cerit in your .env file"
    )


def get_provider() -> str:
    """Get the current LLM provider."""
    global _current_provider
    if _current_provider is None:
        _current_provider = _detect_provider()
    return _current_provider


def set_provider(provider: Optional[str]) -> None:
    """Pin the provider for this run, bypassing detection. Called by init_from_config."""
    global _current_provider, _llm_creative, _llm_precise
    if not provider:
        return
    _current_provider = _normalise_provider(provider) or provider
    _llm_creative = None
    _llm_precise = None


def get_default_model() -> str:
    """Get the default model for the current provider."""
    provider = get_provider()
    return MODELS[provider]["default"]


def get_model() -> str:
    """Get the current model name (or the provider default if unset)."""
    return _current_model or get_default_model()


def set_model(model: str):
    """Set the model to use for LLM calls."""
    global _current_model, _llm_creative, _llm_precise
    _current_model = model
    # Reset instances so they get recreated with new model
    _llm_creative = None
    _llm_precise = None


def _create_llm(provider: str, model: str, temperature: float):
    """Create an LLM instance for the specified provider."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # Codex models are only available on the Responses API and reject the
        # temperature parameter (reasoning-only models).
        is_codex = "codex" in model.lower()
        kwargs: dict = {"model": model}
        if is_codex:
            kwargs["use_responses_api"] = True
        else:
            kwargs["temperature"] = temperature
        base_llm = ChatOpenAI(**kwargs)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model, "temperature": temperature}
        base_llm = ChatAnthropic(**kwargs)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": model, "temperature": temperature}
        base_llm = ChatGoogleGenerativeAI(**kwargs)
    elif provider == "cerit":
        from langchain_openai import ChatOpenAI

        # CERIT uses OpenAI-compatible API with custom base_url
        base_url = os.getenv("CERIT_API_BASE")
        api_key = os.getenv("CERIT_API_KEY")
        kwargs = {
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
            "api_key": api_key,
        }
        base_llm = ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return TrackedLLM(base_llm)


def get_llm_creative():
    """Get the creative LLM instance (for analysis, planning)."""
    global _llm_creative, _current_provider, _current_model
    if _llm_creative is None:
        provider = get_provider()
        model = _current_model or get_default_model()
        config = _get_model_config(model)
        _llm_creative = _create_llm(provider, model, config["creative_temperature"])
    return _llm_creative


def get_llm_precise():
    """Get the precise LLM instance (for code generation, structured output)."""
    global _llm_precise, _current_provider, _current_model
    if _llm_precise is None:
        provider = get_provider()
        model = _current_model or get_default_model()
        config = _get_model_config(model)
        _llm_precise = _create_llm(provider, model, config["precise_temperature"])
    return _llm_precise


def check_api_key():
    """Check if at least one LLM provider API key is set."""
    provider = get_provider()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Please set OPENAI_API_KEY environment variable.\n"
                "You can add it to a .env file in the project root."
            )
    elif provider == "cerit":
        # CERIT requires at least CERIT_API_KEY (base_url has default)
        if not os.getenv("CERIT_API_KEY"):
            raise ValueError(
                "Please set CERIT_API_KEY environment variable.\n"
                "You can add it to a .env file in the project root."
            )
        if not os.getenv("CERIT_API_BASE"):
            raise ValueError(
                "Please set CERIT_API_BASE environment variable.\n"
                "You can add it to a .env file in the project root."
            )
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError(
                "Please set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable.\n"
                "You can add it to a .env file in the project root."
            )
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.\n"
                "You can add it to a .env file in the project root."
            )
