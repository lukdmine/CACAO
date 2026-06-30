"""
Configuration for the CUDA Agentic Optimizer.

Contains paths, constants, and LLM setup.
Supports multiple LLM providers: OpenAI, Claude (Anthropic), Gemini (Google), and CERIT.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== Token Tracker & Retry Logic =====

# Transient exceptions that should be retried with backoff.
# Non-transient errors (auth, invalid model, validation) fail immediately.
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,  # covers network-level failures
)

try:
    from openai import RateLimitError, APITimeoutError, APIConnectionError

    TRANSIENT_ERRORS = TRANSIENT_ERRORS + (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
    )
except ImportError:
    pass


def _extract_status_code(error) -> Optional[int]:
    """Best-effort extraction of HTTP status codes from provider exceptions."""
    for attr in ("status_code", "status", "http_status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(error, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value

    return None


def _extract_retry_after(error) -> Optional[float]:
    """Best-effort extraction of Retry-After information from provider exceptions."""
    for attr in ("retry_after", "retry_after_seconds"):
        value = getattr(error, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                parsed = float(retry_after)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass

    return None


def _is_transient_error(error) -> bool:
    """Return True for retryable provider/network/rate-limit errors."""
    current = error
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, TRANSIENT_ERRORS):
            return True

        status_code = _extract_status_code(current)
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
            return True

        message = str(current).lower()
        transient_markers = (
            "429",
            "rate limit",
            "ratelimit",
            "too many requests",
            "overloaded",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
        if any(marker in message for marker in transient_markers):
            return True

        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )

    return False


def _compute_retry_wait(error, retries: int, base_wait: float) -> float:
    """Exponential backoff with jitter, honoring Retry-After when present."""
    retry_after = _extract_retry_after(error)
    backoff = base_wait * (2 ** (retries - 1))
    jitter = random.uniform(0, 1)
    wait_time = backoff + jitter
    if retry_after is not None:
        # Concurrency limits (e.g. CERIT's max_parallel_requests) clear as soon
        # as an in-flight request returns — but the provider reports Retry-After
        # as the next hourly-quota reset, which can be ~3600s. Ignore it for
        # parallel-slot errors and rely on exponential backoff instead.
        if "max_parallel_requests" not in str(error).lower():
            wait_time = max(wait_time, retry_after)
    return wait_time


class TokenTracker:
    """Single-event-loop token counter. No lock needed under asyncio cooperative scheduling."""

    def __init__(self):
        self.api_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add(self, response):
        self.api_calls += 1
        if (
            hasattr(response, "response_metadata")
            and "token_usage" in response.response_metadata
        ):
            usage = response.response_metadata["token_usage"]
            if usage:
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)

    def save(self, output_dir: Path = None):
        """Persist current token stats to output/token_usage.json (atomic write)."""
        if output_dir is None:
            output_dir = get_output_dir()
        path = output_dir / "token_usage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        data = {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        try:
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def load(self, output_dir: Path = None):
        """Seed tracker from a previously saved token_usage.json (for resume)."""
        if output_dir is None:
            output_dir = get_output_dir()
        path = output_dir / "token_usage.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self.api_calls = data.get("api_calls", 0)
            self.prompt_tokens = data.get("prompt_tokens", 0)
            self.completion_tokens = data.get("completion_tokens", 0)
            self.total_tokens = data.get("total_tokens", 0)
        except (json.JSONDecodeError, OSError):
            pass


global_tracker = TokenTracker()


def get_tracker_stats():
    return {
        "api_calls": global_tracker.api_calls,
        "prompt_tokens": global_tracker.prompt_tokens,
        "completion_tokens": global_tracker.completion_tokens,
        "total_tokens": global_tracker.total_tokens,
    }


class TrackedLLM:
    def __init__(self, llm):
        self._llm = llm

    async def ainvoke(self, *args, **kwargs):
        import asyncio

        retries = 0
        max_retries = 5
        base_wait = 2

        while True:
            try:
                response = await self._llm.ainvoke(*args, **kwargs)
                global_tracker.add(response)

                content = getattr(response, "content", None) or ""
                # Responses API returns content as a list of blocks.
                # Flatten to a plain string so downstream .strip()/string ops work.
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and block.get("text"):
                                parts.append(block["text"])
                        elif isinstance(block, str):
                            parts.append(block)
                    content = "".join(parts)
                    response.content = content
                if not content.strip():
                    reasoning = (
                        getattr(response, "additional_kwargs", None) or {}
                    ).get("reasoning_content")
                    if reasoning and reasoning.strip():
                        from utils.log import log

                        log(
                            "Content empty but reasoning_content found — using as output",
                            "WARN",
                        )
                        response.content = reasoning
                        return response

                    from utils.log import log

                    additional = getattr(response, "additional_kwargs", None) or {}
                    metadata = getattr(response, "response_metadata", None) or {}
                    finish_reason = metadata.get("finish_reason", "N/A")
                    completion_tokens = metadata.get("token_usage", {}).get(
                        "completion_tokens", "N/A"
                    )
                    log(
                        f"Empty content response — reasoning_content: {bool(reasoning)}, "
                        f"additional_kwargs keys: {list(additional.keys())}, "
                        f"finish_reason: {finish_reason}, "
                        f"tokens: {completion_tokens}",
                        "WARN",
                    )

                    if finish_reason == "length":
                        log(
                            "Aborting retries: response truncated by max_tokens (finish_reason=length). "
                            "Increase max_tokens in MODEL_CONFIGS for this model.",
                            "ERROR",
                        )
                        return response

                    retries += 1
                    if retries > max_retries:
                        log(
                            f"LLM returned empty content after {max_retries} retries",
                            "ERROR",
                        )
                        return response
                    wait_time = base_wait * (2 ** (retries - 1))
                    log(
                        f"Retrying {retries}/{max_retries} in {wait_time:.1f}s...",
                        "WARN",
                    )
                    await asyncio.sleep(wait_time)
                    continue

                return response
            except Exception as e:
                if not _is_transient_error(e):
                    raise
                retries += 1
                if retries > max_retries:
                    from utils.log import log

                    log(f"LLM invoke failed after {max_retries} retries: {e}", "ERROR")
                    raise

                wait_time = _compute_retry_wait(e, retries, base_wait)
                from utils.log import log

                log(
                    f"LLM transient error ({e}). Retrying {retries}/{max_retries} in {wait_time:.1f}s...",
                    "WARN",
                )
                await asyncio.sleep(wait_time)

    def with_structured_output(self, schema):
        structured_llm = self._llm.with_structured_output(schema)
        return TrackedStructuredLLM(structured_llm, self._llm, schema)


class TrackedStructuredLLM:
    def __init__(self, structured_llm, base_llm=None, schema=None):
        self._structured_llm = structured_llm
        self._base_llm = base_llm
        self._schema = schema

    async def ainvoke(self, *args, **kwargs):
        import asyncio

        retries = 0
        max_retries = 5
        base_wait = 2

        while True:
            try:
                response = await self._structured_llm.ainvoke(*args, **kwargs)
                global_tracker.api_calls += 1
                if hasattr(response, "response_metadata") and "token_usage" in getattr(
                    response, "response_metadata", {}
                ):
                    usage = response.response_metadata["token_usage"]
                    if usage:
                        global_tracker.prompt_tokens += usage.get("prompt_tokens", 0)
                        global_tracker.completion_tokens += usage.get(
                            "completion_tokens", 0
                        )
                        global_tracker.total_tokens += usage.get("total_tokens", 0)
                return response
            except Exception as e:
                if not _is_transient_error(e):
                    if self._base_llm and self._schema:
                        result = await self._try_raw_fallback_async(*args, **kwargs)
                        if result is not None:
                            return result
                    raise
                retries += 1
                if retries > max_retries:
                    from utils.log import log

                    log(
                        f"Structured LLM invoke failed after {max_retries} retries: {e}",
                        "ERROR",
                    )
                    raise

                wait_time = _compute_retry_wait(e, retries, base_wait)
                from utils.log import log

                log(
                    f"Structured LLM transient error ({e}). Retrying {retries}/{max_retries} in {wait_time:.1f}s...",
                    "WARN",
                )
                await asyncio.sleep(wait_time)

    async def _try_raw_fallback_async(self, *args, **kwargs):
        """Async variant of _try_raw_fallback for thinking models."""
        import re
        from utils.log import log

        try:
            log(
                "Structured output failed — trying raw LLM fallback with JSON parsing",
                "WARN",
            )
            tracked = TrackedLLM(self._base_llm)
            response = await tracked.ainvoke(*args, **kwargs)
            content = response.content or ""

            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
            if json_match:
                raw_json = json_match.group(1).strip()
            else:
                brace_match = re.search(r"\{.*\}", content, re.DOTALL)
                if brace_match:
                    raw_json = brace_match.group(0)
                else:
                    log("No JSON found in raw LLM fallback response", "WARN")
                    return None

            parsed = json.loads(raw_json)
            result = self._schema.model_validate(parsed)
            log(
                "Raw LLM fallback succeeded — parsed structured output from text",
                "SUCCESS",
            )
            return result
        except Exception as fallback_err:
            log(f"Raw LLM fallback also failed: {fallback_err}", "WARN")
            return None


SCRIPT_DIR = Path(__file__).parent
_base_output_dir = SCRIPT_DIR / "output"
_problem_dir = SCRIPT_DIR  # Defaults to script dir for backwards compatibility


def get_output_dir() -> Path:
    """Get the active output directory (defaults to ./output if no problem dir active)."""
    global _base_output_dir
    return _base_output_dir


def set_output_dir(path: Path):
    """Set the active output directory."""
    global _base_output_dir
    _base_output_dir = path


def get_problem_dir() -> Path:
    """Get the active problem directory (where problem.yaml and ref_kernel.cu live)."""
    global _problem_dir
    return _problem_dir


def set_problem_dir(path: Path):
    """Set the active problem directory."""
    global _problem_dir
    _problem_dir = path


# When set, this overrides both the LLM_PROVIDER env var and key-based
# auto-detection. Leave as None to let .env / available API keys decide.
LLM_PROVIDER: Optional[Literal["openai", "anthropic", "gemini", "cerit"]] = None

# NOTE: !!! NOT USED RIGHT NOW !!!
# Reasoning-token cap for cerit kimi-k2.6 — the only model on cerit's vLLM cluster with
# --enable-custom-logit-processor turned on. Sent alongside custom_logit_processor:"true"
# so the server-side KimiK26ThinkingBudgetLogitProcessor forces </think> emission once
# exceeded. Without the cap, kimi can think past max_tokens and return empty content with
# reasoning_content full — the proposal-empty failure mode. Other cerit models reject the
# processor flag with HTTP 400, so the helper gates on this exact model name.
CERIT_KIMI_THINKING_BUDGET: int = 27000

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
    "cerit": {
        "default": "kimi-k2.6",
        "available": [
            "qwen3.5",
            "qwen3.5-122b",
            "kimi-k2.5",
            "kimi-k2.6",
            "glm-5",
            "glm-5.1",
            "deepseek-v3.2",
            "deepseek-v3.2-thinking",
            "gpt-oss-120b",
            "qwen3-coder-next",
            "gemma4",
        ],
    },
}

# ===== Model-Specific Configurations =====
MODEL_CONFIGS = {
    "default": {
        "creative_temperature": 0.3,
        "precise_temperature": 0.0,
        "max_tokens": None,
    },
    "qwen": {
        "creative_temperature": 1.0,
        "precise_temperature": 0.7,
        "max_tokens": None,
    },
    "deepseek": {
        "creative_temperature": 0.4,
        "precise_temperature": 0.0,
        "max_tokens": 64000,
    },
    "claude": {
        "creative_temperature": 0.3,
        "precise_temperature": 0.0,
        "max_tokens": None,
    },
    "kimi": {
        "creative_temperature": 1.0,
        "precise_temperature": 0.6,
        # NOTE: max_tokens is currently not honored by the cerit proxy for kimi-k2.6 —
        # the server caps responses at ~32k regardless of what we send. Use
        # CERIT_KIMI_THINKING_BUDGET above to control reasoning length instead.
        "max_tokens": 96000,
    },
    "glm": {
        "creative_temperature": 1.0,
        "precise_temperature": 0.7,
        "max_tokens": 64000,
    },
}


def _get_model_config(model: str) -> dict:
    """Get the specific configuration parameters for a given model."""
    model_lower = model.lower()
    for prefix, config in MODEL_CONFIGS.items():
        if prefix != "default" and prefix in model_lower:
            return config
    return MODEL_CONFIGS["default"]


# ===== Workflow Configuration =====
MAX_ITERATIONS = 5  # Max iterations per branch (depth mode)
MAX_BRANCH_DEPTH = 2  # Max depth of nested branches (depth mode)
PATH_BUDGET = 0  # Total path iteration budget (path mode, 0 = use depth mode)
MAX_STRATEGIES = 4  # Max strategies per strategize call
TUNER_TIMEOUT = (
    100  # System fallback when neither the user nor problem.yaml sets a budget
)
TUNER_TIMEOUT_OVERRIDE: Optional[int] = (
    None  # Explicit user override (CLI --timeout / RunConfig.timeout). None = not overridden.
)
HISTORY_ITERS = 2  # Number of past iterations to include in LLM context (None = all). best_so_far covers older working kernels.
INCLUDE_BEST_SO_FAR = (
    True  # Include best-performing iteration's kernel + config in LLM context
)


@dataclass(frozen=True)
class OptimizerConfig:
    """Immutable per-run configuration. Built once at startup, replaces mutable globals."""

    output_dir: Path
    problem_dir: Path
    max_iterations: int = MAX_ITERATIONS
    max_branch_depth: int = MAX_BRANCH_DEPTH
    path_budget: int = PATH_BUDGET
    tuner_timeout: Optional[int] = (
        None  # None = let problem.yaml / system default apply
    )
    model: Optional[str] = None
    provider: Optional[str] = None


def init_from_config(cfg: OptimizerConfig):
    """Apply an OptimizerConfig to the module-level globals (called once per run)."""
    global \
        _base_output_dir, \
        _problem_dir, \
        _current_provider, \
        MAX_ITERATIONS, \
        MAX_BRANCH_DEPTH, \
        PATH_BUDGET, \
        TUNER_TIMEOUT_OVERRIDE
    _base_output_dir = cfg.output_dir
    _problem_dir = cfg.problem_dir
    MAX_ITERATIONS = cfg.max_iterations
    MAX_BRANCH_DEPTH = cfg.max_branch_depth
    PATH_BUDGET = cfg.path_budget
    TUNER_TIMEOUT_OVERRIDE = cfg.tuner_timeout
    if cfg.provider:
        _current_provider = cfg.provider
    if cfg.model:
        set_model(cfg.model)
    else:
        set_model(get_default_model())


# ===== LLM Instances =====
# Lazy initialization to avoid import errors when API key not set
_llm_creative: Optional[object] = None
_llm_precise: Optional[object] = None
_current_provider: Optional[str] = None
_current_model: Optional[str] = None


def _detect_provider() -> str:
    """Auto-detect provider based on available API keys."""
    # Check module-level LLM_PROVIDER setting first
    if LLM_PROVIDER in ("openai", "anthropic", "gemini", "cerit"):
        return LLM_PROVIDER

    # Then check environment variable
    explicit_provider = os.getenv("LLM_PROVIDER", "").lower()
    if explicit_provider in ["openai", "anthropic", "gemini", "cerit"]:
        return explicit_provider

    # Auto-detect based on API keys (priority: cerit > claude > openai > gemini)
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
        "Or set LLM_PROVIDER=openai|claude|gemini|cerit in your .env file"
    )


def get_provider() -> str:
    """Get the current LLM provider."""
    global _current_provider
    if _current_provider is None:
        _current_provider = _detect_provider()
    return _current_provider


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


def _create_llm(
    provider: str, model: str, temperature: float, max_tokens: Optional[int] = None
):
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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        base_llm = ChatOpenAI(**kwargs)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model, "temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        base_llm = ChatAnthropic(**kwargs)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": model, "temperature": temperature}
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
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
        _llm_creative = _create_llm(
            provider, model, config["creative_temperature"], config.get("max_tokens")
        )
    return _llm_creative


def get_llm_precise():
    """Get the precise LLM instance (for code generation, structured output)."""
    global _llm_precise, _current_provider, _current_model
    if _llm_precise is None:
        provider = get_provider()
        model = _current_model or get_default_model()
        config = _get_model_config(model)
        _llm_precise = _create_llm(
            provider, model, config["precise_temperature"], config.get("max_tokens")
        )
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
