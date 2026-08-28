"""Which provider exceptions are worth retrying, and how long to wait."""

import random
from typing import Optional

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


def _format_error(error) -> str:
    """Render an exception with its type, HTTP body, and underlying cause chain.

    str(e) alone is near-useless for provider errors: the OpenAI SDK raises
    APIConnectionError with the fixed message "Connection error." and stores the
    actual failure (refused socket, DNS, TLS handshake) in __cause__, so the log
    line ends up saying nothing about what went wrong.
    """
    parts = [f"{type(error).__name__}: {error}"]

    status = _extract_status_code(error)
    if status is not None:
        parts.append(f"status={status}")

    body = getattr(error, "body", None)
    if body:
        parts.append(f"body={body!r}")

    current = error.__cause__ or error.__context__
    seen = {id(error)}
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"caused by {type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__

    return " | ".join(parts)


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
