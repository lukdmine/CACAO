"""Remembering, per run, whether a provider can actually drive a tool loop.

``bind_tools`` is a local operation for every provider in this repo — langchain
attaches the schemas to the request and returns. Nothing raises until the request
reaches the server, so a provider without tool support is discovered by *calling* it,
not by binding.

That discovery has to happen once, not once per step. A branch pays for it on every
iteration otherwise, and there are four branches running at a time: on an endpoint
that answers a tools request with a 5xx, TrackedLLM's retry ladder turns each
rediscovery into about a minute of backoff before the fallback fires.

State is per process and keyed by provider and model, so switching model mid-run
re-evaluates rather than inheriting a verdict about different weights. A successful
step clears the count, so one refusal on a model that normally works does not disable
the loop for the rest of the run.
"""

from __future__ import annotations

from typing import Dict, Tuple

from utils.log import log

# Two consecutive capability failures before the loop is skipped. One is not enough:
# a model that simply chose to answer in prose once is not a model that cannot call
# tools, and disabling on a single sample would strand a capable provider on the
# single-shot path for the whole run.
DISABLE_AFTER = 2

_failures: Dict[Tuple[str, str], int] = {}
_disabled: Dict[Tuple[str, str], bool] = {}


def _key() -> Tuple[str, str]:
    from config import get_model, get_provider

    try:
        return (get_provider(), get_model())
    except Exception:
        return ("unknown", "unknown")


def should_skip_loop() -> bool:
    """True when this provider has already shown it cannot drive the loop."""
    return _disabled.get(_key(), False)


def record_failure(reason: str) -> None:
    """Note a capability failure; disable the loop once they accumulate."""
    key = _key()
    if _disabled.get(key):
        return
    _failures[key] = _failures.get(key, 0) + 1
    if _failures[key] >= DISABLE_AFTER:
        _disabled[key] = True
        log(
            f"{key[0]}/{key[1]} failed to drive the tool loop {_failures[key]}x "
            f"(last: {reason}) — using single-shot authoring for the rest of this run. "
            f"Set AGENTIC_STEPS=False to skip this detection entirely.",
            "WARN",
        )
    else:
        log(
            f"{key[0]}/{key[1]} did not drive the tool loop ({reason}); "
            f"{DISABLE_AFTER - _failures[key]} more before falling back permanently",
            "WARN",
        )


def record_success() -> None:
    """A completed step clears the count — the provider demonstrably works."""
    _failures.pop(_key(), None)


def reset() -> None:
    """Forget everything. For tests, and for a deliberate re-evaluation."""
    _failures.clear()
    _disabled.clear()


def status() -> dict:
    """Current verdicts, for logging and diagnostics."""
    return {
        "failures": {f"{p}/{m}": n for (p, m), n in _failures.items()},
        "disabled": [f"{p}/{m}" for (p, m), off in _disabled.items() if off],
    }
