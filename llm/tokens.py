"""Token accounting across a run.

output_dir is passed in rather than read from config: this module is below
configuration, not beside it.
"""

import json
from pathlib import Path

class TokenTracker:
    """Single-event-loop token counter. No lock needed under asyncio cooperative scheduling."""

    def __init__(self):
        self.api_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def record_usage(self, response) -> None:
        """Add one response's tokens. usage_metadata is langchain's normalised field;
        only langchain_openai also fills response_metadata["token_usage"]."""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.prompt_tokens += usage.get("input_tokens", 0)
            self.completion_tokens += usage.get("output_tokens", 0)
            self.total_tokens += usage.get(
                "total_tokens",
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            )
            return

        raw = (getattr(response, "response_metadata", None) or {}).get("token_usage")
        if raw:
            self.prompt_tokens += raw.get("prompt_tokens", 0)
            self.completion_tokens += raw.get("completion_tokens", 0)
            self.total_tokens += raw.get("total_tokens", 0)

    def add(self, response):
        self.api_calls += 1
        self.record_usage(response)

    def save(self, output_dir: Path):
        """Persist current token stats to output/token_usage.json (atomic write)."""
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
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def load(self, output_dir: Path):
        """Seed tracker from a previously saved token_usage.json (for resume)."""
        path = output_dir / "token_usage.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
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


# Appended to the conversation when a response dies with finish_reason=length and
# empty content: the model reasoned past the token cap without answering. Changing
# the prompt (not just re-rolling) is what makes the retry meaningfully different.
