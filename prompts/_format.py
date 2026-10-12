"""Shared prompt fragments.

Here rather than in nodes/_llm_helper.py, which is where format_strategy used to
live: four prompt modules imported it from there while nodes imported them back,
making prompts and nodes mutually dependent for one formatting helper.
"""


def format_strategy(strategy) -> str:
    """A strategy as a prompt section, from either a Pydantic model or a dict."""
    if not strategy:
        return ""
    get = (
        (lambda k, d: getattr(strategy, k, d))
        if hasattr(strategy, "name")
        else (lambda k, d: strategy.get(k, d))
    )
    return (
        f"## Strategy: {get('name', 'unknown')}\n"
        f"**Description**: {get('description', '')}\n"
        f"**Hypothesis**: {get('hypothesis', '')}\n"
        f"**Key Parameters**: {', '.join(get('key_parameters', []))}"
    )
