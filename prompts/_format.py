"""Shared prompt fragments. Kept out of nodes/ so prompts and nodes do not import each other."""


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


def format_reference(ctx: dict) -> str:
    """The reference implementation, fenced in the language it is written in.

    ref_language: the reference may be CUDA, C, or Python — fencing a Python reference as
    ```cuda misleads the model about what it is reading.
    """
    if not ctx.get("ref_kernel"):
        return ""
    return (
        "## Reference implementation:\n"
        f"```{ctx.get('ref_language', 'cuda')}\n{ctx['ref_kernel']}\n```"
    )


def inputs_hpp_block(text: str) -> str:
    """The I/O boundary listing, with the caveat that NVRTC never sees the file.

    The `inline constexpr` clause is the whole point: a scalar declared there reads as
    available everywhere and is not, so a paraphrase that drops it buys a compile round trip.
    """
    return (
        "## I/O boundary (inputs.hpp) — the buffers and scalars your kernels consume and "
        "produce; kernel arguments are bound to these by name.\n"
        "**Compiled into the host driver, NOT into your kernel.** NVRTC compiles kernels.cu "
        "on its own, so nothing declared here — including every `inline constexpr` — is "
        "visible to a kernel unless it is a `-D` macro or an argument. See the scalar "
        "contract below.\n"
        f"```cpp\n{text}\n```"
    )
