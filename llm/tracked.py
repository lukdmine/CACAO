"""Retry, token tracking and truncation handling around a provider client."""

import asyncio

from utils.log import log

from llm.retry import _compute_retry_wait, _format_error, _is_transient_error
from llm.tokens import global_tracker

TRUNCATION_RETRY_NUDGE = (
    "Your previous attempt consumed the entire token budget on internal reasoning "
    "and produced no output. This time, keep deliberation brief: pick a "
    "straightforward approach, commit to it, and emit the final answer directly. "
    "A complete simple answer beats an unfinished sophisticated one."
)


class TrackedLLM:
    def __init__(self, llm, tools_bound: bool = False):
        self._llm = llm
        self._tools_bound = tools_bound

    async def ainvoke(self, *args, **kwargs):
        import asyncio

        retries = 0
        max_retries = 5
        base_wait = 2

        while True:
            try:
                response = await self._llm.ainvoke(*args, **kwargs)
                global_tracker.add(response)

                # A tool-call response carries its payload in tool_calls and leaves
                # content empty by construction. It must return before the emptiness
                # check below, which would otherwise treat every single tool call as a
                # failed generation and burn five retries with backoff on it.
                if getattr(response, "tool_calls", None):
                    return response

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
                        log(
                            "Content empty but reasoning_content found — using as output",
                            "WARN",
                        )
                        response.content = reasoning
                        return response

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

                    # The agentic loop has its own escalating truncation policy;
                    # retrying here too made the two multiply.
                    if self._tools_bound:
                        return response

                    if finish_reason == "length":
                        # Thinking models (e.g. glm-5.2 behind a ~48k server cap) can
                        # spend the whole budget on reasoning and emit zero content.
                        # Retrying the identical prompt re-runs the same risk, so
                        # append an explicit brevity instruction — the retry prompt
                        # differs, steering the model to answer.
                        log(
                            "Response hit the token cap with no usable content "
                            "(reasoning consumed the budget). Retrying with a "
                            "brevity instruction appended.",
                            "WARN",
                        )
                        if (
                            args
                            and isinstance(args[0], list)
                            and (
                                not args[0]
                                or getattr(args[0][-1], "content", None)
                                != TRUNCATION_RETRY_NUDGE
                            )
                        ):
                            from langchain_core.messages import HumanMessage

                            args = (
                                list(args[0])
                                + [HumanMessage(content=TRUNCATION_RETRY_NUDGE)],
                                *args[1:],
                            )

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
                    log(
                        f"LLM invoke failed (non-transient): {_format_error(e)}",
                        "ERROR",
                    )
                    raise
                retries += 1
                if retries > max_retries:
                    log(
                        f"LLM invoke failed after {max_retries} retries: {_format_error(e)}",
                        "ERROR",
                    )
                    raise

                wait_time = _compute_retry_wait(e, retries, base_wait)
                log(
                    f"LLM transient error ({_format_error(e)}). Retrying {retries}/{max_retries} in {wait_time:.1f}s...",
                    "WARN",
                )
                await asyncio.sleep(wait_time)

    def with_structured_output(self, schema):
        structured_llm = self._llm.with_structured_output(schema)
        return TrackedStructuredLLM(structured_llm, self._llm, schema)

    def bind_tools(self, tools, **kwargs):
        """Bind tools, keeping retry and token tracking around the bound model.

        Raises whatever the provider raises when it has no tool support — the agentic
        loop treats that as a signal to fall back rather than something to retry.

        Keeps transient-error retry and token tracking, drops the empty-content retry.
        """
        return TrackedLLM(self._llm.bind_tools(tools, **kwargs), tools_bound=True)


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
                global_tracker.record_usage(response)
                return response
            except Exception as e:
                if not _is_transient_error(e):
                    will_fallback = bool(self._base_llm and self._schema)
                    log(
                        f"Structured LLM invoke failed (non-transient): {_format_error(e)}",
                        "WARN" if will_fallback else "ERROR",
                    )
                    if will_fallback:
                        result = await self._try_raw_fallback_async(*args, **kwargs)
                        if result is not None:
                            return result
                    raise
                retries += 1
                if retries > max_retries:
                    log(
                        f"Structured LLM invoke failed after {max_retries} retries: {_format_error(e)}",
                        "ERROR",
                    )
                    raise

                wait_time = _compute_retry_wait(e, retries, base_wait)
                log(
                    f"Structured LLM transient error ({_format_error(e)}). Retrying {retries}/{max_retries} in {wait_time:.1f}s...",
                    "WARN",
                )
                await asyncio.sleep(wait_time)

    async def _try_raw_fallback_async(self, *args, **kwargs):
        """Async variant of _try_raw_fallback for thinking models."""
        import re
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
            log(f"Raw LLM fallback also failed: {_format_error(fallback_err)}", "WARN")
            return None
