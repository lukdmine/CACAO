"""Provider resolution and token accounting — both silent-failure surfaces."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

import config
from llm import providers


# -- token accounting -------------------------------------------------------


@pytest.fixture
def tracker():
    return config.TokenTracker()


def test_usage_metadata_is_counted(tracker):
    """anthropic and gemini report only this one."""
    tracker.add(
        AIMessage(
            content="x",
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
    )

    assert (tracker.prompt_tokens, tracker.completion_tokens, tracker.total_tokens) == (
        100,
        20,
        120,
    )
    assert tracker.api_calls == 1


def test_raw_provider_payload_is_still_counted(tracker):
    """langchain_openai passes the provider payload through as token_usage."""
    tracker.add(
        AIMessage(
            content="x",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                }
            },
        )
    )

    assert (tracker.prompt_tokens, tracker.completion_tokens, tracker.total_tokens) == (
        7,
        3,
        10,
    )


def test_usage_metadata_wins_when_both_are_present(tracker):
    tracker.add(
        AIMessage(
            content="x",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            response_metadata={
                "token_usage": {"prompt_tokens": 999, "completion_tokens": 999}
            },
        )
    )

    assert tracker.total_tokens == 2


def test_a_response_with_no_usage_still_counts_the_call(tracker):
    tracker.add(AIMessage(content="x"))

    assert tracker.api_calls == 1
    assert tracker.total_tokens == 0


def test_total_is_derived_when_the_provider_omits_it(tracker):
    """AIMessage requires total_tokens; duck-typed responses need not."""

    class Response:
        usage_metadata = {"input_tokens": 4, "output_tokens": 6}
        response_metadata: dict = {}

    tracker.add(Response())

    assert tracker.total_tokens == 10


# -- provider resolution ----------------------------------------------------


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini", "cerit"])
def test_canonical_provider_names_resolve(name):
    assert providers._normalise_provider(name) == name


def test_claude_resolves_to_anthropic():
    """The name this project's own docs used; it fell through to key detection."""
    assert providers._normalise_provider("claude") == "anthropic"


def test_names_are_case_and_space_insensitive():
    assert providers._normalise_provider("  Anthropic ") == "anthropic"


def test_an_unknown_name_does_not_resolve():
    assert providers._normalise_provider("bedrock") is None
    assert providers._normalise_provider("") is None
    assert providers._normalise_provider(None) is None


def test_the_error_message_names_only_accepted_values(monkeypatch):
    """It used to name "claude", which the accepted set did not contain."""
    for var in (
        "LLM_PROVIDER",
        "CERIT_API_KEY",
        "CERIT_API_BASE",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(providers, "LLM_PROVIDER", None)

    with pytest.raises(ValueError) as excinfo:
        providers._detect_provider()

    message = str(excinfo.value)
    assert "LLM_PROVIDER=openai|anthropic|gemini|cerit" in message


def test_an_unrecognised_env_provider_warns_rather_than_silently_switching(monkeypatch):
    warnings = []
    monkeypatch.setattr(providers, "LLM_PROVIDER", None)
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for var in ("CERIT_API_KEY", "CERIT_API_BASE", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(providers, "log", lambda msg, level="INFO": warnings.append((level, msg)))

    assert providers._detect_provider() == "openai"
    assert any(level == "WARN" and "bedrock" in msg for level, msg in warnings)


def test_env_claude_selects_anthropic_over_key_detection(monkeypatch):
    """LLM_PROVIDER=claude with both keys set used to run entirely on cerit."""
    monkeypatch.setattr(providers, "LLM_PROVIDER", None)
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("CERIT_API_KEY", "cerit-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert providers._detect_provider() == "anthropic"


# -- Claude request shape ---------------------------------------------------


class _Verdict(BaseModel):
    action: str
    reason: str


def _claude(model):
    return providers._create_llm("anthropic", model, 0.3)._llm


def _payload(llm, **kwargs):
    return llm._get_request_payload([HumanMessage(content="x")], **kwargs)


@pytest.mark.parametrize(
    "model", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
)
def test_older_claude_models_keep_their_temperature(model):
    assert _payload(_claude(model))["temperature"] == 0.3


def test_every_newer_claude_model_is_sent_without_temperature():
    """Each of them answers a non-default temperature with a 400."""
    newer = [
        m
        for m in providers.MODELS["anthropic"]["available"]
        if not m.startswith(providers._CLAUDE_WITH_SAMPLING)
    ]
    assert newer

    for model in newer:
        assert _payload(_claude(model)).get("temperature") is None, model


@pytest.mark.parametrize("model", providers.MODELS["anthropic"]["available"])
def test_every_claude_model_is_sent_its_full_output_limit(model):
    """langchain sends 4096 for any model it has no profile for, which included the
    default, claude-opus-4-6."""
    expected = 64000 if model.startswith("claude-haiku-4-5") else 128000

    assert _payload(_claude(model))["max_tokens"] == expected


def test_an_older_claude_model_keeps_its_profiled_limit():
    """claude-sonnet-4-5 allows 64K; sending it 128K would be a 400."""
    assert _payload(_claude("claude-sonnet-4-5"))["max_tokens"] == 64000


def test_newer_claude_structured_output_uses_output_config_not_a_forced_tool():
    """The newest models reject tool_choice "tool", which is how langchain gets
    structured output by default; decide and strategize would 400 on every call."""
    chain = _claude("claude-opus-5-5").with_structured_output(_Verdict)
    binding = chain.first
    payload = _payload(binding.bound, **binding.kwargs)

    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert not payload.get("tool_choice")
    assert not payload.get("tools")


def test_newer_claude_structured_output_parses_past_a_thinking_block():
    chain = _claude("claude-opus-5-5").with_structured_output(_Verdict)
    reply = AIMessage(
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": '{"action": "continue", "reason": "faster"}'},
        ]
    )

    assert chain.last.invoke(reply) == _Verdict(action="continue", reason="faster")


def _reply_from_the_api(monkeypatch, message):
    """Stand in for the network: every ChatClaude call returns ``message``."""
    from langchain_core.outputs import ChatGeneration, ChatResult

    from llm.claude import ChatClaude

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ChatClaude, "_agenerate", _agenerate)


def test_newer_claude_structured_output_works_through_the_tracked_wrapper(monkeypatch):
    """TrackedLLM asks for include_raw=True so it can count the reply's tokens. The tests
    above call ChatClaude directly and never pass it, so a signature without it shipped
    and every analyze, strategize and decide call on these models raised TypeError."""
    from llm import tracked

    counted = config.TokenTracker()
    monkeypatch.setattr(tracked, "global_tracker", counted)
    _reply_from_the_api(
        monkeypatch,
        AIMessage(
            content='{"action": "continue", "reason": "faster"}',
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        ),
    )

    llm = providers._create_llm("anthropic", "claude-opus-5-5", 0.3)
    verdict = asyncio.run(llm.with_structured_output(_Verdict).ainvoke("x"))

    assert verdict == _Verdict(action="continue", reason="faster")
    assert counted.total_tokens == 120


def test_newer_claude_include_raw_reports_a_parse_failure_instead_of_raising(monkeypatch):
    """langchain's include_raw contract, which TrackedStructuredLLM._unwrap relies on to
    reach its raw-JSON fallback."""
    _reply_from_the_api(monkeypatch, AIMessage(content="not json"))
    chain = _claude("claude-opus-5-5").with_structured_output(_Verdict, include_raw=True)

    out = asyncio.run(chain.ainvoke("x"))

    assert out["parsed"] is None
    assert out["parsing_error"] is not None
    assert out["raw"].content == "not json"
