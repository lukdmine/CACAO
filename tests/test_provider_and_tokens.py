"""Provider resolution and token accounting — both silent-failure surfaces."""

import pytest
from langchain_core.messages import AIMessage

import config


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
    assert config._normalise_provider(name) == name


def test_claude_resolves_to_anthropic():
    """The name this project's own docs used; it fell through to key detection."""
    assert config._normalise_provider("claude") == "anthropic"


def test_names_are_case_and_space_insensitive():
    assert config._normalise_provider("  Anthropic ") == "anthropic"


def test_an_unknown_name_does_not_resolve():
    assert config._normalise_provider("bedrock") is None
    assert config._normalise_provider("") is None
    assert config._normalise_provider(None) is None


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
    monkeypatch.setattr(config, "LLM_PROVIDER", None)

    with pytest.raises(ValueError) as excinfo:
        config._detect_provider()

    message = str(excinfo.value)
    assert "LLM_PROVIDER=openai|anthropic|gemini|cerit" in message


def test_an_unrecognised_env_provider_warns_rather_than_silently_switching(monkeypatch):
    warnings = []
    monkeypatch.setattr(config, "LLM_PROVIDER", None)
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for var in ("CERIT_API_KEY", "CERIT_API_BASE", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(config, "log", lambda msg, level="INFO": warnings.append((level, msg)))

    assert config._detect_provider() == "openai"
    assert any(level == "WARN" and "bedrock" in msg for level, msg in warnings)


def test_env_claude_selects_anthropic_over_key_detection(monkeypatch):
    """LLM_PROVIDER=claude with both keys set used to run entirely on cerit."""
    monkeypatch.setattr(config, "LLM_PROVIDER", None)
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("CERIT_API_KEY", "cerit-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert config._detect_provider() == "anthropic"
