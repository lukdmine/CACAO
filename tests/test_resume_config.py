"""A resume continues the run it resumes, on the same model and the same budget.

change-decision used to resume a stopped run with RunConfig(max_iter=...) and nothing
else. The child then auto-detected a provider, and with a CERIT and an Anthropic key
both set, a Claude run came back on cerit's default model — one the endpoint no longer
served, so propose and decide failed with a 400.
"""

import json

import pytest

from api import optimizer
from api.schemas import RunConfig


@pytest.fixture
def problem_dir(tmp_path):
    (tmp_path / "output").mkdir()
    return tmp_path


def _record(problem_dir, **meta):
    (problem_dir / "output" / "run_meta.json").write_text(json.dumps(meta))


CLAUDE_RUN = {
    "model": "claude-opus-5-5",
    "provider": "anthropic",
    "max_iter": 7,
    "max_depth": 3,
    "path_budget": 40,
    "timeout": 300,
}


def test_a_resume_naming_only_the_budget_keeps_the_runs_model(problem_dir):
    _record(problem_dir, **CLAUDE_RUN)

    resumed = optimizer._as_recorded(problem_dir, RunConfig(max_iter=12))

    assert (resumed.provider, resumed.model) == ("anthropic", "claude-opus-5-5")
    assert (resumed.max_depth, resumed.path_budget, resumed.timeout) == (3, 40, 300)
    assert resumed.max_iter == 12  # what the request named wins


def test_an_explicit_provider_never_gets_the_recorded_providers_model(problem_dir):
    _record(problem_dir, **CLAUDE_RUN)

    resumed = optimizer._as_recorded(problem_dir, RunConfig(provider="cerit"))

    assert resumed.provider == "cerit"
    assert resumed.model is None


def test_an_explicit_model_and_provider_win(problem_dir):
    _record(problem_dir, **CLAUDE_RUN)

    resumed = optimizer._as_recorded(
        problem_dir, RunConfig(provider="cerit", model="glm-5.3")
    )

    assert (resumed.provider, resumed.model) == ("cerit", "glm-5.3")
    assert resumed.path_budget == 40


def test_a_run_recorded_before_the_budget_was_keeps_the_defaults(problem_dir):
    _record(problem_dir, model="claude-opus-5-5", provider="anthropic")

    resumed = optimizer._as_recorded(problem_dir, RunConfig())

    assert resumed.provider == "anthropic"
    assert resumed.path_budget == RunConfig().path_budget


def test_no_record_leaves_the_request_alone(problem_dir):
    assert optimizer._as_recorded(problem_dir, RunConfig()) == RunConfig()


def test_start_resume_spawns_with_the_recorded_settings(problem_dir, monkeypatch):
    _record(problem_dir, **CLAUDE_RUN)
    spawned = {}
    monkeypatch.setattr(
        optimizer,
        "_reserve_and_spawn",
        lambda name, pdir, config, *args, **kwargs: spawned.update(config=config),
    )

    optimizer.start_resume("p", problem_dir, RunConfig(max_iter=12), gpu_index=0)

    assert spawned["config"].model == "claude-opus-5-5"
