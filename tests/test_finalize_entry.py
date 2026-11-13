"""Finalize's entry points: the engine-end hook, the config knobs and the CLI flags."""

import asyncio
import json
import os
import sys

import pytest

import config as _cfg
import engine.master as master


@pytest.fixture
def engine_env(tmp_path, monkeypatch):
    """run_optimization_engine on the resume path with every heavy step stubbed."""
    out = tmp_path / "output"
    (out / "branches").mkdir(parents=True)
    problem = tmp_path / "problem"
    problem.mkdir()
    monkeypatch.setattr(master, "get_output_dir", lambda: out)
    monkeypatch.setattr(_cfg, "get_output_dir", lambda: out)
    monkeypatch.setattr(_cfg, "get_problem_dir", lambda: problem)

    async def _ok(*a, **k):
        return True

    async def _none(*a, **k):
        return None

    monkeypatch.setattr(master, "preflight_python_reference", _ok)
    monkeypatch.setattr(master, "time_python_reference", _none)
    monkeypatch.setattr(master, "_write_final_results", lambda: None)
    calls = []

    async def recorder(*, trigger):
        calls.append(trigger)

    monkeypatch.setattr("engine.finalize.run_finalize", recorder)
    return out, calls


def _run(out, monkeypatch, loop_impl):
    branch = out / "branches" / "b1"
    branch.mkdir()
    monkeypatch.setattr(master, "run_branch_loop", loop_impl)
    asyncio.run(master.run_optimization_engine("name: p\n", "", [branch]))


def test_a_normal_run_end_calls_finalize(engine_env, monkeypatch):
    out, calls = engine_env
    monkeypatch.setattr(_cfg, "FINALIZE_AUTO", True)

    async def loop(path):
        return None

    _run(out, monkeypatch, loop)
    assert calls == ["auto"]


def test_a_worker_exception_skips_finalize(engine_env, monkeypatch):
    out, calls = engine_env
    monkeypatch.setattr(_cfg, "FINALIZE_AUTO", True)

    async def loop(path):
        raise asyncio.CancelledError()  # BaseException: the worker task dies

    _run(out, monkeypatch, loop)
    assert calls == []


def test_finalize_auto_off_skips_finalize(engine_env, monkeypatch):
    out, calls = engine_env
    monkeypatch.setattr(_cfg, "FINALIZE_AUTO", False)

    async def loop(path):
        return None

    _run(out, monkeypatch, loop)
    assert calls == []


def test_init_from_config_applies_finalize_overrides(tmp_path, monkeypatch):
    # init_from_config rebinds module globals; have monkeypatch restore every one.
    for name in (
        "FINALIZE_AUTO", "FINALIZE_TUNER_TIMEOUT", "MAX_ITERATIONS", "MAX_BRANCH_DEPTH",
        "PATH_BUDGET", "TUNER_TIMEOUT_OVERRIDE", "_base_output_dir", "_problem_dir",
    ):
        monkeypatch.setattr(_cfg, name, getattr(_cfg, name))
    monkeypatch.setattr(_cfg, "set_provider", lambda p: None)
    monkeypatch.setattr(_cfg, "set_model", lambda m: None)
    base = dict(output_dir=tmp_path / "o", problem_dir=tmp_path, model="m")

    defaults = (_cfg.FINALIZE_AUTO, _cfg.FINALIZE_TUNER_TIMEOUT)
    _cfg.init_from_config(_cfg.OptimizerConfig(**base))
    assert (_cfg.FINALIZE_AUTO, _cfg.FINALIZE_TUNER_TIMEOUT) == defaults == (True, 1000)

    # Overrides must differ from the defaults, or this proves nothing.
    _cfg.init_from_config(
        _cfg.OptimizerConfig(**base, finalize=False, finalize_timeout=60)
    )
    assert (_cfg.FINALIZE_AUTO, _cfg.FINALIZE_TUNER_TIMEOUT) == (False, 60)


def test_cli_accepts_the_finalize_flags(monkeypatch):
    import cli

    monkeypatch.setattr(sys, "argv", ["cli.py", "--dir", "x"])
    a = cli.parse_args()
    assert (a.finalize, a.no_finalize, a.finalize_timeout) == (False, False, None)

    monkeypatch.setattr(
        sys, "argv",
        ["cli.py", "--dir", "x", "--finalize", "--no-finalize", "--finalize-timeout", "60"],
    )
    a = cli.parse_args()
    assert (a.finalize, a.no_finalize, a.finalize_timeout) == (True, True, 60)


# -- cli --finalize ---------------------------------------------------------


def test_run_pid_alive_cases(tmp_path):
    import os

    import cli

    assert cli._run_pid_alive(tmp_path) is False  # no file
    pid_file = tmp_path / "run.pid"
    pid_file.write_text("not a pid")
    assert cli._run_pid_alive(tmp_path) is False  # garbage
    pid_file.write_text("2147483646")
    assert cli._run_pid_alive(tmp_path) is False  # dead pid
    pid_file.write_text(str(os.getpid()))
    assert cli._run_pid_alive(tmp_path) is True


def test_finalize_only_refuses_when_there_is_nothing_to_finalize(tmp_path, monkeypatch):
    import cli
    import utils.branch_bests

    (tmp_path / "output").mkdir()
    monkeypatch.setattr(cli, "check_prerequisites", lambda d: True)
    monkeypatch.setattr(utils.branch_bests, "collect_branch_bests", lambda o, p: [])
    assert asyncio.run(cli._finalize_only(tmp_path)) == 1
    assert not (tmp_path / "output" / "run.pid").exists()


class _Response:
    """What TrackedLLM hands the tracker: langchain's normalised usage."""

    usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


@pytest.fixture
def finalize_only_env(tmp_path, monkeypatch):
    """A finished run's output/ with its token spend on disk, a fresh process's counter,
    and a run_finalize that spends one LLM call and notes what run.pid said meanwhile."""
    import utils.branch_bests

    out = tmp_path / "output"
    out.mkdir()
    (out / "token_usage.json").write_text(json.dumps({
        "api_calls": 10, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
    }), encoding="utf-8")
    monkeypatch.setattr(_cfg, "_base_output_dir", out)
    monkeypatch.setattr(_cfg, "_problem_dir", tmp_path)
    for field in ("api_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
        monkeypatch.setattr(_cfg.global_tracker, field, 0)
    monkeypatch.setattr(utils.branch_bests, "collect_branch_bests", lambda o, p: [object()])
    seen = {}

    async def fake_finalize(*, trigger):
        pid_file = out / "run.pid"
        seen["pid"] = pid_file.read_text(encoding="utf-8") if pid_file.exists() else None
        seen["trigger"] = trigger
        _cfg.global_tracker.add(_Response())

    monkeypatch.setattr("engine.finalize.run_finalize", fake_finalize)
    return out, seen


def _usage(out):
    return json.loads((out / "token_usage.json").read_text(encoding="utf-8"))


def test_cli_finalize_claims_run_pid_and_saves_its_token_spend(finalize_only_env, monkeypatch):
    import cli
    import nodes.merge

    out, seen = finalize_only_env
    monkeypatch.setattr(cli, "check_prerequisites", lambda d: True)
    monkeypatch.setattr(nodes.merge, "merge_node", lambda: {"status": "success"})

    assert asyncio.run(cli._finalize_only(out.parent)) == 0

    # The API reads run.pid to refuse a second finalize over the same manifest.
    assert seen == {"pid": str(os.getpid()), "trigger": "manual"}
    assert not (out / "run.pid").exists()
    usage = _usage(out)
    assert (usage["api_calls"], usage["total_tokens"]) == (11, 1320)


def test_cli_finalize_leaves_a_live_runs_pid_file_alone(tmp_path, monkeypatch):
    import cli

    out = tmp_path / "output"
    out.mkdir()
    (out / "run.pid").write_text(str(os.getppid()), encoding="utf-8")  # alive: our parent
    monkeypatch.setattr(cli, "check_prerequisites", lambda d: True)

    assert asyncio.run(cli._finalize_only(tmp_path)) == 1
    assert (out / "run.pid").read_text(encoding="utf-8") == str(os.getppid())


def test_the_api_finalize_child_saves_its_token_spend(finalize_only_env, monkeypatch):
    import api.optimizer
    import engine.master
    import utils.inputs

    out, seen = finalize_only_env
    monkeypatch.setattr(api.optimizer.os, "setsid", lambda: None)
    monkeypatch.setattr(_cfg, "init_from_config", lambda c: None)
    monkeypatch.setattr(_cfg, "check_api_key", lambda: None)
    monkeypatch.setattr(utils.inputs, "ensure_inputs_hpp", lambda d: None)
    monkeypatch.setattr(engine.master, "_write_final_results", lambda: None)

    api.optimizer._optimizer_target(out.parent, 1, 1, 1, None, finalize_only=True)

    assert seen == {"pid": str(os.getpid()), "trigger": "manual"}
    usage = _usage(out)
    assert (usage["api_calls"], usage["total_tokens"]) == (11, 1320)
