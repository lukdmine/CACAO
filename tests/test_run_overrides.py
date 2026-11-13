"""run_node's keyword-only overrides for finalize, and the budget precedence they add."""

from types import SimpleNamespace

from tests.helpers.run_harness import commands, run_two_case_node


def test_the_finalize_budget_beats_every_other_source(monkeypatch):
    import config
    from nodes.run import _resolve_tuning_budget

    monkeypatch.setattr(config, "TUNER_TIMEOUT_OVERRIDE", 77.0)
    case = SimpleNamespace(name="big", duration_s=5)
    meta = {"tuning": {"duration_s": 10}}
    assert _resolve_tuning_budget(meta, case, 1000.0) == (1000.0, "finalize")
    assert _resolve_tuning_budget(meta, case) == (77.0, "user override")


def test_mode_and_per_case_seed_reach_each_driver(tmp_path):
    run_two_case_node(
        tmp_path,
        run_kwargs={"mode": "mcmc", "seed_results_by_case": {"big": "/s/big/results"}},
    )
    big, small = commands()
    assert big[-3:] == ["", "mcmc", "/s/big/results"]
    assert small[-3:] == ["", "mcmc", ""]


def test_defaults_leave_the_driver_command_alone(tmp_path):
    run_two_case_node(tmp_path)
    for cmd in commands():
        assert len(cmd) == 8
