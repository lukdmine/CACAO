"""The case table the agent is shown, and the rule that comes with it."""

from prompts._cases import cases_block


def test_single_case_adds_nothing():
    assert cases_block({}, None) == ""


def test_table_lists_every_case_with_its_overrides():
    meta = {"cases": [{"name": "t512", "scalars": {"kT": 512}},
                      {"name": "t127", "scalars": {"kT": 127}}]}
    out = cases_block(meta, None)
    assert "t512" in out and "kT=512" in out
    assert "t127" in out and "kT=127" in out
    assert "every case" in out.lower() or "EVERY one" in out


def test_table_carries_measured_times_when_present():
    meta = {"cases": [{"name": "a"}, {"name": "b"}]}
    summary = {
        "cases": {
            "a": {"best_time_us": 100.0, "speedup": 2.0},
            "b": {"best_time_us": 400.0, "speedup": 0.5},
        },
        "geomean_speedup": 1.0,
    }
    out = cases_block(meta, summary)
    assert "100.00" in out and "0.50x" in out
    assert "geometric mean" in out


def test_a_case_that_did_not_run_is_shown_as_such():
    meta = {"cases": [{"name": "a"}, {"name": "b"}]}
    summary = {"cases": {"a": {"best_time_us": 100.0, "speedup": 2.0}, "b": None},
               "geomean_speedup": None, "failed_case": "b"}
    out = cases_block(meta, summary)
    assert "not run" in out


def test_a_missing_baseline_says_so_instead_of_showing_a_partial_mean():
    meta = {"cases": [{"name": "a"}, {"name": "b"}]}
    summary = {"cases": {"a": {"best_time_us": 100.0, "speedup": 2.0},
                         "b": {"best_time_us": 400.0, "speedup": None}},
               "geomean_speedup": None}
    out = cases_block(meta, summary)
    assert "not available" in out and "subset" in out
