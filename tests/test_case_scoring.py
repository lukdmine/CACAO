"""Multi-case scoring.

The rule that carries the most weight is the null geomean: dropping a case that has no
reference time would let a problem score better by having an untimed case.
"""

import math

from utils.results import aggregate_case_summaries


def _summary(best, ref, ok=True):
    speedup = (ref / best) if (ref and best) else None
    return {
        "has_success": ok,
        "num_successful": 100 if ok else 0,
        "num_total": 100,
        "best_config": {"TILE": 16},
        "best_time_us": best,
        "reference_time_us": ref,
        "speedup": speedup,
    }


def test_single_case_passes_todays_dict_through_unchanged():
    one = _summary(470.0, 411.28)
    assert aggregate_case_summaries({"default": one}, "default") == one


def test_geomean_over_two_cases():
    out = aggregate_case_summaries(
        {"a": _summary(100.0, 200.0), "b": _summary(100.0, 800.0)}, "a"
    )
    assert out["geomean_speedup"] == math.sqrt(2.0 * 8.0)
    assert out["speedup"] == out["geomean_speedup"]


def test_flat_keys_describe_the_primary_case():
    out = aggregate_case_summaries(
        {"a": _summary(100.0, 200.0), "b": _summary(100.0, 800.0)}, "a"
    )
    assert out["best_time_us"] == 100.0
    assert out["reference_time_us"] == 200.0
    assert out["cases"]["b"]["reference_time_us"] == 800.0


def test_missing_reference_nulls_the_whole_geomean():
    out = aggregate_case_summaries(
        {"a": _summary(100.0, 200.0), "b": _summary(100.0, None)}, "a"
    )
    assert out["geomean_speedup"] is None
    assert out["speedup"] is None


def test_worst_case_is_the_lowest_speedup():
    out = aggregate_case_summaries(
        {"a": _summary(100.0, 800.0), "b": _summary(100.0, 200.0)}, "a"
    )
    assert out["worst_case"] == "b"


def test_a_case_that_did_not_run_is_reported_as_the_failure():
    out = aggregate_case_summaries({"a": _summary(100.0, 200.0), "b": None}, "a")
    assert out["failed_case"] == "b"
    assert out["geomean_speedup"] is None


def test_a_case_with_no_valid_configuration_is_a_failure():
    out = aggregate_case_summaries(
        {"a": _summary(100.0, 200.0), "b": _summary(None, 200.0, ok=False)}, "a"
    )
    assert out["failed_case"] == "b"
