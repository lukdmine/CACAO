"""What changed in the tuning space between two iterations' PARAMS regions."""

from utils.param_history import diff_parameters

P = 'tuner.AddParameter(kernel, "{n}", std::vector<uint64_t>{{{v}}});\n'


def test_a_narrowed_parameter_is_reported():
    d = diff_parameters(P.format(n="TILE", v="16, 32, 64"), P.format(n="TILE", v="32"))
    assert d.narrowed == {"TILE": (["16", "32", "64"], ["32"])}
    assert "TILE" in d.render()


def test_a_dropped_parameter_is_reported():
    d = diff_parameters(P.format(n="TILE", v="16") + P.format(n="UNROLL", v="1, 2"),
                        P.format(n="TILE", v="16"))
    assert d.dropped == ["UNROLL"]


def test_value_order_does_not_count_as_a_change():
    assert diff_parameters(P.format(n="T", v="32, 16"), P.format(n="T", v="16, 32")).render() == ""


def test_widening_is_silent_unless_asked_for():
    d = diff_parameters(P.format(n="T", v="16"), P.format(n="T", v="16, 32"))
    assert d.render() == "" and "T" in d.render(include_widened=True)


def test_values_compare_numerically():
    assert diff_parameters(P.format(n="T", v="16"), P.format(n="T", v="16.0")).render() == ""


def test_an_added_parameter_is_recorded():
    d = diff_parameters(P.format(n="T", v="16"), P.format(n="T", v="16") + P.format(n="U", v="1"))
    assert d.added == ["U"] and d.render() == ""


# -- parameter_history ------------------------------------------------------

import json

from tests.helpers.run_harness import PROBLEM_YAML
from utils.framework import assemble_framework_cpp
from utils.param_history import parameter_history

_META = {k: v for k, v in PROBLEM_YAML.items() if k != "cases"}


def _results(path, rows):
    """rows: [(TILE value, duration_us)] as a minimal KTT results.json."""
    path.write_text(json.dumps({"Results": [
        {"Status": "Ok", "TotalDuration": 1, "TotalOverhead": 1,
         "Configuration": [{"Name": "TILE", "Value": v}],
         "ComputationResults": [{"Duration": d}]}
        for v, d in rows
    ]}), encoding="utf-8")


def _branch(tmp_path):
    b = tmp_path / "branch"
    (b / "iter1").mkdir(parents=True)
    (b / "iter2").mkdir()
    return b


def test_history_reads_an_iteration_that_only_has_framework_cpp(tmp_path):
    b = _branch(tmp_path)
    old = P.format(n="TILE", v="8, 16, 32")
    (b / "iter1" / "framework.cpp").write_text(
        assemble_framework_cpp(_META, {"kernels": "", "params": old, "launcher": ""},
                               cuda_include="/x"), encoding="utf-8")
    cur = P.format(n="TILE", v="16")
    (b / "iter2" / "region_params.cpp").write_text(cur, encoding="utf-8")
    out = parameter_history(b, cur, tmp_path, _META)
    iter1_line = next(l for l in out.splitlines() if l.strip().startswith("iter1"))
    assert "8" in iter1_line and "16" in iter1_line and "32" in iter1_line


def test_history_reports_best_time_per_value(tmp_path):
    b = _branch(tmp_path)
    cur = P.format(n="TILE", v="8, 16")
    (b / "iter1" / "region_params.cpp").write_text(cur, encoding="utf-8")
    _results(b / "iter1" / "results.json", [("8", 50.0), ("8", 40.0), ("16", 30.0)])
    out = parameter_history(b, cur, tmp_path, _META)
    assert "8 (40.0 us)" in out and "16 (30.0 us)" in out


def test_history_lists_dropped_parameters(tmp_path):
    b = _branch(tmp_path)
    (b / "iter1" / "region_params.cpp").write_text(
        P.format(n="TILE", v="16") + P.format(n="UNROLL", v="1, 2"), encoding="utf-8")
    cur = P.format(n="TILE", v="16")
    (b / "iter2" / "region_params.cpp").write_text(cur, encoding="utf-8")
    out = parameter_history(b, cur, tmp_path, _META)
    tail = out.split("No longer declared:")[1]
    assert "UNROLL" in tail and "iter1" in tail and "TILE" not in tail


def test_history_orders_iterations_numerically(tmp_path):
    b = tmp_path / "branch"
    for n, v in ((2, "8"), (10, "16")):
        (b / f"iter{n}").mkdir(parents=True)
        (b / f"iter{n}" / "region_params.cpp").write_text(P.format(n="TILE", v=v), encoding="utf-8")
    out = parameter_history(b, P.format(n="TILE", v="16"), tmp_path, _META)
    assert out.index("iter2") < out.index("iter10")


def test_history_collapses_identical_iterations(tmp_path):
    b = tmp_path / "branch"
    src = P.format(n="TILE", v="8, 16") + P.format(n="UNROLL", v="1, 2")
    for n in range(1, 26):
        (b / f"iter{n}").mkdir(parents=True)
        (b / f"iter{n}" / "region_params.cpp").write_text(src, encoding="utf-8")
    out = parameter_history(b, src, tmp_path, _META)
    assert out.count("iter1-iter25") == 2
    assert len(out.splitlines()) == 4


def test_an_oversized_history_ends_with_a_truncation_note(tmp_path):
    from utils import param_history as ph

    b = tmp_path / "branch"
    (b / "iter1").mkdir(parents=True)
    src = "".join(P.format(n=f"P{i}", v="1, 2, 3") for i in range(2000))
    (b / "iter1" / "region_params.cpp").write_text(src, encoding="utf-8")
    out = parameter_history(b, src, tmp_path, _META)
    assert len(out) <= ph._MAX_CHARS
    assert out.splitlines()[-1].startswith("... (truncated")


def _write_params(b, n, src):
    (b / f"iter{n}").mkdir(parents=True, exist_ok=True)
    (b / f"iter{n}" / "region_params.cpp").write_text(src, encoding="utf-8")


def test_history_groups_a_b_a_value_sets_separately(tmp_path):
    b = tmp_path / "branch"
    for n, v in ((1, "8"), (2, "16"), (3, "8")):
        _write_params(b, n, P.format(n="TILE", v=v))
    out = parameter_history(b, P.format(n="TILE", v="8"), tmp_path, _META)
    assert len([l for l in out.splitlines() if l.startswith("  iter")]) == 3


def test_a_non_declaring_iteration_breaks_the_run(tmp_path):
    b = tmp_path / "branch"
    _write_params(b, 1, P.format(n="TILE", v="8, 16"))
    _write_params(b, 2, P.format(n="UNROLL", v="1"))
    _write_params(b, 3, P.format(n="TILE", v="8, 16"))
    out = parameter_history(b, P.format(n="TILE", v="8, 16"), tmp_path, _META)
    assert "iter1-iter3" not in out
    assert "iter1: 8, 16" in out and "iter3: 8, 16" in out


def test_a_best_time_difference_breaks_the_run(tmp_path):
    b = tmp_path / "branch"
    src = P.format(n="TILE", v="8, 16")
    for n in (1, 2):
        _write_params(b, n, src)
    _results(b / "iter1" / "results.json", [("8", 40.0), ("16", 30.0)])
    _results(b / "iter2" / "results.json", [("8", 40.0), ("16", 20.0)])
    out = parameter_history(b, src, tmp_path, _META)
    assert "iter1-iter2" not in out
    assert "16 (30.0 us)" in out and "16 (20.0 us)" in out


def test_no_longer_declared_survives_truncation(tmp_path):
    from utils import param_history as ph

    b = tmp_path / "branch"
    big = "".join(P.format(n=f"P{i}", v="1, 2, 3") for i in range(2000))
    _write_params(b, 1, big + P.format(n="GONE", v="1"))
    _write_params(b, 2, big)
    out = parameter_history(b, big, tmp_path, _META)
    assert len(out) <= ph._MAX_CHARS
    assert "(truncated" in out
    assert out.splitlines()[-2:] == ["No longer declared:", "- GONE (last declared in iter1)"]
    assert not out.startswith("\n")
