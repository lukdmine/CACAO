"""KTT 2.3 added a Timestamp field that its deserializer requires.

`Tuner::LoadResults` reads it with `j.at("Timestamp")`, which throws rather than
defaulting, so a results.json written by KTT 2.2 aborts the driver in profile mode
with `[json.exception.out_of_range.403] key 'Timestamp' not found`. Verified against
the real library before this shim was written.
"""

import json

import pytest

from utils.results import ensure_results_loadable


def _write(path, results):
    path.write_text(json.dumps({"Metadata": {}, "Results": results}), encoding="utf-8")


def test_injects_timestamp_into_pre_2_3_results(tmp_path):
    path = tmp_path / "results.json"
    _write(path, [{"Status": "Ok"}, {"Status": "Ok"}])

    assert ensure_results_loadable(path) is True

    written = json.loads(path.read_text(encoding="utf-8"))
    assert [r["Timestamp"] for r in written["Results"]] == ["", ""]


def test_leaves_a_current_file_untouched(tmp_path):
    """No rewrite on a fresh run's own file — the common case is a no-op."""
    path = tmp_path / "results.json"
    _write(path, [{"Status": "Ok", "Timestamp": "2026-08-26T10:00:00"}])
    before = path.read_text(encoding="utf-8")

    assert ensure_results_loadable(path) is False
    assert path.read_text(encoding="utf-8") == before


def test_preserves_an_existing_timestamp_when_only_some_are_missing(tmp_path):
    path = tmp_path / "results.json"
    _write(path, [{"Timestamp": "keep-me"}, {"Status": "Ok"}])

    assert ensure_results_loadable(path) is True

    written = json.loads(path.read_text(encoding="utf-8"))
    assert [r["Timestamp"] for r in written["Results"]] == ["keep-me", ""]


def test_preserves_every_other_field(tmp_path):
    """The rewrite must not drop data the engine reads back out of the same file."""
    path = tmp_path / "results.json"
    result = {
        "Status": "Ok",
        "TotalDuration": 12.5,
        "ComputationResults": [{"Duration": 11.0}],
        "Configuration": [{"Name": "TILE", "Value": 16}],
    }
    _write(path, [result])

    assert ensure_results_loadable(path) is True

    written = json.loads(path.read_text(encoding="utf-8"))["Results"][0]
    assert {k: written[k] for k in result} == result


@pytest.mark.parametrize(
    "payload",
    [
        '{"Metadata": {}}',          # no Results key at all
        '{"Results": "not-a-list"}',  # Results present but the wrong shape
        "not json at all",
    ],
)
def test_malformed_files_are_reported_not_raised(tmp_path, payload):
    """A file the profile node cannot fix is its problem to report, not ours to crash on."""
    path = tmp_path / "results.json"
    path.write_text(payload, encoding="utf-8")
    assert ensure_results_loadable(path) is False


def test_missing_file_is_not_an_error(tmp_path):
    assert ensure_results_loadable(tmp_path / "nope.json") is False
