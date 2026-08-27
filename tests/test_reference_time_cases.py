"""Per-case reference times, and the back-compatibility of the four committed seeds."""

import json

import pytest

from utils.results import load_reference_time, save_reference_time


def test_flat_seed_reads_as_the_primary_case(tmp_path):
    (tmp_path / "reference_time.json").write_text(
        json.dumps({"reference_time_us": 411.28}), encoding="utf-8"
    )
    assert load_reference_time(tmp_path) == 411.28


def test_flat_seed_gives_no_baseline_for_a_named_case(tmp_path):
    (tmp_path / "reference_time.json").write_text(
        json.dumps({"reference_time_us": 411.28}), encoding="utf-8"
    )
    assert load_reference_time(tmp_path, "t127") is None


def test_named_case_round_trips(tmp_path):
    save_reference_time(tmp_path, 411.28)
    save_reference_time(tmp_path, 96.0, "t127")
    assert load_reference_time(tmp_path) == 411.28
    assert load_reference_time(tmp_path, "t127") == 96.0


def test_saving_a_case_preserves_the_primary(tmp_path):
    save_reference_time(tmp_path, 411.28)
    save_reference_time(tmp_path, 96.0, "t127")
    data = json.loads((tmp_path / "reference_time.json").read_text(encoding="utf-8"))
    assert data["reference_time_us"] == 411.28
    assert data["cases"]["t127"] == 96.0


def test_first_write_wins_per_case(tmp_path):
    save_reference_time(tmp_path, 96.0, "t127")
    save_reference_time(tmp_path, 500.0, "t127")
    assert load_reference_time(tmp_path, "t127") == 96.0


def test_first_write_wins_for_the_primary(tmp_path):
    save_reference_time(tmp_path, 411.28)
    save_reference_time(tmp_path, 999.0)
    assert load_reference_time(tmp_path) == 411.28


def test_a_case_saved_first_does_not_block_the_primary(tmp_path):
    """The primary's O_EXCL fast path must not be reached once `cases` exists."""
    save_reference_time(tmp_path, 96.0, "t127")
    save_reference_time(tmp_path, 411.28)
    assert load_reference_time(tmp_path) == 411.28
    assert load_reference_time(tmp_path, "t127") == 96.0


def test_missing_file_is_none(tmp_path):
    assert load_reference_time(tmp_path) is None
    assert load_reference_time(tmp_path, "t127") is None


def test_corrupt_file_is_none_not_an_exception(tmp_path):
    (tmp_path / "reference_time.json").write_text("{not json", encoding="utf-8")
    assert load_reference_time(tmp_path) is None
