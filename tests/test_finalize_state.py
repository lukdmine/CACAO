"""The finalize manifest: model defaults and the save/load round trip."""

import logging

from state.persistence import (
    finalize_dir,
    load_finalize_manifest,
    save_finalize_manifest,
)
from state.types import FINALIZE_STATUSES, FinalistState, FinalizeManifest, Variant


def _manifest():
    src = Variant(dir="branches/a/iter3", summary={"best_time_us": 12.5, "speedup": 2.0})
    fin = FinalistState(
        name="a",
        source_branch="branches/a",
        source_iter=3,
        source=src,
        reported="finalized",
        reported_dir="finalize/a",
    )
    return FinalizeManifest(status="tuning", finalists=[fin], category_bests={"tensor_cores=true": "a"})


def test_round_trip(tmp_path):
    m = _manifest()
    save_finalize_manifest(tmp_path, m)
    assert (finalize_dir(tmp_path) / "finalize.json").is_file()
    assert load_finalize_manifest(tmp_path) == m


def test_reported_dir_survives_round_trip(tmp_path):
    save_finalize_manifest(tmp_path, _manifest())
    loaded = load_finalize_manifest(tmp_path)
    assert loaded.finalists[0].reported_dir == "finalize/a"
    assert loaded.finalists[0].reported == "finalized"


def test_load_absent_returns_none(tmp_path):
    assert load_finalize_manifest(tmp_path) is None


def test_load_corrupt_returns_none_and_warns(tmp_path, caplog):
    (tmp_path / "finalize").mkdir()
    (tmp_path / "finalize" / "finalize.json").write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert load_finalize_manifest(tmp_path) is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_defaults_are_not_shared():
    a, b = FinalistState(name="x", source_branch="b", source_iter=1, source=Variant(dir="d")), FinalistState(
        name="y", source_branch="b", source_iter=1, source=Variant(dir="d")
    )
    a.stages["widen"] = "done"
    assert b.stages["widen"] == "pending"
    assert "done" in FINALIZE_STATUSES
