from tests.helpers.finalize_fixtures import invalid_summary, make_branch, valid_summary
from utils.branch_bests import (
    collect_branch_bests,
    rank_key,
    source_fingerprint,
    summary_rank_key,
)


def _collect(tmp_path):
    return collect_branch_bests(tmp_path, tmp_path)


def test_a_failed_branch_with_an_earlier_valid_iteration_is_eligible(tmp_path):
    make_branch(tmp_path, "a", "failed", [valid_summary(2.0, 50), invalid_summary()])
    (b,) = _collect(tmp_path)
    assert b.best_iter == 1 and b.status == "failed" and b.score == 2.0


def test_an_interrupted_branch_is_eligible(tmp_path):
    make_branch(tmp_path, "a", "running", [valid_summary(1.5, 60)])
    assert [b.name for b in _collect(tmp_path)] == ["a"]


def test_a_branch_with_no_valid_iteration_is_excluded(tmp_path):
    make_branch(tmp_path, "a", "failed", [invalid_summary(), None])
    assert _collect(tmp_path) == []


def test_a_failed_case_makes_an_iteration_invalid(tmp_path):
    bad = dict(valid_summary(9.0, 10), failed_case="small")
    make_branch(tmp_path, "a", "success", [valid_summary(2.0, 50), bad])
    (b,) = _collect(tmp_path)
    assert b.best_iter == 1


def test_bests_are_ranked_by_score(tmp_path):
    make_branch(tmp_path, "slow", "success", [valid_summary(1.2, 90)])
    make_branch(tmp_path, "fast", "success", [valid_summary(1.1, 70), valid_summary(3.0, 30)])
    bests = _collect(tmp_path)
    assert [b.name for b in bests] == ["fast", "slow"]
    assert bests[0].best_iter == 2


def test_without_speedups_ranking_falls_back_to_time(tmp_path):
    make_branch(tmp_path, "a", "success", [valid_summary(None, 80)])
    make_branch(tmp_path, "b", "success", [valid_summary(None, 50)])
    bests = _collect(tmp_path)
    assert [b.name for b in bests] == ["b", "a"]
    assert bests[0].score is None and bests[0].time_us == 50


def test_nested_branches_are_found_with_a_relative_path(tmp_path):
    parent = make_branch(tmp_path, "p", "branching", [valid_summary(1.0, 100)])
    make_branch(tmp_path, "c", "success", [valid_summary(2.0, 40)], parent=parent)
    rels = {b.rel for b in _collect(tmp_path)}
    assert rels == {"branches/p", "branches/p/branches/c"}


def test_summary_rank_key_orders_none_last():
    assert summary_rank_key(None) > summary_rank_key({"speedup": 0.1, "best_time_us": 1e9})
    assert summary_rank_key({"speedup": 2.0, "best_time_us": 9}) < summary_rank_key(
        {"speedup": 1.0, "best_time_us": 1}
    )
    assert summary_rank_key({"speedup": None, "best_time_us": 5}) < summary_rank_key(
        {"speedup": None, "best_time_us": 6}
    )


def test_the_fingerprint_is_stable_and_tracks_scores(tmp_path):
    make_branch(tmp_path, "a", "success", [valid_summary(2.0, 50)])
    first = source_fingerprint(_collect(tmp_path))
    assert first == source_fingerprint(_collect(tmp_path))
    make_branch(tmp_path, "a", "success", [valid_summary(2.5, 40)])
    assert source_fingerprint(_collect(tmp_path)) != first


def test_recorded_run_yields_bests(cov_output, cov_problem):
    bests = collect_branch_bests(cov_output, cov_problem)
    assert bests
    assert [rank_key(b) for b in bests] == sorted(rank_key(b) for b in bests)
    assert all((b.path / f"iter{b.best_iter}").is_dir() for b in bests)


def test_the_strategy_description_is_read_from_the_manifest(tmp_path):
    import json

    d = make_branch(tmp_path, "a", "success", [valid_summary(2.0, 50)])
    make_branch(tmp_path, "b", "success", [valid_summary(1.0, 90)])
    manifest = json.loads((d / "branch.json").read_text())
    manifest["strategy"]["description"] = "tile the loop"
    (d / "branch.json").write_text(json.dumps(manifest))
    by_name = {b.name: b for b in _collect(tmp_path)}
    assert by_name["a"].description == "tile the loop"
    assert by_name["b"].description == ""
