"""A revert must not leave the branch advertising a time it can no longer produce.

`best_time_us` is a running minimum: nodes/run.py only ever lowers it, the worker
copies it onto the manifest, and api/tree.py treats the manifest value as a floor it
can only lower further. Nothing raises it back. So deleting the iteration that set the
record used to leave the number behind, pointing at a kernels.cu and results.json that
revert had just removed — and it is read by the frontend, by --best via nodes/merge.py,
and by sibling branches via state/crossbranch.py as the score to beat.
"""

import pytest

from state.control import revert_branch_on_disk
from state.persistence import (
    load_branch_manifest,
    save_branch_manifest,
    save_iter_state,
)
from state.types import BranchManifest, IterState


def _branch(tmp_path, times):
    """A branch whose iteration N recorded times[N-1] microseconds (None = no result)."""
    branch = tmp_path / "branches" / "b"
    branch.mkdir(parents=True)

    best = min((t for t in times if t is not None), default=None)
    manifest = BranchManifest(current_iter=len(times), status="running")
    manifest.best_time_us = best
    manifest.speedup = (1000.0 / best) if best else None
    save_branch_manifest(branch, manifest)

    for i, t in enumerate(times, start=1):
        state = IterState(iter_num=i, status="decided")
        if t is not None:
            state.results_summary = {"best_time_us": t, "speedup": 1000.0 / t}
        save_iter_state(branch, i, state)
    return branch


def test_best_time_drops_when_the_iteration_that_set_it_is_deleted(tmp_path):
    # iter3 is the record holder; reverting to iter2 destroys it.
    branch = _branch(tmp_path, [900.0, 800.0, 500.0])
    assert load_branch_manifest(branch).best_time_us == 500.0

    revert_branch_on_disk(branch, target_iter=2)

    manifest = load_branch_manifest(branch)
    assert manifest.best_time_us == 800.0
    assert manifest.speedup == pytest.approx(1000.0 / 800.0)


def test_best_time_survives_when_the_record_holder_survives(tmp_path):
    # The record is at iter1, which the revert keeps — the number must not move.
    branch = _branch(tmp_path, [500.0, 800.0, 900.0])

    revert_branch_on_disk(branch, target_iter=2)

    assert load_branch_manifest(branch).best_time_us == 500.0


def test_reverting_past_every_result_clears_the_best(tmp_path):
    """iter1 never produced a timing, so nothing on disk backs a best time."""
    branch = _branch(tmp_path, [None, 800.0, 500.0])

    revert_branch_on_disk(branch, target_iter=1)

    manifest = load_branch_manifest(branch)
    assert manifest.best_time_us is None
    assert manifest.speedup is None


def test_the_deleted_iterations_are_really_gone(tmp_path):
    """Guards the premise: if revert stopped deleting, the recompute would be a no-op."""
    branch = _branch(tmp_path, [900.0, 800.0, 500.0])

    revert_branch_on_disk(branch, target_iter=2)

    assert (branch / "iter1").is_dir()
    assert (branch / "iter2").is_dir()
    assert not (branch / "iter3").exists()


def test_an_iteration_with_no_results_summary_is_skipped_not_fatal(tmp_path):
    branch = _branch(tmp_path, [700.0, None, 500.0])

    revert_branch_on_disk(branch, target_iter=2)

    assert load_branch_manifest(branch).best_time_us == 700.0
