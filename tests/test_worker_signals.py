"""Control-signal handling and the main loop's exit conditions."""

import json
from pathlib import Path

import pytest

from engine.worker import _handle_signal
from state.types import BranchManifest, IterState, StrategyInfo


@pytest.fixture
def branch(tmp_path) -> Path:
    d = tmp_path / "branches" / "tiled"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def manifest() -> BranchManifest:
    return BranchManifest(
        strategy=StrategyInfo(name="tiled"), status="running", current_iter=3
    )


@pytest.fixture
def iter_state() -> IterState:
    return IterState(iter_num=3, status="running")


def _signal(action: str, **extra) -> dict:
    return {"action": action, **extra}


def test_stop_records_where_to_resume_to(branch, manifest, iter_state):
    _handle_signal(_signal("stop"), manifest, iter_state, branch)

    assert manifest.status == "stopped"
    assert manifest.pre_stop_status == "running"


def test_resume_restores_the_pre_stop_status(branch, manifest, iter_state):
    _handle_signal(_signal("stop"), manifest, iter_state, branch)
    _handle_signal(_signal("resume"), manifest, iter_state, branch)

    assert manifest.status == "running"
    assert manifest.pre_stop_status is None


def test_second_stop_does_not_overwrite_the_saved_status(branch, manifest, iter_state):
    """A stop on an already-stopped branch must not make "stopped" what resume
    returns to — that wedges _wait_for_resume, and the run's termination check."""
    _handle_signal(_signal("stop"), manifest, iter_state, branch)
    _handle_signal(_signal("stop"), manifest, iter_state, branch)

    assert manifest.pre_stop_status == "running"

    _handle_signal(_signal("resume"), manifest, iter_state, branch)

    assert manifest.status == "running"
    assert manifest.pre_stop_status is None


def test_stop_stop_resume_resume_leaves_the_branch_runnable(branch, manifest, iter_state):
    """The full sequence from the report: two stops, then two resumes."""
    for action in ("stop", "stop", "resume", "resume"):
        _handle_signal(_signal(action), manifest, iter_state, branch)

    assert manifest.status == "running"


def test_a_repeated_stop_still_records_its_message(branch, manifest, iter_state):
    """Suppressing the status write must not suppress the user's message with it."""
    _handle_signal(_signal("stop"), manifest, iter_state, branch)
    _handle_signal(_signal("stop", content="actually stop now"), manifest, iter_state, branch)

    assert [m["content"] for m in iter_state.user_messages] == ["actually stop now"]


# -- the main loop ----------------------------------------------------------


import pytest_asyncio  # noqa: F401  (asyncio marker support)

from engine.worker import run_branch_loop
from state import save_branch_manifest


@pytest.mark.asyncio
async def test_a_deleted_branch_directory_exits_instead_of_rebuilding_itself(
    tmp_path, monkeypatch
):
    """Every write path mkdirs its parent, so the branch used to rebuild the subtree
    a parent revert had deleted and carry on running."""
    branch = tmp_path / "branches" / "vectorized"
    branch.mkdir(parents=True)
    manifest = BranchManifest(
        strategy=StrategyInfo(name="vectorized"), status="running", current_iter=1
    )
    save_branch_manifest(branch, manifest)
    (branch.parent.parent / "context.json").write_text(
        json.dumps({"analysis": "a"}), encoding="utf-8"
    )

    import shutil

    shutil.rmtree(branch)

    result = await run_branch_loop(branch)

    assert result == []
    assert not branch.exists()


@pytest.mark.asyncio
async def test_a_crash_before_the_iteration_loads_still_marks_the_branch_failed(
    tmp_path, monkeypatch
):
    """A failure loading iter_state used to raise UnboundLocalError from the handler,
    leaving the branch dead with no status."""
    import engine.worker as worker

    branch = tmp_path / "branches" / "tiled"
    branch.mkdir(parents=True)
    manifest = BranchManifest(
        strategy=StrategyInfo(name="tiled"), status="running", current_iter=1
    )
    save_branch_manifest(branch, manifest)
    (branch.parent.parent / "context.json").write_text(
        json.dumps({"analysis": "a"}), encoding="utf-8"
    )

    def boom(*args, **kwargs):
        raise ValueError("corrupt state.json")

    monkeypatch.setattr(worker, "load_iter_state_if_exists", boom)

    result = await run_branch_loop(branch)

    assert result == []
    assert worker.load_branch_manifest(branch).status == "failed"
