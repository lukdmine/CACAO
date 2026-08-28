"""Control-signal handling in the branch worker.

These drive ``engine.worker._handle_signal`` directly rather than through a running
loop: the failure it guards against is a state-machine one — a field that survives
across signals — and a loop would only reproduce it after minutes of real polling.
"""

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
    """A stop delivered to an already-stopped branch must not make "stopped" the
    thing resume comes back to.

    The API applies no status check before writing a stop signal, and the UI leaves
    the button clickable for up to one poll interval after a branch parks, so this
    is reachable without a race. When it stored pre_stop_status="stopped", resume
    restored "stopped" and cleared the field, and the branch could never leave
    _wait_for_resume again — taking the whole run's termination check with it.
    """
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
    """A parent revert rmtree's the subtree; every write path recreates it.

    _atomic_write_json mkdirs its parent, so save_iter_state and
    save_branch_manifest rebuilt a directory the user had just deleted and the
    branch carried on — reappearing in the tree and still spending LLM calls and
    GPU lock time. The cooperative stop cannot help: the rmtree destroys the
    signal file before the worker's next poll.
    """
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
    """iter_state is bound inside the try, so a failure loading it used to raise
    UnboundLocalError out of the handler — and the branch died with its status
    never set to failed."""
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
