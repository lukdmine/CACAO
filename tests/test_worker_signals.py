"""Control-signal handling in the branch worker.

These drive ``engine.worker._handle_signal`` directly rather than through a running
loop: the failure it guards against is a state-machine one — a field that survives
across signals — and a loop would only reproduce it after minutes of real polling.
"""

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
