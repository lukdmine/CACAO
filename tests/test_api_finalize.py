"""The finalize API: start, report, and the guards that keep branch edits out while it runs."""


import os
import subprocess
import sys

import pytest

from state import save_finalize_manifest
from state.types import FinalizeManifest
from tests.helpers.finalize_fixtures import invalid_summary, make_branch, valid_summary
from utils.branch_bests import collect_branch_bests, source_fingerprint


@pytest.fixture
def env(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api
    import api.helpers as ah

    monkeypatch.setattr(ah, "PROBLEMS_DIR", tmp_path)
    d = tmp_path / "p"
    (d / "output").mkdir(parents=True)
    (d / "problem.yaml").write_text("name: p\ngpu: 0\n", encoding="utf-8")
    make_branch(d / "output", "b1", "success", [valid_summary(2.0, 50.0)])
    return TestClient(api.create_app()), d, monkeypatch


def _running(monkeypatch, value=True):
    import api.finalize as af
    import api.helpers as ah
    import api.tree as at

    for mod in (af, ah, at):
        monkeypatch.setattr(mod, "is_problem_running", lambda name: value, raising=False)


def _dead_pid():
    """A pid that belonged to a process that has exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _manifest(d, status, fingerprint="", pid=os.getpid()):
    save_finalize_manifest(
        d / "output",
        FinalizeManifest(status=status, source_fingerprint=fingerprint, pid=pid),
    )


def test_post_refuses_while_running(env):
    c, d, mp = env
    _running(mp)
    assert c.post("/api/problems/p/finalize").status_code == 409


def test_post_refuses_without_valid_results(env):
    c, d, mp = env
    _running(mp, False)
    make_branch(d / "output", "b1", "failed", [invalid_summary()])
    import shutil

    shutil.rmtree(d / "output" / "branches")
    make_branch(d / "output", "bad", "failed", [invalid_summary()])
    r = c.post("/api/problems/p/finalize")
    assert r.status_code == 400
    assert r.json()["detail"] == "No branch has a valid result yet."


def test_post_spawns_a_finalize_only_process(env):
    import api.optimizer as ao

    c, d, mp = env
    _running(mp, False)
    seen = {}

    class FakeProc:
        def is_alive(self):
            return True

    def fake_spawn(problem_dir, config, label, resume, **kw):
        seen.update(label=label, resume=resume, **kw)
        return FakeProc()

    mp.setattr(ao, "_spawn_optimizer", fake_spawn)
    mp.setattr(ao, "_check_gpu_orphans", lambda *a, **k: None)
    mp.setattr(ao, "_is_running_unlocked", lambda n: False)
    try:
        r = c.post("/api/problems/p/finalize", json={"finalize_timeout": 60})
    finally:
        ao._running_problems.pop("p", None)
    assert r.status_code == 200
    assert r.json() == {"status": "finalizing", "problem": "p"}
    assert seen["finalize_only"] is True
    assert seen["resume"] is True
    assert seen["label"] == "optimizer-finalize-p"


def test_get_reports_each_state(env):
    c, d, mp = env
    _running(mp, False)

    def state():
        return c.get("/api/problems/p/finalize").json()

    assert state() == {"state": "not_run", "manifest": None, "overview_markdown": None}

    fp = source_fingerprint(collect_branch_bests(d / "output", d))
    _manifest(d, "done", fp)
    assert state()["state"] == "up_to_date"
    assert state()["manifest"]["status"] == "done"

    _manifest(d, "done", "something-else")
    assert state()["state"] == "stale"

    _manifest(d, "tuning")
    assert state()["state"] == "stopped"

    _manifest(d, "failed")
    assert state()["state"] == "failed"

    _running(mp, True)
    _manifest(d, "tuning")
    assert state()["state"] == "finalizing"

    # A killed finalize leaves this manifest behind; a live normal run is not it.
    _manifest(d, "tuning", pid=_dead_pid())
    assert state()["state"] == "stopped"


BRANCH_CALLS = [
    ("post", "stop", None),
    ("post", "resume", None),
    ("post", "message", {"content": "hi"}),
    ("post", "change-decision", {"target_iter": 1, "content": "x"}),
    ("post", "config", {"max_iter": 3}),
    ("delete", "", None),
]


@pytest.mark.parametrize("method,tail,body", BRANCH_CALLS)
def test_branch_endpoints_refuse_while_finalizing(env, method, tail, body):
    c, d, mp = env
    _running(mp, True)
    _manifest(d, "tuning")
    url = "/api/problems/p/branches/b1" + (f"/{tail}" if tail else "")
    r = c.request(method.upper(), url, json=body)
    assert r.status_code == 409
    assert r.json()["detail"] == "Finalizing; stop it first."


def test_the_tree_etag_changes_with_finalize_json(env):
    c, d, mp = env
    _running(mp, False)
    first = c.get("/api/problems/p/tree").headers["ETag"]
    _manifest(d, "done")
    second = c.get("/api/problems/p/tree")
    assert second.headers["ETag"] != first
    assert second.json()["finalize"]["status"] == "done"
    assert c.get("/api/problems/p/tree", headers={"If-None-Match": second.headers["ETag"]}).status_code == 304


def test_tree_finalize_is_null_without_a_manifest(env):
    c, d, mp = env
    _running(mp, False)
    assert c.get("/api/problems/p/tree").json()["finalize"] is None


def test_status_reports_the_finalizing_phase(env):
    import api.helpers as ah

    c, d, mp = env

    class Alive:
        def is_alive(self):
            return True

    _manifest(d, "tuning")
    ah._running_problems["p"] = ah.RunEntry(process=Alive(), gpu_index=0)
    try:
        _running(mp, True)
        assert c.get("/api/status").json()["running_problems"]["p"]["phase"] == "finalizing"
        _manifest(d, "done")
        assert c.get("/api/status").json()["running_problems"]["p"]["phase"] == "branches"
    finally:
        ah._running_problems.pop("p", None)


def test_a_stale_manifest_does_not_block_a_live_normal_run(env):
    import api.helpers as ah

    c, d, mp = env
    _running(mp, True)
    _manifest(d, "tuning", pid=_dead_pid())

    r = c.post("/api/problems/p/branches/b1/stop")

    assert r.status_code == 200

    class Alive:
        def is_alive(self):
            return True

    ah._running_problems["p"] = ah.RunEntry(process=Alive(), gpu_index=0)
    try:
        assert c.get("/api/status").json()["running_problems"]["p"]["phase"] == "branches"
    finally:
        ah._running_problems.pop("p", None)
