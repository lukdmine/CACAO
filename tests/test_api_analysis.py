"""The root node's analysis: served on its own, not on the tree payload.

It is written once in phase 1 and never changes, while the tree is polled for the
life of a run — folding 7 KB of prose into that payload would re-ship it on every
tick that any branch touched its manifest.
"""

import json

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api
    import api.tree as at

    d = tmp_path / "p"
    (d / "output").mkdir(parents=True)
    (d / "problem.yaml").write_text("name: p\n", encoding="utf-8")
    monkeypatch.setattr(at, "get_problem_dir", lambda name: d)
    return TestClient(api.create_app()), d


def test_analysis_is_null_before_phase_one_finishes(client):
    c, _ = client

    r = c.get("/api/problems/p/analysis")

    assert r.status_code == 200
    assert r.json()["analysis"] is None


def test_analysis_comes_from_context_json(client):
    c, d = client
    (d / "output" / "context.json").write_text(
        json.dumps({"analysis": "The reference is memory bound."}), encoding="utf-8"
    )

    assert c.get("/api/problems/p/analysis").json()["analysis"] == (
        "The reference is memory bound."
    )


def test_analysis_falls_back_to_the_markdown_dump(client):
    """A run killed between the node's own write and the context write leaves only
    analysis.md. Reading it back is the difference between a root panel that has the
    analysis and one that says the run never produced it."""
    c, d = client
    (d / "output" / "analysis.md").write_text("# Analysis\nfrom the dump", encoding="utf-8")

    assert c.get("/api/problems/p/analysis").json()["analysis"] == "# Analysis\nfrom the dump"


def test_context_json_wins_over_the_dump(client):
    c, d = client
    (d / "output" / "analysis.md").write_text("stale", encoding="utf-8")
    (d / "output" / "context.json").write_text(
        json.dumps({"analysis": "canonical"}), encoding="utf-8"
    )

    assert c.get("/api/problems/p/analysis").json()["analysis"] == "canonical"


def test_a_repeat_fetch_is_a_304(client):
    c, d = client
    (d / "output" / "context.json").write_text(
        json.dumps({"analysis": "x"}), encoding="utf-8"
    )

    first = c.get("/api/problems/p/analysis")
    etag = first.headers["ETag"]
    again = c.get("/api/problems/p/analysis", headers={"If-None-Match": etag})

    assert first.status_code == 200
    assert again.status_code == 304
    assert again.content == b""


def test_the_etag_changes_when_the_analysis_lands(client):
    """The panel polls while a run is in phase 1; a frozen etag would keep it empty."""
    c, d = client
    before = c.get("/api/problems/p/analysis").headers["ETag"]
    (d / "output" / "context.json").write_text(
        json.dumps({"analysis": "arrived"}), encoding="utf-8"
    )

    assert c.get("/api/problems/p/analysis").headers["ETag"] != before


def test_an_unreadable_context_json_does_not_500(client):
    """The panel is what you open to see what went wrong; it must still load."""
    c, d = client
    (d / "output" / "context.json").write_text("{ not json", encoding="utf-8")

    r = c.get("/api/problems/p/analysis")

    assert r.status_code == 200
    assert r.json()["analysis"] is None
