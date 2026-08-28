"""output/final_results.json — its contents, and who writes it.

It had two writers with two different shapes, and the entry point most runs go
through had none at all.
"""

import json

import pytest

from nodes.merge import build_final_summary, find_branch_results


def _branch(output_dir, name, *, best_time=None, speedup=None, iters=1, status="success"):
    path = output_dir / "branches" / name
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy": {"name": name},
        "status": status,
        "current_iter": iters,
    }
    if best_time is not None:
        manifest["best_time_us"] = best_time
    if speedup is not None:
        manifest["speedup"] = speedup
    (path / "branch.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _results_json(branch_path, iteration, time_us, config):
    """A results.json in the shape utils.results.get_results_summary reads."""
    iter_dir = branch_path / f"iter{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "results.json").write_text(
        json.dumps(
            {
                "Results": [
                    {
                        "Status": "Ok",
                        "TotalDuration": int(time_us * 1000),
                        "Configuration": [
                            {"Name": n, "Value": v} for n, v in config.items()
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return iter_dir


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "branches").mkdir(parents=True)
    import config as _cfg
    import nodes.merge as merge

    monkeypatch.setattr(_cfg, "get_output_dir", lambda: out)
    monkeypatch.setattr(merge, "get_output_dir", lambda: out)
    return out


def test_best_config_reaches_the_summary(output_dir):
    """It was unconditionally null: find_branch_results never put a best_config key
    on any code path, so best.get("best_config") could only ever be None — for a
    value get_results_summary computes and stores in every results_summary."""
    branch = _branch(output_dir, "tiled", best_time=305.0)
    _results_json(branch, 1, 305.0, {"TILE": 16, "VECTOR": 4})

    results = find_branch_results(output_dir / "branches")

    assert len(results) == 1
    assert results[0]["best_config"] == {"TILE": 16, "VECTOR": 4}


def test_the_fastest_iterations_config_wins(output_dir):
    branch = _branch(output_dir, "tiled", best_time=200.0)
    _results_json(branch, 1, 400.0, {"TILE": 8})
    _results_json(branch, 2, 200.0, {"TILE": 32})
    _results_json(branch, 3, 300.0, {"TILE": 16})

    results = find_branch_results(output_dir / "branches")

    assert results[0]["best_config"] == {"TILE": 32}


def test_a_branch_with_no_results_has_no_config(output_dir):
    _branch(output_dir, "tiled", status="failed")

    results = find_branch_results(output_dir / "branches")

    assert results[0]["best_config"] is None


def test_the_summary_keeps_the_richer_per_branch_fields(output_dir):
    """merge_node's shape dropped total_branches, and every branch's path and
    iteration count. `cli.py --best` runs merge_node, so looking at your results
    silently rewrote the richer file with the poorer one."""
    branch = _branch(output_dir, "tiled", best_time=305.0, speedup=2.5, iters=4)
    _results_json(branch, 1, 305.0, {"TILE": 16})

    results = find_branch_results(output_dir / "branches")
    summary = build_final_summary(results, results[0])

    assert summary["best_branch"] == "tiled"
    assert summary["best_config"] == {"TILE": 16}
    assert summary["best_time_us"] == 305.0
    assert summary["total_branches"] == 1
    assert summary["all_branches"][0]["iterations"] == 4
    assert summary["all_branches"][0]["path"].endswith("branches/tiled")


def test_the_engine_writes_the_file_itself(output_dir, monkeypatch):
    """The API's run target calls run_optimization_engine and nothing else, so every
    run started from the UI's Run button used to finish without this file —
    and GET /api/problems/{name}/results returned {"results": null} forever."""
    import engine.master as master

    monkeypatch.setattr(master, "get_output_dir", lambda: output_dir)
    branch = _branch(output_dir, "tiled", best_time=305.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})

    master._write_final_results()

    written = json.loads((output_dir / "final_results.json").read_text(encoding="utf-8"))
    assert written["best_branch"] == "tiled"
    assert written["best_config"] == {"TILE": 16}


def test_writing_the_summary_never_raises_into_the_run(output_dir, monkeypatch):
    """The run has already succeeded by this point and its per-branch results are on
    disk; a reporting failure must not turn that into a crash."""
    import engine.master as master

    monkeypatch.setattr(master, "get_output_dir", lambda: output_dir)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("nodes.merge.find_branch_results", boom)

    master._write_final_results()  # must not raise

    assert not (output_dir / "final_results.json").exists()
