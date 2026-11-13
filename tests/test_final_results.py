"""output/final_results.json — its contents, and who writes it."""

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
    """It was unconditionally null: find_branch_results never set the key."""
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
    """merge_node's shape dropped total_branches, path and iterations, and `--best`
    runs merge_node — so reading your results rewrote the richer file."""
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
    """The API's run target calls only the engine, so UI runs produced no file."""
    import engine.master as master

    monkeypatch.setattr(master, "get_output_dir", lambda: output_dir)
    branch = _branch(output_dir, "tiled", best_time=305.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})

    master._write_final_results()

    written = json.loads((output_dir / "final_results.json").read_text(encoding="utf-8"))
    assert written["best_branch"] == "tiled"
    assert written["best_config"] == {"TILE": 16}


def test_writing_the_summary_never_raises_into_the_run(output_dir, monkeypatch):
    """A reporting failure must not crash a run that already succeeded."""
    import engine.master as master

    monkeypatch.setattr(master, "get_output_dir", lambda: output_dir)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("nodes.merge.find_branch_results", boom)

    master._write_final_results()  # must not raise

    assert not (output_dir / "final_results.json").exists()


# -- multi-case layout ------------------------------------------------------


def _multicase_problem(tmp_path, monkeypatch, names=("small", "large")):
    """A problem declaring several cases, so results live per case directory."""
    import config as _cfg
    import nodes.merge as merge

    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "problem.yaml").write_text(
        "name: P\ncases:\n" + "".join(f"  - name: {n}\n" for n in names),
        encoding="utf-8",
    )
    monkeypatch.setattr(_cfg, "get_problem_dir", lambda: problem)
    monkeypatch.setattr(merge, "get_problem_dir", lambda: problem)
    return problem


def test_best_config_is_found_when_results_live_per_case(output_dir, tmp_path, monkeypatch):
    """A multi-case problem writes iter_N/case_<name>/results.json and nothing at the
    iteration root, so a scan anchored on the root finds no configuration at all —
    silently reintroducing the null best_config this fix removed."""
    _multicase_problem(tmp_path, monkeypatch)

    branch = _branch(output_dir, "tiled", best_time=305.0)
    case_dir = branch / "iter1" / "case_small"
    case_dir.mkdir(parents=True)
    _results_json(branch, 1, 305.0, {"TILE": 16})
    # Move the results where a multi-case run actually writes them.
    (branch / "iter1" / "results.json").rename(case_dir / "results.json")

    results = find_branch_results(output_dir / "branches")

    assert results[0]["best_config"] == {"TILE": 16}


def test_single_case_problems_still_read_the_iteration_root(output_dir, tmp_path, monkeypatch):
    """The layout every problem that declares no cases uses."""
    import config as _cfg
    import nodes.merge as merge

    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "problem.yaml").write_text("name: P\n", encoding="utf-8")
    monkeypatch.setattr(_cfg, "get_problem_dir", lambda: problem)
    monkeypatch.setattr(merge, "get_problem_dir", lambda: problem)

    branch = _branch(output_dir, "tiled", best_time=305.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})

    results = find_branch_results(output_dir / "branches")

    assert results[0]["best_config"] == {"TILE": 16}


def test_an_unreadable_problem_yaml_falls_back_to_the_root(output_dir, tmp_path, monkeypatch):
    """primary_case_dir must never raise into a summary."""
    import config as _cfg
    import nodes.merge as merge

    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "problem.yaml").write_text("cases: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(_cfg, "get_problem_dir", lambda: problem)
    monkeypatch.setattr(merge, "get_problem_dir", lambda: problem)

    branch = _branch(output_dir, "tiled", best_time=305.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})

    results = find_branch_results(output_dir / "branches")

    assert results[0]["best_config"] == {"TILE": 16}


# -- finalize section -------------------------------------------------------


def _manifest(output_dir, finalists, **kw):
    """Write output/finalize/finalize.json with the given finalists."""
    from state.types import FinalizeManifest, Overview
    from state import save_finalize_manifest

    m = FinalizeManifest(
        status="done",
        finalists=finalists,
        category_bests={"tensor_cores=true": finalists[0].name},
        overview=Overview(headline="h", markdown="## Finalize overview\nbody"),
        **kw,
    )
    save_finalize_manifest(output_dir, m)
    return m


def _finalist(name, source_branch, *, source, finalized=None):
    from state.types import FinalistState, Variant

    reported = "finalized" if finalized else "source"
    sv = Variant(dir=f"{source_branch}/iter1", summary=source)
    fv = Variant(dir=f"finalize/{name}", summary=finalized) if finalized else None
    return FinalistState(
        name=name,
        source_branch=source_branch,
        source_iter=1,
        source=sv,
        finalized=fv,
        reported=reported,
        reported_dir=fv.dir if fv else sv.dir,
    )


def _sum(speedup, time_us, config=None):
    return {"has_success": True, "best_time_us": time_us, "speedup": speedup,
            "best_config": config or {}}


def test_without_a_manifest_the_shape_is_unchanged(output_dir):
    branch = _branch(output_dir, "tiled", best_time=305.0, speedup=2.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})
    results = find_branch_results(output_dir / "branches")

    summary = build_final_summary(results, results[0], None)

    assert set(summary) == {
        "best_branch", "best_config", "best_time_us", "speedup",
        "precise_best_time_us", "speedup_precise",
        "total_branches", "all_branches",
    }


def test_a_better_finalist_becomes_the_top_level_best(output_dir):
    branch = _branch(output_dir, "tiled", best_time=305.0, speedup=2.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})
    fin = _finalist("f1", "branches/tiled", source=_sum(2.0, 305.0),
                    finalized=_sum(3.0, 200.0, {"TILE": 32}))
    manifest = _manifest(output_dir, [fin], attempt=2)
    results = find_branch_results(output_dir / "branches")

    summary = build_final_summary(results, results[0], manifest)

    assert summary["best_source"] == "finalize"
    assert summary["best_branch"] == "tiled"
    assert summary["best_config"] == {"TILE": 32}
    assert summary["best_time_us"] == 200.0
    assert summary["speedup"] == 3.0
    section = summary["finalize"]
    assert section["status"] == "done" and section["attempt"] == 2
    assert section["overview"].startswith("## Finalize overview")
    assert section["category_bests"] == {"tensor_cores=true": "f1"}
    f = section["finalists"][0]
    assert (f["name"], f["source_speedup"], f["finalized_speedup"]) == ("f1", 2.0, 3.0)
    assert f["reported"] == "finalized" and f["reported_dir"] == "finalize/f1"


def test_a_worse_finalist_leaves_the_branch_best(output_dir):
    branch = _branch(output_dir, "tiled", best_time=305.0, speedup=2.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})
    fin = _finalist("f1", "branches/tiled", source=_sum(1.5, 400.0))
    manifest = _manifest(output_dir, [fin])
    results = find_branch_results(output_dir / "branches")

    summary = build_final_summary(results, results[0], manifest)

    assert summary["best_source"] == "branch"
    assert summary["best_config"] == {"TILE": 16}
    assert summary["speedup"] == 2.0
    assert "finalize" in summary


def test_a_finalist_from_a_failed_branch_still_produces_the_file(output_dir, monkeypatch):
    import engine.master as master

    monkeypatch.setattr(master, "get_output_dir", lambda: output_dir)
    _branch(output_dir, "broken", status="failed")
    fin = _finalist("f1", "branches/broken", source=_sum(1.2, 500.0, {"T": 4}))
    _manifest(output_dir, [fin])

    master._write_final_results()

    written = json.loads((output_dir / "final_results.json").read_text(encoding="utf-8"))
    assert written["best_source"] == "finalize"
    assert written["best_branch"] == "broken"
    assert written["best_config"] == {"T": 4}


def test_best_after_finalize_keeps_the_finalize_section(output_dir):
    """--best runs merge_node; it used to overwrite the file without the section."""
    from nodes.merge import merge_node

    branch = _branch(output_dir, "tiled", best_time=305.0, speedup=2.0)
    _results_json(branch, 1, 305.0, {"TILE": 16})
    fin = _finalist("f1", "branches/tiled", source=_sum(2.0, 305.0))
    _manifest(output_dir, [fin])

    merge_node()

    written = json.loads((output_dir / "final_results.json").read_text(encoding="utf-8"))
    assert "finalize" in written and written["best_source"] == "branch"
