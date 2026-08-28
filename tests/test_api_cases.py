"""The tree payload carries per-case detail without breaking single-case readers.

The import test is deliberate: nothing else in the suite imports the api package, so a
NameError in a route signature reached a green run before this existed.
"""

from pathlib import Path

import pytest
import yaml


def test_the_api_package_imports_and_the_app_builds():
    import api

    app = api.create_app()
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/problems/{name}/inputs/{buffer_name}" in paths


def test_the_upload_route_accepts_a_case_parameter():
    import inspect

    from api.problems import upload_input_file

    assert "case" in inspect.signature(upload_input_file).parameters


def test_single_case_payload_is_unchanged():
    from utils.results import aggregate_case_summaries

    one = {"has_success": True, "num_successful": 1, "num_total": 1,
           "best_config": {}, "best_time_us": 100.0,
           "reference_time_us": 200.0, "speedup": 2.0}
    assert "cases" not in aggregate_case_summaries({"default": one}, "default")


def test_multi_case_payload_keeps_the_flat_keys_readers_use():
    from utils.results import aggregate_case_summaries

    def s(best, ref):
        return {"has_success": True, "num_successful": 1, "num_total": 1,
                "best_config": {}, "best_time_us": best,
                "reference_time_us": ref, "speedup": ref / best}

    out = aggregate_case_summaries({"a": s(100.0, 200.0), "b": s(100.0, 800.0)}, "a")
    # api/tree.py and state/history.py read exactly these two keys.
    assert out["best_time_us"] == 100.0
    assert out["speedup"] is not None


def test_primary_case_dir_is_identity_for_a_single_case_problem(tmp_path):
    from api.tree import _primary_case_dir

    (tmp_path / "problem.yaml").write_text(yaml.safe_dump({"name": "p"}), encoding="utf-8")
    iter_dir = tmp_path / "output" / "branches" / "b" / "iter1"
    assert _primary_case_dir(tmp_path, iter_dir) == iter_dir


def test_primary_case_dir_resolves_the_first_case(tmp_path):
    from api.tree import _primary_case_dir

    (tmp_path / "problem.yaml").write_text(
        yaml.safe_dump({"name": "p", "cases": [{"name": "big"}, {"name": "small"}]}),
        encoding="utf-8",
    )
    iter_dir = tmp_path / "iter1"
    assert _primary_case_dir(tmp_path, iter_dir) == iter_dir / "case_big"


def test_primary_case_dir_falls_back_when_problem_yaml_is_unreadable(tmp_path):
    """The tree endpoint polls constantly; a malformed problem.yaml must not 500 it."""
    from api.tree import _primary_case_dir

    (tmp_path / "problem.yaml").write_text("{ not: [valid", encoding="utf-8")
    iter_dir = tmp_path / "iter1"
    assert _primary_case_dir(tmp_path, iter_dir) == iter_dir


def _minimal_request(slug, **over):
    from api.schemas import CreateProblemRequest

    body = {
        "slug": slug,
        "name": slug,
        "description": "",
        "reference_type": "cuda",
        "ref_function": "ref",
        "ref_kernel_code": "__global__ void ref(float* o) {}",
        "grid_x": "N",
        "inputs": {
            "args": [
                {"kind": "scalar", "name": "N", "dtype": "int", "value": 8,
                 "placements": ["host", "runtime"]},
                {"kind": "buffer", "name": "out", "dtype": "float", "size": "N",
                 "access": "write", "init": "zeros", "validate": True},
            ]
        },
    }
    body.update(over)
    return CreateProblemRequest.model_validate(body)


def test_a_save_that_omits_cases_preserves_them(tmp_path):
    """problem.yaml is rebuilt wholesale from the request. A UI with no cases support
    must not delete a hand-written case list — the next run would tune only the primary
    shape and report a perfectly healthy result."""
    from api.problems import _build_problem_data, _write_problem_files

    (tmp_path / "problem.yaml").write_text(
        yaml.safe_dump({"name": "p", "cases": [{"name": "big"}, {"name": "small"}]}),
        encoding="utf-8",
    )
    req = _minimal_request("p")
    _write_problem_files(tmp_path, _build_problem_data(req, 0), req)

    saved = yaml.safe_load((tmp_path / "problem.yaml").read_text(encoding="utf-8"))
    assert [c["name"] for c in saved["cases"]] == ["big", "small"]


def test_a_save_that_sends_cases_replaces_them(tmp_path):
    from api.problems import _build_problem_data, _write_problem_files

    (tmp_path / "problem.yaml").write_text(
        yaml.safe_dump({"name": "p", "cases": [{"name": "old"}]}), encoding="utf-8"
    )
    req = _minimal_request("p", cases=[{"name": "new", "scalars": {"N": 4}}])
    _write_problem_files(tmp_path, _build_problem_data(req, 0), req)

    saved = yaml.safe_load((tmp_path / "problem.yaml").read_text(encoding="utf-8"))
    assert [c["name"] for c in saved["cases"]] == ["new"]


def test_an_explicit_empty_list_clears_them(tmp_path):
    from api.problems import _build_problem_data, _write_problem_files

    (tmp_path / "problem.yaml").write_text(
        yaml.safe_dump({"name": "p", "cases": [{"name": "old"}]}), encoding="utf-8"
    )
    req = _minimal_request("p", cases=[])
    _write_problem_files(tmp_path, _build_problem_data(req, 0), req)

    saved = yaml.safe_load((tmp_path / "problem.yaml").read_text(encoding="utf-8"))
    assert "cases" not in saved


def test_a_problem_that_never_had_cases_does_not_grow_the_stanza(tmp_path):
    from api.problems import _build_problem_data, _write_problem_files

    req = _minimal_request("p")
    _write_problem_files(tmp_path, _build_problem_data(req, 0), req)

    saved = yaml.safe_load((tmp_path / "problem.yaml").read_text(encoding="utf-8"))
    assert "cases" not in saved
