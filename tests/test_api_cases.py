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


# ── Per-case input files ─────────────────────────────────────────────────────


def _file_problem(root, cases):
    """A problem dir with one init=file buffer and the given cases block."""
    d = root / "fp"
    (d / "inputs").mkdir(parents=True)
    (d / "inputs.yaml").write_text(
        yaml.safe_dump(
            {
                "args": [
                    {"kind": "scalar", "name": "N", "dtype": "int", "value": 16},
                    {"kind": "buffer", "name": "A", "dtype": "float", "size": "N",
                     "init": "file", "file_name": "A.bin", "validate": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    (d / "problem.yaml").write_text(
        yaml.safe_dump({"name": "fp", "cases": cases}), encoding="utf-8"
    )
    return d


def _detail(monkeypatch, problem_dir):
    import api.problems as ap

    monkeypatch.setattr(ap, "get_problem_dir", lambda name: problem_dir)
    return ap.get_problem("fp")


def test_detail_lists_what_the_inputs_directory_holds(monkeypatch, tmp_path):
    """A flat listing, not a per-case resolution: the editor has to answer "is THIS
    name on the server" for a name the user typed a second ago, against a case that
    may not exist in problem.yaml yet."""
    d = _file_problem(tmp_path, [{"name": "base"}, {"name": "alt", "files": {"A": "A_alt.bin"}}])
    (d / "inputs" / "A.bin").write_bytes(b"\0" * 64)
    (d / "inputs" / "A_alt.bin").write_bytes(b"\0" * 32)

    assert _detail(monkeypatch, d)["inputs_dir_files"] == {"A.bin": 64, "A_alt.bin": 32}


def test_detail_lists_a_binary_in_a_subdirectory_by_its_relative_path(monkeypatch, tmp_path):
    """file_name may name a subdirectory, so the key has to match what a field holds."""
    d = _file_problem(tmp_path, [])
    (d / "inputs" / "big").mkdir()
    (d / "inputs" / "big" / "A.bin").write_bytes(b"\0" * 8)

    assert _detail(monkeypatch, d)["inputs_dir_files"] == {"big/A.bin": 8}


def test_detail_ignores_the_temp_file_a_rejected_upload_leaves(monkeypatch, tmp_path):
    """A .part that outlived its upload must not read as a binary that is present."""
    d = _file_problem(tmp_path, [])
    (d / "inputs" / "A.bin.part").write_bytes(b"\0" * 3)

    assert _detail(monkeypatch, d)["inputs_dir_files"] == {}


def test_detail_still_reports_the_declared_buffers(monkeypatch, tmp_path):
    """The payload readers have always seen stays exactly as it was."""
    d = _file_problem(tmp_path, [])

    assert _detail(monkeypatch, d)["input_files"]["A"]["file_name"] == "A.bin"


def _client(monkeypatch, problem_dir):
    from fastapi.testclient import TestClient

    import api
    import api.problems as ap

    monkeypatch.setattr(ap, "get_problem_dir", lambda name: problem_dir)
    monkeypatch.setattr(ap, "is_problem_running", lambda name: False)
    return TestClient(api.create_app())


def test_upload_writes_to_the_named_case_override(monkeypatch, tmp_path):
    d = _file_problem(
        tmp_path, [{"name": "base"}, {"name": "alt", "files": {"A": "A_alt.bin"}}]
    )
    c = _client(monkeypatch, d)

    r = c.post("/api/problems/fp/inputs/A?case=alt", content=b"\0" * 64)

    assert r.status_code == 200, r.text
    assert r.json()["file_name"] == "A_alt.bin"
    assert (d / "inputs" / "A_alt.bin").read_bytes() == b"\0" * 64
    # The shared binary the other case reads is untouched.
    assert not (d / "inputs" / "A.bin").exists()


def test_upload_checks_the_byte_count_against_that_case_scalars(monkeypatch, tmp_path):
    """The whole point of routing through the case: 16 floats for base, 8 for alt."""
    d = _file_problem(
        tmp_path,
        [{"name": "base"}, {"name": "alt", "scalars": {"N": 8}, "files": {"A": "A_alt.bin"}}],
    )
    c = _client(monkeypatch, d)

    assert c.post("/api/problems/fp/inputs/A?case=alt", content=b"\0" * 64).status_code == 400
    assert c.post("/api/problems/fp/inputs/A?case=alt", content=b"\0" * 32).status_code == 200


def test_upload_to_an_unknown_case_is_rejected(monkeypatch, tmp_path):
    d = _file_problem(tmp_path, [{"name": "base"}])
    c = _client(monkeypatch, d)

    r = c.post("/api/problems/fp/inputs/A?case=nope", content=b"\0" * 64)

    assert r.status_code == 404
    assert "base" in r.json()["detail"]
