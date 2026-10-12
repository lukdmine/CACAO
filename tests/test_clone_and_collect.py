"""Cloning a problem, and reading its scalars back out."""

import json

import pytest

from utils.files import clone_problem_dir


# -- clone ------------------------------------------------------------------


@pytest.fixture
def problem(tmp_path):
    """A problem directory with a definition, a finished run, and an archive."""
    d = tmp_path / "mmul"
    (d / "output" / "branches" / "tiled" / "iter1").mkdir(parents=True)
    (d / "archive" / "output_1" / "branches").mkdir(parents=True)

    (d / "problem.yaml").write_text("name: GEMM\n", encoding="utf-8")
    (d / "inputs.yaml").write_text("args: []\n", encoding="utf-8")
    (d / "ref_kernel.cu").write_text("// ref\n", encoding="utf-8")
    (d / "run.pid").write_text("1234", encoding="utf-8")
    (d / "run.log").write_text("log", encoding="utf-8")

    (d / "output" / "final_results.json").write_text("{}", encoding="utf-8")
    (d / "output" / "branches" / "tiled" / "iter1" / "kernels.cu").write_text(
        "// tuned\n", encoding="utf-8"
    )
    (d / "output" / "branches" / "tiled" / "iter1" / "driver").write_bytes(b"\x7fELF" * 32)
    (d / "archive" / "output_1" / "branches" / "old.json").write_text("{}", encoding="utf-8")
    return d


def test_clone_copies_the_definition(problem, tmp_path):
    target = clone_problem_dir(problem, tmp_path / "mmul_v2")

    assert (target / "problem.yaml").read_text(encoding="utf-8") == "name: GEMM\n"
    assert (target / "inputs.yaml").exists()
    assert (target / "ref_kernel.cu").exists()


def test_clone_leaves_the_run_history_behind(problem, tmp_path):
    """Copying them duplicated 71 GB for gdn_chunk and left the clone looking
    already-run."""
    target = clone_problem_dir(problem, tmp_path / "mmul_v2")

    assert not (target / "output").exists()
    assert not (target / "archive").exists()


def test_clone_leaves_runtime_files_behind(problem, tmp_path):
    target = clone_problem_dir(problem, tmp_path / "mmul_v2")

    assert not (target / "run.pid").exists()
    assert not (target / "run.log").exists()


def test_clone_survives_a_broken_symlink(problem, tmp_path):
    """symlinks=True copies links as links, so a stale one cannot crash the copy."""
    (problem / "inputs").mkdir()
    (problem / "inputs" / "data.bin").symlink_to(tmp_path / "gone.bin")

    target = clone_problem_dir(problem, tmp_path / "mmul_v2")

    assert (target / "inputs" / "data.bin").is_symlink()


# -- collect_results scalars ------------------------------------------------


def test_scalars_are_read_from_inputs_yaml(tmp_path):
    """They moved to inputs.yaml's `args`; the old location returned {} for every
    problem, and the work formulas raised KeyError into a bare except."""
    import collect_results

    d = tmp_path / "p"
    d.mkdir()
    (d / "problem.yaml").write_text("name: X\n", encoding="utf-8")
    (d / "inputs.yaml").write_text(
        json.dumps(
            {
                "args": [
                    {"kind": "scalar", "name": "CLIENTS", "value": 8192},
                    {"kind": "scalar", "name": "PERIODS", "value": 4096},
                    {"kind": "buffer", "name": "data", "size": "CLIENTS * PERIODS"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert collect_results.problem_scalars(d) == {"CLIENTS": 8192, "PERIODS": 4096}


def test_a_scalar_without_a_value_is_skipped(tmp_path):
    import collect_results

    d = tmp_path / "p"
    d.mkdir()
    (d / "inputs.yaml").write_text(
        json.dumps({"args": [{"kind": "scalar", "name": "N"}]}), encoding="utf-8"
    )

    assert collect_results.problem_scalars(d) == {}


def test_the_repo_problems_still_satisfy_their_work_formulas():
    """The regression that mattered: every METRICS entry raised KeyError."""
    from pathlib import Path

    import collect_results

    checked = 0
    for name, entry in collect_results.METRICS.items():
        problem_dir = Path("problems") / name
        if not problem_dir.exists():
            continue
        scalars = collect_results.problem_scalars(problem_dir)
        assert entry[1](scalars) > 0, name
        checked += 1

    if checked == 0:
        pytest.skip("none of the METRICS problems are present")
