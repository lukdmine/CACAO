"""The python-reference preflight.

run_optimization_engine calls this before anything else, so a failure here takes the
whole run down before a single branch starts. It had no test at all, which is how it
survived a move that left one of its names behind: the body referenced `_cfg`, an
alias that existed only in engine/master.py, and every python-reference run raised
NameError at startup.
"""

import asyncio
import textwrap

import pytest

import config
from utils.ref_timing import preflight_python_reference


def _problem(tmp_path, ref_type="python", ref_body="def f(scalars, buffers):\n    return []\n"):
    (tmp_path / "problem.yaml").write_text(
        f"reference:\n  type: {ref_type}\n  file: ref.py\n  function: f\n", encoding="utf-8"
    )
    if ref_body is not None:
        (tmp_path / "ref.py").write_text(textwrap.dedent(ref_body), encoding="utf-8")
    return (tmp_path / "problem.yaml").read_text(encoding="utf-8")


@pytest.fixture
def problem_dir(tmp_path, monkeypatch):
    """Point config at a scratch problem, restoring whatever was there."""
    previous = config.get_problem_dir()
    config.set_problem_dir(tmp_path)
    yield tmp_path
    config.set_problem_dir(previous)


def test_an_importable_reference_passes(problem_dir):
    yaml_text = _problem(problem_dir)
    assert asyncio.run(preflight_python_reference(yaml_text)) is True


def test_a_reference_that_cannot_import_fails_the_run(problem_dir):
    """The whole point: a missing dependency surfaces here, not 40 minutes in."""
    yaml_text = _problem(
        problem_dir, ref_body="import a_module_that_does_not_exist\n\ndef f(s, b):\n    return []\n"
    )
    assert asyncio.run(preflight_python_reference(yaml_text)) is False


def test_a_missing_reference_file_fails_the_run(problem_dir):
    yaml_text = _problem(problem_dir, ref_body=None)
    assert asyncio.run(preflight_python_reference(yaml_text)) is False


@pytest.mark.parametrize("ref_type", ["cuda", "cpu_c"])
def test_a_non_python_reference_is_not_preflighted(problem_dir, ref_type):
    yaml_text = _problem(problem_dir, ref_type=ref_type, ref_body=None)
    assert asyncio.run(preflight_python_reference(yaml_text)) is True


def test_unparseable_problem_yaml_does_not_block_the_run(problem_dir):
    """Nothing else has validated problem.yaml yet at this point; failing here would
    report a YAML error as a reference problem."""
    assert asyncio.run(preflight_python_reference("reference: [unclosed\n")) is True
