"""Reading problem.yaml's reference stanza."""

import pytest

from utils.problem import load_problem_yaml, reference_source, reference_spec


@pytest.mark.parametrize(
    "meta, expected",
    [
        ({}, ("cuda", "ref_kernel.cu", "")),
        ({"reference": {"type": "cpu_c"}}, ("cpu_c", "ref_cpu.c", "")),
        ({"reference": {"type": "python"}}, ("python", "ref.py", "")),
        ({"reference": {"type": "CUDA", "function": "gemm"}}, ("cuda", "ref_kernel.cu", "gemm")),
        ({"reference": {"type": "cpu_c", "file": "other.c"}}, ("cpu_c", "other.c", "")),
        ({"reference": None}, ("cuda", "ref_kernel.cu", "")),
    ],
)
def test_defaults_follow_the_reference_type(meta, expected):
    """The old per-site defaults did not: "ref_kernel.cu" regardless of type in the
    CLI and API, "" in the worker."""
    assert tuple(reference_spec(meta)) == expected


def test_a_missing_problem_yaml_is_empty_not_an_error(tmp_path):
    assert load_problem_yaml(tmp_path) == {}


def test_unparseable_yaml_is_empty_not_an_error(tmp_path):
    (tmp_path / "problem.yaml").write_text("reference: [unclosed\n", encoding="utf-8")
    assert load_problem_yaml(tmp_path) == {}


def test_reference_source_reads_the_declared_file(tmp_path):
    (tmp_path / "problem.yaml").write_text(
        "reference: {type: cpu_c, file: ref_cpu.c}\n", encoding="utf-8"
    )
    (tmp_path / "ref_cpu.c").write_text("void f(float* a) {}\n", encoding="utf-8")

    assert "void f" in reference_source(tmp_path)


def test_reference_source_falls_back_to_the_type_default(tmp_path):
    """reference.file omitted used to give the worker "" and hand every prompt an
    empty reference kernel."""
    (tmp_path / "problem.yaml").write_text("reference: {type: cuda}\n", encoding="utf-8")
    (tmp_path / "ref_kernel.cu").write_text("__global__ void k() {}\n", encoding="utf-8")

    assert "__global__" in reference_source(tmp_path)


def test_reference_source_is_empty_when_the_file_is_missing(tmp_path):
    (tmp_path / "problem.yaml").write_text("reference: {type: cuda}\n", encoding="utf-8")
    assert reference_source(tmp_path) == ""
