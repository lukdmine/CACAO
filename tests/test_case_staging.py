"""Per-case staging: what lands in a case directory before the compiler runs."""

from pathlib import Path

import pytest
import yaml

from models.cases import CaseSpec
from utils.cases import case_list, stage_case
from utils.inputs import load_inputs_spec

MMUL = Path(__file__).resolve().parent.parent / "problems" / "mmul"


@pytest.fixture
def mmul_bits():
    if not (MMUL / "inputs.yaml").exists():
        pytest.skip("mmul not present")
    meta = yaml.safe_load((MMUL / "problem.yaml").read_text(encoding="utf-8"))
    return load_inputs_spec(MMUL / "inputs.yaml"), meta


def _regions():
    return {"kernels": "// k", "params": "// p", "launcher": "// l"}


def test_single_case_stages_at_the_iteration_root(tmp_path, mmul_bits):
    spec, meta = mmul_bits
    iter_dir = tmp_path / "iter1"
    iter_dir.mkdir()
    (iter_dir / "kernels.cu").write_text("// kernel\n", encoding="utf-8")
    cases = case_list(meta)

    out = stage_case(MMUL, iter_dir, spec, meta, cases, cases[0], _regions())

    assert out == iter_dir
    assert (iter_dir / "inputs.hpp").exists()
    assert (iter_dir / "framework.cpp").exists()


def test_multi_case_stages_into_subdirectories_with_its_own_scalars(tmp_path, mmul_bits):
    spec, meta = mmul_bits
    meta = dict(meta, cases=[{"name": "big", "scalars": {"kSizeM": 2048}},
                             {"name": "small", "scalars": {"kSizeM": 256}}])
    iter_dir = tmp_path / "iter1"
    iter_dir.mkdir()
    (iter_dir / "kernels.cu").write_text("// kernel\n", encoding="utf-8")
    cases = case_list(meta)

    small = stage_case(MMUL, iter_dir, spec, meta, cases, cases[1], _regions())

    assert small == iter_dir / "case_small"
    assert "kSizeM = 256" in (small / "inputs.hpp").read_text(encoding="utf-8")
    assert (small / "kernels.cu").read_text(encoding="utf-8") == "// kernel\n"


def test_staged_inputs_yaml_carries_the_resolved_scalars(tmp_path, mmul_bits):
    """python_ref_runner reads this file relative to the driver's cwd, so it must be
    the case's values — not the declared ones."""
    spec, meta = mmul_bits
    meta = dict(meta, cases=[{"name": "a"}, {"name": "small", "scalars": {"kSizeM": 256}}])
    iter_dir = tmp_path / "iter1"
    iter_dir.mkdir()
    (iter_dir / "kernels.cu").write_text("// kernel\n", encoding="utf-8")
    cases = case_list(meta)

    small = stage_case(MMUL, iter_dir, spec, meta, cases, cases[1], _regions())

    staged = yaml.safe_load((small / "inputs.yaml").read_text(encoding="utf-8"))
    values = {a["name"]: a.get("value") for a in staged["args"] if a["kind"] == "scalar"}
    assert values["kSizeM"] == 256


def test_the_regions_reach_the_assembled_driver(tmp_path, mmul_bits):
    spec, meta = mmul_bits
    iter_dir = tmp_path / "iter1"
    iter_dir.mkdir()
    (iter_dir / "kernels.cu").write_text("// kernel\n", encoding="utf-8")
    cases = case_list(meta)

    out = stage_case(MMUL, iter_dir, spec, meta, cases, cases[0],
                     {"kernels": "// KERNELS_MARK", "params": "// PARAMS_MARK",
                      "launcher": "// LAUNCHER_MARK"})

    fw = (out / "framework.cpp").read_text(encoding="utf-8")
    assert "KERNELS_MARK" in fw and "PARAMS_MARK" in fw and "LAUNCHER_MARK" in fw


def test_reference_build_extras_uses_the_cases_scalar_values():
    """A cpu_c reference compiled with another case's sizes computes the wrong answer,
    and validation then reports a correct kernel as broken."""
    from utils.build import reference_build_extras

    problem = Path(__file__).resolve().parent.parent / "problems" / "covariance"
    if not (problem / "inputs.yaml").exists():
        pytest.skip("covariance not present")
    meta = yaml.safe_load((problem / "problem.yaml").read_text(encoding="utf-8")) or {}
    if str((meta.get("reference") or {}).get("type", "cuda")).lower() != "cpu_c":
        pytest.skip("covariance is not a cpu_c reference")

    spec = load_inputs_spec(problem / "inputs.yaml")
    first = spec.scalars[0].name
    _, flags = reference_build_extras(problem, CaseSpec(name="x", scalars={first: 7}))
    assert f"-D{first}=7" in flags
