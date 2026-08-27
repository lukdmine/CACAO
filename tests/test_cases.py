"""Case resolution: the `cases:` block in problem.yaml.

A case is a named partial override of the declared inputs. The rules that matter are
that an unknown override name is an error (a typo'd `kt: 127` would otherwise tune five
identical cases and every signal would look healthy), and that a problem with one case
keeps today's on-disk layout.
"""

from pathlib import Path

import pytest

from models.cases import CaseSpec
from utils.cases import case_dir, case_duration, case_list


def test_absent_cases_block_yields_one_implicit_case():
    cases = case_list({"name": "X"})
    assert len(cases) == 1
    assert cases[0].name == "default"
    assert cases[0].scalars == {}


def test_declared_cases_are_parsed_in_order():
    cases = case_list(
        {"cases": [{"name": "t512", "scalars": {"kT": 512}},
                   {"name": "t127", "scalars": {"kT": 127}, "duration_s": 30}]}
    )
    assert [c.name for c in cases] == ["t512", "t127"]
    assert cases[0].scalars == {"kT": 512}
    assert cases[1].duration_s == 30


def test_duplicate_case_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate case name"):
        case_list({"cases": [{"name": "a"}, {"name": "a"}]})


def test_case_name_must_be_directory_safe():
    with pytest.raises(ValueError, match="directory name"):
        case_list({"cases": [{"name": "kT=512"}]})


def test_single_case_uses_the_iteration_root():
    cases = case_list({})
    assert case_dir(Path("/x/iter1"), cases, cases[0]) == Path("/x/iter1")


def test_one_declared_case_also_uses_the_iteration_root():
    cases = case_list({"cases": [{"name": "t512", "scalars": {"kT": 512}}]})
    assert case_dir(Path("/x/iter1"), cases, cases[0]) == Path("/x/iter1")


def test_multiple_cases_each_get_a_subdirectory():
    cases = case_list({"cases": [{"name": "t512"}, {"name": "t127"}]})
    assert case_dir(Path("/x/iter1"), cases, cases[0]) == Path("/x/iter1/case_t512")
    assert case_dir(Path("/x/iter1"), cases, cases[1]) == Path("/x/iter1/case_t127")


def test_case_duration_falls_back_to_the_problem_budget():
    meta = {"tuning": {"duration_s": 120}, "cases": [{"name": "a"}, {"name": "b", "duration_s": 30}]}
    cases = case_list(meta)
    assert case_duration(meta, cases[0]) == 120
    assert case_duration(meta, cases[1]) == 30


def test_case_duration_is_none_when_nothing_declares_one():
    cases = case_list({})
    assert case_duration({}, cases[0]) is None


def _spec(**over):
    from models.inputs import InputsSpec

    return InputsSpec.model_validate({
        "args": [
            {"kind": "scalar", "name": "kT", "dtype": "int", "value": 512,
             "placements": ["host", "runtime"]},
            {"kind": "buffer", "name": "q", "dtype": "float", "size": "kT * 128",
             "access": "read", "init": "file", "file_name": "q_512.bin"},
            {"kind": "buffer", "name": "o", "dtype": "float", "size": "kT * 128",
             "access": "write", "init": "zeros", "validate": True},
        ],
        **over,
    })


def test_for_case_applies_scalar_and_file_overrides():
    spec = _spec()
    out = spec.for_case(CaseSpec(name="t127", scalars={"kT": 127}, files={"q": "q_127.bin"}))
    assert [s.value for s in out.scalars] == [127]
    assert [b.file_name for b in out.buffers if b.name == "q"] == ["q_127.bin"]


def test_for_case_does_not_mutate_the_original():
    spec = _spec()
    spec.for_case(CaseSpec(name="t127", scalars={"kT": 127}))
    assert spec.scalars[0].value == 512


def test_for_case_coerces_to_the_declared_dtype():
    out = _spec().for_case(CaseSpec(name="a", scalars={"kT": 127.0}))
    assert out.scalars[0].value == 127
    assert isinstance(out.scalars[0].value, int)


def test_for_case_rejects_an_unknown_scalar():
    with pytest.raises(ValueError, match="unknown scalar"):
        _spec().for_case(CaseSpec(name="a", scalars={"kt": 127}))


def test_for_case_rejects_a_file_override_on_a_non_file_buffer():
    with pytest.raises(ValueError, match="init=file"):
        _spec().for_case(CaseSpec(name="a", files={"o": "o.bin"}))


def test_implicit_case_is_an_identity_copy():
    spec = _spec()
    out = spec.for_case(CaseSpec())
    assert out.model_dump() == spec.model_dump()


def test_implicit_case_regenerates_mmul_inputs_hpp_byte_for_byte(tmp_path):
    """The compatibility claim, tested against a real problem rather than a fixture."""
    import yaml
    from utils.inputs import generate_inputs_hpp, load_inputs_spec

    problem = Path(__file__).resolve().parent.parent / "problems" / "mmul"
    if not (problem / "inputs.yaml").exists():
        pytest.skip("mmul not present")
    spec = load_inputs_spec(problem / "inputs.yaml")
    reference = (yaml.safe_load((problem / "problem.yaml").read_text(encoding="utf-8")) or {}).get("reference")

    direct = generate_inputs_hpp(spec, reference, problem)
    via_case = generate_inputs_hpp(spec.for_case(CaseSpec()), reference, problem)
    assert direct == via_case


def test_ensure_inputs_hpp_uses_the_primary_case(tmp_path):
    """The generated header is what every prompt shows as the I/O boundary and what the
    fast compile check parses -D macros from, so it must be a shape a case actually runs."""
    import yaml
    from utils.inputs import ensure_inputs_hpp

    problem = tmp_path / "p"
    problem.mkdir()
    (problem / "problem.yaml").write_text(yaml.safe_dump({
        "name": "p",
        "reference": {"type": "cuda", "function": "ref", "file": "ref_kernel.cu"},
        "cases": [{"name": "small", "scalars": {"N": 64}},
                  {"name": "big", "scalars": {"N": 4096}}],
    }), encoding="utf-8")
    (problem / "inputs.yaml").write_text(yaml.safe_dump({
        "args": [
            {"kind": "scalar", "name": "N", "dtype": "int", "value": 1024,
             "placements": ["host", "runtime"]},
            {"kind": "buffer", "name": "out", "dtype": "float", "size": "N",
             "access": "write", "init": "zeros", "validate": True},
        ]
    }), encoding="utf-8")

    header = ensure_inputs_hpp(problem).read_text(encoding="utf-8")
    assert "N = 64" in header       # the primary case
    assert "N = 1024" not in header  # not the declared value


def test_ensure_inputs_hpp_is_unchanged_without_cases(tmp_path):
    import yaml
    from utils.inputs import ensure_inputs_hpp

    problem = tmp_path / "p"
    problem.mkdir()
    (problem / "problem.yaml").write_text(yaml.safe_dump({
        "name": "p",
        "reference": {"type": "cuda", "function": "ref", "file": "ref_kernel.cu"},
    }), encoding="utf-8")
    (problem / "inputs.yaml").write_text(yaml.safe_dump({
        "args": [
            {"kind": "scalar", "name": "N", "dtype": "int", "value": 1024,
             "placements": ["host", "runtime"]},
            {"kind": "buffer", "name": "out", "dtype": "float", "size": "N",
             "access": "write", "init": "zeros", "validate": True},
        ]
    }), encoding="utf-8")

    assert "N = 1024" in ensure_inputs_hpp(problem).read_text(encoding="utf-8")
