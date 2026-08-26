"""The compile check has to agree with KTT, so it is tested against kernels KTT judged."""

import pytest

from utils import nvrtc

CUDA_INCLUDE = "/usr/include"


def test_parses_parameters_from_a_real_params_region(params_region):
    params = nvrtc.parse_parameters(params_region)
    assert set(params) == {
        "REG_TILE_N",
        "BLOCK_DIM",
        "ROWS_UNROLL",
        "SPLIT_K",
        "NUM_MEAN_BLOCKS",
    }
    assert params["BLOCK_DIM"] == ["4", "8", "16"]


def test_detects_constraints(params_region):
    assert nvrtc.has_constraints(params_region) is True
    assert nvrtc.has_constraints('tuner.AddParameter(kernel, "X", {1});') is False


def test_values_are_not_round_tripped_through_numbers():
    # KTT pastes the value text into the #define verbatim; reformatting it here would
    # silently change what the kernel compiles against.
    params = nvrtc.parse_parameters(
        'tuner.AddParameter(kernel, "SCALE", std::vector<double>{0.50, 1.0});'
    )
    assert params["SCALE"] == ["0.50", "1.0"]


@pytest.mark.parametrize(
    "capability,expected",
    [
        ("8.6", "--gpu-architecture=compute_86"),
        ("9.0", "--gpu-architecture=compute_90"),
        ("7", "--gpu-architecture=compute_70"),
        (None, "--gpu-architecture=compute_52"),
        ("nonsense", "--gpu-architecture=compute_52"),
    ],
)
def test_arch_option_matches_ktt_format(capability, expected):
    assert nvrtc.arch_option(capability) == expected


def test_parses_scalar_defines_from_inputs_hpp(cov_problem):
    text = (cov_problem / "inputs.hpp").read_text(encoding="utf-8")
    assert nvrtc.parse_defines_from_inputs_hpp(text) == ["-DCOV_M=8192", "-DCOV_K=512"]


def test_no_defines_when_the_problem_has_no_scalars():
    assert nvrtc.parse_defines_from_inputs_hpp("struct Inputs { int x; };") == []


def test_prefix_matches_ktt_generate_prefix():
    # KernelConfiguration.cpp:25 -> "#define " + name + " " + value + "\n"
    assert nvrtc._generate_prefix({"A": "1", "B": "2"}) == "#define A 1\n#define B 2\n"


def test_samples_both_ends_of_each_range():
    samples = nvrtc.sample_configurations({"A": ["1", "2", "4"], "B": ["8", "16"]})
    labels = [label for label, _ in samples]
    configs = [config for _, config in samples]
    assert labels == ["minimum values", "maximum values"]
    assert configs == [{"A": "1", "B": "8"}, {"A": "4", "B": "16"}]


def test_single_valued_ranges_compile_once():
    samples = nvrtc.sample_configurations({"A": ["1"]})
    assert len(samples) == 1


@pytest.mark.skipif(not nvrtc.available(), reason="libnvrtc not available")
def test_validated_kernel_compiles(good_kernel, params_region, cov_problem):
    """iter8 ran 103/103 configurations. If this fails, the checker is wrong."""
    defines = nvrtc.parse_defines_from_inputs_hpp(
        (cov_problem / "inputs.hpp").read_text(encoding="utf-8")
    )
    checks = nvrtc.check_kernel(
        good_kernel,
        params_region,
        cuda_include=CUDA_INCLUDE,
        scalar_defines=defines,
        compute_capability="8.6",
    )
    assert checks and all(c.ok for c in checks), [c.log for c in checks if not c.ok]


@pytest.mark.skipif(not nvrtc.available(), reason="libnvrtc not available")
def test_broken_kernel_fails(broken_kernel, params_region, cov_problem):
    """iter9 was recorded as 'all 218 configs failed compilation'."""
    defines = nvrtc.parse_defines_from_inputs_hpp(
        (cov_problem / "inputs.hpp").read_text(encoding="utf-8")
    )
    checks = nvrtc.check_kernel(
        broken_kernel,
        params_region,
        cuda_include=CUDA_INCLUDE,
        scalar_defines=defines,
        compute_capability="8.6",
    )
    assert not all(c.ok for c in checks)
    assert any(c.log for c in checks if not c.ok), "a failure must carry diagnostics"


@pytest.mark.skipif(not nvrtc.available(), reason="libnvrtc not available")
def test_diagnostics_point_into_kernels_cu_not_the_prefix():
    # A prefix of N parameters shifts every reported line by N; an unshifted number
    # sends the model editing the wrong line.
    source = "\n".join(
        ['extern "C" __global__ void k(float* o) {', "    this_is_not_valid;", "}"]
    )
    params = 'tuner.AddParameter(kernel, "A", std::vector<uint64_t>{1});\n'
    checks = nvrtc.check_kernel(source, params, cuda_include=CUDA_INCLUDE)
    assert not checks[0].ok
    assert "(2)" in checks[0].log, checks[0].log


def test_shift_marks_prefix_lines_rather_than_reporting_zero():
    assert "(prefix)" in nvrtc._shift_line_numbers("kernels.cu(1): error", 3)


def test_missing_nvrtc_skips_the_check_instead_of_failing(monkeypatch):
    # An unavailable compiler must not look like a broken kernel: that would send the
    # branch into a fix loop for a defect that does not exist.
    monkeypatch.setattr(nvrtc, "_load", lambda: None)
    result = nvrtc.compile_source("junk", {}, [])
    assert result.ok is True
    assert "unavailable" in result.log.lower()


def test_format_notes_unevaluated_constraints():
    check = nvrtc.NvrtcCheck("maximum values", {"A": "4"}, ok=False, log="error: x")
    text = nvrtc.format_checks([check], constrained=True)
    assert "FAIL" in text and "A=4" in text and "AddConstraint" in text


def test_format_reports_passes_compactly():
    check = nvrtc.NvrtcCheck("minimum values", {"A": "1"}, ok=True, log="")
    text = nvrtc.format_checks([check])
    assert text.startswith("NVRTC: PASS")
