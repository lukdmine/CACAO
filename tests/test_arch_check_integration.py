"""Real NVRTC: what an architecture option does to inline PTX."""

import pytest

from utils import nvrtc

pytestmark = pytest.mark.integration

KERNEL = r"""
extern "C" __global__ void k(float* dst, const float* src) {
    __shared__ float buf[32];
    unsigned smem = (unsigned)__cvta_generic_to_shared(&buf[threadIdx.x]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem), "l"(src + threadIdx.x));
    asm volatile("cp.async.wait_all;\n");
    dst[threadIdx.x] = buf[threadIdx.x];
}
"""


@pytest.fixture(autouse=True)
def _need_nvrtc():
    if not nvrtc.available():
        pytest.skip("NVRTC unavailable")


def _compile(arch_option):
    return nvrtc.compile_source(KERNEL, {}, [arch_option], "cp.async")


def test_inline_cp_async_fails_below_sm80_with_a_real_arch():
    assert not _compile(nvrtc.real_arch_option(75)).ok
    assert _compile(nvrtc.real_arch_option(80)).ok


def test_ptx_only_compile_does_not_assemble_inline_ptx():
    # The spec assumes a compute_XX compile never assembles inline PTX, which is why
    # the check uses sm_XX. If this fails, sm_XX is still right but the reason is not.
    result = _compile("--gpu-architecture=compute_75")
    assert result.ok, result.log


def test_supported_archs_lists_real_architectures():
    archs = nvrtc.supported_archs()
    assert archs and all(isinstance(a, int) for a in archs)
    assert 80 in archs or 86 in archs
