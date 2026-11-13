from models.finalize import ArchReport, KernelTags
from utils.kernel_tags import (
    CATEGORY_BEST_TAGS,
    CATEGORY_TAGS,
    active_lines,
    category_keys,
    full_tags,
    static_tags,
)

WMMA = "wmma::fragment<wmma::matrix_a, 16, 16, 16, half> a;\n"
TC_KEY = "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active"


def test_constants():
    assert CATEGORY_TAGS == ("tensor_cores", "precision")
    assert CATEGORY_BEST_TAGS == ("tensor_cores", "precision", "min_arch")


def test_plain_wmma_is_tensor_cores():
    assert static_tags(WMMA, config={}).tensor_cores is True


def test_wmma_under_toggle():
    src = "#if USE_TC\n" + WMMA + "#endif\n"
    assert static_tags(src, config={"USE_TC": "0"}).tensor_cores is False
    assert static_tags(src, config={"USE_TC": "1"}).tensor_cores is True
    assert static_tags(src, config={}).tensor_cores is False


def test_else_branch_decides_precision():
    src = "#if USE_TC\n__half h;\n#else\nfloat acc = 0;\n#endif\n"
    assert static_tags(src, config={"USE_TC": "0"}).precision == "fp32"
    assert static_tags(src, config={"USE_TC": "1"}).precision == "fp16"


def test_unknown_condition_keeps_both_branches():
    src = "#if TILE > 16\nasm(\"mma.sync\");\n#else\n__nv_bfloat16 x;\n#endif\n"
    lines = active_lines(src, {})
    assert any("mma.sync" in l for l in lines) and any("bfloat16" in l for l in lines)
    tags = static_tags(src, config={})
    assert tags.tensor_cores and tags.precision == "bf16"


def test_condition_forms():
    cfg = {"USE_A": "1", "USE_B": "0"}
    src = (
        "#ifdef USE_A\na1\n#endif\n#ifndef USE_A\na2\n#endif\n"
        "#if !USE_B\nb1\n#endif\n#if defined(USE_A)\nd1\n#endif\n"
        "#if USE_B == 0\ne0\n#endif\n#if USE_A == 0\ne1\n#endif\n"
    )
    assert [l.strip() for l in active_lines(src, cfg)] == ["a1", "b1", "d1", "e0"]


def test_elif_chain_and_nesting():
    src = (
        "#if USE_A\none\n#elif USE_B\ntwo\n#else\nthree\n#endif\n"
        "#if USE_A\n#if USE_B\nnested\n#endif\n#endif\n"
    )
    f = lambda cfg: [l.strip() for l in active_lines(src, cfg)]
    assert f({"USE_A": "1", "USE_B": "1"}) == ["one", "nested"]
    assert f({"USE_A": "0", "USE_B": "1"}) == ["two"]
    assert f({"USE_A": "0", "USE_B": "0"}) == ["three"]


def test_async_copy_inline_asm():
    src = 'asm volatile("cp.async.ca.shared.global [%0], [%1], 16;");\n'
    assert static_tags(src, config={}).async_copy is True
    assert static_tags("float x;\n", config={}).async_copy is False


def test_precision_order_and_dtypes():
    assert static_tags("namespace precision::tf32 {}\n__half h;", config={}).precision == "tf32"
    assert static_tags("double d;", config={}).precision == "fp64"
    assert static_tags("float x;", config={}).precision == "fp32"
    assert static_tags("float x;", config={}, buffer_dtypes=["float", "half"]).precision == "fp16"


def test_sm_scaled_from_launcher_or_params():
    assert static_tags("", config={}, launcher_src="n = GetMaxComputeUnits();").sm_scaled
    assert static_tags("", config={}, params_src="GetMaxComputeUnits").sm_scaled
    assert not static_tags("", config={}).sm_scaled


def test_full_tags_merges_arch_and_smem():
    tags = full_tags(
        KernelTags(), smem_bytes=4096,
        arch=ArchReport(winner_min_arch=80), ncu_metrics=None,
    )
    assert tags.smem_bytes == 4096 and tags.min_arch == 80
    assert full_tags(KernelTags(), smem_bytes=None, arch=None, ncu_metrics=None).min_arch is None


def test_idle_tensor_pipe_clears_tag():
    static = KernelTags(tensor_cores=True)
    tags = full_tags(static, smem_bytes=None, arch=None, ncu_metrics={TC_KEY: 0})
    assert tags.tensor_cores is False
    assert "tensor-core code present, tensor pipe idle" in tags.note
    assert static.tensor_cores is True  # input untouched
    busy = full_tags(static, smem_bytes=None, arch=None, ncu_metrics={TC_KEY: 12.5})
    assert busy.tensor_cores is True


def test_category_keys():
    tags = KernelTags(tensor_cores=True, precision="fp16", min_arch=80)
    assert category_keys(tags, ["tensor_cores", "precision", "min_arch"]) == [
        "tensor_cores=true", "precision=fp16", "min_arch=80",
    ]
    assert category_keys(KernelTags(), ["min_arch", "precision"]) == ["precision=fp32"]


def test_comments_do_not_tag():
    assert static_tags("// double buffering\nfloat x;", config={}).precision == "fp32"
    assert static_tags("/* uses\n wmma:: later */\nfloat x;", config={}).tensor_cores is False
    assert static_tags("// wmma::\nfloat x;", config={}).tensor_cores is False


def test_string_contents_blanked_for_precision_only():
    assert static_tags('printf("double");', config={}).precision == "fp32"
    asm_cp = 'asm volatile("cp.async.ca.shared.global [%0], [%1], 16;");'
    assert static_tags(asm_cp, config={}).async_copy is True
    assert static_tags('asm("mma.sync.aligned.m16n8k8");', config={}).tensor_cores is True
    assert static_tags('asm("ldmatrix.sync.aligned.x4");', config={}).tensor_cores is True
    assert static_tags('const char* u = "http://x"; double d;', config={}).precision == "fp64"


def test_half2_is_fp16():
    assert static_tags("__half2 v;", config={}).precision == "fp16"
    assert static_tags("half2 v;", config={}).precision == "fp16"


def test_literal_if_zero_and_one():
    src = "#if 0\nwmma::x\n#else\nfloat a;\n#endif\n#if 1\nasm(\"cp.async\");\n#endif\n"
    tags = static_tags(src, config={})
    assert tags.tensor_cores is False and tags.async_copy is True


def test_sm_scaled_ignores_comments():
    assert not static_tags("", config={}, launcher_src="// GetMaxComputeUnits").sm_scaled
