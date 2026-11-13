"""check_archs records the lowest real architecture each configuration compiles for."""

from utils import arch_check, nvrtc

PARAMS = """
tuner.AddParameter(kernel, "BLOCK", std::vector<uint64_t>{64, 128});
tuner.AddParameter(kernel, "USE_ASYNC", std::vector<uint64_t>{0, 1});
tuner.AddParameter(kernel, "USE_VEC", std::vector<uint64_t>{0, 1});
"""


def _fake_compile(seen):
    def fake(kernel_src, config, options, label="config"):
        seen.append(list(options))
        bad = "--gpu-architecture=sm_75" in options and config.get("USE_ASYNC") == "1"
        return nvrtc.NvrtcCheck(label=label, config=config, ok=not bad, log="", options=options)

    return fake


def _run(monkeypatch, winner, archs=(75, 80, 86)):
    seen = []
    monkeypatch.setattr(nvrtc, "compile_source", _fake_compile(seen))
    report = arch_check.check_archs(
        "// kernel",
        PARAMS,
        winning_config=winner,
        cuda_include="/usr/include",
        scalar_defines=["-DN=4"],
        archs=list(archs),
    )
    return report, seen


def test_toggle_needing_newer_arch_is_recorded(monkeypatch):
    report, _ = _run(monkeypatch, {"BLOCK": "64", "USE_ASYNC": "0", "USE_VEC": "0"})
    assert report.per_toggle["USE_ASYNC"] == 80
    assert report.per_toggle["USE_VEC"] == 75
    assert report.base_min_arch == 75
    assert report.winner_min_arch == 75
    assert report.archs == [75, 80, 86]
    assert report.skipped == ""


def test_winner_enabling_the_toggle_needs_the_newer_arch(monkeypatch):
    report, _ = _run(monkeypatch, {"BLOCK": "128", "USE_ASYNC": "1", "USE_VEC": "0"})
    assert report.winner_min_arch == 80


def test_only_real_architectures_are_requested(monkeypatch):
    _, seen = _run(monkeypatch, {"USE_ASYNC": "1"})
    flat = [o for opts in seen for o in opts]
    assert any("sm_" in o for o in flat)
    assert not any("compute_" in o for o in flat)
    assert all("-I/usr/include" in opts and "-DN=4" in opts for opts in seen)


def test_no_arch_compiles_gives_none(monkeypatch):
    monkeypatch.setattr(
        nvrtc,
        "compile_source",
        lambda src, cfg, opts, label="c": nvrtc.NvrtcCheck(label, cfg, False, "", opts),
    )
    report = arch_check.check_archs(
        "k", PARAMS, winning_config={}, cuda_include="/i", scalar_defines=[], archs=[80]
    )
    assert report.base_min_arch is None
    assert report.winner_min_arch is None
    assert report.per_toggle["USE_ASYNC"] is None


def test_empty_archs_is_skipped():
    report = arch_check.check_archs(
        "k", PARAMS, winning_config={}, cuda_include="/i", scalar_defines=[], archs=[]
    )
    assert report.skipped
    assert report.archs == []
