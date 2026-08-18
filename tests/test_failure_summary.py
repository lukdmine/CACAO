"""Compile diagnostics must survive the trip from the tuner to the retry.

summarize_failures() exists because excerpting a raw 957 KB tuner log handed the model
one arbitrary configuration's block, cut mid-way: in one real run the tail showed the
wmma errors while the float2 errors, spread over 1,332 lines, never reached the retry
at all.

The summary it produces was then itself excerpted at 30 lines, which clipped a quarter
of recorded summaries — the worst dropping 5 of 31 distinct diagnostics. Same failure,
smaller scale.
"""

import pytest

from state.history import _fmt_run_output
from utils.results import FAILURE_SUMMARY_HEADING, summarize_failures


def _fake_summary(n_diagnostics: int) -> str:
    """A summary shaped like the real one: header, note, fence, one line each."""
    lines = [f"{FAILURE_SUMMARY_HEADING} {n_diagnostics} distinct", "", "note", "```"]
    lines += [f"x{i:<4} kernels.cu:{i}: error: diagnostic number {i}" for i in range(n_diagnostics)]
    lines.append("```")
    return "\n".join(lines)


def test_every_diagnostic_reaches_the_prompt():
    summary = _fake_summary(31)
    rendered = _fmt_run_output(summary)
    for i in range(31):
        assert f"diagnostic number {i}" in rendered, f"diagnostic {i} was dropped"


def test_a_summary_is_not_clipped_at_the_raw_output_limit():
    # 31 diagnostics is a real observed count; the old 30-line cap lost 5 of them.
    rendered = _fmt_run_output(_fake_summary(31))
    assert "diagnostic number 30" in rendered


def test_raw_output_is_still_excerpted():
    """Unbounded driver output must stay bounded — the cap is only lifted for summaries."""
    raw = "\n".join(f"[Info] Launching configuration {i} / 5000" for i in range(5000))
    rendered = _fmt_run_output(raw)
    assert len(rendered.splitlines()) < 60
    assert "last 30 lines" in rendered


def test_summary_still_has_a_runaway_guard():
    rendered = _fmt_run_output(_fake_summary(5000))
    assert len(rendered.splitlines()) < 260, "an enormous summary must still be bounded"


def test_compile_error_output_is_excerpted_from_the_head():
    # g++ leads with the root cause and follows with cascading repeats, so the head is
    # the diagnostic part.
    raw = "[COMPILE ERROR] Host compilation failed.\n" + "\n".join(
        f"cascade line {i}" for i in range(200)
    )
    rendered = _fmt_run_output(raw)
    assert "[COMPILE ERROR]" in rendered
    assert "first 30 lines" in rendered


def test_real_summary_shape_round_trips():
    """Against output shaped like KTT's, not a hand-written fixture."""
    tuner_output = "\n".join(
        f"[Warning] Kernel run failed with reason: compilation failed\n"
        f"default_program({40 + i}): error: identifier \"thing{i}\" is undefined"
        for i in range(31)
    )
    summary = summarize_failures(tuner_output, kernel_offset=5)
    assert summary is not None
    rendered = _fmt_run_output(summary)
    for i in range(31):
        assert f'thing{i}"' in rendered, f"diagnostic for thing{i} was dropped"
