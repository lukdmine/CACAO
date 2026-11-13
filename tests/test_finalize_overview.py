"""The run overview: prose from the model, every number from code.

The LLM is a fake; what matters is that a number the facts do not hold never reaches
overview.md, and that an LLM failure still leaves a readable overview behind.
"""

import json
from pathlib import Path

import pytest

import nodes.finalize_overview as ov
from models.finalize import ArchReport, KernelTags, OverviewDraft
from nodes.finalize_overview import (
    build_facts,
    numbers_in,
    render_markdown,
    unknown_numbers,
    write_overview,
)
from state.types import FinalistState, FinalizeManifest, Overview, OverviewFact, Variant
from tests.helpers.finalize_fixtures import invalid_summary, make_branch, valid_summary
from utils.branch_bests import collect_branch_bests


class FakeLLM:
    """Replies with each draft in turn (an Exception instance is raised instead)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def facts(*pairs):
    return [OverviewFact(label=a, value=b) for a, b in pairs]


def setup(tmp_path, monkeypatch, *replies, speedup=11.57, time_us=159.6):
    output = tmp_path / "output"
    output.mkdir()
    make_branch(output, "tiled", "success", [valid_summary(speedup, time_us)])
    make_branch(output, "naive", "failed", [valid_summary(speedup / 2 if speedup else None, time_us * 2)])
    llm = FakeLLM(*replies)
    monkeypatch.setattr(ov, "get_llm_precise", lambda: llm)
    bests = collect_branch_bests(output, tmp_path)
    return output, bests, llm


def manifest_for(bests, **kw):
    b = next((b for b in bests if b.name == "tiled"), None)
    source = Variant(
        dir="branches/tiled/iter1", summary=b.summary if b else None,
        tags=KernelTags(tensor_cores=True, min_arch=80),
    )
    f = FinalistState(
        name="tiled", source_branch="branches/tiled", source_iter=1, source=source,
        reported_dir=source.dir, **kw,
    )
    return FinalizeManifest(status="overview", finalists=[f])


def test_a_number_not_in_the_facts_is_flagged():
    f = facts(("tiled source score", "11.57x"))
    assert unknown_numbers("It reached 12.5x.", f) == ["12.5"]
    assert numbers_in("a 12.5x and 3 b") == ["12.5", "3"]


def test_rounded_numbers_match():
    f = facts(("tiled source score", "11.57x"), ("tiled source time", "159.6 µs"))
    assert unknown_numbers("About 11.6x in 160 µs.", f) == []


def test_time_only_facts_pass_the_number_check(tmp_path, monkeypatch):
    output, bests, _ = setup(tmp_path, monkeypatch, speedup=None)
    f = build_facts(manifest_for(bests), bests)
    assert any("µs" in x.value for x in f)
    assert not any("x" == x.value[-1:] for x in f)
    assert unknown_numbers("Best at 159.6 µs.", f) == []


def draft(headline="Tiled wins at 11.6x", findings=(), dead_ends=(), caveats=()):
    return OverviewDraft(
        headline=headline, findings=list(findings), dead_ends=list(dead_ends),
        caveats=list(caveats),
    )


async def test_a_second_offence_drops_the_item_and_rebuilds_the_headline(tmp_path, monkeypatch):
    bad = draft("Tiled hit 99.9x", ["Good: 11.6x speedup", "Bad: 77x faster"], ["Dead 4.2x"])
    output, bests, llm = setup(tmp_path, monkeypatch, bad, bad)

    o = await write_overview(manifest_for(bests), bests, output, caveats=["code caveat"])

    assert len(llm.calls) == 2
    retry_text = " ".join(str(m.content) for m in llm.calls[1])
    assert "99.9" in retry_text and "77" in retry_text
    assert o.headline == "Best: tiled at 11.57x"
    assert o.findings == ["Good: 11.6x speedup"]
    assert o.dead_ends == []
    assert o.caveats == ["code caveat"]


async def test_a_clean_retry_is_kept(tmp_path, monkeypatch):
    output, bests, llm = setup(
        tmp_path, monkeypatch, draft("Tiled hit 99.9x"), draft("Tiled wins at 11.6x")
    )
    o = await write_overview(manifest_for(bests), bests, output, caveats=[])
    assert o.headline == "Tiled wins at 11.6x"


async def test_limits_trim_a_long_reply(tmp_path, monkeypatch):
    long = draft(findings=["f"] * 9, dead_ends=["d"] * 9, caveats=["c"] * 9)
    output, bests, _ = setup(tmp_path, monkeypatch, long)
    o = await write_overview(manifest_for(bests), bests, output, caveats=[])
    assert (len(o.findings), len(o.dead_ends)) == (5, 3)


async def test_an_llm_failure_still_writes_an_overview(tmp_path, monkeypatch):
    output, bests, _ = setup(tmp_path, monkeypatch, RuntimeError("no provider"))

    o = await write_overview(manifest_for(bests), bests, output, caveats=["code caveat"])

    assert o.headline == "Best: tiled at 11.57x"
    assert o.findings == [] and o.caveats == ["code caveat"]
    assert o.facts
    text = (output / "finalize" / "overview.md").read_text(encoding="utf-8")
    assert "Best: tiled at 11.57x" in text and "code caveat" in text


async def test_no_valid_result_gets_its_caveat(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    llm = FakeLLM()
    monkeypatch.setattr(ov, "get_llm_precise", lambda: llm)

    o = await write_overview(
        FinalizeManifest(status="overview"), [], output,
        caveats=["No branch produced a valid result."],
    )

    assert llm.calls == []
    assert "No branch produced a valid result." in o.caveats
    assert o.headline


def fail_branch(output: Path, name: str = "broken") -> Path:
    """A failed branch that never validated, with one recorded failure analysis."""
    branch = make_branch(output, name, "failed", [invalid_summary()])
    path = branch / "iter1" / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["decision"] = {
        "action": "stop",
        "reasoning": "every retry overflowed shared memory",
        "feedback": "",
        "error_analysis": {
            "error_type": "compilation",
            "root_cause": "the shared tile overflows the block's limit",
            "suggested_fix": "smaller tiles",
        },
    }
    path.write_text(json.dumps(state), encoding="utf-8")
    return branch


async def test_a_failed_only_branch_reaches_the_facts_and_the_llm(tmp_path, monkeypatch):
    output, bests, llm = setup(tmp_path, monkeypatch, draft())
    fail_branch(output)

    o = await write_overview(manifest_for(bests), bests, output, caveats=[])

    f = {x.label: x.value for x in o.facts}
    assert f["Branch broken"] == "failed, no valid result"
    assert (f["Branches"], f["Branches with a valid result"]) == ("3", "2")
    sent = " ".join(str(m.content) for m in llm.calls[0])
    assert "the shared tile overflows the block's limit" in sent  # its branch_errors
    assert "every retry overflowed shared memory" in sent  # its last reasoning


async def test_a_run_with_no_valid_result_still_gets_a_draft(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    fail_branch(output)
    llm = FakeLLM(draft(
        "Nothing validated: the only branch overflowed shared memory",
        dead_ends=["broken: its shared tile never fit"],
    ))
    monkeypatch.setattr(ov, "get_llm_precise", lambda: llm)

    o = await write_overview(
        FinalizeManifest(status="overview"), [], output,
        caveats=["No branch produced a valid result."],
    )

    assert len(llm.calls) == 1
    assert "the shared tile overflows the block's limit" in str(llm.calls[0][-1].content)
    assert o.headline == "Nothing validated: the only branch overflowed shared memory"
    assert o.dead_ends == ["broken: its shared tile never fit"]
    assert o.caveats == ["No branch produced a valid result."]
    assert {x.label: x.value for x in o.facts}["Branch broken"] == "failed, no valid result"


async def test_a_failed_finalize_with_nothing_to_rank_is_not_called_no_valid_result(tmp_path, monkeypatch):
    """The scan may be what failed, so an empty bests list proves nothing about them."""
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(ov, "get_llm_precise", lambda: FakeLLM())

    o = await write_overview(
        FinalizeManifest(status="failed", error="RuntimeError: cannot read branches"), [],
        output, caveats=["Finalize failed: RuntimeError: cannot read branches"],
    )

    assert o.headline == "Finalize failed"


async def test_a_failed_scan_claims_no_branch_lacks_a_result(tmp_path, monkeypatch):
    """bests is empty because the scan raised, not because nothing validated."""
    output, _, _ = setup(tmp_path, monkeypatch, draft("Finalize could not read the branches"))

    o = await write_overview(
        FinalizeManifest(status="failed", error="RuntimeError: x"), [], output,
        caveats=[], scanned=False,
    )

    f = {x.label: x.value for x in o.facts}
    assert f["Branch tiled"] == "success, result not read"
    assert "Branches with a valid result" not in f


async def test_agentic_steps_off_is_never_silent(tmp_path, monkeypatch):
    import engine.finalize as fin
    import config

    monkeypatch.setattr(config, "AGENTIC_STEPS", False)
    output, bests, _ = setup(tmp_path, monkeypatch, draft())
    caveats = fin.overview_caveats(manifest_for(bests), bests)
    assert "Widening did not run: AGENTIC_STEPS is off." in caveats


def test_failed_widening_is_a_caveat_with_its_first_line():
    import engine.finalize as fin

    m = manifest_for([])
    m.finalists[0].stages["widen"] = "failed"
    m.finalists[0].error = "ran out of tool calls\nsecond line"
    caveats = fin.overview_caveats(m, [object()])
    assert "tiled: widening failed (ran out of tool calls); its source result stands." in caveats


async def test_overview_md_is_written_with_its_sections(tmp_path, monkeypatch):
    d = draft(findings=["Tiling helped"], dead_ends=["Naive was slow"], caveats=["Model note"])
    output, bests, _ = setup(tmp_path, monkeypatch, d)

    o = await write_overview(manifest_for(bests), bests, output, caveats=["Code note"])

    text = (output / "finalize" / "overview.md").read_text(encoding="utf-8")
    assert text == o.markdown
    for part in ("# Run overview", "**Tiled wins at 11.6x**", "## Findings",
                 "## Dead ends", "## Caveats", "| Fact | Value |", "Model note", "Code note"):
        assert part in text
    assert "—" not in text


def test_empty_sections_are_omitted():
    md = render_markdown(Overview(headline="H", facts=facts(("a", "1"))))
    assert "## Findings" not in md and "## Caveats" not in md
    assert "| a | 1 |" in md


def test_facts_cover_counts_finalists_categories_and_branches(tmp_path, monkeypatch):
    output, bests, _ = setup(tmp_path, monkeypatch)
    m = manifest_for(bests)
    m.category_bests = {"tensor_cores=true": "tiled"}
    m.finalists[0].kernel_diff_lines = 14
    f = {x.label: x.value for x in build_facts(m, bests)}
    assert f["Finalists"] == "1"
    assert f["tiled source score"] == "11.57x"
    assert f["tiled source time"] == "159.6 µs"
    assert f["tiled minimum arch"] == "sm_80"
    assert f["tiled kernel diff lines"] == "14"
    assert f["Best with tensor_cores=true"] == "tiled"
    assert f["Branch naive"].startswith("failed")


def test_facts_name_the_searcher_of_the_retune(tmp_path, monkeypatch):
    output, bests, _ = setup(tmp_path, monkeypatch)
    m = manifest_for(bests)
    m.finalists[0].finalized = Variant(
        dir="finalize/tiled/iter1", summary=valid_summary(12.0, 150.0), searcher="mcmc (seeded)"
    )
    f = {x.label: x.value for x in build_facts(m, bests)}
    assert f["tiled searcher"] == "mcmc (seeded)"
