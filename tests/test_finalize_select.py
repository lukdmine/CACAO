"""Stage A of finalize: forced finalists by category, completed by one LLM call."""

import asyncio
from pathlib import Path

from models.finalize import FinalistPick, FinalistPicks, KernelTags
from nodes import finalize_select as fs
from utils.branch_bests import BranchBest


def _best(name, speedup, rel=None, description=""):
    return BranchBest(
        name=name,
        path=Path("/x") / name,
        rel=rel or f"branches/{name}",
        status="success",
        best_iter=1,
        summary={"speedup": speedup, "best_time_us": 100.0 / speedup},
        score=speedup,
        time_us=100.0 / speedup,
        description=description,
    )


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls, self.messages = result, error, 0, None

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        if self.error:
            raise self.error
        return self.result


def _use(monkeypatch, llm):
    monkeypatch.setattr(fs, "get_llm_precise", lambda: llm)


def _five():
    bests = [_best(f"tc{i}", 5 - i) for i in range(4)] + [_best("cuda", 1.0)]
    tags = {b.rel: KernelTags(tensor_cores=b.name != "cuda") for b in bests}
    return bests, tags


def test_the_best_cuda_core_kernel_survives_four_faster_tensor_core_kernels():
    bests, tags = _five()
    forced = {b.name: why for b, why in fs.forced_finalists(bests, tags, 4)}
    assert forced["tc0"] == "overall best"
    assert forced["cuda"] == "best with tensor_cores=false"


def test_forced_finalists_respect_the_cap():
    bests = [_best("a", 3), _best("b", 2), _best("c", 1)]
    tags = {
        "branches/a": KernelTags(tensor_cores=True, precision="fp16"),
        "branches/b": KernelTags(tensor_cores=False, precision="fp32"),
        "branches/c": KernelTags(tensor_cores=False, precision="bf16"),
    }
    assert len(fs.forced_finalists(bests, tags, 2)) == 2
    assert [b.name for b, _ in fs.forced_finalists(bests, tags, 1)] == ["a"]


def test_llm_additions_fill_up_to_the_cap(monkeypatch):
    bests, tags = _five()
    picks = FinalistPicks(
        picks=[FinalistPick(branch=n, reason=f"why {n}") for n in ("tc1", "tc2", "tc3")]
    )
    _use(monkeypatch, FakeLLM(picks))
    out = asyncio.run(fs.select_finalists(bests, tags, cap=4))
    assert [b.name for b, _, _ in out] == ["tc0", "cuda", "tc1", "tc2"]
    assert [forced for _, _, forced in out] == [True, True, False, False]
    assert out[2][1] == "why tc1"


def test_a_forced_finalist_keeps_its_forced_reason_as_a_prefix(monkeypatch):
    bests, tags = _five()
    picks = FinalistPicks(picks=[FinalistPick(branch="cuda", reason="plain fp32")])
    _use(monkeypatch, FakeLLM(picks))
    out = {b.name: why for b, why, _ in asyncio.run(fs.select_finalists(bests, tags, cap=4))}
    assert out["cuda"] == "best with tensor_cores=false; plain fp32"


def test_a_blank_llm_reason_falls_back(monkeypatch):
    bests, tags = _five()
    picks = FinalistPicks(picks=[FinalistPick(branch="tc1", reason="  ")])
    _use(monkeypatch, FakeLLM(picks))
    out = {b.name: why for b, why, _ in asyncio.run(fs.select_finalists(bests, tags, cap=4))}
    assert out["tc1"] == "chosen by the model"


def test_strategy_descriptions_reach_the_prompt(monkeypatch):
    bests, tags = _five()
    bests[1] = _best("tc1", 4, description="double buffered\nwarp tiles")
    llm = FakeLLM(FinalistPicks(picks=[]))
    _use(monkeypatch, llm)
    asyncio.run(fs.select_finalists(bests, tags, cap=4))
    assert "double buffered warp tiles" in llm.messages[1].content


def test_a_forced_finalist_without_an_llm_reason_keeps_the_forced_one(monkeypatch):
    bests, tags = _five()
    _use(monkeypatch, FakeLLM(FinalistPicks(picks=[])))
    out = asyncio.run(fs.select_finalists(bests, tags, cap=4))
    assert dict((b.name, why) for b, why, _ in out)["cuda"] == (
        "best with tensor_cores=false"
    )


def test_an_unknown_branch_from_the_llm_is_ignored(monkeypatch):
    bests, tags = _five()
    picks = FinalistPicks(picks=[FinalistPick(branch="ghost", reason="?")])
    _use(monkeypatch, FakeLLM(picks))
    out = asyncio.run(fs.select_finalists(bests, tags, cap=4))
    assert "ghost" not in [b.name for b, _, _ in out]


def test_an_llm_failure_falls_back_to_the_forced_set(monkeypatch):
    bests, tags = _five()
    _use(monkeypatch, FakeLLM(error=RuntimeError("boom")))
    out = asyncio.run(fs.select_finalists(bests, tags, cap=4))
    assert [b.name for b, _, _ in out] == ["tc0", "cuda"]
    assert all(forced for _, _, forced in out)


def test_no_bests_means_no_llm_call(monkeypatch):
    llm = FakeLLM(FinalistPicks(picks=[]))
    _use(monkeypatch, llm)
    assert asyncio.run(fs.select_finalists([], {}, cap=4)) == []
    assert llm.calls == 0


def test_names_are_unique_and_directory_safe():
    bests = [
        _best("tiled", 3, rel="branches/tiled"),
        _best("tiled", 2, rel="branches/x/branches/tiled"),
        _best("a/b c", 1, rel="branches/ab"),
    ]
    names = fs.finalist_names(bests)
    assert names["branches/tiled"] == "tiled"
    assert names["branches/x/branches/tiled"] == "tiled_2"
    assert names["branches/ab"] == "a_b_c"
