"""Replay harness, framework round-trip, and the wiring into config and the worker."""

import json

import pytest
from langchain_core.messages import AIMessage

import config as _cfg
from agentic import replay
from agentic.loop import StepOutcome, run_agentic_step
from agentic.tools import Toolbox
from agentic.workspace import Workspace


# -- replay ----------------------------------------------------------------


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A trace from a completed step, plus a fresh workspace to replay it into."""

    def fake_check(self):
        self.checks_run += 1
        self.check_passed = True
        self.last_check = "Compilation check: PASS"
        return self.last_check

    monkeypatch.setattr(Toolbox, "check_compilation", fake_check)

    live_dir = tmp_path / "live"
    live_dir.mkdir()
    ws = Workspace(live_dir)
    ws.reset()
    box = Toolbox(ws, branch_path=tmp_path, output_dir=tmp_path, problem_dir=tmp_path, meta={})

    turns = [
        [
            {"tool": "write_file", "args": {"name": "kernels.cu", "content": "// recorded\n"}},
            {"tool": "write_file", "args": {"name": "region_kernels.cpp", "content": "// k\n"}},
            {"tool": "write_file", "args": {"name": "region_params.cpp", "content": "// p\n"}},
            {"tool": "write_file", "args": {"name": "region_launcher.cpp", "content": "// l\n"}},
        ],
        [{"tool": "check_compilation", "args": {}}],
        [{"tool": "end_step", "args": {"summary": "recorded run"}}],
    ]
    trace = tmp_path / "step_trace.jsonl"
    return {"turns": turns, "trace": trace, "box": box, "tmp": tmp_path}


async def test_a_live_step_can_be_replayed_into_the_same_workspace(recorded, monkeypatch):
    await run_agentic_step(
        replay.ScriptedLLM(recorded["turns"]),
        "sys",
        "usr",
        recorded["box"],
        trace_path=recorded["trace"],
    )

    def fake_check(self):
        self.check_passed = True
        return "PASS"

    monkeypatch.setattr(Toolbox, "check_compilation", fake_check)

    replay_dir = recorded["tmp"] / "replay"
    replay_dir.mkdir()
    ws2 = Workspace(replay_dir)
    ws2.reset()
    box2 = Toolbox(
        ws2,
        branch_path=recorded["tmp"],
        output_dir=recorded["tmp"],
        problem_dir=recorded["tmp"],
        meta={},
    )

    result = await replay.replay_step(recorded["trace"], box2)
    assert result.outcome is StepOutcome.COMPLETED
    assert ws2.read("kernels.cu") == "// recorded\n"
    assert box2.end_summary == "recorded run"


async def test_turns_are_reconstructed_in_order(recorded):
    await run_agentic_step(
        replay.ScriptedLLM(recorded["turns"]), "s", "u", recorded["box"], trace_path=recorded["trace"]
    )
    turns = replay.turns_from_trace(recorded["trace"])
    assert [len(t) for t in turns] == [4, 1, 1]
    assert turns[0][0]["tool"] == "write_file"
    assert turns[-1][0]["tool"] == "end_step"


async def test_summary_is_human_readable(recorded):
    await run_agentic_step(
        replay.ScriptedLLM(recorded["turns"]), "s", "u", recorded["box"], trace_path=recorded["trace"]
    )
    text = replay.summarize_trace(recorded["trace"])
    assert "start (budget" in text
    assert "write_file" in text
    assert "completed after 6 calls" in text


def test_corrupt_trace_lines_are_skipped(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"event":"start"}\nnot json\n{"event":"tool_call","turn":1,"tool":"x"}\n', encoding="utf-8")
    assert len(replay.load_trace(trace)) == 2


def test_missing_trace_is_empty(tmp_path):
    assert replay.load_trace(tmp_path / "nope.jsonl") == []
    assert "No trace records" in replay.summarize_trace(tmp_path / "nope.jsonl")


async def test_exhausted_script_stops_driving(recorded):
    llm = replay.ScriptedLLM([[{"tool": "list_iterations", "args": {}}]])
    result = await run_agentic_step(llm, "s", "u", recorded["box"], budget=10)
    assert result.outcome is StepOutcome.NO_TOOL_CALLS


# -- framework round-trip --------------------------------------------------


def test_regions_survive_an_assemble_extract_round_trip(cov_branch):
    """Seeding an agentic step from a legacy iteration depends on this being exact."""
    import re

    import yaml

    from utils.framework import assemble_framework_cpp, extract_regions

    path = cov_branch / "iter8" / "framework.cpp"
    if not path.exists():
        pytest.skip("recorded framework.cpp not present")
    original = path.read_text(encoding="utf-8")
    meta = yaml.safe_load(
        (cov_branch.parents[2] / "problem.yaml").read_text(encoding="utf-8")
    )
    include = re.search(r'SetCompilerOptions\("-I([^"]+)"', original).group(1)
    rebuilt = assemble_framework_cpp(meta, extract_regions(original), cuda_include=include)
    assert rebuilt == original


def test_empty_region_extracts_as_empty():
    from utils.framework import assemble_framework_cpp, extract_regions

    meta = {
        "grid": {"x": 1},
        "reference": {"type": "cpu_c", "file": "ref_cpu.c"},
    }
    assembled = assemble_framework_cpp(
        meta, {"kernels": "int a;", "params": "int b;", "launcher": ""}, cuda_include="/usr/include"
    )
    assert extract_regions(assembled)["launcher"] == ""


# -- config wiring ---------------------------------------------------------


class _FakeChatModel:
    def __init__(self, response):
        self._response = response
        self.bound = None

    def bind_tools(self, tools, **kwargs):
        self.bound = tools
        return self

    async def ainvoke(self, messages, **kwargs):
        return self._response


async def test_tool_call_response_is_not_retried_as_empty():
    """The landmine: a tool-call reply has empty content by construction.

    Without an early return it hits the empty-content retry path and burns five
    attempts with exponential backoff on every single tool call.
    """
    response = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {}, "id": "1"}],
    )
    model = _FakeChatModel(response)
    tracked = _cfg.TrackedLLM(model)

    import time

    started = time.monotonic()
    out = await tracked.ainvoke([])
    assert out is response
    assert time.monotonic() - started < 1.0, "retry/backoff ran on a tool-call response"


def test_bind_tools_keeps_the_tracking_wrapper():
    tracked = _cfg.TrackedLLM(_FakeChatModel(AIMessage(content="x")))
    bound = tracked.bind_tools([str])
    assert isinstance(bound, _cfg.TrackedLLM)


def test_agentic_flags_exist_with_safe_defaults():
    assert isinstance(_cfg.AGENTIC_STEPS, bool)
    assert _cfg.STEP_TOOL_BUDGET > 0
    assert _cfg.CROSS_BRANCH_ACCESS in ("off", "index", "log", "errors")


# -- worker dispatch -------------------------------------------------------


def test_worker_routes_legacy_statuses_to_the_merged_step(monkeypatch):
    from engine.worker import _dispatch_status
    from state.types import IterState

    monkeypatch.setattr(_cfg, "AGENTIC_STEPS", True)
    for status in ("implementing", "configuring"):
        assert _dispatch_status(IterState(iter_num=1, status=status)) == "authoring"
    # Everything else passes through untouched.
    assert _dispatch_status(IterState(iter_num=1, status="running")) == "running"


def test_worker_keeps_the_old_nodes_when_the_flag_is_off(monkeypatch):
    from engine.worker import _dispatch_status
    from state.types import IterState

    monkeypatch.setattr(_cfg, "AGENTIC_STEPS", False)
    assert _dispatch_status(IterState(iter_num=1, status="implementing")) == "implementing"
    assert _dispatch_status(IterState(iter_num=1, status="configuring")) == "configuring"


def test_transition_table_accepts_the_merged_step_and_the_old_ones():
    from state.types import validate_transition

    for target in ("running", "proposing", "deciding"):
        validate_transition("authoring", target)
    validate_transition("planning", "authoring")
    # Output directories written before the merge must still resume.
    validate_transition("planning", "implementing")
    validate_transition("implementing", "configuring")
    validate_transition("configuring", "running")

    with pytest.raises(ValueError):
        validate_transition("authoring", "planning")


def test_skip_implement_becomes_a_config_only_scope():
    from state.persistence import create_initial_iter_state
    from state.types import IterState

    prev = IterState(iter_num=1, status="decided", next_status="configuring", kernel_code="// k")
    nxt = create_initial_iter_state(2, prev)
    assert nxt.authoring_scope == "config_only"
    assert nxt.kernel_code == "// k"

    prev_full = IterState(iter_num=1, status="decided", next_status="implementing")
    assert create_initial_iter_state(2, prev_full).authoring_scope == "full"


async def test_trace_records_the_models_own_text(recorded):
    """The trace held what the model called but not what it said, which is the part
    that explains why. reasoning_content is saved for the single-shot nodes already."""
    from langchain_core.messages import AIMessage

    class Talkative(replay.ScriptedLLM):
        async def ainvoke(self, messages, **kwargs):
            msg = await super().ainvoke(messages, **kwargs)
            if msg.tool_calls:
                msg.content = "Writing the four files now."
                msg.additional_kwargs = {"reasoning_content": "step by step..."}
            return msg

    trace = recorded["tmp"] / "talk.jsonl"
    await run_agentic_step(
        Talkative(recorded["turns"]), "sys", "usr", recorded["box"], trace_path=trace
    )
    records = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    said = [r for r in records if r.get("event") == "assistant"]
    assert said, "the model's text was not recorded"
    assert said[0]["text"] == "Writing the four files now."
    assert said[0]["reasoning"] == "step by step..."
    assert "write_file" in said[0]["tools"]
    assert "said:" in replay.summarize_trace(trace)


async def test_silent_turns_add_no_trace_noise(recorded):
    trace = recorded["tmp"] / "quiet.jsonl"
    await run_agentic_step(
        replay.ScriptedLLM(recorded["turns"]), "sys", "usr", recorded["box"], trace_path=trace
    )
    records = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    assert not [r for r in records if r.get("event") == "assistant"]


async def test_replay_ignores_assistant_records(recorded):
    """turns_from_trace must still reconstruct only the tool calls."""
    trace = recorded["tmp"] / "mixed.jsonl"
    await run_agentic_step(
        replay.ScriptedLLM(recorded["turns"]), "sys", "usr", recorded["box"], trace_path=trace
    )
    with trace.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "assistant", "turn": 1, "text": "hi", "reasoning": ""}) + "\n")
    assert [len(t) for t in replay.turns_from_trace(trace)] == [4, 1, 1]
