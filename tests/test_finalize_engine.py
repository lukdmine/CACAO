"""The finalize orchestrator: stages A to D over a finished run.

Every stage's worker (selection, widening, the tuner, NCU, the arch check) is a
recording fake, so these tests cover the orchestration alone: which stage runs for which
finalist, what the manifest records, what survives a stop, and that nothing under
output/branches/ is ever written.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import engine.finalize as fin
from models.finalize import ArchReport, KernelTags
from nodes.finalize_widen import WidenResult
from state import FinalizeManifest, load_finalize_manifest, save_finalize_manifest
from state.types import FinalistState, Overview, Variant
from tests.helpers.finalize_fixtures import invalid_summary, make_branch, valid_summary
from utils.branch_bests import collect_branch_bests, source_fingerprint

PROBLEM_YAML = (
    "grid: {x: 1024}\n"
    "reference: {type: cuda, function: ref_kernel, file: ref_kernel.cu}\n"
    "validation: {tolerance: 0.0001}\n"
)

KERNEL = 'extern "C" __global__ void k(float* o)\n{\n    o[0] = TILE;\n}\n'


def params(*values) -> str:
    vals = ", ".join(str(v) for v in values)
    return f'tuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{{{vals}}});\n'


def results_json(tile: int) -> dict:
    # No Timestamp: a pre-2.3 KTT wrote it, as it did every recorded covariance result.
    return {
        "Metadata": {},
        "Results": [
            {
                "KernelName": "k",
                "Status": "Ok",
                "TotalDuration": 5000,
                "TotalOverhead": 200000,
                "Configuration": [{"Name": "TILE", "Value": tile}],
                "ComputationResults": [
                    {"Duration": 100000, "CompilationData": {"LocalMemorySize": 4096}}
                ],
            }
        ],
    }


def add_branch(output: Path, name: str, speedup, time_us: float = 100.0, *, tile: int = 16):
    """A branch whose iter2 validated (iter1 did not), with the files a real one has."""
    summary = valid_summary(speedup, time_us)
    summary["best_config"] = {"TILE": tile}
    branch = make_branch(output, name, "success", [invalid_summary(), summary])
    it = branch / "iter2"
    (it / "kernels.cu").write_text(KERNEL, encoding="utf-8")
    (it / "region_kernels.cpp").write_text("ktt::KernelId kernel = 0;\n", encoding="utf-8")
    (it / "region_params.cpp").write_text(params(8, 16, 32), encoding="utf-8")
    (it / "region_launcher.cpp").write_text("// launcher\n", encoding="utf-8")
    (it / "results.json").write_text(json.dumps(results_json(tile)), encoding="utf-8")
    state = json.loads((it / "state.json").read_text(encoding="utf-8"))
    state["ncu_metrics"] = {"source_metric": 1.0}
    (it / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return branch


def rescore(output: Path, name: str, speedup: float, time_us: float) -> None:
    """The branch's best iteration now scores differently (the run moved on)."""
    path = output / "branches" / name / "iter2" / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["results_summary"].update(speedup=speedup, best_time_us=time_us)
    path.write_text(json.dumps(state), encoding="utf-8")


def snapshot(root: Path) -> dict:
    """Every file's sha1 and every directory, so a new file or folder shows too."""
    snap = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        snap[rel] = hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else "dir"
    return snap


def lay_out(iter_dir: Path, source_iter_dir: Path, params_src=None) -> None:
    """What a committed widening step leaves: the four owned files in iter_dir."""
    iter_dir = Path(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)
    for name in ("kernels.cu", "region_kernels.cpp", "region_params.cpp", "region_launcher.cpp"):
        text = (Path(source_iter_dir) / name).read_text(encoding="utf-8")
        (iter_dir / name).write_text(text, encoding="utf-8")
    if params_src is not None:
        (iter_dir / "region_params.cpp").write_text(params_src, encoding="utf-8")


class Fakes:
    def __init__(self):
        self.calls = {
            k: [] for k in ("select", "widen", "seed", "count", "run", "profile", "arch", "overview")
        }
        self.widen_result = WidenResult(ok=True, diff_lines=4, summary="widened")
        self.widen_params = None  # region_params.cpp the widened kernel declares
        self.count = 10
        self.run_summary = {}  # finalist name -> the retune's summary
        self.run_error = None
        self.select_error = None
        self.overview_error = None

    def reset(self):
        for v in self.calls.values():
            v.clear()

    async def select(self, bests, tags, *, cap):
        self.calls["select"].append(
            SimpleNamespace(bests=[b.name for b in bests], tags=tags, cap=cap)
        )
        if self.select_error is not None:
            raise self.select_error
        return [(b, f"why {b.name}", i == 0) for i, b in enumerate(bests[:cap])]

    async def overview(self, manifest, bests, output_dir, *, caveats, scanned=True):
        self.calls["overview"].append(SimpleNamespace(
            status=manifest.status, bests=[b.name for b in bests], caveats=caveats,
            scanned=scanned,
        ))
        if self.overview_error is not None:
            raise self.overview_error
        return Overview(headline="stub overview")

    async def widen(self, **kw):
        self.calls["widen"].append(SimpleNamespace(**kw))
        if self.widen_result.ok:
            lay_out(kw["iter_dir"], kw["source_iter_dir"], self.widen_params)
        return self.widen_result

    def seed(self, **kw):
        self.calls["seed"].append(SimpleNamespace(**kw))
        lay_out(kw["iter_dir"], kw["source_iter_dir"])

    def count_for(self, iter_dir, meta, problem_dir, *, gpu_index, ref_file):
        self.calls["count"].append(Path(iter_dir))
        return self.count

    async def run(self, state, **kw):
        self.calls["run"].append(SimpleNamespace(state=state, **kw))
        if self.run_error is not None:
            raise self.run_error
        state.results_summary = self.run_summary.get(state.strategy.name, valid_summary(9.0, 10.0))
        state.status = "profiling"
        return state

    async def profile(self, state):
        self.calls["profile"].append(state)
        state.ncu_metrics = {"profiled": 1.0}
        return state

    def arch(self, kernel_src, params_src, *, winning_config, cuda_include, scalar_defines, archs):
        self.calls["arch"].append(
            SimpleNamespace(kernel_src=kernel_src, winning_config=winning_config, archs=archs)
        )
        return ArchReport(archs=list(archs), base_min_arch=75, winner_min_arch=80)


@pytest.fixture
def env(tmp_path, monkeypatch):
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "problem.yaml").write_text(PROBLEM_YAML, encoding="utf-8")
    (problem / "ref_kernel.cu").write_text("// ref\n", encoding="utf-8")
    output = problem / "output"
    output.mkdir()
    monkeypatch.setattr(config, "_base_output_dir", output)
    monkeypatch.setattr(config, "_problem_dir", problem)
    monkeypatch.setattr(config, "AGENTIC_STEPS", True)

    fakes = Fakes()
    monkeypatch.setattr(fin, "select_finalists", fakes.select)
    monkeypatch.setattr(fin, "write_overview", fakes.overview)
    monkeypatch.setattr(fin, "widen_finalist", fakes.widen)
    monkeypatch.setattr(fin, "seed_unchanged", fakes.seed)
    monkeypatch.setattr(fin, "count_for_iteration", fakes.count_for)
    monkeypatch.setattr(fin, "run_node", fakes.run)
    monkeypatch.setattr(fin, "profile_node", fakes.profile)
    monkeypatch.setattr(fin, "check_archs", fakes.arch)
    monkeypatch.setattr(fin.nvrtc, "supported_archs", lambda: [75, 80, 86])
    monkeypatch.setattr(fin, "resolve_cuda_include", lambda: "/usr/include")
    return SimpleNamespace(problem=problem, output=output, fakes=fakes)


def by_name(manifest) -> dict:
    return {f.name: f for f in manifest.finalists}


async def test_finalize_records_both_variants_and_reports_the_faster(env):
    add_branch(env.output, "fast", 4.0, 25.0)
    add_branch(env.output, "slow", 2.0, 50.0)
    env.fakes.run_summary = {"fast": valid_summary(5.0, 20.0), "slow": valid_summary(2.5, 40.0)}

    m = await fin.run_finalize(trigger="manual")

    assert m.status == "done" and m.finished_at and m.attempt == 1 and m.trigger == "manual"
    assert m.source_fingerprint == source_fingerprint(collect_branch_bests(env.output, env.problem))
    f = by_name(m)["fast"]
    assert (f.source_branch, f.source_iter) == ("branches/fast", 2)
    assert f.source.dir == "branches/fast/iter2"
    assert f.finalized.dir == "finalize/fast/iter1"
    assert f.reported == "finalized" and f.reported_dir == "finalize/fast/iter1"
    assert f.stages == {"widen": "done", "tune": "done", "verify": "done"}
    assert f.kernel_diff_lines == 4
    assert f.finalized.summary["speedup"] == 5.0
    # Both variants are arch-checked and tagged (D13); only the reported one is profiled.
    assert f.source.arch.winner_min_arch == 80 and f.finalized.arch.winner_min_arch == 80
    assert f.source.tags.smem_bytes == 4096 and f.source.tags.min_arch == 80
    assert f.finalized.tags is not None
    assert f.source.ncu_metrics == {"source_metric": 1.0}
    assert f.finalized.ncu_metrics == {"profiled": 1.0}
    assert len(env.fakes.calls["arch"]) == 4
    assert env.fakes.calls["arch"][0].archs == [75, 80, 86]
    assert {"TILE": "16"} in [c.winning_config for c in env.fakes.calls["arch"]]

    run = env.fakes.calls["run"][0]
    assert Path(run.state.branch_path) == env.output / "finalize" / "fast"
    assert run.state.iter_num == 1 and run.budget_override_s == 1000
    assert run.mode == "exhaustive" and not run.seed_results_by_case
    assert [Path(s.branch_path).name for s in env.fakes.calls["profile"]] == ["fast", "slow"]

    assert m.category_bests == {
        "tensor_cores=false": "fast",
        "precision=fp32": "fast",
        "min_arch=80": "fast",
    }
    assert load_finalize_manifest(env.output) == m


@pytest.mark.parametrize(
    "retuned", [valid_summary(3.0, 33.0), valid_summary(4.0, 25.0)], ids=["slower", "tie"]
)
async def test_a_slower_retune_reports_the_source(env, retuned):
    add_branch(env.output, "b", 4.0, 25.0)
    env.fakes.run_summary = {"b": retuned}

    m = await fin.run_finalize(trigger="manual")

    f = by_name(m)["b"]
    assert f.reported == "source" and f.reported_dir == "branches/b/iter2"
    assert f.finalized.dir == "finalize/b/iter1" and f.finalized.summary == retuned
    assert f.stages == {"widen": "done", "tune": "done", "verify": "done"}
    # The source's NCU reading comes from its own state.json; NCU never runs on a branch.
    assert env.fakes.calls["profile"] == []
    assert f.source.ncu_metrics == {"source_metric": 1.0}
    assert f.finalized.tags is not None and f.finalized.ncu_metrics is None


async def test_nothing_under_branches_changes(env):
    add_branch(env.output, "a", 4.0, 25.0)
    add_branch(env.output, "b", 2.0, 50.0)
    # MCMC with a reachable seed: the one path that hands a branch's results.json (a
    # pre-2.3 one, which the driver could not load) to the tuner.
    env.fakes.count = 10**6
    env.fakes.widen_params = params(8, 16, 32, 64)
    before = snapshot(env.output / "branches")

    m = await fin.run_finalize(trigger="manual")

    assert m.status == "done"
    assert all(r.seed_results_by_case for r in env.fakes.calls["run"])
    assert snapshot(env.output / "branches") == before


async def test_resume_after_a_stop_skips_finished_finalists(env):
    add_branch(env.output, "A", 4.0, 25.0)
    add_branch(env.output, "B", 2.0, 50.0)
    fp = source_fingerprint(collect_branch_bests(env.output, env.problem))
    lay_out(env.output / "finalize" / "B" / "iter1", env.output / "branches" / "B" / "iter2")
    a = FinalistState(
        name="A",
        source_branch="branches/A",
        source_iter=2,
        stages={"widen": "done", "tune": "done", "verify": "done"},
        source=Variant(dir="branches/A/iter2", summary=valid_summary(4.0, 25.0), tags=KernelTags()),
        finalized=Variant(
            dir="finalize/A/iter1", summary=valid_summary(5.0, 20.0), tags=KernelTags()
        ),
        reported="finalized",
        reported_dir="finalize/A/iter1",
    )
    b = FinalistState(
        name="B",
        source_branch="branches/B",
        source_iter=2,
        stages={"widen": "done", "tune": "pending", "verify": "pending"},
        source=Variant(dir="branches/B/iter2", summary=valid_summary(2.0, 50.0)),
        reported_dir="branches/B/iter2",
    )
    save_finalize_manifest(
        env.output,
        FinalizeManifest(status="tuning", attempt=1, source_fingerprint=fp, finalists=[a, b]),
    )

    m = await fin.run_finalize(trigger="manual")

    assert m.status == "done" and m.attempt == 1
    assert env.fakes.calls["select"] == []
    assert env.fakes.calls["widen"] == [] and env.fakes.calls["seed"] == []
    assert [r.state.strategy.name for r in env.fakes.calls["run"]] == ["B"]
    assert Path(env.fakes.calls["run"][0].state.branch_path) == env.output / "finalize" / "B"
    assert len(env.fakes.calls["arch"]) == 2  # B's two variants; A was already verified
    found = by_name(m)
    assert found["A"].finalized.dir == "finalize/A/iter1"
    assert found["B"].stages == {"widen": "done", "tune": "done", "verify": "done"}


async def test_auto_finalize_is_a_no_op_when_up_to_date(env):
    add_branch(env.output, "b", 4.0, 25.0)
    first = await fin.run_finalize(trigger="auto")
    on_disk = (env.output / "finalize" / "finalize.json").read_text(encoding="utf-8")
    env.fakes.reset()

    again = await fin.run_finalize(trigger="auto")

    assert again == first
    assert all(not calls for calls in env.fakes.calls.values())
    assert (env.output / "finalize" / "finalize.json").read_text(encoding="utf-8") == on_disk


async def test_refinalize_reuses_unchanged_finalists(env):
    add_branch(env.output, "A", 4.0, 25.0)
    add_branch(env.output, "B", 2.0, 50.0)
    await fin.run_finalize(trigger="auto")
    rescore(env.output, "B", 3.0, 33.0)
    env.fakes.reset()

    m = await fin.run_finalize(trigger="auto")

    assert m.status == "done" and m.attempt == 2
    assert len(env.fakes.calls["select"]) == 1
    widened = [Path(w.iter_dir) for w in env.fakes.calls["widen"]]
    assert widened == [env.output / "finalize" / "B" / "iter2"]
    assert [r.state.strategy.name for r in env.fakes.calls["run"]] == ["B"]
    found = by_name(m)
    assert found["A"].finalized.dir == "finalize/A/iter1"
    assert found["A"].stages == {"widen": "done", "tune": "done", "verify": "done"}
    assert found["B"].finalized.dir == "finalize/B/iter2"
    assert found["B"].source.summary["speedup"] == 3.0


async def test_a_failure_during_selection_selects_again_on_resume(env):
    add_branch(env.output, "A", 4.0, 25.0)
    add_branch(env.output, "B", 2.0, 50.0)
    await fin.run_finalize(trigger="auto")
    rescore(env.output, "B", 3.0, 33.0)
    env.fakes.select_error = RuntimeError("selection blew up")

    failed = await fin.run_finalize(trigger="auto")

    assert failed.status == "failed" and "selection blew up" in failed.error
    assert failed.attempt == 2 and failed.source_fingerprint == ""
    env.fakes.select_error = None
    env.fakes.reset()

    m = await fin.run_finalize(trigger="manual")

    # The attempt that never finished selecting wrote nothing of its own, so its number
    # is reused, and A, finished in attempt 1, is still reused rather than redone.
    assert m.status == "done" and m.attempt == 2 and m.error == ""
    assert len(env.fakes.calls["select"]) == 1
    assert [r.state.strategy.name for r in env.fakes.calls["run"]] == ["B"]
    assert by_name(m)["A"].finalized.dir == "finalize/A/iter1"


async def test_agentic_steps_off_skips_widening_and_retunes_the_source(env, monkeypatch):
    monkeypatch.setattr(config, "AGENTIC_STEPS", False)
    add_branch(env.output, "b", 4.0, 25.0)

    m = await fin.run_finalize(trigger="manual")

    f = by_name(m)["b"]
    assert env.fakes.calls["widen"] == []
    seed = env.fakes.calls["seed"][0]
    assert Path(seed.iter_dir) == env.output / "finalize" / "b" / "iter1"
    assert Path(seed.source_iter_dir) == env.output / "branches" / "b" / "iter2"
    assert f.stages == {"widen": "skipped", "tune": "done", "verify": "done"}
    assert [r.state.strategy.name for r in env.fakes.calls["run"]] == ["b"]
    assert f.finalized.dir == "finalize/b/iter1"


async def test_a_widen_failure_keeps_the_source_and_records_why(env):
    add_branch(env.output, "b", 4.0, 25.0)
    env.fakes.widen_result = WidenResult(ok=False, diagnosis="the step ran out of tool calls")

    m = await fin.run_finalize(trigger="manual")

    f = by_name(m)["b"]
    assert m.status == "done"
    assert f.stages == {"widen": "failed", "tune": "skipped", "verify": "done"}
    assert f.error == "the step ran out of tool calls"
    assert f.reported == "source" and f.reported_dir == "branches/b/iter2"
    assert f.finalized is None
    assert env.fakes.calls["count"] == [] and env.fakes.calls["run"] == []
    assert len(env.fakes.calls["arch"]) == 1 and f.source.tags is not None


async def test_a_retune_with_no_valid_result_fails_its_stage(env):
    add_branch(env.output, "b", 4.0, 25.0)
    env.fakes.run_summary = {"b": invalid_summary()}

    m = await fin.run_finalize(trigger="manual")

    f = by_name(m)["b"]
    assert f.stages == {"widen": "done", "tune": "failed", "verify": "done"}
    assert f.error.startswith("The retune produced no valid configuration")
    assert f.reported == "source"
    assert f.finalized.dir == "finalize/b/iter1" and f.finalized.tags is None
    assert len(env.fakes.calls["arch"]) == 1


async def test_a_retune_whose_driver_did_not_build_says_so(env, monkeypatch):
    add_branch(env.output, "b", 4.0, 25.0)

    async def build_fails(state, **kw):
        # What run_node returns when g++ rejects framework.cpp: no tuner run at all.
        state.run_output = "[COMPILE ERROR] Host compilation of framework.cpp failed.\n\nerror: x"
        state.results_summary = invalid_summary()
        state.status = "proposing"
        return state

    monkeypatch.setattr(fin, "run_node", build_fails)

    f = by_name(await fin.run_finalize(trigger="manual"))["b"]

    assert f.stages["tune"] == "failed"
    assert "did not build" in f.error and "no valid configuration" not in f.error


@pytest.mark.parametrize(
    "count, widened, searcher",
    [
        (10, None, "exhaustive"),
        (10**6, params(8, 16, 32, 64), "mcmc (seeded)"),
        (10**6, params(8, 32), "mcmc (unseeded)"),  # 16, the source's best, is gone
    ],
)
async def test_the_searcher_is_recorded_on_the_finalized_variant(env, count, widened, searcher):
    add_branch(env.output, "b", 4.0, 25.0, tile=16)
    env.fakes.count = count
    env.fakes.widen_params = widened

    f = by_name(await fin.run_finalize(trigger="manual"))["b"]

    assert f.finalized.searcher == searcher
    assert load_finalize_manifest(env.output).finalists[0].finalized.searcher == searcher


async def test_no_valid_result_ends_without_selecting(env):
    make_branch(env.output, "x", "failed", [invalid_summary()])

    m = await fin.run_finalize(trigger="auto")

    assert m.status == "done" and m.finished_at
    assert m.finalists == [] and m.category_bests == {}
    assert env.fakes.calls["select"] == []
    assert load_finalize_manifest(env.output) == m


async def test_an_unreachable_seed_runs_mcmc_unseeded(env):
    add_branch(env.output, "b", 4.0, 25.0, tile=16)
    env.fakes.count = 10**6  # far past what the budget can exhaust
    env.fakes.widen_params = params(8, 32)  # 16, the source's best, is gone

    await fin.run_finalize(trigger="manual")

    run = env.fakes.calls["run"][0]
    assert run.mode == "mcmc" and not run.seed_results_by_case


async def test_a_reachable_seed_is_a_migrated_copy_outside_the_branch(env):
    add_branch(env.output, "b", 4.0, 25.0, tile=16)
    env.fakes.count = 10**6
    env.fakes.widen_params = params(8, 16, 32, 64)

    await fin.run_finalize(trigger="manual")

    run = env.fakes.calls["run"][0]
    seed = env.output / "finalize" / "b" / "iter1" / "seed_default"
    assert run.mode == "mcmc" and run.seed_results_by_case == {"default": str(seed)}
    copied = json.loads(Path(f"{seed}.json").read_text(encoding="utf-8"))
    assert all("Timestamp" in r for r in copied["Results"])


async def test_an_unexpected_error_fails_the_manifest_and_a_rerun_resumes(env, monkeypatch):
    add_branch(env.output, "b", 4.0, 25.0)

    real = fin._category_bests
    broken = [True]

    def boom(finalists):
        if broken[0]:
            raise RuntimeError("bests exploded")
        return real(finalists)

    monkeypatch.setattr(fin, "_category_bests", boom)

    failed = await fin.run_finalize(trigger="auto")

    assert failed.status == "failed" and "bests exploded" in failed.error
    assert load_finalize_manifest(env.output).status == "failed"
    # The overview still runs after a failure, and says what status it found.
    assert failed.overview is not None and env.fakes.calls["overview"][-1].status == "failed"
    broken[0] = False
    env.fakes.reset()

    m = await fin.run_finalize(trigger="manual")

    assert m.status == "done" and m.attempt == 1 and m.error == ""
    assert env.fakes.calls["widen"] == [] and env.fakes.calls["run"] == []


async def test_the_overview_runs_before_done_with_the_code_caveats(env):
    add_branch(env.output, "b", 4.0, 25.0)
    env.fakes.widen_result = WidenResult(ok=False, diagnosis="out of tool calls\nmore")

    m = await fin.run_finalize(trigger="manual")

    call = env.fakes.calls["overview"][0]
    assert call.status == "overview" and m.status == "done"
    assert call.caveats == ["b: widening failed (out of tool calls); its source result stands."]
    assert m.overview.headline == "stub overview"


async def test_no_valid_result_still_gets_an_overview(env):
    make_branch(env.output, "x", "failed", [invalid_summary()])

    m = await fin.run_finalize(trigger="auto")

    assert m.status == "done"
    assert env.fakes.calls["overview"][0].caveats == ["No branch produced a valid result."]


async def test_an_early_failure_leaves_the_existing_manifest_alone(env, monkeypatch):
    add_branch(env.output, "b", 4.0, 25.0)
    done = await fin.run_finalize(trigger="manual")
    before = (env.output / "finalize" / "finalize.json").read_bytes()

    def boom(*a, **k):
        raise RuntimeError("cannot read branches")

    monkeypatch.setattr(fin, "collect_branch_bests", boom)

    assert await fin.run_finalize(trigger="manual") is None
    assert (env.output / "finalize" / "finalize.json").read_bytes() == before
    assert load_finalize_manifest(env.output).finalists[0].name == done.finalists[0].name


async def test_an_early_failure_with_no_manifest_records_it(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("cannot read branches")

    monkeypatch.setattr(fin, "collect_branch_bests", boom)

    m = await fin.run_finalize(trigger="manual")

    assert m.status == "failed" and "cannot read branches" in m.error
    assert load_finalize_manifest(env.output).status == "failed"


async def test_a_failed_finalize_names_its_failure_in_the_overview(env, monkeypatch):
    add_branch(env.output, "b", 4.0, 25.0)

    def boom(finalists):
        raise RuntimeError("bests exploded\nsecond line")

    monkeypatch.setattr(fin, "_category_bests", boom)

    m = await fin.run_finalize(trigger="manual")

    assert m.status == "failed"
    caveats = env.fakes.calls["overview"][-1].caveats
    assert "Finalize failed: RuntimeError: bests exploded" in caveats
    assert "No branch produced a valid result." not in caveats


async def test_a_failed_branch_scan_is_not_reported_as_no_valid_result(env, monkeypatch):
    add_branch(env.output, "b", 4.0, 25.0)

    def boom(*a, **k):
        raise RuntimeError("cannot read branches")

    monkeypatch.setattr(fin, "collect_branch_bests", boom)

    m = await fin.run_finalize(trigger="manual")

    assert m.status == "failed"
    call = env.fakes.calls["overview"][-1]
    assert call.caveats == ["Finalize failed: RuntimeError: cannot read branches"]
    assert call.scanned is False


async def test_a_crash_in_one_finalists_stage_does_not_stop_the_others(env, monkeypatch):
    add_branch(env.output, "A", 4.0, 25.0)
    add_branch(env.output, "B", 2.0, 50.0)
    real_run = env.fakes.run

    async def run(state, **kw):
        if state.strategy.name == "A":
            raise RuntimeError("tuner exploded")
        return await real_run(state, **kw)

    monkeypatch.setattr(fin, "run_node", run)

    m = await fin.run_finalize(trigger="manual")

    a, b = by_name(m)["A"], by_name(m)["B"]
    assert m.status == "done"
    assert a.stages == {"widen": "done", "tune": "failed", "verify": "done"}
    assert "tuner exploded" in a.error and a.reported == "source"
    assert b.stages == {"widen": "done", "tune": "done", "verify": "done"}
    assert env.fakes.calls["overview"][0].caveats[0].startswith("A: retuning failed")



def test_should_auto_finalize():
    assert fin.should_auto_finalize([None, None], False)
    assert fin.should_auto_finalize([], False)
    assert not fin.should_auto_finalize([None, RuntimeError("worker died")], False)
    assert not fin.should_auto_finalize([None], True)


def test_finalist_working_state_points_at_the_finalist(tmp_path):
    root = tmp_path / "finalize" / "b"
    state = fin.finalist_working_state(
        name="b", finalist_root=root, attempt=3, problem_yaml="x: 1", gpu_info={"name": "gpu"}
    )
    assert fin.finalist_dir(tmp_path, "b") == root
    assert (state.iter_num, state.current_iter, state.max_iter) == (3, 3, 3)
    assert state.status == "running" and state.strategy.name == "b"
    assert Path(state.branch_path) == root and state.gpu_info == {"name": "gpu"}
