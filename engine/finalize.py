"""Finalize: turn a finished run's best kernels into re-tuned, verified, tagged finalists.

Four stages (design spec 5.4 to 5.7): select finalists from every branch's best valid
iteration, re-widen each one's tuning space, re-tune it with the finalize budget, then
arch-check, profile and tag both of its variants. Finalists run one after another.

Everything lands in ``output/finalize/``: ``finalize.json`` (the manifest, saved after
every stage transition of every finalist) and ``<finalist>/iter<K>/`` (K = the
attempt). Nothing here writes under ``output/branches/``: a branch iteration is only
read, and the one file the driver would otherwise need migrated in place (an MCMC
seed) is copied out first.

A stop leaves the stage reached on disk, and the next call resumes there.
Re-finalizing after the branches moved reuses every finished finalist whose source
did not change (D9).
"""

import asyncio
import json
import os
import shutil
import traceback
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml

import config as _cfg
from agentic.workspace import owned_files_from_dir
from nodes.finalize_overview import write_overview
from nodes.finalize_select import finalist_names, select_finalists
from nodes.finalize_widen import seed_unchanged, widen_finalist
from nodes.profile import profile_node
from nodes.run import run_node
from state import (
    FinalizeManifest,
    StrategyInfo,
    WorkingState,
    finalize_dir,
    load_finalize_manifest,
    save_finalize_manifest,
)
from state.types import FinalistState, Variant
from utils import nvrtc
from utils.arch_check import check_archs
from utils.branch_bests import (
    BranchBest,
    collect_branch_bests,
    is_valid_summary,
    outranks,
    source_fingerprint,
    summary_rank_key,
)
from utils.cases import case_dir, case_list, primary_case_dir
from utils.framework import resolve_cuda_include
from utils.inputs import load_inputs_spec
from utils.kernel_tags import CATEGORY_BEST_TAGS, category_keys, full_tags, static_tags
from utils.landscape import best_record, load_records
from utils.log import log
from utils.problem import gpu_index as problem_gpu_index, reference_spec
from utils.results import ensure_results_loadable
from utils.space import (
    choose_searcher,
    count_for_iteration,
    estimate_exhaustive_s,
    seed_is_reachable,
)

_FINISHED = ("done", "skipped")


def should_auto_finalize(outcomes: list, abandoned_work: bool) -> bool:
    """Whether the run ended by itself (D1): no engine task raised and the monitor did
    not shut down with work still queued."""
    if abandoned_work:
        return False
    return not any(isinstance(o, BaseException) for o in outcomes)


def finalist_dir(output_dir: Path, name: str) -> Path:
    return finalize_dir(output_dir) / name


def finalist_working_state(
    *,
    name: str,
    finalist_root: Path,
    attempt: int,
    problem_yaml: str,
    gpu_info: Optional[dict],
) -> WorkingState:
    """A WorkingState that points run_node and profile_node at ``finalist_root/iter<attempt>``.

    Both resolve their directory as ``branch_path/iter{iter_num}`` and write no
    branch.json, so a finalist directory stands in for a branch.
    """
    return WorkingState(
        analysis="",
        gpu_info=gpu_info,
        strategy=StrategyInfo(name=name),
        problem_yaml=problem_yaml,
        branch_path=str(finalist_root),
        iter_num=attempt,
        current_iter=attempt,
        max_iter=attempt,
        status="running",
    )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _score(summary: Optional[dict]) -> float:
    """What source_fingerprint records per branch: the speedup, else the time."""
    s = summary or {}
    value = s["speedup"] if s.get("speedup") is not None else (s.get("best_time_us") or 0.0)
    return round(value, 6)


def _str_config(summary: Optional[dict]) -> Dict[str, str]:
    """A summary's best configuration with string values, as the probes take it."""
    return {k: str(v) for k, v in ((summary or {}).get("best_config") or {}).items()}


def _reported(f: FinalistState) -> Variant:
    return f.finalized if f.reported == "finalized" and f.finalized is not None else f.source


def _reusable(old: FinalizeManifest, bests: List[BranchBest]) -> List[FinalistState]:
    """Finished finalists of an earlier attempt whose source iteration and score are
    unchanged (D9). Their ``finalized.dir`` keeps naming the attempt that produced it."""
    current = {(b.rel, b.best_iter): b for b in bests}
    keep = []
    for f in old.finalists:
        b = current.get((f.source_branch, f.source_iter))
        if (
            b is not None
            and _score(b.summary) == _score(f.source.summary)
            and all(s in _FINISHED for s in f.stages.values())
        ):
            keep.append(f)
    return keep


def _reselected(f: FinalistState, name: str, reason: str, forced: bool) -> FinalistState:
    """A reused finalist under this attempt's selection: its measurements stand, but the
    reason (and the tag note built from it) is this selection's, since an "overall best"
    from the last attempt may not be one now."""
    for v in (f.source, f.finalized):
        if v is not None and v.tags is not None and f.reason and v.tags.note.startswith(f.reason):
            v.tags.note = reason + v.tags.note[len(f.reason):]
    return f.model_copy(update={"name": name, "reason": reason, "forced": forced})


def _category_bests(finalists: List[FinalistState]) -> Dict[str, str]:
    """For each tag value among the reported variants, the finalist that scores best."""
    best: Dict[str, str] = {}
    for f in sorted(finalists, key=lambda f: summary_rank_key(_reported(f).summary)):
        tags = _reported(f).tags
        if tags is not None:
            for key in category_keys(tags, CATEGORY_BEST_TAGS):
                best.setdefault(key, f.name)
    return best


def _load_or_start(
    old: Optional[FinalizeManifest], fp: str, bests: List[BranchBest], trigger: str
) -> FinalizeManifest:
    if old is not None and old.status != "done" and old.source_fingerprint == fp:
        log(f"Resuming finalize attempt {old.attempt} from '{old.status}'")
        old.error = ""
        old.finished_at = ""
        return old
    attempt, carried = 1, []
    if old is not None:
        # source_fingerprint is set in the same save that records the selection, so an
        # attempt without one stopped having written nothing but this file.
        attempt = old.attempt + 1 if old.source_fingerprint else old.attempt
        carried = _reusable(old, bests)
    return FinalizeManifest(
        status="selecting", attempt=attempt, trigger=trigger, started_at=_now(), finalists=carried
    )


def _searcher_label(mode: str, seeds: Dict[str, str], case_names: List[str]) -> str:
    """What the retune searched with, as Variant.searcher records it."""
    if mode != "mcmc":
        return mode
    if not seeds:
        return "mcmc (unseeded)"
    if len(seeds) == len(case_names):
        return "mcmc (seeded)"
    return f"mcmc (seeded for {', '.join(n for n in case_names if n in seeds)})"


def _tune_failure(state: WorkingState) -> str:
    """Why a retune left no valid result, worded by what actually failed."""
    if (state.run_output or "").startswith("[COMPILE ERROR]"):
        # run_node stops before the tuner and saves the compiler's output in place of
        # the tuner's.
        return (
            "The retune's driver did not build: host compilation of framework.cpp "
            "failed, and the compiler output is in tuner_output.txt."
        )
    failed_case = (state.results_summary or {}).get("failed_case")
    if failed_case:
        return (
            f"The retune produced no valid configuration for case {failed_case} "
            "(see that case's tuner_output.txt)."
        )
    return "The retune produced no valid configuration (see tuner_output.txt)."


def _first_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[0]


def overview_caveats(
    manifest: FinalizeManifest, bests: List[BranchBest], *, scanned: bool = True
) -> List[str]:
    """The caveats the overview always carries, built from the manifest: the model never
    gets to decide whether a limitation is worth mentioning.

    ``scanned`` is whether the branch scan behind ``bests`` completed. Empty bests from a
    scan that raised say nothing about the branches, so they get no "no valid result".
    """
    caveats: List[str] = []
    if manifest.status == "failed":
        caveats.append(f"Finalize failed: {_first_line(manifest.error) or 'no error recorded'}")
    if scanned and not bests:
        caveats.append("No branch produced a valid result.")
    if manifest.finalists and not _cfg.AGENTIC_STEPS:
        caveats.append("Widening did not run: AGENTIC_STEPS is off.")
    for f in manifest.finalists:
        for stage, text in (("widen", "widening"), ("tune", "retuning"), ("verify", "verification")):
            if f.stages.get(stage) == "failed":
                caveats.append(
                    f"{f.name}: {text} failed ({_first_line(f.error)}); its source result stands."
                )
    return caveats


class _Finalize:
    """One finalize call: the run-wide facts every stage needs, and the stages."""

    def __init__(self, manifest: FinalizeManifest, bests: List[BranchBest], fp: str):
        self.m = manifest
        self.bests = bests
        self.fp = fp
        self.output_dir = Path(_cfg.get_output_dir())
        self.problem_dir = Path(_cfg.get_problem_dir())
        self.problem_yaml = (self.problem_dir / "problem.yaml").read_text(encoding="utf-8")
        self.meta = yaml.safe_load(self.problem_yaml) or {}
        self.cases = case_list(self.meta)
        self.gpu_index = problem_gpu_index(self.meta)
        # Absolute, as run_node passes it: the driver's cwd is the staged case directory.
        self.ref_file = self.problem_dir / reference_spec(self.meta).file
        self.gpu_info = _load_json(self.output_dir / "context.json").get("gpu_info")
        self.budget_s = float(_cfg.FINALIZE_TUNER_TIMEOUT)

    @cached_property
    def archs(self) -> List[int]:
        supported = nvrtc.supported_archs()
        return [a for a in _cfg.FINALIZE_ARCHS if a in supported]

    @cached_property
    def cuda_include(self) -> str:
        return resolve_cuda_include()

    @cached_property
    def buffer_dtypes(self) -> List[str]:
        try:
            return [b.dtype for b in load_inputs_spec(self.problem_dir / "inputs.yaml").buffers]
        except Exception as e:
            log(f"Could not read buffer dtypes from inputs.yaml: {e}", "WARN")
            return []

    # --- bookkeeping ---

    def _save(self) -> None:
        save_finalize_manifest(self.output_dir, self.m)

    def _enter(self, status: str) -> None:
        self.m.status = status
        self._save()

    def _iter_dir(self, f: FinalistState) -> Path:
        return finalist_dir(self.output_dir, f.name) / f"iter{self.m.attempt}"

    def _state(self, f: FinalistState) -> WorkingState:
        return finalist_working_state(
            name=f.name,
            finalist_root=finalist_dir(self.output_dir, f.name),
            attempt=self.m.attempt,
            problem_yaml=self.problem_yaml,
            gpu_info=self.gpu_info,
        )

    def _static_tags(self, vdir: Path, summary: Optional[dict]):
        files = owned_files_from_dir(vdir)
        return static_tags(
            files.get("kernels.cu", ""),
            config=_str_config(summary),
            launcher_src=files.get("region_launcher.cpp", ""),
            params_src=files.get("region_params.cpp", ""),
            buffer_dtypes=self.buffer_dtypes,
        )

    # --- the flow ---

    async def run(self) -> FinalizeManifest:
        if not self.m.source_fingerprint:
            await self._select()
        for f in self.m.finalists:
            if f.stages.get("widen") == "pending":
                await self._guarded(f, "widen", self._widen)
            if f.stages.get("tune") == "pending":
                await self._guarded(f, "tune", self._tune)
            if f.stages.get("verify") == "pending":
                await self._guarded(f, "verify", self._verify)
        self.m.category_bests = _category_bests(self.m.finalists)
        await self._finish()
        return self.m

    async def _guarded(self, f: FinalistState, stage: str, run) -> None:
        """Run one finalist's stage; an unexpected error fails that stage and nothing
        else. Left to propagate it would end the run before the other finalists, the
        category bests and the overview, and, with the stage still pending, would hit
        the same finalist again on every resume."""
        try:
            await run(f)
        except Exception as e:
            log(f"[{f.name}] Stage {stage} crashed: {type(e).__name__}: {e}\n{traceback.format_exc()}", "ERROR")
            f.stages[stage] = "failed"
            if stage == "widen" and f.stages.get("tune") == "pending":
                f.stages["tune"] = "skipped"
            f.error = f"{type(e).__name__}: {e}"
            self._save()

    async def _select(self) -> None:
        self._enter("selecting")
        if not self.bests:
            log("No branch has a valid result; there is nothing to finalize", "WARN")
            self.m.finalists = []
        else:
            tags = {
                b.rel: self._static_tags(b.path / f"iter{b.best_iter}", b.summary)
                for b in self.bests
            }
            cap = int(_cfg.FINALIZE_MAX_FINALISTS)
            picks = await select_finalists(self.bests, tags, cap=cap)
            names = finalist_names(self.bests)
            carried = {(f.source_branch, f.source_iter): f for f in self.m.finalists}
            finalists = []
            for b, reason, forced in picks:
                old = carried.get((b.rel, b.best_iter))
                if old is not None:
                    finalists.append(_reselected(old, names[b.rel], reason, forced))
                    continue
                source = Variant(dir=f"{b.rel}/iter{b.best_iter}", summary=b.summary)
                finalists.append(FinalistState(
                    name=names[b.rel], source_branch=b.rel, source_iter=b.best_iter,
                    reason=reason, forced=forced, source=source, reported_dir=source.dir,
                ))
            self.m.finalists = finalists
            reused = sum(1 for f in finalists if all(s in _FINISHED for s in f.stages.values()))
            log(f"Finalists: {', '.join(f.name for f in finalists)} ({reused} reused)", "INFO")
        # Recorded with the selection, in one save: it is what marks selection as done.
        self.m.source_fingerprint = self.fp
        self._save()

    async def _widen(self, f: FinalistState) -> None:
        self._enter("widening")
        iter_dir = self._iter_dir(f)
        source_iter = self.output_dir / f.source.dir
        if not _cfg.AGENTIC_STEPS:
            # No tool loop to widen with: retune the source space as it is (spec 5.5).
            log(f"[{f.name}] AGENTIC_STEPS is off; retuning the source space unchanged", "WARN")
            seed_unchanged(
                iter_dir=iter_dir, source_iter_dir=source_iter,
                problem_dir=self.problem_dir, meta=self.meta,
            )
            f.stages["widen"] = "skipped"
        else:
            result = await widen_finalist(
                iter_dir=iter_dir,
                source_branch=self.output_dir / f.source_branch,
                source_iter_dir=source_iter,
                problem_dir=self.problem_dir,
                output_dir=self.output_dir,
                meta=self.meta,
                problem_yaml=self.problem_yaml,
                gpu_info=self.gpu_info or {},
                budget_s=self.budget_s,
                gpu_index=self.gpu_index,
                ref_file=self.ref_file,
            )
            if result.ok:
                f.stages["widen"] = "done"
                f.kernel_diff_lines = result.diff_lines
            else:
                f.stages["widen"] = "failed"
                f.stages["tune"] = "skipped"
                f.error = result.diagnosis
                log(f"[{f.name}] Widening failed; the source result stands", "WARN")
        self._save()

    def _seeds(self, f: FinalistState, iter_dir: Path) -> Dict[str, str]:
        """Per case, a results base whose best configuration starts the MCMC chain.

        Only for a case whose source best is still a point of the widened space (an
        unreachable seed would fail the tune rather than start somewhere sensible). The
        file is copied into the finalist's directory before KTT sees it: a results.json
        from before KTT 2.3 has to be migrated to load, and migrating it in place would
        write into the source branch.
        """
        params_src = owned_files_from_dir(iter_dir).get("region_params.cpp", "")
        summary = f.source.summary or {}
        per_case = summary.get("cases") or {}
        source_iter = self.output_dir / f.source.dir
        seeds: Dict[str, str] = {}
        for case in self.cases:
            # A multi-case summary's top-level best_config is the primary case's.
            own = (per_case.get(case.name) or {}) if per_case else summary
            best_config = own.get("best_config")
            src = case_dir(source_iter, self.cases, case) / "results.json"
            if best_config is None or not src.is_file():
                continue
            if not seed_is_reachable(best_config, params_src):
                log(
                    f"[{f.name}] The source's best for case {case.name} is not in the "
                    "widened space; MCMC starts unseeded",
                    "WARN",
                )
                continue
            dest = iter_dir / f"seed_{case.name}.json"
            shutil.copyfile(src, dest)
            ensure_results_loadable(dest)
            seeds[case.name] = str(dest.with_suffix(""))
        return seeds

    async def _tune(self, f: FinalistState) -> None:
        self._enter("tuning")
        iter_dir = self._iter_dir(f)
        # Synchronous g++ build plus a driver run; off the loop, as run_node builds.
        count = await asyncio.to_thread(
            count_for_iteration, iter_dir, self.meta, self.problem_dir,
            gpu_index=self.gpu_index, ref_file=self.ref_file,
        )
        source_primary = primary_case_dir(self.problem_dir, self.output_dir / f.source.dir)
        estimate = (
            estimate_exhaustive_s(count, source_primary / "results.json")
            if count is not None else None
        )
        mode = choose_searcher(estimate, self.budget_s)
        seeds = self._seeds(f, iter_dir) if mode == "mcmc" else {}
        searcher = _searcher_label(mode, seeds, [c.name for c in self.cases])
        log(
            f"[{f.name}] {count if count is not None else 'unknown'} configurations, "
            f"exhaustive estimate {'unknown' if estimate is None else f'{estimate:.0f} s'} "
            f"against {self.budget_s:.0f} s: {searcher}"
        )
        state = await run_node(
            self._state(f),
            budget_override_s=self.budget_s,
            mode=mode,
            seed_results_by_case=seeds or None,
        )
        summary = state.results_summary
        f.finalized = Variant(
            dir=iter_dir.relative_to(self.output_dir).as_posix(), summary=summary, searcher=searcher
        )
        if is_valid_summary(summary):
            f.stages["tune"] = "done"
            if outranks(summary, f.source.summary):
                f.reported, f.reported_dir = "finalized", f.finalized.dir
        else:
            # Failed rather than done, so a later re-finalize tries again (D9 reuses only
            # finished work); the source result stands meanwhile.
            f.stages["tune"] = "failed"
            f.error = _tune_failure(state)
            log(f"[{f.name}] {f.error} The source result stands.", "WARN")
        self._save()

    async def _probe(self, f: FinalistState, v: Variant, ncu_metrics: Optional[dict]) -> None:
        """Arch check, shared memory and tags for one variant, from its own files."""
        vdir = self.output_dir / v.dir
        files = owned_files_from_dir(vdir)
        primary = primary_case_dir(self.problem_dir, vdir)
        header = primary / "inputs.hpp"
        if not header.is_file():
            header = self.problem_dir / "inputs.hpp"
        defines = nvrtc.parse_defines_from_inputs_hpp(
            header.read_text(encoding="utf-8") if header.is_file() else ""
        )
        # A few NVRTC compiles per arch: off the loop.
        v.arch = await asyncio.to_thread(
            check_archs,
            files.get("kernels.cu", ""),
            files.get("region_params.cpp", ""),
            winning_config=_str_config(v.summary),
            cuda_include=self.cuda_include,
            scalar_defines=defines,
            archs=self.archs,
        )
        best = best_record(load_records(primary / "results.json"))
        v.ncu_metrics = ncu_metrics
        v.tags = full_tags(
            self._static_tags(vdir, v.summary),
            smem_bytes=best.smem if best is not None else None,
            arch=v.arch,
            ncu_metrics=ncu_metrics,
            note=f.reason,
        )

    async def _verify(self, f: FinalistState) -> None:
        self._enter("verifying")
        # The source's reading was taken by its branch; NCU is never pointed at a branch.
        source_ncu = _load_json(self.output_dir / f.source.dir / "state.json").get("ncu_metrics")
        await self._probe(f, f.source, source_ncu)
        if f.finalized is not None and is_valid_summary(f.finalized.summary):
            ncu = None
            if f.reported == "finalized":
                ncu = (await profile_node(self._state(f))).ncu_metrics
            await self._probe(f, f.finalized, ncu)
        f.stages["verify"] = "done"
        self._save()

    async def _finish(self) -> None:
        self._enter("overview")
        self.m.overview = await write_overview(
            self.m, self.bests, self.output_dir, caveats=overview_caveats(self.m, self.bests)
        )
        self.m.status = "done"
        self.m.finished_at = _now()
        self._save()


async def run_finalize(*, trigger: Literal["auto", "manual"]) -> Optional[FinalizeManifest]:
    """Run, resume or reuse finalize over the active problem's output. Never raises: an
    unexpected exception is recorded as ``status="failed"`` and ``error``, and a later
    call resumes from the stage each finalist reached."""
    output_dir = Path(_cfg.get_output_dir())
    manifest: Optional[FinalizeManifest] = None
    bests: List[BranchBest] = []
    scanned = False
    try:
        bests = collect_branch_bests(output_dir, _cfg.get_problem_dir())
        scanned = True
        fp = source_fingerprint(bests)
        old = load_finalize_manifest(output_dir)
        up_to_date = old is not None and old.status == "done" and old.source_fingerprint == fp
        if up_to_date and trigger == "auto":
            log("Finalize is up to date with the branches; nothing to do")
            return old
        manifest = _load_or_start(old, fp, bests, trigger)
        manifest.pid = os.getpid()
        save_finalize_manifest(output_dir, manifest)
        return await _Finalize(manifest, bests, fp).run()
    except Exception as e:
        log(f"Finalize failed: {type(e).__name__}: {e}\n{traceback.format_exc()}", "ERROR")
        if manifest is None:
            # Failed before anything was loaded: a manifest on disk is the reuse record
            # of an earlier attempt, and a fresh empty one saved over it would make the
            # next finalize redo every finalist.
            if load_finalize_manifest(output_dir) is not None:
                return None
            manifest = FinalizeManifest(trigger=trigger, started_at=_now())
        manifest.status = "failed"
        manifest.error = f"{type(e).__name__}: {e}"
        manifest.finished_at = _now()
        try:
            manifest.overview = await write_overview(
                manifest, bests, output_dir,
                caveats=overview_caveats(manifest, bests, scanned=scanned),
                scanned=scanned,
            )
        except Exception as overview_error:
            log(f"Could not write the overview of a failed finalize: {overview_error}", "ERROR")
        try:
            save_finalize_manifest(output_dir, manifest)
        except Exception as save_error:
            log(f"Could not record the failure in finalize.json: {save_error}", "ERROR")
        return manifest


async def run_manual_finalize() -> Optional[FinalizeManifest]:
    """``run_finalize(trigger="manual")`` in a process of its own: cli --finalize and
    the API's finalize-only child.

    The token counter is loaded from output/ first and saved after, as
    run_optimization_engine does around a run. Without that, the run's
    token_usage.json and the UI's counter never see what finalize spent (its selection,
    widen steps and overview).
    """
    output_dir = Path(_cfg.get_output_dir())
    _cfg.global_tracker.load(output_dir)
    try:
        return await run_finalize(trigger="manual")
    finally:
        _cfg.global_tracker.save(output_dir)
