"""Finalize stage E: a short overview of the whole run.

Code builds a facts table and the model writes prose around it (headline, findings,
dead ends, caveats). Every number the prose carries must appear in the facts: a model
asked to summarize a run will happily invent a speedup, and the overview is the one
page a user reads. A draft with a foreign number gets one retry naming it; after that
the offending items are dropped rather than shipped.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm_precise
from models.finalize import OverviewDraft
from prompts.finalize import build_overview
from state.crossbranch import branch_errors, iter_branches
from state.types import FinalistState, FinalizeManifest, Overview, OverviewFact, Variant
from utils.branch_bests import BranchBest, rank_key, summary_rank_key
from utils.kernel_tags import CATEGORY_TAGS, category_keys
from utils.log import log

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_LIMITS = {"findings": 5, "dead_ends": 3, "caveats": 3}


def _score(v: Optional[float]) -> str:
    return f"{v:.2f}x"


def _time(v: Optional[float]) -> str:
    return f"{v:.1f} µs"


def _result(summary: Optional[dict]) -> Optional[str]:
    """The speedup when there is one, else the time; None when neither is known."""
    s = summary or {}
    if s.get("speedup") is not None:
        return _score(s["speedup"])
    if s.get("best_time_us") is not None:
        return _time(s["best_time_us"])
    return None


def _reported(f: FinalistState) -> Variant:
    return f.finalized if f.reported == "finalized" and f.finalized is not None else f.source


def _tag_text(v: Variant) -> Optional[str]:
    return ", ".join(category_keys(v.tags, CATEGORY_TAGS)) if v.tags is not None else None


@dataclass(frozen=True)
class BranchRow:
    """One branch of the run, with or without a valid result."""

    name: str  # the strategy name, as BranchBest names it
    ref: str  # state.crossbranch's name for it ('parent/child'), which branch_errors takes
    path: Path
    status: str
    description: str
    summary: Optional[dict]  # its best valid iteration's summary; None if none validated


def _row_key(r: BranchRow) -> tuple:
    return summary_rank_key(r.summary)


def _row_from_best(b: BranchBest) -> BranchRow:
    # rel is "branches/a/branches/b"; crossbranch names that branch "a/b".
    return BranchRow(
        name=b.name, ref="/".join(b.rel.split("/")[1::2]), path=b.path, status=b.status,
        description=b.description, summary=b.summary,
    )


def branch_rows(output_dir: Path, bests: List[BranchBest]) -> List[BranchRow]:
    """Every branch in the tree, best first; the ones that never validated last.

    The bests alone would hide the dead ends, which are what a reader most needs
    explained, and a run where nothing validated would have nothing to summarize.
    """
    by_path = {Path(b.path).resolve(): b for b in bests}
    rows: List[BranchRow] = []
    try:
        for ref, path, manifest in iter_branches(output_dir):
            strategy = manifest.get("strategy") or {}
            best = by_path.pop(Path(path).resolve(), None)
            rows.append(BranchRow(
                name=strategy.get("name") or path.name,
                ref=ref,
                path=Path(path),
                status=manifest.get("status", ""),
                description=strategy.get("description") or "",
                summary=best.summary if best is not None else None,
            ))
    except Exception as e:
        log(f"Could not list the run's branches; the overview covers the bests only: {e}", "WARN")
    # A best the scan did not reach still belongs in the table.
    rows += [_row_from_best(b) for b in by_path.values()]
    return sorted(rows, key=_row_key)


def build_facts(
    manifest: FinalizeManifest,
    bests: List[BranchBest],
    *,
    branches: Optional[List[BranchRow]] = None,
    scanned: bool = True,
) -> List[OverviewFact]:
    """The facts table. ``branches`` is every branch of the run (``branch_rows``); without
    it the table lists only the branches in ``bests``. ``scanned`` False means the scan
    behind ``bests`` raised, so a branch missing from it is unread, not resultless."""
    out: List[OverviewFact] = []
    rows = branches if branches is not None else sorted(map(_row_from_best, bests), key=_row_key)
    no_result = "no valid result" if scanned else "result not read"

    def add(label: str, value) -> None:
        out.append(OverviewFact(label=label, value=str(value)))

    if branches is not None:
        add("Branches", len(branches))
    if scanned:
        add("Branches with a valid result", len(bests))
    add("Finalists", len(manifest.finalists))
    for f in manifest.finalists:
        for tag, v in (("source", f.source), ("finalized", f.finalized)):
            s = (v.summary or {}) if v is not None else {}
            if s.get("speedup") is not None:
                add(f"{f.name} {tag} score", _score(s["speedup"]))
            if s.get("best_time_us") is not None:
                add(f"{f.name} {tag} time", _time(s["best_time_us"]))
        add(f"{f.name} reported variant", f.reported)
        if f.finalized is not None and f.finalized.searcher:
            add(f"{f.name} searcher", f.finalized.searcher)
        if f.kernel_diff_lines is not None:
            add(f"{f.name} kernel diff lines", f.kernel_diff_lines)
        rep = _reported(f)
        if rep.tags is not None and rep.tags.min_arch is not None:
            add(f"{f.name} minimum arch", f"sm_{rep.tags.min_arch}")
        src_tags = _tag_text(f.source)
        fin_tags = _tag_text(f.finalized) if f.finalized is not None else None
        if fin_tags is not None and fin_tags != src_tags:
            add(f"{f.name} source tags", src_tags or "none")
            add(f"{f.name} finalized tags", fin_tags)
        elif src_tags is not None:
            add(f"{f.name} tags", src_tags)
    for key, name in manifest.category_bests.items():
        add(f"Best with {key}", name)
    for r in rows:
        add(f"Branch {r.name}", ", ".join([r.status, _result(r.summary) or no_result]))
    return out


def numbers_in(text: str) -> List[str]:
    return _NUMBER.findall(text)


def _decimals(n: str) -> int:
    return len(n.split(".")[1]) if "." in n else 0


def _same_number(a: str, b: str) -> bool:
    """Equal once the more precise of the two is rounded to the other's places, so
    "11.6" and "160" match "11.57" and "159.6"."""
    places = min(_decimals(a), _decimals(b))
    return round(float(a), places) == round(float(b), places)


def unknown_numbers(text: str, facts: List[OverviewFact]) -> List[str]:
    known: List[str] = []
    for fact in facts:
        known += numbers_in(fact.label) + numbers_in(fact.value)
    return [n for n in numbers_in(text) if not any(_same_number(n, k) for k in known)]


def render_markdown(overview: Overview) -> str:
    lines = ["# Run overview", "", f"**{overview.headline}**", ""]
    for title, items in (
        ("Findings", overview.findings),
        ("Dead ends", overview.dead_ends),
        ("Caveats", overview.caveats),
    ):
        if items:
            lines += [f"## {title}", ""] + [f"- {i}" for i in items] + [""]
    lines += ["| Fact | Value |", "| --- | --- |"]
    for fact in overview.facts:
        lines.append(f"| {fact.label.replace('|', '/')} | {fact.value.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def _code_headline(manifest: FinalizeManifest, bests: List[BranchBest]) -> str:
    ranked = sorted(manifest.finalists, key=lambda f: summary_rank_key(_reported(f).summary))
    if ranked:
        name, summary = ranked[0].name, _reported(ranked[0]).summary
    elif bests:
        best = min(bests, key=rank_key)
        name, summary = best.name, best.summary
    elif manifest.status == "failed":
        # Possibly before the branch scan finished, so "no valid result" may be untrue.
        return "Finalize failed"
    else:
        return "No valid result"
    return f"Best: {name} at {_result(summary) or 'an unknown score'}"


def _last_reasoning(path: Path) -> str:
    """The reasoning recorded by the branch's last iteration's decision."""
    iters = sorted(
        (int(p.name[4:]) for p in path.glob("iter*") if p.name[4:].isdigit()), reverse=True
    )
    for n in iters:
        try:
            state = json.loads((path / f"iter{n}" / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        reasoning = ((state.get("decision") or {}).get("reasoning") or "").strip()
        if reasoning:
            return " ".join(reasoning.split())[:600]
    return ""


def _branch_text(rows: List[BranchRow]) -> str:
    lines = []
    for r in rows:
        desc = " ".join(r.description.split()) or "no description"
        reasoning = _last_reasoning(r.path) or "none recorded"
        lines.append(f"- {r.name} ({r.status}): {desc}\n  last reasoning: {reasoning}")
    return "\n".join(lines)


def _error_text(rows: List[BranchRow], output_dir: Path) -> str:
    """Failure analyses of the failed branches and of every branch that never validated:
    a branch that stopped or ran out of iterations without a result is a dead end too."""
    parts = []
    for r in rows:
        if r.status == "failed" or r.summary is None:
            try:
                parts.append(f"### {r.name}\n{branch_errors(output_dir, r.ref)}")
            except Exception as e:
                log(f"Could not read the errors of branch {r.name}: {e}", "WARN")
    return "\n".join(parts)


def _foreign(draft: OverviewDraft, facts: List[OverviewFact]) -> List[str]:
    text = " ".join([draft.headline, *draft.findings, *draft.dead_ends, *draft.caveats])
    return unknown_numbers(text, facts)


async def _draft(messages: list) -> OverviewDraft:
    return await get_llm_precise().with_structured_output(OverviewDraft).ainvoke(messages)


async def write_overview(
    manifest: FinalizeManifest,
    bests: List[BranchBest],
    output_dir: Path,
    *,
    caveats: List[str],
    scanned: bool = True,
) -> Overview:
    """Build, check and write ``output/finalize/overview.md``. Never raises: with no
    usable model reply the overview is the code-built headline, the facts and the
    caveats. ``scanned`` is False when the branch scan behind ``bests`` raised."""
    output_dir = Path(output_dir)
    headline = _code_headline(manifest, bests)
    findings: List[str] = []
    dead_ends: List[str] = []
    model_caveats: List[str] = []
    facts: List[OverviewFact] = []
    try:
        rows = branch_rows(output_dir, bests)
        facts = build_facts(manifest, bests, branches=rows, scanned=scanned)
        # Whenever a branch exists, including when none validated: why nothing worked
        # is the one thing such a run's reader needs, and only the prose can say it.
        if rows:
            system, user = build_overview({
                "facts": "\n".join(f"- {f.label}: {f.value}" for f in facts),
                "branches": _branch_text(rows),
                "errors": _error_text(rows, output_dir),
            })
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            draft = await _draft(messages)
            bad = _foreign(draft, facts)
            if bad:
                log(f"Overview draft has numbers outside the facts: {bad}; retrying", "WARN")
                messages += [
                    HumanMessage(content=(
                        f"These numbers are not in the facts table: {', '.join(bad)}. "
                        "Rewrite the whole overview using only numbers from the table, "
                        "or none."
                    )),
                ]
                draft = await _draft(messages)
            def ok(text: str) -> bool:
                return not unknown_numbers(text, facts)

            # A headline with a foreign number keeps the code-built one.
            if ok(draft.headline):
                headline = draft.headline.strip() or headline
            findings = [s for s in draft.findings if ok(s)][: _LIMITS["findings"]]
            dead_ends = [s for s in draft.dead_ends if ok(s)][: _LIMITS["dead_ends"]]
            model_caveats = [s for s in draft.caveats if ok(s)][: _LIMITS["caveats"]]
    except Exception as e:
        log(f"Overview LLM step failed, writing the code-built overview: {type(e).__name__}: {e}", "WARN")
        findings, dead_ends, model_caveats = [], [], []
    overview = Overview(
        headline=headline,
        findings=findings,
        dead_ends=dead_ends,
        caveats=model_caveats + list(caveats),
        facts=facts,
    )
    try:
        overview.markdown = render_markdown(overview)
        target = output_dir / "finalize"
        target.mkdir(parents=True, exist_ok=True)
        (target / "overview.md").write_text(overview.markdown, encoding="utf-8")
    except Exception as e:
        log(f"Could not write overview.md: {e}", "ERROR")
    return overview
