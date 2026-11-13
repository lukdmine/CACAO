"""Finalize stage A: choose up to ``cap`` finalist kernels.

Code forces the overall best plus the best branch per category value (design D7), so
the best CUDA-core kernel survives even when four tensor-core kernels outscore it.
One structured LLM call then spends the remaining slots, with a reason for each.
"""

import re
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm_precise
from models.finalize import FinalistPicks, KernelTags
from prompts.finalize import build_select
from utils.branch_bests import BranchBest, rank_key
from utils.kernel_tags import CATEGORY_TAGS, category_keys
from utils.log import log


def forced_finalists(
    bests: List[BranchBest], tags: Dict[str, KernelTags], cap: int
) -> List[Tuple[BranchBest, str]]:
    """The overall best, then for each category key the best branch of each value
    that no chosen finalist already represents. ``tags`` is keyed by ``rel``."""
    ranked = sorted(bests, key=rank_key)
    if not ranked or cap <= 0:
        return []
    chosen: List[Tuple[BranchBest, str]] = [(ranked[0], "overall best")]

    def labels(b: BranchBest) -> List[str]:
        t = tags.get(b.rel)
        return category_keys(t, CATEGORY_TAGS) if t is not None else []

    for key in CATEGORY_TAGS:
        prefix = key + "="
        # ranked is best first, so the first branch seen with a value is that value's
        # best and the dict's order is the rank order of those bests.
        best_per_value: Dict[str, BranchBest] = {}
        for b in ranked:
            for label in labels(b):
                if label.startswith(prefix):
                    best_per_value.setdefault(label, b)
        for label, b in best_per_value.items():
            if len(chosen) >= cap:
                return chosen
            if any(label in labels(c) for c, _ in chosen):
                continue
            chosen.append((b, f"best with {label}"))
    return chosen[:cap]


def finalist_names(bests: List[BranchBest]) -> Dict[str, str]:
    """``rel`` to a directory-safe name, unique across ``bests``."""
    names: Dict[str, str] = {}
    used = set()
    for b in bests:
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", b.name).strip(".") or "branch"
        name, n = base, 1
        while name in used:
            n += 1
            name = f"{base}_{n}"
        used.add(name)
        names[b.rel] = name
    return names


def _fmt(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def _table(bests: List[BranchBest], tags: Dict[str, KernelTags]) -> str:
    rows = []
    for b in sorted(bests, key=rank_key):
        t = tags.get(b.rel)
        tag_text = ", ".join(category_keys(t, CATEGORY_TAGS)) if t else "untagged"
        rows.append(
            f"- {b.name} | status={b.status} | speedup={_fmt(b.score, '.2f')} | "
            f"time_us={_fmt(b.time_us, '.1f')} | {tag_text} | "
            f"{' '.join(b.description.split()) or 'no description'}"
        )
    return "\n".join(rows)


async def select_finalists(
    bests: List[BranchBest], tags: Dict[str, KernelTags], *, cap: int
) -> List[Tuple[BranchBest, str, bool]]:
    """Each finalist as (branch, reason, forced)."""
    if not bests:
        return []
    forced = forced_finalists(bests, tags, cap)
    result: List[Tuple[BranchBest, str, bool]] = [(b, why, True) for b, why in forced]
    if len(result) >= cap:
        return result

    ctx = {
        "branches": _table(bests, tags),
        "forced": "\n".join(f"- {b.name} ({why})" for b, why in forced),
        "cap": cap,
    }
    system, user = build_select(ctx)
    try:
        picks = (
            await get_llm_precise()
            .with_structured_output(FinalistPicks)
            .ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        )
    except Exception as e:
        log(f"Finalist selection LLM call failed, keeping the forced set: {e}", "WARN")
        return result

    by_name = {b.name: b for b in bests}
    index = {b.rel: i for i, (b, _, _) in enumerate(result)}
    for pick in picks.picks:
        b = by_name.get(pick.branch)
        if b is None:
            log(f"Finalist pick names an unknown branch: {pick.branch}", "WARN")
            continue
        if b.rel in index:
            # A forced finalist always says why it was forced; the model's reason
            # is appended to it.
            i = index[b.rel]
            if pick.reason.strip():
                result[i] = (b, f"{result[i][1]}; {pick.reason.strip()}", True)
        elif len(result) < cap:
            index[b.rel] = len(result)
            result.append((b, pick.reason.strip() or "chosen by the model", False))
    return result
