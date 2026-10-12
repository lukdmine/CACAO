"""The multi-case block shared by every prompt that shows results.

Empty for a single-case problem, so no existing prompt changes shape unless the problem
declares cases. The rule text is generated rather than written per problem: it was prose
in problem.yaml before this feature and nothing enforced it.
"""

from __future__ import annotations

from utils.cases import case_list


def cases_block(meta: dict, summary: dict | None) -> str:
    cases = case_list(meta or {})
    if len(cases) <= 1:
        return ""

    per_case = (summary or {}).get("cases") or {}
    rows = []
    for case in cases:
        overrides = ", ".join(f"{k}={v}" for k, v in case.scalars.items()) or "—"
        result = per_case.get(case.name)
        if result is None:
            time_s, speed = "not run", "—"
        else:
            best = result.get("best_time_us")
            speedup = result.get("speedup")
            time_s = f"{best:,.2f} µs" if best else "no valid config"
            speed = f"{speedup:.2f}x" if speedup else "—"
        rows.append(f"| `{case.name}` | {overrides} | {time_s} | {speed} |")

    geo = (summary or {}).get("geomean_speedup")
    geo_line = (
        f"\nScore: **{geo:.2f}x**, the geometric mean over all cases."
        if geo
        else "\nScore: not available — at least one case has no measured baseline, and "
        "the geometric mean is deliberately withheld rather than computed over a subset."
    )

    return (
        "## Input cases — your kernel is built and validated at EVERY one of these\n\n"
        "One `kernels.cu` and one set of driver regions serve all of them; each case gets "
        "its own tuned configuration. **A kernel that fails at any case fails the "
        "iteration.** Size every grid with a ceiling division and bound every load and "
        "store against the runtime scalars — do not assume a shape divides your tile "
        "size.\n\n"
        "| case | overrides | best time | speedup |\n"
        "|---|---|---|---|\n" + "\n".join(rows) + "\n" + geo_line + "\n"
    )
