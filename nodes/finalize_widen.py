"""Stage B of finalize: re-widen one finalist's tuning space with the authoring tool loop.

Built from the authoring parts rather than from author_node, which is shaped around a
branch iteration (a WorkingState, iteration N-1 of its own branch, a route to decide).
The workspace lives in the finalist's ``output/finalize/<name>/iter<K>`` and is seeded
from the source branch's best iteration. The Toolbox's ``branch_path`` names the source
branch, so tuning_landscape, list_iterations and parameter_history read the source's
history while every write lands in the finalist's staging. Nothing here writes under
``output/branches/``.

A step that fails leaves the source result standing; the caller records the diagnosis.
"""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config as _cfg
from config import get_llm_precise
from agentic.loop import StepResult, run_agentic_step
from agentic.tools import FinalizeToolContext, Toolbox, schemas_for
from agentic.workspace import KERNEL_FILE, REGION_KEYS, Workspace, format_workspace, owned_files_from_dir
from utils.cases import stage_engine_files
from utils.files import save_output
from utils.framework import assemble_framework_cpp
from utils.inputs import load_inputs_spec, scalar_contract_text
from utils.log import log
from utils.rules import parse_rules
import prompts.finalize


# The D3 rule, where the model reads the files it applies to. author's intro offers
# write_file for a structural change, which is the one kind of change widening forbids.
_EDIT_INTRO = (
    "Change `kernels.cu` with `edit_file` only, and only in the two ways the limits "
    "allow. The region files may be edited or rewritten, whichever fits."
)


@dataclass
class WidenResult:
    ok: bool
    diagnosis: str = ""
    diff_lines: Optional[int] = None  # +/- lines of kernels.cu against the source (D3)
    summary: str = ""


def _kernel_diff_lines(old: str, new: str) -> int:
    """Lines added plus lines removed between two versions of a kernel.

    The ``---``/``+++`` file headers are skipped by position, not by prefix: a removed
    line that itself starts with ``--`` reads ``---`` in the diff and is a real change.
    """
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""))
    return sum(1 for line in diff[2:] if line.startswith(("+", "-")))


def _assemble_framework(iter_dir: Path, meta: dict) -> None:
    """framework.cpp from the committed regions, as run_node expects to find it."""
    framework_cpp = assemble_framework_cpp(meta, {
        key: (iter_dir / name).read_text(encoding="utf-8")
        for name, key in REGION_KEYS.items()
        if (iter_dir / name).exists()
    })
    save_output(iter_dir, framework_cpp, "framework.cpp")


def _iteration_number(source_iter_dir: Path) -> Optional[int]:
    m = re.fullmatch(r"iter(\d+)", Path(source_iter_dir).name)
    return int(m.group(1)) if m else None


def seed_unchanged(*, iter_dir: Path, source_iter_dir: Path, problem_dir: Path, meta: dict) -> None:
    """Lay the source's files out in the finalist directory with no LLM call.

    The route when AGENTIC_STEPS is False: stage B has no tool loop to run, so the
    source space is retuned as it is rather than leaving the run without a finalize.
    """
    iter_dir = Path(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)
    stage_engine_files(problem_dir, iter_dir, meta)

    # Through the workspace so the same filters apply as on the widened route.
    ws = Workspace(iter_dir)
    ws.reset()
    ws.seed(owned_files_from_dir(source_iter_dir))
    ws.commit()
    _assemble_framework(iter_dir, meta)


def _failure(result: StepResult, ws: Workspace, toolbox: Toolbox) -> str:
    """Why the step produced nothing to tune, in the words author_node would use.

    Only reached when a salvage condition fails, and the parenthetical names the first
    that does. The outcome's own wording describes a step that wrote nothing, which a
    step that edited and then stalled is not, so the parenthetical says what state the
    files were left in.
    """
    if result.outcome.aborted or not ws.complete():
        detail = result.outcome.diagnosis
    else:
        # BUDGET_EXHAUSTED's own wording assumes the files were incomplete.
        detail = "the step ran out of tool calls"
    if toolbox.mutations == 0:
        detail += " (no file was written)"
    elif not ws.complete():
        detail += f" (missing {', '.join(ws.missing())})"
    elif not toolbox.check_passed:
        detail += " (the last compilation check did not pass)"
    if toolbox.internal_errors:
        detail += (
            f" (after {toolbox.internal_errors} internal tool error(s), "
            "an engine bug, not the model's)"
        )
    if result.error:
        detail += f": {result.error}"
    return detail


async def widen_finalist(
    *,
    iter_dir: Path,
    source_branch: Path,
    source_iter_dir: Path,
    problem_dir: Path,
    output_dir: Path,
    meta: dict,
    problem_yaml: str,
    gpu_info: dict,
    budget_s: float,
    gpu_index: int,
    ref_file: Path,
) -> WidenResult:
    iter_dir = Path(iter_dir)
    iter_dir.mkdir(parents=True, exist_ok=True)
    inputs_hpp = stage_engine_files(problem_dir, iter_dir, meta)

    # --- workspace, seeded with the source's best iteration ---
    source_files = owned_files_from_dir(source_iter_dir)
    ws = Workspace(iter_dir)
    ws.reset()
    seeded = ws.seed(source_files)
    log(f"Widening from {source_iter_dir}: seeded {', '.join(seeded) or 'nothing'}", "INFO")
    # check_compilation reports any parameter narrowed against this, which is the one
    # direction widening must not go.
    previous_params_src = ws.read("region_params.cpp") if "region_params.cpp" in seeded else None

    # --- prompt ---
    ctx = {
        "current_files": format_workspace(ws, _EDIT_INTRO) if seeded else "",
        "budget_s": budget_s,
        "inputs_hpp": inputs_hpp,
        "problem_yaml": problem_yaml,
        "rules_block": parse_rules(meta).prompt_block(),
        "source_iter": _iteration_number(source_iter_dir),
    }
    # The inputs.hpp block tells the model the scalar contract follows; as author_node.
    inputs_yaml = Path(problem_dir) / "inputs.yaml"
    if inputs_yaml.exists():
        try:
            ctx["scalar_contract"] = scalar_contract_text(load_inputs_spec(inputs_yaml))
        except Exception as e:
            log(f"Could not derive the scalar contract from inputs.yaml: {e}", "WARN")
    system, user = prompts.finalize.build_widen(ctx)
    save_output(iter_dir, f"# System Prompt\n\n{system}\n\n# User Message\n\n{user}", "prompt_finalize.md")
    log(f"Prompt: {len(system) + len(user):,} bytes")

    # --- run the step ---
    toolbox = Toolbox(
        ws,
        branch_path=Path(source_branch),
        output_dir=Path(output_dir),
        problem_dir=Path(problem_dir),
        meta=meta,
        gpu_info=gpu_info,
        previous_params_src=previous_params_src,
        finalize=FinalizeToolContext(
            source_iter_dir=Path(source_iter_dir),
            budget_s=budget_s,
            gpu_index=gpu_index,
            ref_file=Path(ref_file),
        ),
    )
    # A finalist's source is a measured iteration by construction, so the history
    # tools always have something to read.
    schemas = schemas_for(has_history=True, finalize=True)
    result = await run_agentic_step(
        get_llm_precise(),
        system,
        user,
        toolbox,
        budget=getattr(_cfg, "STEP_TOOL_BUDGET", 50),
        trace_path=iter_dir / "step_trace.jsonl",
        schemas=schemas,
        truncation_retries=getattr(_cfg, "STEP_TRUNCATION_RETRIES", 3),
    )
    log(
        f"Widen step {result.outcome.value}: {result.tool_calls} tool calls over "
        f"{result.turns} turns ({dict(result.tools_used)})"
    )

    # author_node's rule, less its change requirement: end_step in finalize mode
    # enforces only these two (a space already as wide as plausible has nothing to
    # widen), and a step that stalled or ran out of calls after meeting them still did
    # the work. With nothing changed, the commit below is the seeded source files.
    salvageable = ws.complete() and toolbox.check_passed
    if not (result.ok or salvageable):
        detail = _failure(result, ws, toolbox)
        log(f"Widening failed: {detail}", "ERROR")
        return WidenResult(ok=False, diagnosis=detail, summary=result.summary)
    if not result.ok:
        log(
            f"Widen step ended {result.outcome.value} with all four files present and a "
            "passing check; committing rather than discarding the work.",
            "WARN",
        )

    # --- commit ---
    written = ws.commit()
    _assemble_framework(iter_dir, meta)
    diff_lines = _kernel_diff_lines(
        source_files.get(KERNEL_FILE, ""),
        (iter_dir / KERNEL_FILE).read_text(encoding="utf-8"),
    )
    log(
        f"Committed: {', '.join(written)} + framework.cpp "
        f"(kernels.cu diff: {diff_lines} line(s))",
        "SUCCESS",
    )
    if result.summary:
        log(f"Step summary: {result.summary}")
    return WidenResult(ok=True, diff_lines=diff_lines, summary=result.summary)
