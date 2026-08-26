"""Author node — writes the kernel and its tuning driver in one agentic step.

Replaces the implement -> configure pair. They were split, but the kernel and the
parameter space are one contract: a tile size in the kernel has to match a launcher's
thread count and a parameter declared in the params region, and two independent LLM
calls authoring two halves of that is a failure mode by construction. It is also the
only shape in which a compile check means anything, since the kernel cannot be
NVRTC-compiled without the ``#define``s the params region declares.

A step that cannot author its files fails the iteration and routes to decide, which is
what the branch already does with an implementation that comes back empty. It does not
quietly re-run the work through the two single-shot calls: a run once lost its tool
loop for its last 33 authoring steps that way, because the silent path made the
failure look like a success to everything downstream. ``AGENTIC_STEPS`` remains as a deliberate
switch back to those calls; nothing turns it off on your behalf.
"""

from pathlib import Path
from typing import Optional

import yaml

import config as _cfg
from config import get_llm_precise, get_problem_dir, get_output_dir
from utils.files import create_iter_dir, save_output
from utils.framework import assemble_framework_cpp
from utils.inputs import load_inputs_spec, scalar_contract_text
from utils.log import log
from utils.rules import parse_rules
from state.types import WorkingState
from agentic.loop import StepResult, run_agentic_step
from agentic.tools import Toolbox, schemas_for
from agentic.workspace import ALL_FILES, REGION_KEYS, Workspace
from nodes._llm_helper import build_prompt_context
import prompts.author


def _stage_engine_files(state: WorkingState, iter_dir: Path, meta: dict) -> str:
    """Place the files the engine owns beside the ones the LLM writes.

    Identical to configure_node's staging, and for the same reason: inputs.hpp is
    generated once per run from inputs.yaml, so the LLM neither owns it nor can author
    it, and its absence is not something an iteration can recover from.
    """
    import shutil

    problem_dir = get_problem_dir()
    inputs_src = problem_dir / "inputs.hpp"
    if not inputs_src.exists():
        raise FileNotFoundError(
            f"{inputs_src} not found — the problem's I/O boundary was never generated. "
            "It comes from inputs.yaml at the start of a run; check that inputs.yaml exists."
        )
    shutil.copyfile(inputs_src, iter_dir / "inputs.hpp")

    ref = meta.get("reference", {})
    if str(ref.get("type", "cuda")).lower() == "python":
        ref_py = problem_dir / ref.get("file", "ref.py")
        if not ref_py.exists():
            raise FileNotFoundError(
                f"{ref_py} not found — python reference file required by problem.yaml"
            )
        shutil.copyfile(ref_py, iter_dir / "ref.py")
        shutil.copyfile(problem_dir / "inputs.yaml", iter_dir / "inputs.yaml")

    return inputs_src.read_text(encoding="utf-8")


def _seed_from_previous(ws: Workspace, branch_path: Optional[Path], iteration: int) -> list:
    """Carry the previous iteration's files in so the step can edit rather than rewrite.

    Region files did not exist before this node, so an iteration following the old
    configure path is reconstructed from its framework.cpp markers where possible.
    """
    if not branch_path or iteration <= 1:
        return []
    prev = Path(branch_path) / f"iter{iteration - 1}"
    if not prev.is_dir():
        return []

    contents = {}
    kernel = prev / "kernels.cu"
    if kernel.exists():
        contents["kernels.cu"] = kernel.read_text(encoding="utf-8")

    for name in ALL_FILES:
        if name == "kernels.cu":
            continue
        staged = prev / name
        if staged.exists():
            contents[name] = staged.read_text(encoding="utf-8")

    if len(contents) <= 1:
        framework = prev / "framework.cpp"
        if framework.exists():
            from utils.framework import extract_regions

            try:
                regions = extract_regions(framework.read_text(encoding="utf-8"))
            except Exception:
                regions = {}
            for name, key in REGION_KEYS.items():
                if regions.get(key):
                    contents[name] = regions[key]

    return ws.seed(contents)


_LANGUAGE = {"kernels.cu": "cuda"}


def _format_workspace(ws: Workspace) -> str:
    """The files the step starts from, ready to edit, as a prompt section."""
    parts = []
    for name in ALL_FILES:
        if not ws.exists(name):
            continue
        body = ws.read(name)
        parts.append(f"### {name}\n```{_LANGUAGE.get(name, 'cpp')}\n{body}\n```")
    if not parts:
        return ""
    return (
        "## Current files (already in your workspace)\n\n"
        "Change them with edit_file or write_file, whichever fits: edit_file for a "
        "localised change to a large file, write_file when the change is structural "
        "or the file is small.\n\n" + "\n\n".join(parts)
    )


def _task_text(state: WorkingState, seeded: list) -> str:
    """The one-paragraph statement of what this iteration is for.

    Retry is the only mode that needs evidence in the prompt: the model is fixing a
    specific failure and cannot ask for something it does not know exists. Everything
    else is a tool call away.
    """
    scope = getattr(state, "authoring_scope", "full") or "full"
    lines = []

    # The decision, its feedback, the proposal and the run output all arrive in the
    # iteration-history section above this one; repeating them here duplicated ~16 KB
    # per call in the node this replaced.
    authoritative = (
        "The feedback and suggested fix above are the authoritative instruction for "
        "this iteration — they reflect the decision made after reviewing the results. "
        "Apply them directly rather than re-planning. Where they conflict with the "
        "plan or the strategy, follow the feedback."
    )

    if state.mode == "retry":
        lines.append(
            "The previous attempt failed. The failure and its analysis are above; the "
            "current files are shown below. Make the smallest correct fix.\n\n"
            + authoritative
        )
    elif state.mode == "followup":
        lines.append(
            "Apply the change requested in the most recent decision above. The current "
            "files are shown below — edit them rather than rewriting.\n\n"
            + authoritative
        )
    elif seeded:
        lines.append(
            "Continue from the previous iteration. Its files are shown below — edit "
            "them to apply this iteration's change."
        )
    else:
        lines.append("Write the first implementation of this strategy.")

    if scope == "config_only":
        lines.append(
            "Keep `kernels.cu` as it is; only the framework regions need to change."
        )
    return "\n".join(lines)


async def _legacy(state: WorkingState) -> WorkingState:
    """The two single-shot calls this node replaces, unchanged."""
    from nodes.implement import implement_node
    from nodes.configure import configure_node

    scope = getattr(state, "authoring_scope", "full") or "full"
    if scope != "config_only":
        state = await implement_node(state)
        if state.status == "deciding":  # implement failed and routed itself
            return state
    return await configure_node(state)


def _fail_iteration(state: WorkingState, detail: str, result: StepResult) -> WorkingState:
    """End the iteration on a step that produced nothing, and say why.

    Routed to decide rather than propose: propose reads a kernel and its results to
    suggest the next change, and neither exists here. This is the same route
    implement_node takes when its single call comes back empty, and decide handles it
    the same way — it reads the failure as a generation failure rather than a strategy
    one and retries with a smaller ask.
    """
    log(f"Authoring step failed: {detail}", "ERROR")
    lines = [
        f"[AUTHORING FAILED] The authoring step produced no usable files: {detail}.",
        "",
        f"The step ran {result.turns} turn(s) and {result.tool_calls} tool call(s).",
        "No kernel was compiled and no tuner run happened, so there are no results for "
        "this iteration. This is a failure to generate code, not a failure of the "
        "strategy — the previous iteration's kernel is still the branch's best.",
    ]
    if result.truncations:
        lines += [
            "",
            f"The model's reply was cut off at its output token limit {result.truncations} "
            "time(s) before it reached a tool call. It is reasoning until it runs out of "
            "room to answer. The next attempt has to ask for less in one go: a single "
            "targeted edit rather than a full rewrite.",
        ]
    if result.text:
        lines += ["", "Last thing the model said before it stopped:", result.text.strip()[:1000]]
    state.run_output = "\n".join(lines)
    state.status = "deciding"
    return state


async def author_node(state: WorkingState) -> WorkingState:
    iteration = state.iter_num
    strategy = state.strategy
    branch_name = strategy.name if strategy else "default"
    branch_path = Path(state.branch_path) if state.branch_path else None

    print("\n" + "=" * 60)
    print(f"  NODE: Author [{branch_name}] (iter {iteration}/{state.max_iter})")
    print("=" * 60)

    iter_dir = (
        create_iter_dir(branch_path, iteration)
        if branch_path
        else get_output_dir() / f"iter{iteration}"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)

    meta = yaml.safe_load(state.problem_yaml) if state.problem_yaml else {}
    inputs_hpp = _stage_engine_files(state, iter_dir, meta)

    if not _cfg.AGENTIC_STEPS:
        log("AGENTIC_STEPS disabled — using single-shot implement + configure", "INFO")
        return await _legacy(state)

    # --- workspace ---
    ws = Workspace(iter_dir)
    ws.reset()
    seeded = _seed_from_previous(ws, branch_path, iteration)
    if seeded:
        log(f"Seeded from iter{iteration - 1}: {', '.join(seeded)}", "INFO")

    # --- prompt ---
    # propose and decide exist to tell this step what to change; an empty history
    # threw their output away and every retry repeated the mistake that caused it.
    # One iteration deep and no kernel bodies: the previous files are already loaded
    # into the workspace, and older iterations are a read_file / list_iterations away.
    ctx = build_prompt_context(
        state,
        iteration_history_fields=[
            {"name": "decision", "limit": 1},
            {"name": "feedback", "limit": 1},
            {"name": "proposal", "limit": 1},
            {"name": "run_output", "limit": 1},
            {"name": "results_summary", "limit": 1},
        ],
    )
    ctx["inputs_hpp"] = inputs_hpp
    ctx["rules_block"] = parse_rules(meta).prompt_block()
    ctx["task"] = _task_text(state, seeded)
    # Inlined rather than read back: the model read all four on turn one every time,
    # so the tokens are paid regardless and the read-back only costs a round trip.
    ctx["current_files"] = _format_workspace(ws) if seeded else ""
    # The reference is orientation for a first implementation; afterwards the branch's
    # own kernel is what is being changed and the reference is a read_file away.
    if iteration > 1:
        ctx["ref_kernel"] = ""

    inputs_yaml = get_problem_dir() / "inputs.yaml"
    if inputs_yaml.exists():
        try:
            ctx["scalar_contract"] = scalar_contract_text(load_inputs_spec(inputs_yaml))
        except Exception as e:
            log(f"Could not derive the scalar contract from inputs.yaml: {e}", "WARN")

    # Bind only tools that can return something. From run state, not iteration number:
    # a sibling can be ahead of a branch restarting after a revert.
    access = getattr(_cfg, "CROSS_BRANCH_ACCESS", "errors")
    has_siblings = False
    if access != "off" and branch_path:
        from state.crossbranch import branch_index

        index = branch_index(get_output_dir(), exclude_path=branch_path)
        # An index listing only branches with no result yet is nothing to investigate.
        has_siblings = bool(index) and "best " in index
        if has_siblings:
            ctx["branch_index"] = index

    has_history = bool(
        branch_path and (Path(branch_path) / f"iter{iteration - 1}" / "state.json").exists()
    )

    system, user = prompts.author.build(ctx)
    save_output(iter_dir, f"# System Prompt\n\n{system}\n\n# User Message\n\n{user}", "prompt_author.md")
    log(f"Prompt: {len(system) + len(user):,} bytes")

    # --- run the step ---
    toolbox = Toolbox(
        ws,
        branch_path=branch_path or iter_dir.parent,
        output_dir=get_output_dir(),
        problem_dir=get_problem_dir(),
        meta=meta,
        gpu_info=getattr(state, "gpu_info", None),
    )
    schemas = schemas_for(access, has_history=has_history, has_siblings=has_siblings)
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
        f"Step {result.outcome.value}: {result.tool_calls} tool calls over "
        f"{result.turns} turns ({dict(result.tools_used)}); "
        f"{len(schemas)} tools bound (history={has_history}, siblings={has_siblings})"
    )

    if result.outcome.aborted:
        detail = result.outcome.diagnosis
        if toolbox.internal_errors:
            # A tool of ours raised and left the model with nothing to converge on.
            # Worth saying out loud: the fix for that is here, not in the branch.
            detail += (
                f" (after {toolbox.internal_errors} internal tool error(s) — "
                "an engine bug, not the model's)"
            )
        if result.error:
            detail += f": {result.error}"
        return _fail_iteration(state, detail, result)

    if not ws.complete():
        return _fail_iteration(
            state,
            f"{result.outcome.diagnosis} — missing {', '.join(ws.missing())}",
            result,
        )

    # --- commit ---
    written = ws.commit()
    framework_cpp = assemble_framework_cpp(meta, {
        key: (iter_dir / name).read_text(encoding="utf-8")
        for name, key in REGION_KEYS.items()
        if (iter_dir / name).exists()
    })
    save_output(iter_dir, framework_cpp, "framework.cpp")

    state.kernel_code = (iter_dir / "kernels.cu").read_text(encoding="utf-8")
    state.framework_cpp = framework_cpp
    log(f"Committed: {', '.join(written)} + framework.cpp", "SUCCESS")
    if result.summary:
        log(f"Step summary: {result.summary}")

    if toolbox.check_passed:
        state.status = "running"
        return state

    # Budget ran out on code that does not compile. Routing to propose mirrors what
    # run.py does with a host-compile failure, and skips a tuner run whose outcome is
    # already known.
    log("Step ended without a passing compilation check — routing to propose", "WARN")
    state.run_output = (
        "[COMPILE ERROR] The authoring step ended without a passing compilation "
        "check (tool budget exhausted).\n\n" + (toolbox.last_check or "No check was run.")
    )
    state.status = "proposing"
    return state
