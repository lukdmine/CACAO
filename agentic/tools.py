"""Tool schemas and dispatch for the authoring step.

Schema classes are named in snake_case because langchain uses the class name as the
tool name and the docstring as its description; the names below are what the model
sees. Descriptions are deliberately terse — the step prompt states the contract once,
and repeating it per tool inflates every request for no gain.

Preconditions raise ToolError, which the loop turns into an ordinary tool result. The
model reads the message and corrects on its next turn; it never sees a tool vanish.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic.workspace import ALL_FILES, ToolError, Workspace
from utils.log import log

_GREP_MAX_HITS = 40
_READ_MAX_BYTES = 64_000


# -------------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------------


class write_file(BaseModel):
    """Create or fully replace one of the four files this step owns."""

    name: str = Field(description=f"One of: {', '.join(ALL_FILES)}")
    content: str = Field(description="Full file contents.")


class edit_file(BaseModel):
    """Replace an exact, unique snippet in a file that already exists."""

    name: str = Field(description=f"One of: {', '.join(ALL_FILES)}")
    old_text: str = Field(description="Exact text to replace; must occur exactly once.")
    new_text: str = Field(description="Replacement text.")


class read_file(BaseModel):
    """Read a file from this step, or from an earlier iteration of this branch."""

    name: str = Field(description="File name, e.g. kernels.cu.")
    iteration: Optional[int] = Field(
        default=None, description="Earlier iteration number. Omit for the current step."
    )


class check_compilation(BaseModel):
    """Compile the driver (g++) and the kernel (NVRTC), as the tuner will."""


class end_step(BaseModel):
    """Finish the step. Requires a passing check_compilation."""

    summary: str = Field(description="One sentence: what changed and why.")


class tuning_landscape(BaseModel):
    """Query all measured configurations of an earlier iteration."""

    iteration: int = Field(description="Iteration number in this branch.")
    mode: str = Field(description="'top' (fastest configs), 'sweep' (one parameter), 'failures'.")
    param: Optional[str] = Field(default=None, description="Parameter name, for mode='sweep'.")
    n: int = Field(default=10, description="Row count for mode='top'.")
    case: Optional[str] = Field(
        default=None,
        description=(
            "Which input case's results to read. Omit for the primary case. Only "
            "meaningful when the problem declares several input cases."
        ),
    )


class grep_tuner_output(BaseModel):
    """Search an earlier iteration's raw tuner log."""

    iteration: int = Field(description="Iteration number in this branch.")
    pattern: str = Field(description="Regular expression.")


class list_iterations(BaseModel):
    """This branch's iterations: which ran, which failed, how fast."""


class branch_log(BaseModel):
    """What another branch tried, one line per iteration."""

    branch: str = Field(description="Branch name from the index in the prompt.")


class branch_errors(BaseModel):
    """Failures another branch hit and what it concluded caused them."""

    branch: str = Field(description="Branch name from the index in the prompt.")


# Tools available at each CROSS_BRANCH_ACCESS level. No level exposes another branch's
# kernel or framework regions; see state.crossbranch.branch_errors for why.
_CROSS_BRANCH_TOOLS = {
    "off": [],
    "index": [],
    "log": [branch_log],
    "errors": [branch_log, branch_errors],
}

# Always available: they act on the workspace, which exists from the first turn.
_CORE_SCHEMAS = [
    write_file,
    edit_file,
    read_file,
    check_compilation,
    end_step,
]

# Only meaningful once this branch has a completed iteration to read.
_HISTORY_SCHEMAS = [
    tuning_landscape,
    grep_tuner_output,
    list_iterations,
]


def schemas_for(
    cross_branch_access: str = "errors",
    *,
    has_history: bool = True,
    has_siblings: bool = True,
) -> List[type]:
    """Tool schemas to bind for one step.

    A tool that cannot succeed is not free: the model calls it, spends a round trip,
    and gets an error that was knowable before the request was sent. Observed on a live
    run's first iteration — three `grep_tuner_output` calls each answered
    "iter1 has no tuner_output.txt", plus two `branch_log` calls against siblings that
    were themselves on iteration 1. Two branches spent 22 and 23 of a 25-call budget
    that way.

    So history tools bind only once this branch has a completed iteration, and
    cross-branch tools only once some other branch has a result worth reading. Both are
    computed from the run's actual state rather than assumed from the iteration number:
    branches run concurrently, and a revert can leave a sibling ahead of a branch that
    is starting over.
    """
    schemas = list(_CORE_SCHEMAS)
    if has_history:
        schemas += _HISTORY_SCHEMAS
    if has_siblings:
        schemas += _CROSS_BRANCH_TOOLS.get(cross_branch_access, [])
    return schemas


# Default binding, used by the replay harness and tests.
SCHEMAS = schemas_for("errors")


# -------------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------------


class Toolbox:
    """Executes tool calls against one iteration's context.

    Holds the mutable step state the loop needs to decide when a step may end:
    whether a compilation check has passed since the last edit, and whether end_step
    was reached.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        branch_path: Path,
        output_dir: Path,
        problem_dir: Path,
        meta: dict,
        gpu_info: Optional[dict] = None,
        rules=None,
    ):
        self.ws = workspace
        self.branch_path = Path(branch_path)
        self.output_dir = Path(output_dir)
        self.problem_dir = Path(problem_dir)
        self.meta = meta or {}
        self.gpu_info = gpu_info or {}

        if rules is None:
            from utils.rules import parse_rules

            rules = parse_rules(self.meta)
        self.rules = rules

        self.check_passed = False
        self.checks_run = 0
        self.mutations = 0
        self.ended = False
        self.end_summary = ""
        self.last_check = ""
        # Exceptions raised inside a tool — our bugs, not the model's failure to drive
        # the loop. Kept apart so a defect on our side cannot be misread as a provider
        # that lacks tool support.
        self.internal_errors = 0

    # -- helpers -----------------------------------------------------------

    def _iter_dir(self, iteration: int) -> Path:
        return self.branch_path / f"iter{iteration}"

    def _handlers(self) -> Dict[str, Callable[..., str]]:
        return {
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "read_file": self.read_file,
            "check_compilation": self.check_compilation,
            "end_step": self.end_step,
            "tuning_landscape": self.tuning_landscape,
            "grep_tuner_output": self.grep_tuner_output,
            "list_iterations": self.list_iterations,
            "branch_log": self.branch_log,
            "branch_errors": self.branch_errors,
        }

    def dispatch(self, name: str, args: dict) -> str:
        """Run one tool call. Never raises — errors are results the model acts on."""
        handler = self._handlers().get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self._handlers())}."
        try:
            return handler(**(args or {}))
        except ToolError as e:
            return f"ERROR: {e}"
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except Exception as e:  # a tool bug must not kill the branch
            # Distinct from ToolError: this is our defect, not the model's. It gets
            # counted so the caller does not read the resulting dead-end as evidence
            # that the provider cannot drive tools — a broken check_compilation
            # otherwise blames the model for a step it drove correctly.
            self.internal_errors += 1
            log(f"Tool '{name}' raised: {e}", "WARN")
            return f"ERROR: {name} failed: {e}"

    # -- file tools --------------------------------------------------------

    def write_file(self, name: str, content: str) -> str:
        result = self.ws.write(name, content)
        # Any write invalidates the previous verdict; otherwise a model could pass a
        # check, rewrite the kernel, and end the step on a stale result.
        self.check_passed = False
        self.mutations += 1
        return result

    def edit_file(self, name: str, old_text: str, new_text: str) -> str:
        result = self.ws.edit(name, old_text, new_text)
        self.check_passed = False
        self.mutations += 1
        return result

    def read_file(self, name: str, iteration: Optional[int] = None) -> str:
        if iteration is None:
            return self.ws.read(name)
        if "/" in name or "\\" in name or ".." in name:
            raise ToolError("name must be a bare file name.")
        target = self._iter_dir(iteration) / name
        if not target.exists():
            raise ToolError(f"iter{iteration}/{name} does not exist in this branch.")
        data = target.read_text(encoding="utf-8", errors="replace")
        if len(data) > _READ_MAX_BYTES:
            return data[:_READ_MAX_BYTES] + f"\n\n[truncated at {_READ_MAX_BYTES:,} bytes]"
        return data

    # -- compilation -------------------------------------------------------

    def _case_headers(self) -> list:
        """(case_name, inputs.hpp text) for every case, primary first.

        NVRTC never sees inputs.hpp — it sees the -D macros parsed out of it — so where a
        `define`-placement scalar varies per case, each case is a genuinely different
        compile. Where the varying scalars are `runtime`-placed, every case yields the
        same defines and the extra checks collapse to one on their own, with no special
        case needed here.

        Falls back to the problem's generated inputs.hpp if the spec cannot be resolved:
        a compile check that refuses to run is worse than one that checks the primary.
        """
        from utils.cases import case_list
        from utils.inputs import generate_inputs_hpp, load_inputs_spec

        try:
            cases = case_list(self.meta)
            spec = load_inputs_spec(self.problem_dir / "inputs.yaml")
            reference = self.meta.get("reference")
            return [
                (c.name, generate_inputs_hpp(spec.for_case(c), reference, self.problem_dir))
                for c in cases
            ]
        except Exception:
            src = self.problem_dir / "inputs.hpp"
            text = src.read_text(encoding="utf-8") if src.exists() else ""
            return [("default", text)]

    def check_compilation(self) -> str:
        """Host-compile the assembled driver and NVRTC-compile the kernel."""
        from utils.build import compile_framework, reference_build_extras
        from utils.framework import assemble_framework_cpp, resolve_cuda_include
        from utils import nvrtc

        missing = self.ws.missing()
        if missing:
            raise ToolError(
                f"Cannot compile yet — not written: {', '.join(missing)}."
            )

        self.checks_run += 1
        parts: List[str] = []

        # --- problem rules, before anything expensive ---
        # A rule that is only stated in the prompt is a rule that loses to a speedup.
        # Checking it here is what makes "do not drop to fp16 accumulation" a
        # constraint rather than a suggestion, and failing before compiling keeps the
        # reason for the failure unambiguous.
        violations = self.rules.violations(self.ws.read("kernels.cu"))
        if violations:
            self.check_passed = False
            return (
                "Compilation check: FAIL\n\n"
                "Problem rules: VIOLATED\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\n\nThese rules are non-negotiable for this problem. Rewrite the "
                "kernel without the forbidden construct; a faster kernel that breaks "
                "them is not a valid result."
            )
        if self.rules.forbid:
            parts.append("Problem rules: PASS")

        # --- host compile of the assembled driver ---
        framework_cpp = assemble_framework_cpp(self.meta, self.ws.regions())
        (self.ws.staging / "framework.cpp").write_text(framework_cpp, encoding="utf-8")

        # The primary case only: this runs on every check and the inner loop has to stay
        # fast. end_step is where every case is gated.
        headers = self._case_headers()
        primary_header = headers[0][1]
        if primary_header:
            (self.ws.staging / "inputs.hpp").write_text(primary_header, encoding="utf-8")

        from utils.cases import case_list

        try:
            primary_case = case_list(self.meta)[0]
        except Exception:
            primary_case = None
        extra_sources, extra_flags = reference_build_extras(self.problem_dir, primary_case)
        build = compile_framework(
            self.ws.staging, extra_sources=extra_sources, extra_flags=extra_flags
        )
        host_ok = build.ok
        if host_ok:
            parts.append("Driver (g++): PASS")
        else:
            parts.append("Driver (g++): FAIL\n" + (build.stderr or "").strip()[:6000])

        # --- NVRTC compile of the kernel, exactly as KTT will ---
        params_src = self.ws.read("region_params.cpp")
        scalar_defines = (
            nvrtc.parse_defines_from_inputs_hpp(primary_header) if primary_header else []
        )
        checks = nvrtc.check_kernel(
            self.ws.read("kernels.cu"),
            params_src,
            cuda_include=resolve_cuda_include(),
            scalar_defines=scalar_defines,
            compute_capability=self.gpu_info.get("compute_capability"),
        )
        kernel_ok = all(c.ok for c in checks)
        parts.append(nvrtc.format_checks(checks, constrained=nvrtc.has_constraints(params_src)))

        self.check_passed = host_ok and kernel_ok
        verdict = "PASS" if self.check_passed else "FAIL"
        self.last_check = f"Compilation check: {verdict}\n\n" + "\n\n".join(parts)

        if not self.check_passed:
            # Said at the moment the choice is made; a prompt line did not stop the
            # model rewriting whole files after a failed check.
            self.last_check += (
                "\n\nIf these are localised errors, edit_file is usually the smaller "
                "change; use write_file when the fix is structural."
            )
        return self.last_check

    def end_step(self, summary: str = "") -> str:
        if not self.ws.complete():
            raise ToolError(
                f"Cannot end the step — not written: {', '.join(self.ws.missing())}."
            )
        if self.mutations == 0:
            # The workspace is seeded with the previous iteration's files so the step
            # can edit rather than rewrite. Ending without touching them would spend an
            # iteration re-running work that has already been measured.
            raise ToolError(
                "Cannot end the step — nothing was changed. This iteration exists to "
                "make a change; edit or write at least one file."
            )
        if not self.check_passed:
            raise ToolError(
                "Cannot end the step — run check_compilation and get a PASS first."
                if self.checks_run == 0
                else "Cannot end the step — the last check_compilation did not pass, "
                "or a file changed after it. Run it again."
            )
        # The complete gate. check_compilation covers the primary case on every call so
        # the inner loop stays fast; this is the one place a kernel that compiles only at
        # the primary shape must not get through. headers[1:] because the primary was
        # already checked by the check_compilation this gate requires.
        from utils import nvrtc
        from utils.framework import resolve_cuda_include

        headers = self._case_headers()
        for case_name, header in headers[1:]:
            checks = nvrtc.check_kernel(
                self.ws.read("kernels.cu"),
                self.ws.read("region_params.cpp"),
                cuda_include=resolve_cuda_include(),
                scalar_defines=nvrtc.parse_defines_from_inputs_hpp(header),
                compute_capability=self.gpu_info.get("compute_capability"),
            )
            bad = [c for c in checks if not c.ok]
            if bad:
                raise ToolError(
                    f"The kernel compiles at case '{headers[0][0]}' but not at case "
                    f"'{case_name}':\n"
                    + "\n".join(f"- {c.label}: {c.log.strip()}" for c in bad)
                    + "\n\nEvery case is built and validated, and a kernel that fails "
                    "any of them fails the iteration. Fix it so it compiles at all of "
                    "them, then run check_compilation again."
                )

        self.ended = True
        self.end_summary = summary or ""
        return "Step complete."

    # -- read/query tools --------------------------------------------------

    def tuning_landscape(
        self,
        iteration: int,
        mode: str = "top",
        param: Optional[str] = None,
        n: int = 10,
        case: Optional[str] = None,
    ) -> str:
        from utils.cases import case_dir, case_list
        from utils.landscape import describe

        cases = case_list(self.meta)
        target = (
            cases[0] if case is None else next((c for c in cases if c.name == case), None)
        )
        if target is None:
            raise ToolError(
                f"No case named '{case}'. This problem's cases are: "
                + ", ".join(c.name for c in cases)
            )
        results = case_dir(self._iter_dir(iteration), cases, target) / "results.json"
        if not results.exists():
            where = "" if len(cases) == 1 else f" case '{target.name}'"
            raise ToolError(
                f"iter{iteration}{where} has no results.json (it may not have run)."
            )
        return describe(results, mode=mode, param=param, n=n)

    def grep_tuner_output(self, iteration: int, pattern: str) -> str:
        path = self._iter_dir(iteration) / "tuner_output.txt"
        if not path.exists():
            raise ToolError(f"iter{iteration} has no tuner_output.txt.")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}")

        hits = []
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{i}: {line.rstrip()}")
                if len(hits) >= _GREP_MAX_HITS:
                    break
        if not hits:
            return f"No match for /{pattern}/ in iter{iteration}/tuner_output.txt."
        header = f"{len(hits)} match(es)" + (
            f", showing the first {_GREP_MAX_HITS}" if len(hits) >= _GREP_MAX_HITS else ""
        )
        return header + ":\n" + "\n".join(hits)

    def list_iterations(self) -> str:
        from state.crossbranch import list_iterations as _list

        return _list(self.branch_path)

    def branch_log(self, branch: str) -> str:
        from state.crossbranch import branch_log as _log

        return _log(self.output_dir, branch)

    def branch_errors(self, branch: str) -> str:
        from state.crossbranch import branch_errors as _errors

        return _errors(self.output_dir, branch)
