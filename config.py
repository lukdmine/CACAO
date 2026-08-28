"""Run configuration: paths, workflow limits, and the agentic-step knobs.

The LLM stack lives in llm/. It is re-exported here because ~40 call sites import
it from config, and because config is what init_from_config seeds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from llm import (  # noqa: F401  (re-exported for existing call sites)
    LLM_PROVIDER,
    MODELS,
    MODEL_CONFIGS,
    PROVIDERS,
    TokenTracker,
    TrackedLLM,
    TrackedStructuredLLM,
    check_api_key,
    get_default_model,
    get_llm_creative,
    get_llm_precise,
    get_model,
    get_provider,
    get_tracker_stats,
    global_tracker,
    set_model,
)
from llm.providers import (  # noqa: F401
    _detect_provider,
    _normalise_provider,
    set_provider,
)

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
_base_output_dir = SCRIPT_DIR / "output"
_problem_dir = SCRIPT_DIR  # Defaults to script dir for backwards compatibility


def get_output_dir() -> Path:
    """Get the active output directory (defaults to ./output if no problem dir active)."""
    global _base_output_dir
    return _base_output_dir


def set_output_dir(path: Path):
    """Set the active output directory."""
    global _base_output_dir
    _base_output_dir = path


def get_problem_dir() -> Path:
    """Get the active problem directory (where problem.yaml and ref_kernel.cu live)."""
    global _problem_dir
    return _problem_dir


def set_problem_dir(path: Path):
    """Set the active problem directory."""
    global _problem_dir
    _problem_dir = path


# ===== Workflow Configuration =====
MAX_ITERATIONS = 5  # Max iterations per branch (depth mode)
MAX_BRANCH_DEPTH = 2  # Max depth of nested branches (depth mode)
PATH_BUDGET = 0  # Total path iteration budget (path mode, 0 = use depth mode)
MAX_STRATEGIES = 4  # Max strategies per strategize call
TUNER_TIMEOUT = (
    100  # System fallback when neither the user nor problem.yaml sets a budget
)
TUNER_TIMEOUT_OVERRIDE: Optional[int] = (
    None  # Explicit user override (CLI --timeout / RunConfig.timeout). None = not overridden.
)
HISTORY_ITERS = 2  # Number of past iterations to include in LLM context (None = all). best_so_far covers older working kernels.
INCLUDE_BEST_SO_FAR = (
    True  # Include best-performing iteration's kernel + config in LLM context
)

# ===== Agentic Steps =====
# When True, implement+configure run as one tool loop (nodes/author.py) that can
# compile-check and fix before the iteration is spent. False restores the two
# single-shot calls. This is the only way back to them: a step that cannot author its
# files fails the iteration and tells decide why, rather than silently re-running the
# work down a quieter path.
AGENTIC_STEPS = True
# Tool calls allowed in one authoring step — a runaway guard, not a working limit.
# Recorded steps run a median of 12 calls; at 25 three of 72 were cut off mid-fix and
# ten more came within five calls of it. A step stopped here still costs the iteration.
STEP_TOOL_BUDGET = 50
# How many times a reply cut off at the model's output token cap is corrected before
# the step gives up. Thinking models spend the whole cap reasoning about a "write the
# kernel" prompt and never reach the call; being told so is usually enough. The
# correction escalates, so raising this past the number of distinct corrections
# (agentic.loop._TRUNCATION_NUDGES) just repeats the last one.
STEP_TRUNCATION_RETRIES = 3
# How much of a sibling branch a step may read. Parallel branches are parallel *bets*;
# a branch that can read the current leader's kernel converges on it and the run buys
# one attempt instead of four. So no level exposes another branch's code.
#   "errors" — index + iteration log + recorded failure analyses (default)
#   "log"    — index + iteration log
#   "index"  — the one-line-per-branch header only
#   "off"    — nothing about other branches
CROSS_BRANCH_ACCESS = "errors"


@dataclass(frozen=True)
class OptimizerConfig:
    """Immutable per-run configuration. Built once at startup, replaces mutable globals."""

    output_dir: Path
    problem_dir: Path
    max_iterations: int = MAX_ITERATIONS
    max_branch_depth: int = MAX_BRANCH_DEPTH
    path_budget: int = PATH_BUDGET
    tuner_timeout: Optional[int] = (
        None  # None = let problem.yaml / system default apply
    )
    model: Optional[str] = None
    provider: Optional[str] = None


def init_from_config(cfg: OptimizerConfig):
    """Apply an OptimizerConfig to the module-level globals (called once per run)."""
    global \
        _base_output_dir, \
        _problem_dir, \
        MAX_ITERATIONS, \
        MAX_BRANCH_DEPTH, \
        PATH_BUDGET, \
        TUNER_TIMEOUT_OVERRIDE
    _base_output_dir = cfg.output_dir
    _problem_dir = cfg.problem_dir
    MAX_ITERATIONS = cfg.max_iterations
    MAX_BRANCH_DEPTH = cfg.max_branch_depth
    PATH_BUDGET = cfg.path_budget
    TUNER_TIMEOUT_OVERRIDE = cfg.tuner_timeout
    set_provider(cfg.provider)
    if cfg.model:
        set_model(cfg.model)
    else:
        set_model(get_default_model())
