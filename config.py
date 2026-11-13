"""Run configuration: paths, workflow limits, and the agentic-step knobs.

The LLM stack lives in llm/. It is re-exported here because ~40 call sites import
it from config, and because config is what init_from_config seeds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from llm import (  # noqa: F401  (re-exported for existing call sites)
    MODELS,
    MODEL_CONFIGS,
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
from llm.providers import set_provider

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
MAX_ITERATIONS = 5  # Max iterations per branch; depth mode only (PATH_BUDGET = 0)
MAX_BRANCH_DEPTH = 2  # Max depth of nested branches; bounds both modes
PATH_BUDGET = 20  # Total iterations per root-to-leaf path; 0 = fall back to depth mode
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
# One tool loop (nodes/author.py) for implement+configure; False is the only route back
# to the two single-shot calls.
AGENTIC_STEPS = True
# Tool calls allowed in one authoring step — a runaway guard, not a working limit.
STEP_TOOL_BUDGET = 50
# Replies cut off at the model's output token cap that are corrected before the step
# gives up. The correction escalates, so raising this past the number of distinct ones
# (agentic.loop._TRUNCATION_NUDGES) just repeats the last.
STEP_TRUNCATION_RETRIES = 3
# How much of a sibling branch a step may read. No level exposes another branch's code.
#   "errors" — index + iteration log + recorded failure analyses (default)
#   "log"    — index + iteration log
#   "index"  — the one-line-per-branch header only
#   "off"    — nothing about other branches
CROSS_BRANCH_ACCESS = "errors"

# ===== Precise re-measure =====
# After a successful tune the run node replays the best config this many times through
# the driver's remeasure mode and records the min as precise_best_time_us. The search
# itself stays single-shot. 0 turns it off; the CACAO_PRECISE_RUNS env var overrides.
PRECISE_RUNS = 25
# Wall-clock cap on one re-measure (seconds). Past it the single-shot numbers stand.
PRECISE_TIMEOUT_S = 300

# ===== Finalize =====
# The post-run pass (engine/finalize.py): re-tunes the best variants, widens toggles and
# checks other GPU architectures. Writes only under output/finalize/.
# On: the merge step reports its output in final_results.json. --no-finalize skips it.
FINALIZE_AUTO = True
# Tuner wall-clock budget per finalist per case (seconds).
FINALIZE_TUNER_TIMEOUT = 1000
# Architectures (sm_XX) the portability check compiles for.
FINALIZE_ARCHS = (75, 80, 86, 89, 90)
# Finalists taken from the branch ranking.
FINALIZE_MAX_FINALISTS = 4


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
    # None = leave the module defaults (FINALIZE_AUTO / FINALIZE_TUNER_TIMEOUT) alone.
    finalize: Optional[bool] = None
    finalize_timeout: Optional[int] = None


def init_from_config(cfg: OptimizerConfig):
    """Apply an OptimizerConfig to the module-level globals (called once per run)."""
    global \
        _base_output_dir, \
        _problem_dir, \
        MAX_ITERATIONS, \
        MAX_BRANCH_DEPTH, \
        PATH_BUDGET, \
        TUNER_TIMEOUT_OVERRIDE, \
        FINALIZE_AUTO, \
        FINALIZE_TUNER_TIMEOUT
    _base_output_dir = cfg.output_dir
    _problem_dir = cfg.problem_dir
    MAX_ITERATIONS = cfg.max_iterations
    MAX_BRANCH_DEPTH = cfg.max_branch_depth
    PATH_BUDGET = cfg.path_budget
    TUNER_TIMEOUT_OVERRIDE = cfg.tuner_timeout
    if cfg.finalize is not None:
        FINALIZE_AUTO = cfg.finalize
    if cfg.finalize_timeout is not None:
        FINALIZE_TUNER_TIMEOUT = cfg.finalize_timeout
    set_provider(cfg.provider)
    if cfg.model:
        set_model(cfg.model)
    else:
        set_model(get_default_model())
