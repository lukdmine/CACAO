#!/usr/bin/env python3
"""
CUDA Agentic Kernel Optimizer — CLI entry point.

Run ``python cli.py --help`` for the full list of options.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import config
from config import (
    SCRIPT_DIR,
    get_default_model,
    get_provider,
    MAX_ITERATIONS,
    MAX_BRANCH_DEPTH,
    PATH_BUDGET,
    TUNER_TIMEOUT,
    check_api_key,
    OptimizerConfig,
    init_from_config,
)
from utils.files import (
    load_file,
    archive_output_dir,
    clone_problem_dir,
    prune_archives,
)
from utils.log import log, TeeWriter
from utils.problem import reference_spec


def log_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60, flush=True)
    print(f"  {title}")
    print("=" * 60, flush=True)


def _report_final(merge_result: dict) -> int:
    """Print the verdict and return the process exit code.

    The verdict is ``merge_node``'s. cli.py used to re-derive it and demanded status
    exactly "success", so a run whose fastest branch was a parent that spawned children
    ("branching") printed OPTIMIZATION FAILED and exited 1 — while the engine had
    already written a final_results.json naming that branch the winner.
    """
    log_section("FINAL RESULT")
    if merge_result.get("status") == "success":
        log("OPTIMIZATION COMPLETE", "SUCCESS")
        log(f"Results saved to: {config.get_output_dir()}")
        return 0

    log("OPTIMIZATION FAILED", "ERROR")
    log(f"Check {config.get_output_dir()} for details")
    return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CUDA Agentic Kernel Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing problem.yaml and reference source files (default: current directory)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Maximum iterations per branch (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_BRANCH_DEPTH,
        help=f"Maximum branch nesting depth (default: {MAX_BRANCH_DEPTH})",
    )
    parser.add_argument(
        "--path-budget",
        type=int,
        default=PATH_BUDGET,
        help=f"Total iteration budget per root-to-leaf path (0 = disabled; when >0, still bounded by --max-depth, default: {PATH_BUDGET})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            f"Override the tuner wall-clock budget (seconds). If unset, falls back to "
            f"problem.yaml tuning.duration_s, then system default {TUNER_TIMEOUT}s."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "gemini", "cerit"],
        default=None,
        help="LLM provider (default: auto-detect from env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: provider's default model)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean output directory before running",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted run (skips analysis/strategy, resumes incomplete branches)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Only display the best results from the current output directory and exit (skips optimization)",
    )
    parser.add_argument(
        "--clone",
        type=str,
        metavar="NEW_NAME",
        help="Clone this problem to a new directory with the given name and exit",
    )
    parser.add_argument(
        "--prune-archives",
        action="store_true",
        help="Delete regenerable files (tuner input dumps, compiled drivers) from "
             "archive/output_* and exit. Kernels, results and decisions are kept.",
    )
    return parser.parse_args()


def check_prerequisites(problem_dir: Path):
    """Check that all required files and dependencies are available."""
    log("Checking prerequisites...")

    # Check problem.yaml
    problem_path = problem_dir / "problem.yaml"
    if not problem_path.exists():
        log(f"problem.yaml not found at {problem_path}", "ERROR")
        return False
    log(f"problem.yaml found in {problem_dir}", "SUCCESS")

    # Check configured reference source file
    try:
        problem_config = yaml.safe_load(problem_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        log(f"Failed to parse problem.yaml: {e}", "ERROR")
        return False

    ref_file = reference_spec(problem_config).file
    ref_path = problem_dir / ref_file
    if not ref_path.exists():
        log(f"Reference source not found at {ref_path}", "ERROR")
        return False
    log(f"Reference source found: {ref_file}", "SUCCESS")

    # Regenerate inputs.hpp from the canonical inputs.yaml before anything reads it.
    from utils.inputs import ensure_inputs_hpp

    try:
        ensure_inputs_hpp(problem_dir)
        log("inputs.hpp generated from inputs.yaml", "SUCCESS")
    except FileNotFoundError as e:
        log(str(e), "ERROR")
        return False
    except Exception as e:
        log(f"Failed to generate inputs.hpp: {e}", "ERROR")
        return False

    # Check libktt.so (the framework driver links against it)
    libktt_path = SCRIPT_DIR / "libktt.so"
    if not libktt_path.exists():
        log(f"libktt.so not found at {libktt_path}", "ERROR")
        log("Framework driver cannot link — ensure libktt.so is symlinked", "ERROR")
        return False
    log("libktt.so found", "SUCCESS")

    # Check CUDA environment
    from utils.cuda_env import get_env, format_summary

    env = get_env()
    log(format_summary(env))

    for warning in env.warnings:
        log(warning, "WARN")
    for error in env.errors:
        log(error, "ERROR")

    # nvcc is required
    if not env.nvcc:
        return False

    return True


async def main():
    """Main entry point."""
    args = parse_args()

    log_section("CUDA Agentic Kernel Optimizer")
    log("Architecture: Queue-based parallel branches with isolated state")

    # Set dynamic output directory based on problem dir
    problem_dir = Path(args.dir).resolve()

    # Handle --prune-archives fast bypass. Touches only archive/, never output/.
    if args.prune_archives:
        removed, freed = prune_archives(problem_dir)
        if removed:
            log(
                f"Pruned {removed} regenerable file(s) from archived runs, "
                f"freed {freed / 1e9:.2f} GB. Kernels, results and decisions kept.",
                "SUCCESS",
            )
        else:
            log("Nothing to prune — no archived runs, or already pruned.")
        return 0

    # Handle --clone fast bypass
    if args.clone:
        import re

        new_name = args.clone
        if not re.match(r"^[a-z0-9_]+$", new_name):
            log("Name must be lowercase alphanumeric with underscores", "ERROR")
            return 1
        target_dir = problem_dir.parent / new_name
        if target_dir.exists():
            log(f"Problem '{new_name}' already exists", "ERROR")
            return 1
        clone_problem_dir(problem_dir, target_dir)
        log(f"Cloned '{problem_dir.name}' → '{new_name}'", "SUCCESS")
        return 0

    # Handle --best fast bypass. It reports what is already on disk, so it belongs above
    # the provider block with the other read-only bypasses: init_from_config and
    # get_default_model both raise "No LLM provider detected", and reading a finished
    # run must not need an API key. Only the two paths merge_node reads are set.
    if args.best:
        config.set_output_dir(problem_dir / "output")
        config.set_problem_dir(problem_dir)

        log_section("AGGREGATING RESULTS")
        from nodes.merge import merge_node

        return _report_final(merge_node())

    # Build immutable config and apply (must happen before check_api_key/get_provider)
    cfg = OptimizerConfig(
        output_dir=problem_dir / "output",
        problem_dir=problem_dir,
        max_iterations=args.max_iter,
        max_branch_depth=args.max_depth,
        path_budget=args.path_budget,
        tuner_timeout=args.timeout,
        model=args.model,
        provider=args.provider,
    )
    # init_from_config and get_default_model both reach _detect_provider, which is what
    # raises "No LLM provider detected". They have to be inside the try, or the message
    # below is unreachable and a fresh checkout with no API key gets a raw traceback
    # instead of the four lines telling it which variable to set.
    try:
        init_from_config(cfg)
        model = args.model or get_default_model()
        check_api_key()
        provider = get_provider()
        log(f"Using LLM provider: {provider}", "SUCCESS")
    except ValueError as e:
        log(str(e), "ERROR")
        return 1

    # GPU Detection
    log_section("Available GPUs")
    from utils.gpu_info import detect_gpus, format_gpu_table

    log("Detecting GPU specs (this may take a few seconds)...")
    gpus = detect_gpus()
    if gpus:
        log(format_gpu_table(gpus))
    else:
        log("No GPUs detected automatically.", "WARN")

    # Log configuration
    log(f"Script directory: {SCRIPT_DIR}")
    log(f"Problem directory: {problem_dir}")
    log(f"Model: {model}")
    mode = "path-budget" if args.path_budget > 0 else "depth"
    log(
        f"Iteration config: mode={mode} "
        f"path_budget={args.path_budget} max_iter={args.max_iter} max_depth={args.max_depth}"
    )
    if args.path_budget > 0 and args.max_iter != MAX_ITERATIONS:
        log(f"--max-iter={args.max_iter} is IGNORED in path-budget mode", "WARN")
    if args.timeout is not None:
        log(f"Tuner budget override: {args.timeout}s")
    else:
        log(
            f"Tuner budget: from problem.yaml tuning.duration_s (else system default {TUNER_TIMEOUT}s)"
        )

    # Check prerequisites
    if not check_prerequisites(problem_dir):
        return 1

    # Load input files
    problem_yaml = load_file(problem_dir / "problem.yaml")
    problem_config = yaml.safe_load(problem_yaml) or {}
    # Through reference_spec, as check_prerequisites already is: the hardcoded
    # "ref_kernel.cu" default here resolved a file a python or cpu_c problem does not
    # have, and load_file returns "" for a miss — so analyze and strategize ran against
    # no reference at all, silently.
    ref_file = reference_spec(problem_config).file
    ref_kernel = load_file(problem_dir / ref_file)

    log(f"Loaded problem.yaml ({len(problem_yaml)} chars)")
    log(f"Loaded reference source ({ref_file}, {len(ref_kernel)} chars)")

    # Handle resume vs fresh run
    if args.resume:
        log_section("RESUMING PREVIOUS RUN")

        from engine.resume import get_resume_states

        # Build resume states for incomplete branches
        log("\nScanning branch status...")
        resume_states = get_resume_states()

        if not resume_states:
            log(
                "Cannot resume: no valid prior run found, or all branches completed!",
                "ERROR",
            )
            log("Run without --resume to start fresh", "INFO")
            return 1

        log(f"\nResuming {len(resume_states)} incomplete branch(es)...")
        log("Flow: analyze(skip) → strategize(skip) → [resume branches]\n")
    else:
        resume_states = None
        # Fresh run
        # Clean output directory
        if not args.no_clean:
            # Moves the previous run to archive/output_N rather than deleting it.
            archive_output_dir()

        log("Starting optimization workflow...")
        log("Flow: analyze → strategize → [parallel branches]\n")

    from engine.master import run_optimization_engine

    log_path = problem_dir / "output" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Append whenever the previous output/ was kept. Keyed on --resume alone, --no-clean
    # truncated the run.log it had just promised to keep — the only record of the
    # earlier run's wall clock.
    log_mode = "a" if (args.resume or args.no_clean) else "w"
    log(f"Logging to {log_path}")

    log("Starting Queue Execution Engine...")
    with log_path.open(log_mode, encoding="utf-8") as _log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeWriter(old_stdout, _log_file)
        sys.stderr = TeeWriter(old_stderr, _log_file)
        try:
            await run_optimization_engine(problem_yaml, ref_kernel, resume_states)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # After all branches complete, aggregate results from disk. merge_node prints the
    # per-branch summary, the iteration timeline and the winner, and writes
    # final_results.json — the same call --best makes, so both paths agree on what
    # counts as a success.
    log_section("AGGREGATING RESULTS")

    from nodes.merge import merge_node

    return _report_final(merge_node())


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
