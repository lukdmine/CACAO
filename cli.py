#!/usr/bin/env python3
"""
CUDA Agentic Kernel Optimizer — CLI entry point.

Run ``python cli.py --help`` for the full list of options.
"""

import argparse
import shutil
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
from utils.files import load_file, clean_output_dir
from utils.log import log, TeeWriter


def log_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60, flush=True)
    print(f"  {title}")
    print("=" * 60, flush=True)


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
        problem_config = yaml.safe_load(problem_path.read_text()) or {}
    except Exception as e:
        log(f"Failed to parse problem.yaml: {e}", "ERROR")
        return False

    ref_file = problem_config.get("reference", {}).get("file", "ref_kernel.cu")
    ref_path = problem_dir / ref_file
    if not ref_path.exists():
        log(f"Reference source not found at {ref_path}", "ERROR")
        return False
    log(f"Reference source found: {ref_file}", "SUCCESS")

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
        shutil.copytree(
            problem_dir,
            target_dir,
            ignore=shutil.ignore_patterns("run.pid", "run.log", "requeue"),
        )
        log(f"Cloned '{problem_dir.name}' → '{new_name}'", "SUCCESS")
        return 0

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
    init_from_config(cfg)
    model = args.model or get_default_model()

    # Check API key and detect provider
    try:
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

    # Handle --best fast bypass
    if args.best:
        log_section("AGGREGATING RESULTS")
        from nodes.merge import merge_node

        best_state = merge_node()
        log_section("FINAL RESULT")
        if best_state.get("status") == "success":
            log("OPTIMIZATION SUCCESSFUL", "SUCCESS")
        else:
            log("OPTIMIZATION FAILED", "ERROR")
            log(f"Check {config.get_output_dir()} for details")
        return 0

    # Check prerequisites
    if not check_prerequisites(problem_dir):
        return 1

    # Load input files
    problem_yaml = load_file(problem_dir / "problem.yaml")
    problem_config = yaml.safe_load(problem_yaml) or {}
    ref_file = problem_config.get("reference", {}).get("file", "ref_kernel.cu")
    ref_kernel = load_file(problem_dir / ref_file)

    log(f"Loaded problem.yaml ({len(problem_yaml)} chars)")
    log(f"Loaded reference source ({ref_file}, {len(ref_kernel)} chars)")

    # Handle resume vs fresh run
    if args.resume:
        log_section("RESUMING PREVIOUS RUN")

        from utils.resume import get_resume_states

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
            log("Cleaning output directory...")
            clean_output_dir()
            log("Output directory cleaned", "SUCCESS")

        log("Starting optimization workflow...")
        log("Flow: analyze → strategize → [parallel branches]\n")

    from engine.master import run_optimization_engine

    log_path = problem_dir / "output" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.resume else "w"
    log(f"Logging to {log_path}")

    log("Starting Queue Execution Engine...")
    with log_path.open(log_mode) as _log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeWriter(old_stdout, _log_file)
        sys.stderr = TeeWriter(old_stderr, _log_file)
        try:
            await run_optimization_engine(problem_yaml, ref_kernel, resume_states)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # After all branches complete, aggregate results from disk
    log_section("AGGREGATING RESULTS")

    from nodes.merge import find_branch_results, get_best_branch_result
    from utils.files import save_json

    output_dir = config.get_output_dir()
    branches_dir = output_dir / "branches"
    branch_results = find_branch_results(branches_dir)

    log(f"Found {len(branch_results)} branch result(s)")

    if not branch_results:
        log("No branch results found!", "ERROR")
        log("Check output/branches/ for details")
        return 1

    # Print summary of each branch (including nested sub-branches)
    for result in branch_results:
        branch_name = result.get("branch_name", "unknown")
        branch_path = result.get("branch_path", "")
        status = result.get("status", "unknown")
        best_time = result.get("best_time_us")
        speedup = result.get("speedup")
        iterations = result.get("iterations", 0)

        # Determine nesting level from path
        depth = branch_path.count("/branches/") if branch_path else 0
        indent = "  " * depth

        status_icon = "✓" if status == "success" else "✗"
        time_str = f"{best_time:.2f} µs" if best_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

        log(
            f"{indent}{status_icon} {branch_name}: {time_str} (speedup: {speedup_str}, iters: {iterations})"
        )

    # Find best overall result
    best = get_best_branch_result(branch_results)

    # Print final status
    log_section("FINAL RESULT")

    if best and best.get("status") == "success":
        log("OPTIMIZATION COMPLETE", "SUCCESS")
        log(f"Results saved to: {output_dir}")

        best_time = best.get("best_time_us")
        log(f"\nBest branch: {best.get('branch_name')}")
        if best_time is not None:
            log(f"Best time: {best_time:.2f} µs")
        if best.get("speedup"):
            log(f"Speedup: {best.get('speedup'):.2f}x")

        # Save final summary
        final_summary = {
            "best_branch": best.get("branch_name"),
            "best_config": best.get("best_config"),
            "best_time_us": best.get("best_time_us"),
            "speedup": best.get("speedup"),
            "total_branches": len(branch_results),
            "all_branches": [
                {
                    "name": r.get("branch_name"),
                    "path": r.get("branch_path"),
                    "status": r.get("status"),
                    "best_time_us": r.get("best_time_us"),
                    "speedup": r.get("speedup"),
                    "iterations": r.get("iterations"),
                }
                for r in branch_results
            ],
        }
        save_json(output_dir / "final_results.json", final_summary)

        return 0
    else:
        log("OPTIMIZATION FAILED", "ERROR")
        if best:
            log(f"Best attempt: {best.get('branch_name')} ({best.get('status')})")
        log("Check output/ for details")
        return 1


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
