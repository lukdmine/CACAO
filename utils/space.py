"""How big a tuning space is, what exhausting it would cost, and which searcher fits.

Finalize re-tunes a branch's best kernel over a widened space. A small space is better
searched exhaustively (the answer is then the true optimum, not a sample), a large one
needs MCMC; the choice turns on whether the whole space fits in the tuner budget.
"""

import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from utils.cases import case_list, stage_case
from utils.framework import load_regions
from utils.inputs import load_inputs_spec
from utils.build import compile_framework, driver_command, reference_build_extras
from utils.nvrtc import parse_parameters

# Exhaustive only when the estimate leaves 10% of the budget spare: the per-config cost
# is a median from a previous run, and a space that just barely fits would be cut short.
EXHAUSTIVE_MARGIN = 0.9

_COUNT_LINE = re.compile(r"^CACAO_CONFIG_COUNT\s+(\d+)\s*$", re.MULTILINE)


def parse_config_count(output: str) -> Optional[int]:
    m = _COUNT_LINE.search(output or "")
    return int(m.group(1)) if m else None


def count_configurations(
    driver: Path,
    cwd: Path,
    *,
    gpu_index: int,
    kernel_file: Path,
    ref_file: Path,
    timeout_s: float = 120.0,
) -> Optional[int]:
    """Ask a compiled driver how many configurations its space holds (count mode).

    None when it exits non-zero, times out, or prints no count line.
    """
    cmd = driver_command(
        driver,
        platform=0,
        device=gpu_index,
        duration=0,
        tolerance=0,
        output_base="count",
        kernel_file=kernel_file,
        ref_file=ref_file,
        mode="count",
    )
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_s
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return parse_config_count(proc.stdout)


def count_for_iteration(
    iter_dir: Path,
    meta: dict,
    problem_dir: Path,
    *,
    gpu_index: int,
    ref_file: Path,
) -> Optional[int]:
    """Stage and compile the primary case of an iteration, then count its space.

    Mirrors run_node pass 1, so the driver counted is the one a run would tune.
    ``ref_file`` must be absolute (run_node passes ``problem_dir / ref_file``) or valid
    from the staged case directory, which is the driver's cwd.
    """
    iter_dir = Path(iter_dir)
    problem_dir = Path(problem_dir)
    cases = case_list(meta)
    case = cases[0]
    spec = load_inputs_spec(problem_dir / "inputs.yaml")
    cdir = stage_case(
        problem_dir, iter_dir, spec, meta, cases, case, load_regions(iter_dir)
    )
    extra_sources, extra_flags = reference_build_extras(problem_dir, case)
    build = compile_framework(
        cdir, extra_sources=extra_sources, extra_flags=extra_flags
    )
    if not build.ok:
        return None
    return count_configurations(
        build.binary,
        cdir,
        gpu_index=gpu_index,
        kernel_file=cdir / "kernels.cu",
        ref_file=ref_file,
    )


def per_config_cost_s(results_path: Path) -> Optional[float]:
    """Median seconds one configuration cost in a previous run, or None.

    KTT reports TotalDuration and TotalOverhead in microseconds. TotalOverhead already
    includes compilation, validation and data-movement overhead; compilation usually
    dominates for a small kernel. The median is over ALL results, failed ones included,
    because an exhaustive search visits the failing configurations too.
    """
    try:
        data = json.loads(Path(results_path).read_text(encoding="utf-8"))
        costs = [
            (float(r["TotalDuration"]) + float(r["TotalOverhead"])) / 1e6
            for r in data.get("Results", [])
        ]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return statistics.median(costs) if costs else None


def estimate_exhaustive_s(count: int, results_path: Path) -> Optional[float]:
    cost = per_config_cost_s(results_path)
    return None if cost is None else count * cost


def choose_searcher(estimate_s: Optional[float], budget_s: float) -> str:
    if estimate_s is not None and estimate_s <= EXHAUSTIVE_MARGIN * budget_s:
        return "exhaustive"
    return "mcmc"


def _same_value(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def seed_is_reachable(best_config: Dict[str, Any], params_src: str) -> bool:
    """Whether a recorded best configuration is still a point of the (widened) space.

    A seed naming a dropped parameter or a removed value is not in the space, and
    handing it to the searcher would fail or silently start somewhere else.

    Nor is a seed that lacks a declared parameter, which is what widening leaves when it
    turns a literal back into a parameter. KTT locates a seed by one index per tree level
    (ConfigurationTree::GetIndicesFromConfiguration), so a short seed returns too few and
    ConfigurationNode::ComputeLocalIndex reads past the end of the vector: undefined
    behaviour, with the assert compiled out of a release build.
    """
    declared = parse_parameters(params_src)
    if any(name not in best_config for name in declared):
        return False
    for name, value in best_config.items():
        if name not in declared:
            return False
        if not any(_same_value(value, v) for v in declared[name]):
            return False
    return True
