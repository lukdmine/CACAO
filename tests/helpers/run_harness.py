"""Drive nodes/run.run_node against a synthetic two-case problem with a stubbed tuner.

Real KTT is not involved: the host compile and the driver subprocess are both replaced,
so this exercises case ordering, fail-fast and summary assembly and nothing else.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml

_ORDER: list = []


def order_log() -> list:
    return list(_ORDER)


PROBLEM_YAML = {
    "name": "twocase",
    "global_size_type": "cuda",
    "grid": {"x": "N", "y": "1", "z": "1"},
    "reference": {"type": "cuda", "function": "ref", "file": "ref_kernel.cu"},
    "validation": {"tolerance": 1e-4},
    "tuning": {"duration_s": 10},
    "cases": [
        {"name": "big", "scalars": {"N": 1024}},
        {"name": "small", "scalars": {"N": 128}, "duration_s": 5},
    ],
}

INPUTS_YAML = {
    "headers": [],
    "shared_setup": "",
    "args": [
        {"kind": "scalar", "name": "N", "dtype": "int", "value": 1024,
         "placements": ["host", "runtime"]},
        {"kind": "buffer", "name": "out", "dtype": "float", "size": "N",
         "access": "write", "init": "zeros", "validate": True},
    ],
}


def _results_json(best_us: float, ok: bool) -> dict:
    return {
        "Metadata": {"TimeUnit": "Microseconds"},
        "Results": [{
            "KernelName": "K",
            "Status": "Ok" if ok else "ComputationFailed",
            "Configuration": [{"Name": "TILE", "Value": 16, "ValueType": "UnsignedInt"}],
            "ComputationResults": ([{"Duration": best_us}] if ok else []),
            "TotalDuration": best_us if ok else 0.0,
        }],
    }


def run_two_case_node(
    tmp_path: Path,
    compile_framework=None,
    fail_tuning_for: str | None = None,
    cases: list | None = None,
):
    """Build a problem + iteration under tmp_path and run run_node against it."""
    import config
    from utils.build import BuildResult

    _ORDER.clear()

    meta = dict(PROBLEM_YAML)
    if cases is not None:
        if cases:
            meta["cases"] = cases
        else:
            meta.pop("cases")

    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "problem.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    (problem / "inputs.yaml").write_text(yaml.safe_dump(INPUTS_YAML), encoding="utf-8")
    (problem / "ref_kernel.cu").write_text(
        "__global__ void ref(float* o, int n) {}\n", encoding="utf-8"
    )

    output = problem / "output"
    branch = output / "branches" / "b"
    iter_dir = branch / "iter1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "kernels.cu").write_text("// kernel\n", encoding="utf-8")
    for name in ("kernels", "params", "launcher"):
        (iter_dir / f"region_{name}.cpp").write_text(f"// {name}\n", encoding="utf-8")

    import nodes.run as run_mod

    def _compile(d, **kw):
        _ORDER.append(f"compile:{Path(d).name}")
        return BuildResult(True, Path(d) / "driver", "", [])

    async def _drive(cmd, cwd, budget_s, watchdog_s, tracker, branch_name="default"):
        name = Path(cwd).name
        _ORDER.append(f"tune:{name}")
        ok = fail_tuning_for is None or fail_tuning_for not in name
        (Path(cwd) / "results.json").write_text(
            json.dumps(_results_json(100.0, ok)), encoding="utf-8"
        )
        return "Tuning done: 1 configurations -> results\n"

    patches = [
        (run_mod, "compile_framework", compile_framework or _compile),
        (run_mod, "_execute_driver", _drive),
    ]
    saved = [(m, n, getattr(m, n)) for m, n, _ in patches]
    for m, n, v in patches:
        setattr(m, n, v)

    prev_problem, prev_output = config.get_problem_dir(), config.get_output_dir()
    config.set_problem_dir(problem)
    config.set_output_dir(output)
    try:
        from state.types import WorkingState

        state = WorkingState(
            analysis="", iter_num=1, status="running", max_iter=1,
            branch_path=str(branch),
        )
        return asyncio.run(run_mod.run_node(state))
    finally:
        for m, n, v in saved:
            setattr(m, n, v)
        config.set_problem_dir(prev_problem)
        config.set_output_dir(prev_output)
