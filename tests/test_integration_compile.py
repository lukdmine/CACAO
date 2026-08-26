"""The compile check against the real toolchain.

Everything else mocks `check_compilation` to keep the suite fast. This module does not:
it links the assembled driver against libktt.so with g++ and runs the kernel through
libnvrtc, on iterations whose real outcome is recorded. If the check disagrees with
what KTT actually did, the whole feature is worse than useless — it would send
branches chasing defects that are not there, or pass code that cannot run.

Slow (a g++ link takes seconds). Deselect with `-m "not integration"`.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from agentic.tools import Toolbox
from agentic.workspace import Workspace
from utils import nvrtc
from utils.framework import extract_regions

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBKTT = REPO_ROOT / "libktt.so"


def _requirements():
    if not LIBKTT.exists():
        pytest.skip("libktt.so not present — driver cannot be linked")
    if not nvrtc.available():
        pytest.skip(f"libnvrtc not available: {nvrtc.unavailable_reason()}")


def _run_check(branch: Path, problem: Path, iteration: str):
    _requirements()
    meta = yaml.safe_load((problem / "problem.yaml").read_text(encoding="utf-8"))
    iter_src = branch / iteration
    if not (iter_src / "framework.cpp").exists():
        pytest.skip(f"recorded {iteration} not present")

    tmp_root = Path(tempfile.mkdtemp())
    try:
        iter_dir = tmp_root / "iter1"
        iter_dir.mkdir(parents=True)
        ws = Workspace(iter_dir)
        ws.reset()

        regions = extract_regions((iter_src / "framework.cpp").read_text(encoding="utf-8"))
        ws.write("kernels.cu", (iter_src / "kernels.cu").read_text(encoding="utf-8"))
        ws.write("region_kernels.cpp", regions.get("kernels") or "// none")
        ws.write("region_params.cpp", regions.get("params") or "// none")
        ws.write("region_launcher.cpp", regions.get("launcher") or "// none")

        box = Toolbox(
            ws,
            branch_path=tmp_root,
            output_dir=tmp_root,
            problem_dir=problem,
            meta=meta,
            gpu_info={"compute_capability": "8.6"},
        )
        return box, box.check_compilation()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_validated_iteration_passes_the_real_check(cov_branch, cov_problem):
    """iter8 ran 103/103 configurations on the GPU."""
    box, output = _run_check(cov_branch, cov_problem, "iter8")
    assert box.check_passed is True, output
    assert "Driver (g++): PASS" in output
    assert "NVRTC: PASS" in output


def test_broken_iteration_fails_the_real_check(cov_branch, cov_problem):
    """iter9 was recorded as 'all 218 configs failed compilation'."""
    box, output = _run_check(cov_branch, cov_problem, "iter9")
    assert box.check_passed is False
    assert "NVRTC: FAIL" in output
    # The driver itself was fine; only the kernel was broken. Saying so is what makes
    # the failure actionable.
    assert "Driver (g++): PASS" in output


def test_failure_names_the_configuration_it_used(cov_branch, cov_problem):
    # A bare pass/fail cannot be acted on when constraints are not evaluated: the model
    # has to see which configuration failed to judge whether KTT would run it.
    _, output = _run_check(cov_branch, cov_problem, "iter9")
    assert "Defines:" in output
    assert "REG_TILE_N=" in output


def test_later_validated_iterations_also_pass(cov_branch, cov_problem):
    for iteration in ("iter11", "iter12"):
        box, output = _run_check(cov_branch, cov_problem, iteration)
        assert box.check_passed is True, f"{iteration}: {output}"
