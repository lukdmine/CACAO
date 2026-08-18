"""Shared fixtures.

Where a test can run against real run output already in the repo it does, rather than
against a synthetic fixture: the parsers here exist to survive what KTT and the LLM
actually emit, and a fixture written from the same assumptions as the parser proves
nothing. Tests that need that data skip cleanly when it is absent.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# A real completed run: three branches, nested sub-branches, iterations that passed and
# iterations that failed to compile.
COVARIANCE = REPO_ROOT / "problems" / "covariance"
COV_OUTPUT = COVARIANCE / "output"
COV_BRANCH = COV_OUTPUT / "branches" / "register_coarsened_syrk"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"recorded run data not present: {path}")
    return path


@pytest.fixture(scope="session")
def cov_problem() -> Path:
    return _require(COVARIANCE)


@pytest.fixture(scope="session")
def cov_output() -> Path:
    return _require(COV_OUTPUT)


@pytest.fixture(scope="session")
def cov_branch() -> Path:
    return _require(COV_BRANCH)


@pytest.fixture(scope="session")
def params_region() -> str:
    """The PARAMS region of a real, validated iteration."""
    from utils.framework import extract_regions

    fw = _require(COV_BRANCH / "iter8" / "framework.cpp")
    return extract_regions(fw.read_text(encoding="utf-8")).get("params", "")


@pytest.fixture(scope="session")
def good_kernel() -> str:
    """iter8: validated, 305 µs, 103/103 configurations passed."""
    return _require(COV_BRANCH / "iter8" / "kernels.cu").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def broken_kernel() -> str:
    """iter9: recorded as 'all 218 configs failed compilation'."""
    return _require(COV_BRANCH / "iter9" / "kernels.cu").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def isolate_capability_registry():
    """Reset the provider tool-capability verdicts around every test.

    The registry is process-global on purpose — one run, one verdict per provider — so
    without this a test that exercises a fallback disables the loop for every test
    that runs after it.
    """
    from agentic import capability

    capability.reset()
    yield
    capability.reset()


@pytest.fixture
def workspace(tmp_path):
    from agentic.workspace import Workspace

    ws = Workspace(tmp_path / "iter1")
    (tmp_path / "iter1").mkdir(parents=True, exist_ok=True)
    ws.reset()
    return ws


@pytest.fixture
def filled_workspace(workspace):
    """A workspace with all four files present, so gating can be tested past it."""
    workspace.write("kernels.cu", 'extern "C" __global__ void k(float* o) { o[0] = TILE; }\n')
    workspace.write("region_kernels.cpp", "ktt::KernelId kernel = 0;\n")
    workspace.write(
        "region_params.cpp",
        'tuner.AddParameter(kernel, "TILE", std::vector<uint64_t>{16, 32});\n',
    )
    workspace.write("region_launcher.cpp", "// default launcher\n")
    return workspace
