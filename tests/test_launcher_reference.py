"""The launcher patterns in the prompt must compile AND run.

Teaching an API from its header is not the same as knowing it works. The first draft
of the convergence loop used `UploadBuffer`, which the header documents as uploading a
vector argument into a compute buffer; against the real tuner it failed all six
configurations with `Buffer for argument with id 2 already exists`. Nothing short of
running it would have caught that, and a wrong example in the prompt is worse than no
example — every branch follows it confidently.

Slow: builds a driver and runs the GPU. Deselect with `-m "not integration"`.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

from prompts._launcher_reference import LAUNCHER_RULES

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBLEM = REPO_ROOT / "problems" / "flood"

# A flood fill: 4-connected reachability run to a fixed point. The problem class the
# single-launch pattern gets wrong, so the pattern that fixes it is what gets tested.
KERNEL = """
extern "C" __global__ void flood_expand(const int* __restrict__ heights,
                                        unsigned int* __restrict__ flooded,
                                        unsigned int* __restrict__ changed)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= MAP_W || y >= MAP_H) return;
    const int idx = y * MAP_W + x;
    if (flooded[idx]) return;
    if (heights[idx] > LEVEL) return;
    bool nb = (x == SRC_X && y == SRC_Y);
    if (x > 0)          nb |= flooded[idx - 1] != 0u;
    if (x < MAP_W - 1)  nb |= flooded[idx + 1] != 0u;
    if (y > 0)          nb |= flooded[idx - MAP_W] != 0u;
    if (y < MAP_H - 1)  nb |= flooded[idx + MAP_W] != 0u;
    if (nb) { flooded[idx] = 1u; *changed = 1u; }
}
"""

REGION_KERNELS = """
const ktt::KernelDefinitionId expandDef = tuner.AddKernelDefinitionFromFile(
    "flood_expand", kernelFile, ndRange, ktt::DimensionVector(16, 16));
auto changedId = tuner.AddArgumentVector(std::vector<unsigned int>(1, 0u),
                                         ktt::ArgumentAccessType::ReadWrite);
ktt::KernelId kernel = tuner.CreateSimpleKernel("Flood", expandDef);
tuner.SetArguments(expandDef, {in.heights, in.flooded, changedId});
"""

# More than one value so the tuner produces several configurations: a single-config
# run makes "all configurations validated" a much weaker claim than it looks.
# Several configurations, all with a round bound large enough to converge. 2048 is
# NOT: on this map it stops short and the result silently fails validation, which is
# the trap the launcher rules warn about rather than something to reproduce here.
REGION_PARAMS = """
tuner.AddParameter(kernel, "MAX_ROUNDS", std::vector<uint64_t>{4096, 8192});
tuner.AddParameter(kernel, "UNUSED_TILE", std::vector<uint64_t>{1, 2});
"""

# Verbatim shape of the loop taught in LAUNCHER_RULES.
REGION_LAUNCHER = """
tuner.SetLauncher(kernel, [expandDef, changedId](ktt::ComputeInterface& ci) {
    const auto& cfg = ci.GetCurrentConfiguration();
    const uint64_t maxRounds =
        ktt::ParameterPair::GetParameterValue<uint64_t>(cfg.GetPairs(), "MAX_ROUNDS");
    unsigned int changed = 1u;
    for (uint64_t round = 0; changed != 0u && round < maxRounds; ++round) {
        const unsigned int zero = 0u;
        ci.UpdateBuffer(changedId, &zero, sizeof(zero));
        ci.RunKernel(expandDef);
        ci.DownloadBuffer(changedId, &changed, sizeof(changed));
    }
});
"""


def _requirements():
    if not (REPO_ROOT / "libktt.so").exists():
        pytest.skip("libktt.so not present")
    if not (PROBLEM / "inputs.hpp").exists():
        pytest.skip("flood problem not present")
    from utils import nvrtc

    if not nvrtc.available():
        pytest.skip("libnvrtc not available")


@pytest.fixture(scope="module")
def built():
    """Build the driver from the documented patterns; yield its staging directory."""
    _requirements()
    from agentic.tools import Toolbox
    from agentic.workspace import Workspace

    meta = yaml.safe_load((PROBLEM / "problem.yaml").read_text(encoding="utf-8"))
    root = Path(tempfile.mkdtemp())
    try:
        iter_dir = root / "iter1"
        iter_dir.mkdir(parents=True)
        ws = Workspace(iter_dir)
        ws.reset()
        ws.write("kernels.cu", KERNEL)
        ws.write("region_kernels.cpp", REGION_KERNELS)
        ws.write("region_params.cpp", REGION_PARAMS)
        ws.write("region_launcher.cpp", REGION_LAUNCHER)

        box = Toolbox(
            ws,
            branch_path=root,
            output_dir=root,
            problem_dir=PROBLEM,
            meta=meta,
            gpu_info={"compute_capability": "8.6"},
        )
        output = box.check_compilation()
        assert box.check_passed, output
        yield ws.staging
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_documented_patterns_compile(built):
    assert (built / "driver").exists()


def test_convergence_loop_actually_validates(built):
    """Runs the tuner. Every configuration must succeed and match the CPU reference.

    A single-launch flood fill compiles and fails validation on every configuration;
    that is the failure this pattern exists to prevent, so passing here is the whole
    point of the test.
    """
    from utils.landscape import load_records

    proc = subprocess.run(
        [
            "./driver", "0", "0", "10", "1e-6", "results",
            "kernels.cu", str(PROBLEM / "ref_cpu.c"),
        ],
        cwd=str(built),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "already exists" not in proc.stdout, (
        "buffer management regression — UploadBuffer instead of UpdateBuffer?\n"
        + proc.stdout[-2000:]
    )

    records = load_records(built / "results.json")
    assert records, proc.stdout[-2000:]
    ok = [r for r in records if r.ok]
    assert len(ok) == len(records), (
        f"only {len(ok)}/{len(records)} configurations validated\n" + proc.stdout[-3000:]
    )


def test_prompt_teaches_update_not_upload():
    assert "UpdateBuffer(changedId" in LAUNCHER_RULES
    assert "Do NOT use `UploadBuffer`" in LAUNCHER_RULES
    # RunKernel takes a definition id; passing the composite KernelId compiles and
    # silently runs one definition.
    assert "KernelDefinitionId — NOT the KernelId" in LAUNCHER_RULES
