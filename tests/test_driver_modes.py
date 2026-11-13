"""driver_command keywords and the searcher modes the driver skeleton knows."""

from tests.helpers.run_harness import PROBLEM_YAML
from utils.build import driver_command
from utils.framework import assemble_framework_cpp

BASE = dict(platform=0, device=1, duration=10, tolerance=1e-4, output_base="results",
            kernel_file="kernels.cu", ref_file="ref.cu")

MINIMAL_META = {k: v for k, v in PROBLEM_YAML.items() if k != "cases"}


def test_default_command_is_unchanged():
    assert driver_command("./driver", **BASE) == [
        "./driver", "0", "1", "10", "0.0001", "results", "kernels.cu", "ref.cu"]


def test_profile_results_append_alone():
    assert driver_command("./driver", **BASE, profile_results="results")[8:] == ["results"]


def test_a_mode_leaves_the_profile_slot_empty():
    cmd = driver_command("./driver", **BASE, mode="mcmc", seed_results="/s/results")
    assert cmd[8:] == ["", "mcmc", "/s/results"]


def test_the_skeleton_knows_every_mode():
    src = assemble_framework_cpp(MINIMAL_META, {})
    for token in ("RandomSearcher", "DeterministicSearcher", "McmcSearcher",
                  "CACAO_CONFIG_COUNT", "InitializeConfigurationData"):
        assert token in src
