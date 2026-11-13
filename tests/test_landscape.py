"""Landscape queries, run against the 103-configuration result set they exist for."""

import json

import pytest

from utils import landscape

# A real iteration with failures: 11 ValidationFailed, 3 Ok, and two parameter values
# that never produce a working kernel.
MIXED = (
    "problems/flood_copy/output/branches/tile_local_fixed_point/branches/"
    "warp_autonomous_relax/iter1/results.json"
)


@pytest.fixture
def mixed(cov_output):
    from tests.conftest import REPO_ROOT

    path = REPO_ROOT / MIXED
    if not path.exists():
        pytest.skip("recorded results with failures not present")
    return path


def test_loads_every_configuration_not_just_the_best(all_ok):
    records = landscape.load_records(all_ok)
    assert len(records) == 103
    assert all(r.ok for r in records)


def test_top_reports_registers_spill_and_shared_memory(all_ok):
    out = landscape.top_configs(all_ok, n=5)
    header = next(l for l in out.splitlines() if l.startswith("| #"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    row = next(l for l in out.splitlines() if l.startswith("| 1 "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    # iter8's fastest kernel declares __shared__ arrays and spills nothing.
    assert cells[cols.index("smem B")] == "4672"
    assert cells[cols.index("spill B")] == "0"
    assert "103/103 configurations succeeded" in out


def _write_results(tmp_path, configs):
    """A results.json in KTT's shape; configs is [(duration_us, local, private)]."""
    results = [
        {
            "Status": "Ok",
            "Configuration": [{"Name": "T", "Value": 16 * (i + 1)}],
            "ComputationResults": [
                {
                    "Duration": dur,
                    "CompilationData": {
                        "RegistersCount": 32,
                        "LocalMemorySize": local,
                        "PrivateMemorySize": private,
                    },
                }
            ],
        }
        for i, (dur, local, private) in enumerate(configs)
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"Results": results}), encoding="utf-8")
    return path


def test_spill_reads_private_memory(tmp_path):
    rec = landscape.load_records(_write_results(tmp_path, [(10.0, 4096, 64)]))[0]
    assert rec.smem == 4096 and rec.local_mem == 64


def test_best_record_is_the_fastest_ok_record(tmp_path):
    records = landscape.load_records(
        _write_results(tmp_path, [(30.0, 0, 0), (10.0, 512, 0), (20.0, 0, 0)])
    )
    best = landscape.best_record(records)
    assert best.duration == 10.0 and best.smem == 512
    assert landscape.best_record([]) is None


def test_top_reports_the_spread(all_ok):
    out = landscape.top_configs(all_ok, n=3)
    assert "spread" in out


def test_top_row_count_is_bounded(all_ok):
    out = landscape.top_configs(all_ok, n=10_000)
    assert out.count("\n|") <= landscape._MAX_ROWS + 2


def test_sweep_groups_by_value(all_ok):
    out = landscape.param_sweep(all_ok, "BLOCK_DIM")
    for value in ("4", "8", "16"):
        assert f"| {value} |" in out


def test_sweep_flags_an_optimum_at_the_range_edge(all_ok):
    # NUM_MEAN_BLOCKS is {64, 128} and 64 wins: the best value is an endpoint, so the
    # range may be clipped and the fix is the params region, not the kernel.
    out = landscape.param_sweep(all_ok, "NUM_MEAN_BLOCKS")
    assert "edge of the" in out


def test_sweep_stays_quiet_on_an_interior_optimum(all_ok):
    # BLOCK_DIM is {4, 8, 16} and 8 wins: nothing to widen.
    assert "edge of the" not in landscape.param_sweep(all_ok, "BLOCK_DIM")


def test_sweep_names_available_parameters_when_asked_for_a_bad_one(all_ok):
    out = landscape.param_sweep(all_ok, "NOPE")
    assert "Unknown parameter" in out and "BLOCK_DIM" in out


def test_failures_identify_dead_parameter_values(mixed):
    out = landscape.failure_breakdown(mixed)
    assert "ValidationFailed: 11" in out
    assert "BACKOFF=32" in out
    assert "CTRCAD=8" in out


def test_failures_say_so_when_everything_worked(all_ok):
    assert "Every configuration succeeded" in landscape.failure_breakdown(all_ok)


def test_missing_file_is_not_an_error(tmp_path):
    assert landscape.load_records(tmp_path / "nope.json") == []
    assert "No results" in landscape.top_configs(tmp_path / "nope.json")


def test_corrupt_file_is_not_an_error(tmp_path):
    bad = tmp_path / "results.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert landscape.load_records(bad) == []


def test_all_failed_points_at_the_failure_mode(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "Results": [
                    {
                        "Status": "CompilationFailed",
                        "Configuration": [{"Name": "T", "Value": 16}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert "mode='failures'" in landscape.top_configs(path)


def test_describe_rejects_an_unknown_mode(all_ok):
    assert "Unknown mode" in landscape.describe(all_ok, mode="sideways")


def test_describe_sweep_without_a_param_lists_them(all_ok):
    assert "SPLIT_K" in landscape.describe(all_ok, mode="sweep")
