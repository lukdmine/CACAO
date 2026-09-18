"""NCU profile CSV parsing: whitelist filter and multi-kernel attribution."""

from nodes.profile import parse_ncu_csv

_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"


def _csv(header, units, *rows):
    def q(fields):
        return ",".join(f'"{f}"' for f in fields)

    return "\n".join([q(header), q(units), *(q(r) for r in rows)])


def test_junk_columns_and_rollup_duplicates_are_dropped():
    # --page raw always emits device attributes, bookkeeping and one column per
    # rollup variant; only whitelisted columns may survive into the LLM prompt.
    out = parse_ncu_csv(
        _csv(
            [
                "ID",
                "Kernel Name",
                "Block Size",
                "device__attribute_display_name",
                "nvlink__count_logical",
                "profiler__replayer_passes",
                _DRAM,
                "dram__throughput.min.pct_of_peak_sustained_elapsed",
                "launch__registers_per_thread",
                "launch__grid_size",
            ],
            ["", "", "", "", "", "", "%", "%", "register/thread", ""],
            ["0", "gemm_tiled", "(16, 16, 1)", "RTX 5050", "0", "8", "6.16", "6.16", "63", "512"],
        )
    )
    assert out == {_DRAM: 6.16, "launch__registers_per_thread": 63.0}


def test_single_kernel_does_not_emit_kernel_name():
    out = parse_ncu_csv(
        _csv(
            ["Kernel Name", _DRAM],
            ["", "%"],
            ["stage_a", "6.16"],
        )
    )
    assert out == {_DRAM: 6.16}


def test_multi_kernel_rows_are_prefixed_by_kernel_name():
    out = parse_ncu_csv(
        _csv(
            ["Kernel Name", _DRAM, "launch__waves_per_multiprocessor"],
            ["", "%", ""],
            ["stage_a", "6.16", "6.4"],
            ["stage_b", "1.5", "2.0"],
        )
    )
    assert set(out) == {
        f"stage_a/{_DRAM}",
        "stage_a/launch__waves_per_multiprocessor",
        f"stage_b/{_DRAM}",
        "stage_b/launch__waves_per_multiprocessor",
    }
    assert out[f"stage_b/{_DRAM}"] == 1.5


def test_duplicate_kernel_names_get_suffix():
    out = parse_ncu_csv(
        _csv(
            ["Kernel Name", _DRAM],
            ["", "%"],
            ["k", "1.0"],
            ["k", "2.0"],
        )
    )
    assert out == {f"k/{_DRAM}": 1.0, f"k#2/{_DRAM}": 2.0}


def test_empty_cells_and_no_data_are_skipped():
    out = parse_ncu_csv(
        _csv(
            ["Kernel Name", _DRAM, "launch__occupancy_limit_registers"],
            ["", "%", "block"],
            ["k", "no data", "4"],
        )
    )
    assert out == {"launch__occupancy_limit_registers": 4.0}
