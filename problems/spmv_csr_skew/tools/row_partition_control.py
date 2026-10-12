#!/usr/bin/env python3
"""Build the two control matrices that isolate row-partition cost from data movement.

Both carry mawi's exact nonzeros, columns and values. Only row_offsets differs:

    mawimerge   594 380 uniform rows of 32 nonzeros
    mawisplit   mawi's real rows, except the 7 397 164-nonzero row chopped into 32s

Running tools/bench_cusparse on all three separates three hypotheses that look alike
from the outside -- "the giant row serialises a warp", "the gather thrashes", and "the
per-row cost dominates". Measured here on an RTX 3090, CUDA 12.0:

    mawi        18 571 154 rows, avg 1.02      14 155 us     26.5 GB/s
    mawisplit   18 802 315 rows, max 65 467    14 146 us     26.6 GB/s
    mawimerge      594 380 rows, all 32        // 257 us    878.5 GB/s

Splitting the giant row changes nothing; merging the short ones is 55x faster on
identical data. The cost is the row boundaries, not the nonzeros.

    python problems/spmv_csr_skew/tools/row_partition_control.py
    make -C problems/spmv_csr_skew/tools && \
        problems/spmv_csr_skew/tools/bench_cusparse mawi mawisplit mawimerge
"""

import json
import pathlib

import numpy as np

CHUNK = 32


def main() -> int:
    d = pathlib.Path(__file__).resolve().parent.parent / "inputs"
    meta = json.loads((d / "mawi.meta.json").read_text())
    nnz = meta["NNZ"]
    rp = np.fromfile(d / "mawi.rowptr.bin", dtype=np.int32).astype(np.int64)

    big = int(np.diff(rp).argmax())
    split = np.concatenate([rp[: big + 1],
                            np.arange(rp[big], rp[big + 1], CHUNK, dtype=np.int64)[1:],
                            rp[big + 1:]])
    merge = np.arange(0, nnz + CHUNK, CHUNK, dtype=np.int64)
    merge[-1] = nnz

    for name, offsets in (("mawisplit", split), ("mawimerge", merge)):
        assert offsets[0] == 0 and offsets[-1] == nnz and np.all(np.diff(offsets) >= 0)
        offsets.astype(np.int32).tofile(d / f"{name}.rowptr.bin")
        for part in ("col", "val"):
            (d / f"{name}.{part}.bin").write_bytes((d / f"mawi.{part}.bin").read_bytes())
        (d / f"{name}.meta.json").write_text(
            json.dumps({**meta, "NROWS": len(offsets) - 1}, indent=2) + "\n")
        print(f"{name}: {len(offsets) - 1:,} rows, {nnz:,} nonzeros, "
              f"longest row {int(np.diff(offsets).max()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
