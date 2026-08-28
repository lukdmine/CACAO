#!/usr/bin/env python3
"""Fetch SuiteSparse matrices and write the CSR binaries this problem loads.

    python problems/spmv_csr_skew/tools/fetch_matrices.py --all
    python problems/spmv_csr_skew/tools/fetch_matrices.py --matrix transient kron_g500

Writes, into ``problems/spmv_csr_skew/inputs/``:

    <name>.rowptr.bin   int32,   NROWS + 1 entries
    <name>.col.bin      int32,   NNZ entries
    <name>.val.bin      float32, NNZ entries
    <name>.meta.json    the shape, the row-length statistics, the (A, XMOD) palette

and prints the ``cases:`` block to paste into problem.yaml.

Two decisions worth knowing about.

**Symmetry is not expanded.** For a ``symmetric`` Matrix Market file only the stored
triangle is used, which is what CUSP did in the thesis being compared against: its
kron_g500-logn21 count of 91 042 010 and mawi count of 19 020 160 are exactly the stored
entry counts, not the 2x expansions. Expanding would double the nonzeros and change the
row-length distribution, and the published numbers would no longer be comparable.

**The values in the file are discarded and regenerated.** SpMV performance depends on the
sparsity pattern, not on what the numbers are, so the values are rebuilt to make fp32
validation exact instead of approximate:

    values[j] = a_j * 2^-ceil(log2(n_r))     a_j in {1..A},  n_r = nonzeros in j's row
    x[i]      = b_i                          b_i in {1..XMOD}   (generated in inputs.hpp)

with ``A * XMOD * max_nnz_per_row < 2^24``. Every product is then a small integer times a
power of two, every partial sum of a row is an integer multiple of that same power of two
below 2^24, and both are exactly representable in fp32. Summation order stops mattering:
serial, tree, segmented and atomic kernels all return bitwise identical results, so the
validation tolerance is a bug detector rather than a fudge factor. Row sums land in
(0.5, A*XMOD] no matter how long the row is, so one absolute tolerance is equally strict
on a 1-nonzero row and a 7.4-million-nonzero one.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://suitesparse-collection-website.herokuapp.com/MM"

# name -> (SuiteSparse group/matrix, is it one of the cases enabled in problem.yaml)
MATRICES = {
    "transient":  ("Freescale/transient",         True),
    "rail4284":   ("Mittelmann/rail4284",         True),
    "mawi":       ("MAWI/mawi_201512012345",      True),
    "GL7d19":     ("JGD_GL7d/GL7d19",             False),
    "kron_g500":  ("DIMACS10/kron_g500-logn21",   False),
}

# Tried in order; the first whose A*XMOD*max_nnz_per_row stays under 2^24 wins. More
# distinct values means a wrong gather is likelier to produce a wrong answer, so wider
# palettes are preferred; the widest matrices force the narrowest palette.
PALETTES = [(4, 4), (2, 4), (2, 2), (1, 2)]

EXACT_LIMIT = 1 << 24
CHUNK = 1 << 27  # 128 MiB of matrix text per parse call


def _splitmix64(idx: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser. Deterministic, and independent of numpy version."""
    h = (idx.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


def download(group: str, dest: Path) -> Path:
    """Download <group>.tar.gz unless it is already on disk."""
    name = group.split("/")[-1]
    tgz = dest / f"{name}.tar.gz"
    if tgz.exists() and tgz.stat().st_size > 0:
        print(f"  cached  {tgz.name} ({tgz.stat().st_size / 2**20:.1f} MiB)")
        return tgz

    url = f"{BASE}/{group}.tar.gz"
    print(f"  GET     {url}")
    tmp = tgz.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                pct = 100.0 * done / total
                print(f"\r          {done / 2**20:8.1f} / {total / 2**20:.1f} MiB "
                      f"({pct:5.1f}%)", end="", flush=True)
        print()
    tmp.rename(tgz)
    return tgz


def extract_mtx(tgz: Path, dest: Path) -> Path:
    """Pull the matrix's own .mtx out of the tarball.

    Several SuiteSparse archives ship auxiliary vectors beside the matrix -- rail4284
    carries _b, _c, _hi, _lo and _z0 -- so match the basename exactly rather than any
    file ending in .mtx.
    """
    stem = tgz.name[: -len(".tar.gz")]
    with tarfile.open(tgz, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if Path(m.name).name == f"{stem}.mtx"]
        if len(members) != 1:
            raise RuntimeError(f"{tgz.name}: expected {stem}.mtx, found "
                               f"{[m.name for m in tar.getmembers() if m.name.endswith('.mtx')]}")
        member = members[0]
        out = dest / Path(member.name).name
        if out.exists() and out.stat().st_size == member.size:
            print(f"  cached  {out.name} ({out.stat().st_size / 2**20:.1f} MiB)")
            return out
        print(f"  extract {member.name} ({member.size / 2**20:.1f} MiB)")
        src = tar.extractfile(member)
        with open(out, "wb") as f:
            while True:
                block = src.read(1 << 22)
                if not block:
                    break
                f.write(block)
    return out


def read_coordinates(mtx: Path):
    """Return (rows, cols, nnz, row_idx, col_idx) with 0-based int32 indices.

    Streams the file so a 2 GiB .mtx never lands in memory as text. Entry VALUES are
    parsed (they have to be, to find the field boundaries) and thrown away.
    """
    f = open(mtx, "rb")
    banner = f.readline().decode("ascii", "replace").strip()
    parts = banner.lower().split()
    if len(parts) < 5 or parts[0] != "%%matrixmarket" or parts[2] != "coordinate":
        raise RuntimeError(f"{mtx.name}: not a coordinate Matrix Market file: {banner!r}")
    field, symmetry = parts[3], parts[4]
    nfields = {"pattern": 2, "real": 3, "integer": 3, "complex": 4}[field]

    line = f.readline()
    while line.startswith(b"%"):
        line = f.readline()
    nrows, ncols, nnz = (int(t) for t in line.split())
    print(f"  header  {nrows:,} x {ncols:,}, {nnz:,} stored entries "
          f"({field}, {symmetry}; stored triangle used as-is)")

    row_idx = np.empty(nnz, dtype=np.int32)
    col_idx = np.empty(nnz, dtype=np.int32)
    filled = 0
    tail = b""
    while True:
        block = f.read(CHUNK)
        if not block:
            break
        block = tail + block
        cut = block.rfind(b"\n")
        if cut < 0:
            tail = block
            continue
        tail = block[cut + 1:]
        text = block[:cut].decode("ascii", "replace")
        flat = np.fromstring(text, dtype=np.float64, sep=" ")
        if flat.size % nfields:
            raise RuntimeError(f"{mtx.name}: ragged entry block near entry {filled}")
        entries = flat.reshape(-1, nfields)
        n = entries.shape[0]
        if filled + n > nnz:
            raise RuntimeError(f"{mtx.name}: more entries than the header declares")
        row_idx[filled:filled + n] = entries[:, 0].astype(np.int32) - 1
        col_idx[filled:filled + n] = entries[:, 1].astype(np.int32) - 1
        filled += n
        print(f"\r  parse   {filled:,} / {nnz:,} entries", end="", flush=True)
    if tail.strip():
        entries = np.fromstring(tail.decode("ascii", "replace"),
                                dtype=np.float64, sep=" ").reshape(-1, nfields)
        n = entries.shape[0]
        row_idx[filled:filled + n] = entries[:, 0].astype(np.int32) - 1
        col_idx[filled:filled + n] = entries[:, 1].astype(np.int32) - 1
        filled += n
    print()
    f.close()
    if filled != nnz:
        raise RuntimeError(f"{mtx.name}: header says {nnz} entries, read {filled}")
    return nrows, ncols, nnz, row_idx, col_idx


def build_csr(nrows: int, nnz: int, row_idx: np.ndarray, col_idx: np.ndarray):
    """Sort to (row, column) order and produce row_offsets + sorted column indices."""
    if nrows and int(row_idx.max(initial=0)) >= nrows:
        raise RuntimeError("an entry names a row outside the declared dimensions")
    print("  sort    by (row, column)")
    order = np.lexsort((col_idx, row_idx))
    col_sorted = col_idx[order]
    row_sorted = row_idx[order]
    counts = np.bincount(row_sorted, minlength=nrows).astype(np.int64)
    row_offsets = np.empty(nrows + 1, dtype=np.int64)
    row_offsets[0] = 0
    np.cumsum(counts, out=row_offsets[1:])
    assert row_offsets[-1] == nnz
    return row_offsets, col_sorted, counts


def make_values(counts: np.ndarray, nnz: int, palette: tuple[int, int]) -> np.ndarray:
    """values[j] = a_j * 2^-ceil(log2(n_r)); see the module docstring for why."""
    A = palette[0]
    per_entry_len = np.repeat(counts, counts)          # row length, once per nonzero
    assert per_entry_len.size == nnz

    # p = ceil(log2(n)) = #{k in [0, 24) : 2^k < n}. Exact, no floating-point log.
    p = np.zeros(nnz, dtype=np.int32)
    for k in range(25):
        p += (per_entry_len > (np.int64(1) << k))
    scale = np.ldexp(np.ones(nnz, dtype=np.float64), -p)   # exact: a power of two

    a = (_splitmix64(np.arange(nnz, dtype=np.uint64)) % np.uint64(A)) + np.uint64(1)
    values = (a.astype(np.float64) * scale).astype(np.float32)

    # The exactness claim, checked rather than argued: a_j * 2^-p must survive the
    # narrowing to fp32 unchanged, and the largest row must stay inside 2^24 units.
    exact = a.astype(np.float64) * scale
    if not np.array_equal(values.astype(np.float64), exact):
        raise RuntimeError("value construction lost bits in the fp32 narrowing")
    if int(counts.max(initial=0)) * A * palette[1] >= EXACT_LIMIT:
        raise RuntimeError("row sums can exceed 2^24 units; fp32 SpMV would not be exact")
    return values


def choose_palette(max_nnz_row: int) -> tuple[int, int]:
    for A, B in PALETTES:
        if A * B * max(max_nnz_row, 1) < EXACT_LIMIT:
            return A, B
    raise RuntimeError(
        f"longest row holds {max_nnz_row:,} nonzeros; no palette keeps fp32 SpMV exact "
        f"(the limit is 2^24 = {EXACT_LIMIT:,} coefficient-units per row)")


def convert(name: str, group: str, work: Path, out: Path) -> dict:
    print(f"\n=== {name}  ({group}) ===")
    mtx = extract_mtx(download(group, work), work)
    nrows, ncols, nnz, row_idx, col_idx = read_coordinates(mtx)
    row_offsets, col_sorted, counts = build_csr(nrows, nnz, row_idx, col_idx)
    del row_idx, col_idx

    max_row = int(counts.max()) if nrows else 0
    palette = choose_palette(max_row)
    values = make_values(counts, nnz, palette)

    if row_offsets[-1] > np.iinfo(np.int32).max:
        raise RuntimeError(f"{name}: {nnz:,} nonzeros overflow the int32 offsets the "
                           "problem declares")
    (out / f"{name}.rowptr.bin").write_bytes(row_offsets.astype(np.int32).tobytes())
    (out / f"{name}.col.bin").write_bytes(col_sorted.astype(np.int32).tobytes())
    (out / f"{name}.val.bin").write_bytes(values.tobytes())

    traffic = 4 * (nrows + 1) + 8 * nnz + 4 * nrows + 4 * ncols
    meta = {
        "matrix": group,
        "NROWS": nrows,
        "NCOLS": ncols,
        "NNZ": nnz,
        "XMOD": palette[1],
        "value_palette_A": palette[0],
        "max_nnz_per_row": max_row,
        "avg_nnz_per_row": round(nnz / nrows, 4) if nrows else 0.0,
        "empty_rows": int((counts == 0).sum()),
        "compulsory_traffic_MiB": round(traffic / 2**20, 2),
        "roofline_us_at_850GBs": round(traffic / 850e9 * 1e6, 2),
        "symmetry_expanded": False,
    }
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote   {name}.rowptr.bin / .col.bin / .val.bin "
          f"({(4 * (nrows + 1) + 8 * nnz) / 2**20:.1f} MiB)")
    print(f"  stats   avg {meta['avg_nnz_per_row']} / row, max {max_row:,}, "
          f"{meta['empty_rows']:,} empty rows, palette A={palette[0]} XMOD={palette[1]}")
    print(f"  bound   {meta['compulsory_traffic_MiB']} MiB compulsory -> "
          f"{meta['roofline_us_at_850GBs']} us at 850 GB/s")
    return meta


def case_block(name: str, meta: dict) -> str:
    return (f"  - name: {name}\n"
            f"    scalars: {{NROWS: {meta['NROWS']}, NCOLS: {meta['NCOLS']}, "
            f"NNZ: {meta['NNZ']}, XMOD: {meta['XMOD']}}}\n"
            f"    files:\n"
            f"      row_offsets: {name}.rowptr.bin\n"
            f"      col_indices: {name}.col.bin\n"
            f"      values: {name}.val.bin\n")


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", nargs="+", choices=sorted(MATRICES),
                    help="matrices to fetch (default: the ones problem.yaml enables)")
    ap.add_argument("--all", action="store_true",
                    help="fetch every matrix, including the optional extra cases")
    ap.add_argument("--work", type=Path, default=here / "inputs" / ".download",
                    help="where tarballs and .mtx files are cached")
    args = ap.parse_args()

    if args.all:
        wanted = sorted(MATRICES)
    elif args.matrix:
        wanted = args.matrix
    else:
        wanted = [n for n, (_, enabled) in MATRICES.items() if enabled]

    out = here / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    metas = {}
    for name in wanted:
        group, _ = MATRICES[name]
        metas[name] = convert(name, group, args.work, out)

    print("\n" + "=" * 74)
    print("cases: block for problem.yaml (order defines the primary case)\n")
    for name in wanted:
        print(case_block(name, metas[name]), end="")
    print("=" * 74)
    print(f"\nCached downloads are in {args.work} — delete that directory to reclaim "
          "the .tar.gz and .mtx files; the .bin files are what the problem reads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
