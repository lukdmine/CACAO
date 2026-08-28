# spmv_csr_skew

`y = A*x` on raw CSR, one launch, nothing precomputed — on the three matrices where the
published CUSP autotuner (Bráblík 2024) loses to cuSPARSE.

Full analysis: [`docs/local/spmv/THESIS_ANALYSIS.md`](../../docs/local/spmv/THESIS_ANALYSIS.md).

## Setup

The matrices are ~450 MiB of binaries and are not committed. Build them once:

```bash
conda activate ktt
python problems/spmv_csr_skew/tools/fetch_matrices.py          # transient, rail4284, mawi
python problems/spmv_csr_skew/tools/fetch_matrices.py --all    # + GL7d19, kron_g500
```

That downloads from SuiteSparse, keeps the stored triangle as-is (no symmetric
expansion, matching what CUSP read in the thesis), sorts to `(row, column)` order, and
regenerates the values so fp32 SpMV on them is bitwise order-independent — see §7 of the
analysis. It prints the `cases:` block for `problem.yaml` and writes a `.meta.json` per
matrix. Downloads are cached under `inputs/.download/`; delete it to reclaim ~600 MiB.

## Run

```bash
python cli.py --dir problems/spmv_csr_skew
```

Three cases are tuned per iteration (60 s + 90 s + 180 s by default). Any case that
fails to validate fails the iteration.

## Measurement tools

```bash
make -C problems/spmv_csr_skew/tools

tools/bench_cusparse                  # the incumbent, all matrices, ALG_DEFAULT/ALG1/ALG2
tools/floors mawi 18571154 19020160 18571154   # what each ingredient costs alone
tools/row_partition_control.py        # mawi with only its row partition changed
```

`bench_cusparse` also recomputes `y` on the CPU and reports any row that differs, so it
doubles as a check that the binaries are intact.

## Baselines

`output/reference_time.json` is seeded, not measured at run time — the warp-per-row
oracle in `ref_kernel.cu` is a correctness reference, not a competitor.

| case | seeded baseline | source |
|---|---|---|
| `transient` | 18.43 µs | `bench_cusparse` on this RTX 3090, CUDA 12.0 |
| `rail4284` | 121.86 µs | same |
| `mawi` | 1100 µs | thesis Table B.1b, cuSPARSE 12.4 on an RTX 3080 |

`mawi` is the exception because local cuSPARSE 12.0 takes 14 155 µs on that shape while
12.4 takes 1056 µs on a slower card — a version difference, not a hardware one, and
seeding the local number would report 30x for a mediocre kernel. Re-run
`bench_cusparse` and replace it if this machine moves past CUDA 12.4.

For orientation: a correct `mawi` kernel has roughly 400–500 µs of unavoidable traffic,
against 1056 µs for the best published cuSPARSE and 2702 µs for the thesis's own tuned
kernel. `transient` and `rail4284` are already within ~20% of their stream cost under
cuSPARSE — they are there to stop a `mawi`-shaped kernel from regressing the ordinary
case, not to be won.
