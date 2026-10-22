# library_bench — time a vendor library on a problem's real inputs, seed the baseline

One command per (problem, GPU) pair:

```
python -m tools.library_bench.bench problems/<name> --op <op>
```

This times a vendor-library implementation (cuSPARSE, cuDNN, ...) on the
problem's *real* input data, verifies the result against the problem's
reference kernel, and seeds `problems/<name>/output/reference_time.json` —
the plain, first-write-wins slot the engine already uses for baselines
(ggml, torch/python). From then on every tuned run's `speedup` is against
the library, in `--best`, `final_results.json`, and the web UI. The engine
never learns a library was involved.

## Workflow

1. Bench once per (problem, GPU): the command above.
2. Run the tuning engine as usual (`cli.py ...`) — the seeded baseline is
   picked up automatically.
3. Re-bench (`--force`) when the problem definition or the GPU changes.
   `python -m tools.library_bench.bench <problem> --check` reports a stale
   baseline (it fingerprints the input scalars + input blobs and the GPU).
4. A run on a problem without a seeded baseline silently uses KTT's coarse
   reference time — the expected-baseline comment in the problem's
   `problem.yaml` plus `--check` are the guards against that.

Timing discipline (symmetric with the engine's re-measure mode): CUDA events
around the op call only — no H2D/D2H, no descriptor setup — warmup 20, then
min over 100 reps (median on stderr). The op's algorithm sweep happens
un-timed in setup, so "library at its best" is exact, not guessed.

## Verification gate

Before seeding, the bench runs the problem's own `ref_kernel.cu` on the same
inputs and compares the op's output elementwise within
`validation.tolerance`. On mismatch it prints the worst index and values and
refuses to seed — benchmarking the wrong math is the one failure mode that
would invalidate everything downstream. For calibrating a new problem's
tolerance, use `--tolerance <x>` and read the reported worst diff.

## Adding a problem

Nothing. Any framework-mode problem with `reference.type: cuda` and exactly
one validated buffer works — the tool derives the whole boundary (buffers,
scalars, ref-kernel launch) from `inputs.yaml`/`problem.yaml`. v1 limits:
`reference.type: cuda` only, exactly one validated buffer.

## Adding a library

One file, `ops/<name>.cu`, implementing the contract in `op_contract.h`:

- `op_setup(spec)` — H2D, descriptors, algorithm sweep (untimed)
- `op_run()` — the single timed call
- `op_output(dst)` — copy the validated result out (drives the gate)
- `op_teardown()`

Inputs arrive as host buffers by name (the real dumped `cacao_in_*` data)
plus all scalars. If the build needs extra include dirs or libraries
(`-lcusparse`, cudnn_frontend paths, ...), put them in
`ops/<name>.flags.txt`, whitespace-separated, `#` comments allowed, `$VARS`
expanded from the environment (an unset var is a clear error).

`ops/passthrough.cu` (a plain device memcpy) is the reference implementation
and the smoke test: `pytest tests/test_library_bench.py`.

## Notes

- Runs hold two advisory flocks (`/tmp/cacao_bench_gpu<N>.lock` +
  per-problem, fixed acquire order) so concurrent invocations serialize for
  both timing fidelity and scratch-dir safety. The engine's own GPU lock is
  in-process and cannot interoperate across processes — do not run a tuning
  session on the same GPU at the same time.
- Scratch lives in `<problem>/output/_bench/` (gitignored with `output/`):
  generated launcher, dump tool, binaries, gate outputs.
- The cuDNN op additionally needs cuDNN >= 9 and the `cudnn_frontend` headers
  (https://github.com/NVIDIA/cudnn-frontend releases, matching the installed
  cuDNN major). Known-good combination: `pip install nvidia-cudnn-cu12`
  (9.27) + frontend v1.31.0 + CUDA toolkit 12.9. Set three env vars before
  benching (see the comment in `ops/cudnn_conv2d.flags.txt`):
  `CUDNN_FRONTEND_INCLUDE`, `CUDNN_INCLUDE`, `CUDNN_LIB`. These are a
  user-managed prerequisite of the op, never of the engine.
