# CACAO — CUDA Agentic Coding Autotuning Optimizer

CACAO is an LLM-driven system that searches for fast CUDA kernel implementations across multiple strategy branches in parallel. Given a problem specification (`problem.yaml` + `inputs.yaml`) and a reference implementation (a CUDA kernel, a sequential C function, or a Python function), the agent iteratively writes a CUDA kernel and a tuning parameter space, lets the [KTT](https://github.com/HiPerCoRe/KTT) autotuner search that space, profiles the best configuration with NVIDIA NCU, and uses the timings + profiler counters to propose the next iteration.

## Quick Start

> **First-time setup?** See [SETUP.md](SETUP.md) for building KTT, creating the conda env, installing dependencies, and configuring CUDA + your LLM API key. The steps below assume that setup is complete.

```bash
# 1. Activate the conda environment
conda activate ktt

# 2. Run the optimizer
python cli.py --dir problems/mmul
```

### CLI Options

```bash
python cli.py --dir problems/mmul --resume         # resume an interrupted run
python cli.py --dir problems/mmul --max-iter 5 --max-depth 2
python cli.py --dir problems/mmul --best           # show results without running
python cli.py --dir problems/mmul --prune-archives # shrink archived runs on disk
```

`--dir` may be relative or absolute — the optimizer can be invoked from any working directory.

A fresh run **archives** the previous one rather than deleting it: `output/` is renamed
to `archive/output_N` and the run starts with an empty `output/`. Nothing prunes
automatically; `--prune-archives` strips only regenerable files (tuner input dumps,
compiled drivers) from `archive/`, never from `output/`, and never kernels or results.

## Smoke test

With `CERIT_API_KEY` and `CERIT_API_BASE` set in `.env`:

```bash
python cli.py --dir problems/mmul --provider cerit --model qwen3.5 --max-iter 2 --max-depth 1
```

## Web UI

A visual dashboard for tracking, controlling, and inspecting optimization runs in real-time.

```bash
# Terminal 1: Backend API
python server.py                    # http://localhost:8003

# Terminal 2: Frontend
cd frontend && npm install && npm run dev   # http://localhost:5003
```

Features: tree visualization of optimization branches, live log streaming, llm outputs, NCU metrics, problem creation dialog.

## Security model

CACAO is a research tool intended for use within a **trusted private network**. The HTTP API (`server.py`) is unauthenticated — anyone who can reach port 8003 can create problems, run optimizations, and read or delete files inside `problems/`. The engine compiles and executes LLM-generated CUDA code under the server user's identity.

**Do not expose port 8003 to the public internet, and do not run CACAO on `problem.yaml` files from untrusted sources.** See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the full discussion.

## Architecture

```
cli.py / server.py
    |
engine/master.py                    # Phase 1: analyze -> strategize -> dispatch
    |-- nodes/analyze.py            # LLM: understand the reference kernel
    |-- nodes/strategize.py         # LLM: generate N parallel strategies
    |
engine/worker.py                    # Phase 2: per-branch optimization loop
    |-- nodes/plan.py               # LLM: detailed implementation plan
    |-- nodes/author.py             # LLM tool loop: kernel + driver regions, compile-checked
    |     \-- nodes/implement.py    # LLM: write kernels.cu          (AGENTIC_STEPS=False only)
    |     \-- nodes/configure.py    # LLM: fill framework.cpp regions (AGENTIC_STEPS=False only)
    |-- nodes/run.py                # subprocess: compile + run the KTT C++ driver
    |-- nodes/profile.py            # subprocess: run NCU profiler
    |-- nodes/propose.py            # LLM: analyze results, propose changes
    |-- nodes/decide.py             # LLM: continue / retry / branch / stop
```

When every branch is done, `cli.py` writes `output/final_results.json` — the best
configuration and a per-branch summary. `nodes/merge.py` rebuilds the same summary
from what is already on disk and backs `--best`.

`author.py` runs the kernel and its KTT driver regions as one tool loop with an NVRTC
compile check inside the step, so a compile error costs a retry rather than a whole
iteration. `implement.py` + `configure.py` are the two single-shot calls it replaced;
`AGENTIC_STEPS=False` in `config.py` is the only thing that selects them.

### State Model

State is split into typed Pydantic models, one writer per file:

| File | Model | Purpose |
|------|-------|---------|
| `output/context.json` | `Context` | Shared problem data (written once) |
| `output/branches/<name>/branch.json` | `BranchManifest` | Branch identity, control, results (worker-owned) |
| `output/branches/<name>/branch_config.json` | `BranchConfig` | Settings the UI owns (`max_iter`); the worker only reads it |
| `output/branches/<name>/iterN/state.json` | `IterState` | Per-iteration work products |

Each iteration progresses: `planning -> implementing -> configuring -> running -> profiling -> proposing -> deciding -> decided`

After deciding, the branch can `continue`, `retry`, `branch` (spawn sub-strategies), or `stop`.

### LLM Providers

Supports **Anthropic** (Claude), **OpenAI**, **Gemini** (Google), and **CERIT** (OpenAI-compatible endpoint). All calls go through `TrackedLLM` wrappers with retry and token tracking.

Provider selection, highest priority first:

1. `--provider <name>` CLI flag
2. `LLM_PROVIDER` value hardcoded in `config.py` (`None` by default — leave as-is to fall through)
3. `LLM_PROVIDER` env var in `.env`
4. Auto-detect from whichever API key is set in `.env`. When multiple are set, this order wins: **CERIT > Anthropic > OpenAI > Gemini**.

The model is set via `--model`; otherwise the provider's default model is used (see `MODELS` in `config.py`).

## Adding a Problem

Create a directory under `problems/` with:
- `problem.yaml` — GPU index, grid, reference, validation tolerance, and an optional `rules:` block constraining what a kernel may do
- `inputs.yaml` — the I/O boundary (scalars, buffers, which buffers are validated); `inputs.hpp` is generated from it
- One reference: `ref_kernel.cu` (CUDA), `ref_cpu.c` (a C function linked into the driver), or `ref.py` (Python, e.g. a torch or numpy oracle)

See [docs/PROBLEM_YAML_GUIDE.md](docs/PROBLEM_YAML_GUIDE.md) and [docs/REFERENCE_IMPLEMENTATION_GUIDE.md](docs/REFERENCE_IMPLEMENTATION_GUIDE.md) for format details.

Or use the web UI's "New Problem" dialog.

## License

See [LICENSE](LICENSE).
