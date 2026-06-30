# CACAO — CUDA Agentic Coding Autotuning Optimizer

CACAO is an LLM-driven system that searches for fast CUDA kernel implementations across multiple strategy branches in parallel. Given a problem specification (`problem.yaml`) and a reference implementation (a CUDA kernel or a sequential C/C++ function), the agent iteratively writes a CUDA kernel and a tuning parameter space, lets the [KTT](https://github.com/HiPerCoRe/KTT) autotuner search that space, profiles the best configuration with NVIDIA NCU, and uses the timings + profiler counters to propose the next iteration.

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
```

`--dir` may be relative or absolute — the optimizer can be invoked from any working directory.

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
    |-- nodes/implement.py          # LLM: write optimized kernel.cu
    |-- nodes/configure.py          # LLM: write KTT tuning params
    |-- nodes/run.py                # subprocess: run pyktt autotuner
    |-- nodes/profile.py            # subprocess: run NCU profiler
    |-- nodes/propose.py            # LLM: analyze results, propose changes
    |-- nodes/decide.py             # LLM: continue / retry / branch / stop
```

### State Model

State is split into three typed Pydantic models:

| File | Model | Purpose |
|------|-------|---------|
| `output/context.json` | `Context` | Shared problem data (written once) |
| `output/branches/<name>/branch.json` | `BranchManifest` | Branch identity, control, results |
| `output/branches/<name>/iter_N/state.json` | `IterState` | Per-iteration work products |

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
- `problem.yaml` — kernel interface, scalars, vectors, validation tolerance
- One reference: either `ref_kernel.cu` (CUDA reference kernel) or `ref_cpu.c` (sequential C/C++ reference, compiled at runtime via GCC)

See [docs/PROBLEM_YAML_GUIDE.md](docs/PROBLEM_YAML_GUIDE.md) and [docs/REFERENCE_IMPLEMENTATION_GUIDE.md](docs/REFERENCE_IMPLEMENTATION_GUIDE.md) for format details.

Or use the web UI's "New Problem" dialog.

## Project Structure

```
cuda-agentic-optimizer/
|-- cli.py                  # CLI entry point
|-- server.py               # FastAPI server entry point
|-- config.py               # LLM providers, constants, token tracking
|-- tuner.py                # Python KTT tuner (pyktt bindings)
|-- requirements.txt        # Python dependencies
|-- pyktt.so -> KTT/...     # Symlink to KTT Python bindings
|-- libktt.so -> KTT/...    # Symlink to KTT core library
|
|-- engine/                 # Orchestration (master + worker)
|-- nodes/                  # LLM and subprocess nodes (10 nodes)
|-- state/                  # Pydantic models, persistence, control
|-- models/                 # Structured output schemas
|-- utils/                  # GPU info, logging, file I/O, resume
|-- api/                    # FastAPI server modules
|-- prompts/                # LLM prompt templates (one per node)
|-- problems/               # Problem definitions (problem.yaml + ref_kernel.cu)
|
|-- frontend/               # React + Vite + TypeScript + shadcn/ui
|-- KTT/                    # Kernel Tuning Toolkit (build dependency)
|-- docs/                   # Problem format guides
|
|-- SETUP.md                # First-time setup guide
```

## License

See [LICENSE](LICENSE).
