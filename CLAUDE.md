# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

An **agentic CUDA kernel optimizer**: given a reference CUDA kernel and a `problem.yaml` spec, an LLM agent iteratively rewrites the kernel, benchmarks it with `pyktt` (the KTT autotuner), profiles with NCU, and uses the results to guide the next iteration — branching into multiple strategies in parallel.

---

## Running the Backend

```bash
# Activate the conda environment (required for pyktt/KTT compatibility)
conda activate ktt

# Install Python deps (requires Python 3.10 for pyktt compatibility)
pip install -r requirements.txt

# Copy .env and set your LLM provider key
cp .env.example .env   # or create manually
# Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
#             or CERIT_API_KEY + CERIT_API_BASE

# Run the optimizer CLI on a problem directory
python cli.py --dir problems/mmul

# Options
python cli.py --dir problems/mmul --resume       # resume interrupted run
python cli.py --dir problems/mmul --max-iter 3 --max-depth 1
python cli.py --dir problems/mmul --best         # show results without running

# Start the FastAPI backend server (for frontend)
python server.py              # http://localhost:8003
python server.py --port 8080
```

`pyktt.so` must be symlinked into the project root for the tuner to work.

---

## Running the Frontend

```bash
cd frontend

npm install
npm run dev       # dev server at http://localhost:5003
npm run build     # production build
npm run lint      # ESLint
npm run preview   # preview production build
```

The frontend polls `http://localhost:8003` — start the Python server first.

---

## Type Generation (Frontend <-> Backend Sync)

The TypeScript API types are auto-generated from Python Pydantic models:

```bash
python frontend/scripts/generate_types.py
```

Run this whenever `state/types.py` or `models/` Pydantic models change.

---

## Adding a New Problem

Create a directory under `problems/` with:
- `problem.yaml` — defines GPU hardware, kernel interface, scalars, vectors, validation tolerance
- `ref_kernel.cu` — reference CUDA implementation that the optimizer benchmarks against

See `docs/PROBLEM_YAML_GUIDE.md` and `docs/REFERENCE_IMPLEMENTATION_GUIDE.md` for format docs.

---

## Architecture

### Core Data Flow

```
cli.py
  └─ engine/master.py          # Phase 1: analyze → strategize → dispatch
       ├─ nodes/analyze.py     # LLM: analyze problem + reference kernel
       ├─ nodes/strategize.py  # LLM: generate N parallel optimization strategies
       └─ engine/worker.py     # Phase 2: per-branch loop (4 parallel async coroutines)
            ├─ nodes/plan.py         # LLM: detailed implementation plan
            ├─ nodes/implement.py    # LLM: write kernel.cu
            ├─ nodes/configure.py    # LLM: write params.json for KTT tuner
            ├─ nodes/run.py          # subprocess: run pyktt tuner, get timing
            ├─ nodes/profile.py      # subprocess: run ncu profiler
            ├─ nodes/propose.py      # LLM: analyze results, propose next changes
            └─ nodes/decide.py       # LLM: continue / retry / branch / stop
```

### State Architecture

State is split into three typed Pydantic models:

| File | Model | Purpose |
|------|-------|---------|
| `output/context.json` | `Context` | Shared problem data (written once, read by all branches) |
| `output/branches/<name>/branch.json` | `BranchManifest` | Branch identity, control, aggregated results |
| `output/branches/<name>/iter_N/state.json` | `IterState` | Per-iteration work products (kernel, params, results, decision) |

Nodes receive a `WorkingState` (composed from all three), then the worker decomposes it back after each step. See `state/types.py` for full field definitions.

### Iteration Status Machine

Each iteration progresses through these statuses in order:
`planning → implementing → configuring → running → profiling → proposing → deciding → decided`

After `deciding`, `next_status` on the manifest drives the branch-level outcome:
- `continue` → increment `current_iter`, start `planning` again
- `retry` → restart current iter at `implementing`
- `branch` → master spawns sub-strategies (up to `MAX_BRANCH_DEPTH`)
- `stop/success/failed` → terminal

### LLM Provider Configuration

Provider is set via `LLM_PROVIDER` in `config.py` (default: `"claude"`). Auto-detected from env vars if not set. All LLM calls go through `TrackedLLM` / `TrackedStructuredLLM` wrappers in `config.py` that handle retries with exponential backoff and token usage tracking.

Supported providers: `openai`, `anthropic`, `gemini`, `cerit` (OpenAI-compatible endpoint).

### Output Directory Layout

```
problems/<name>/output/
├── context.json                   # shared problem context
├── final_results.json             # best result summary
└── branches/
    └── <strategy_name>/
        ├── branch.json            # BranchManifest
        ├── iter_1/
        │   ├── state.json         # IterState
        │   ├── kernel.cu          # generated kernel
        │   ├── params.json        # KTT tuner config
        │   ├── results.json       # timing results
        │   └── ncu_profile.csv    # NCU metrics (if profiled)
        └── branches/              # sub-branches (recursive)
```

### Server + Frontend

`server.py` is a thin entry point that creates the FastAPI app via `api.create_app()`. The server is split into an `api/` package:

| Module | Responsibility |
|--------|---------------|
| `api/__init__.py` | App factory, CORS, router mounting |
| `api/schemas.py` | Pydantic request/response models |
| `api/helpers.py` | Shared state (`_running_problems`, `RunEntry`), utilities |
| `api/problems.py` | Problem CRUD (list, create, update, delete, detail, logs) |
| `api/tree.py` | Tree scanning, results, status, GPU endpoints |
| `api/optimizer.py` | `/run`, `/stop`, `/resume` with per-GPU locking |
| `api/branches.py` | Branch control (stop, resume, message, revert, config, delete) |

Multiple problems can run concurrently on different GPUs (per-GPU `RunEntry` registry).

Frontend stack: React 19 + Vite + TypeScript + shadcn/ui + Tailwind v4 + React Flow (`@xyflow/react`) + Zustand. Tree layout uses `dagre`.
