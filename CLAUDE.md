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
- `problem.yaml` — GPU index, grid, reference (type/function/file), validation tolerance
- `inputs.yaml` — the I/O boundary spec (scalars, buffers, which buffers validate); `inputs.hpp` is generated from it by `utils/inputs.py`
- the reference implementation: `ref_kernel.cu` (`reference.type: cuda`), `ref_cpu.c` (`reference.type: cpu_c`, a C function linked into the driver — pointer args only, scalars as `-D` macros), **or** `ref.py` (`reference.type: python`, `f(scalars, buffers)`; extra imports like `torch` are optional per-problem deps — `pip install torch` — checked at engine start)

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
            ├─ nodes/author.py       # LLM tool loop: kernels.cu + framework regions, with compile check
            │    └─ falls back to ↓ when the provider can't drive tools
            ├─ nodes/implement.py    # LLM: write kernels.cu          (single-shot fallback)
            ├─ nodes/configure.py    # LLM: fill framework.cpp regions (single-shot fallback)
            ├─ nodes/run.py          # subprocess: run pyktt tuner, get timing
            ├─ nodes/profile.py      # subprocess: run ncu profiler
            ├─ nodes/propose.py      # LLM: analyze results, propose next changes
            └─ nodes/decide.py       # LLM: continue / retry / branch / stop
```

### State Architecture

State is split into three typed Pydantic models, plus one input file:

| File | Model | Purpose |
|------|-------|---------|
| `output/context.json` | `Context` | Shared problem data (written once, read by all branches) |
| `output/branches/<name>/branch.json` | `BranchManifest` | Branch identity, control, aggregated results |
| `output/branches/<name>/branch_config.json` | `BranchConfig` | Settings the frontend owns (`max_iter`) — API writes, worker only reads |
| `output/branches/<name>/iter_N/state.json` | `IterState` | Per-iteration work products (kernel, params, results, decision) |

Nodes receive a `WorkingState` (composed from all three), then the worker decomposes it back after each step. See `state/types.py` for full field definitions.

**One writer per file.** The worker holds `BranchManifest` in memory for the duration of a node — an LLM call or a tuner run — and rewrites `branch.json` wholesale afterwards. So anything the user can edit while a branch runs must live in `branch_config.json`, which the worker never writes: a setting kept on the manifest gets reverted by that write minutes later. The worker picks the file up in `_compose_working_state`, alongside the fresh reads of `problem.yaml` and the reference kernel.

### Iteration Status Machine

Each iteration progresses through these statuses in order:
`planning → implementing → configuring → running → profiling → proposing → deciding → decided`

After `deciding`, `next_status` on the manifest drives the branch-level outcome:
- `continue` → increment `current_iter`, new iteration starts at `implementing` (planning runs once per branch); `configuring` if `skip_implement`
- `retry` → increment `current_iter`, new iteration starts at `implementing` with the previous decision/feedback/run_output carried over (fix_errors prompt); `configuring` if `skip_implement`
- `branch` → master spawns sub-strategies (up to `MAX_BRANCH_DEPTH`)
- `stop/success/failed` → terminal

**`implementing` and `configuring` are dispatch aliases now.** With `AGENTIC_STEPS`
on, `engine/worker.py:_dispatch_status` routes both to `nodes/author.py`, which does
the work of both in one tool loop. `decide.py` and its prompt still emit the old
statuses and are untouched; the scope distinction rides on `IterState.authoring_scope`
(`skip_implement` → `"config_only"`). Nothing writes `authoring` to disk, so state
files and the frontend see the same statuses they always did.

### Agentic Authoring Step

`nodes/author.py` runs the kernel and its driver regions as one tool loop instead of
two single-shot calls. It exists because a compile error otherwise costs a full
iteration — four LLM calls and one of the user's `max_iter` slots — and because the
kernel cannot be NVRTC-compiled without the `#define`s the params region declares, so
checking it before `configure` runs is impossible.

| Piece | Where |
|---|---|
| Loop driver, budget, `step_trace.jsonl` | `agentic/loop.py` |
| Tool schemas + dispatch + preconditions | `agentic/tools.py` |
| Staged workspace (`.staging/`, commit on `end_step`) | `agentic/workspace.py` |
| Offline replay of a recorded step | `agentic/replay.py` |
| NVRTC check (ctypes → `libnvrtc.so`) | `utils/nvrtc.py` |
| `results.json` queries | `utils/landscape.py` |
| Cross-branch reads | `state/crossbranch.py` |
| Per-problem kernel rules | `utils/rules.py` |

The LLM owns four files — `kernels.cu` plus `region_kernels.cpp` / `region_params.cpp`
/ `region_launcher.cpp` — and never the framework skeleton, which is spliced by
`assemble_framework_cpp` as before. `utils/framework.extract_regions` is its inverse,
used to seed a step from an iteration that predates the region files.

Knobs in `config.py`:
- `AGENTIC_STEPS` — kill switch back to the two single-shot calls.
- `STEP_TOOL_BUDGET` — tool calls per step (50); a runaway guard, median use is 12.
- `CROSS_BRANCH_ACCESS` — `errors` / `log` / `index` / `off`. **No level exposes
  another branch's kernel or framework regions.** Parallel branches are parallel bets;
  a branch that can read the leader's code converges on it. Failure analyses prune dead
  ends without supplying a solution, which is the only sharing that pays.

The loop falls back to `implement_node` + `configure_node` when the model emits no
tool calls on turn 1, the provider errors, or the budget runs out with files missing.
Worst case is the previous behaviour.

**`bind_tools` never raises.** For every provider here it attaches the schemas locally
and returns — cerit is `ChatOpenAI` with a custom `base_url`, so a server without tool
support is only discovered by calling it. `agentic/capability.py` remembers that
verdict per `(provider, model)` for the run and skips the loop after two consecutive
capability failures, so the discovery cost is paid once instead of on every iteration
of all four branches. A completed step clears the count. The registry is process-global,
so tests reset it via an autouse fixture.

**Watch out:** a tool-call response has empty `content` by construction. `TrackedLLM.ainvoke`
returns early on `response.tool_calls` — without that, its empty-content retry path
fires five times with backoff on every single tool call.

### Kernel Rules

`problem.yaml` takes a `rules:` block constraining what a kernel may do — the
mechanism that stops a branch winning by quietly dropping to fp16 accumulation, which
still validates whenever `validation.tolerance` is loose enough. `text` rules go in the
prompt; `forbid` regexes are checked before the compiler runs and fail the check. See
`docs/PROBLEM_YAML_GUIDE.md`.

### Tests

```bash
conda activate ktt
python -m pytest tests/ -q                       # 148 tests, ~9 s
python -m pytest tests/ -m "not integration" -q  # skip the real g++/NVRTC link
```

No live LLM calls: `agentic/replay.ScriptedLLM` drives the loop deterministically, and
the same class replays a recorded `step_trace.jsonl`. The `integration` marker covers
the real toolchain — it links the driver against `libktt.so` and NVRTC-compiles
recorded kernels whose real outcome is known.

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
        ├── branch.json            # BranchManifest (worker-owned)
        ├── branch_config.json     # BranchConfig (frontend-owned)
        ├── iter1/
        │   ├── state.json         # IterState
        │   ├── kernels.cu         # generated kernel(s)
        │   ├── framework.cpp      # assembled KTT C++ driver (engine skeleton + LLM regions)
        │   ├── driver             # compiled driver binary
        │   ├── results.json       # timing results
        │   ├── tuner_output.txt   # KTT stdout/stderr
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
