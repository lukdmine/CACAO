# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

An **agentic CUDA kernel optimizer**: given a reference CUDA kernel and a `problem.yaml` spec, an LLM agent iteratively rewrites the kernel, benchmarks it with KTT (via a compiled C++ driver), profiles with NCU, and uses the results to guide the next iteration — branching into multiple strategies in parallel.

---

## Running the Backend

```bash
# Activate the conda environment (Python 3.10). setup.sh does not create it —
# it aborts unless a venv or conda env is already active.
conda activate ktt

# Install Python deps (Python 3.10)
pip install -r requirements.txt

# Copy .env and set your LLM provider key
cp .env.example .env   # or create manually
# Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
#             or CERIT_API_KEY + CERIT_API_BASE

# Run the optimizer CLI on a problem directory
python cli.py --dir problems/mmul

# Options (python cli.py --help for the full list)
python cli.py --dir problems/mmul --resume           # resume interrupted run
python cli.py --dir problems/mmul --path-budget 30   # a root-to-leaf path shares 30 (default 20)
python cli.py --dir problems/mmul --path-budget 0 --max-iter 3 --max-depth 1  # depth mode
python cli.py --dir problems/mmul --timeout 300      # tuner wall-clock budget override (seconds)
python cli.py --dir problems/mmul --no-clean         # fresh run without archiving output/
python cli.py --dir problems/mmul --finalize --finalize-timeout 300   # finalize an existing run's output
python cli.py --dir problems/mmul --no-finalize     # skip the automatic finalize at the end of a run
python cli.py --dir problems/mmul --clone mmul_v2    # clone the problem and exit
python cli.py --dir problems/mmul --best             # show results (needs no API key)
python cli.py --dir problems/mmul --prune-archives   # shrink archive/output_* on disk

# Start the FastAPI backend server (for frontend)
python server.py              # binds 0.0.0.0:8003 — --host 127.0.0.1 to restrict
python server.py --port 8080
```

`libktt.so` must be symlinked into the project root for the driver to link. KTT is
pinned to **v2.3.1** in `setup.sh`; `utils/framework.py` generates C++ against that
API, so the pin is a compatibility contract, not a preference. The Python bindings
(`pyktt`) are not built and not used — `setup.sh` deliberately omits `--python`.

KTT 2.3 added a `Timestamp` field to every serialised `KernelResult` and reads it back
with `j.at("Timestamp")`, which throws rather than defaulting. Only profile mode hands
a `results.json` back to KTT (`Tuner::LoadResults`), and only its own iteration's, so a
fresh run never hits it — but resuming a run tuned before the upgrade does, and the
driver aborts with `[json.exception.out_of_range.403]`. `utils/results.ensure_results_loadable`
injects the missing field before `nodes/profile.py` launches the driver.

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

Run this whenever `state/types.py`, `models/strategy.py`, `models/decision.py` or
`models/finalize.py` change; those are the only four the generator reads.
`models/inputs.py` and `models/cases.py` are not covered; their TypeScript counterparts
are hand-written in `frontend/src/api/client.ts`. `models/regions.py` never crosses HTTP
and has none.

---

## Adding a New Problem

Create a directory under `problems/` with:
- `problem.yaml` — GPU index, grid, reference (type/function/file), validation tolerance
- `inputs.yaml` — the I/O boundary spec (scalars, buffers, which buffers validate); `inputs.hpp` is generated from it by `utils/inputs.py`. A buffer with `init: file` + `file_name` reads a raw little-endian binary of its dtype from `inputs/<file_name>` (byte count must equal `size * sizeof(dtype)`); the UI upload puts files there via `POST /api/problems/{name}/inputs/{buffer}`. Buffer
  `dtype` is `int`, `float` or `half`; scalars are `int` or `float` only. A `half` buffer is
  `half_float::half` host-side (KTT vendors it) and `__half` in the kernel, which must
  `#include <cuda_fp16.h>` — cacao emits only the host spelling. A `cpu_c` reference works
  too (g++ builds `ref_cpu.c` as C++ with `-I KTT/Source`) but must include
  `<Utility/External/half.hpp>`. **A validated half buffer is compared BIT-EXACTLY and
  `validation.tolerance` is ignored for it** — `ResultValidator` dispatches on
  `std::is_floating_point`, false for the class `half_float::half`, so half takes the
  integral `!=` overload. Pair a half output with a `cuda` reference doing identical
  arithmetic, or keep the validated output `float`. There is no bf16: KTT's
  `ArgumentDataType` has no equivalent
- the reference implementation: `ref_kernel.cu` (`reference.type: cuda`), `ref_cpu.c` (`reference.type: cpu_c`, a C function linked into the driver — pointer args only, scalars as `-D` macros), **or** `ref.py` (`reference.type: python`, `f(scalars, buffers)`; extra imports like `torch` are optional per-problem deps — `pip install torch` — checked at engine start)

See `docs/PROBLEM_YAML_GUIDE.md`, `docs/INPUTS_YAML_GUIDE.md` and
`docs/REFERENCE_IMPLEMENTATION_GUIDE.md` for format docs.

---

## Architecture

### Core Data Flow

```
cli.py
  └─ engine/master.py          # Phase 1: analyze → strategize → dispatch; Phase 3 hook at the end
       ├─ engine/finalize.py   # Phase 3: re-tune the best variants, widen toggles, check archs (nodes/finalize_*)
       ├─ nodes/analyze.py     # LLM: analyze problem + reference kernel
       ├─ nodes/strategize.py  # LLM: generate N parallel optimization strategies
       └─ engine/worker.py     # Phase 2: per-branch loop (4 parallel async coroutines)
            ├─ nodes/plan.py         # LLM: detailed implementation plan
            ├─ nodes/author.py       # LLM tool loop: kernels.cu + framework regions, with compile check
            ├─ nodes/implement.py    # LLM: write kernels.cu           (AGENTIC_STEPS=False only)
            ├─ nodes/configure.py    # LLM: fill framework.cpp regions (AGENTIC_STEPS=False only)
            ├─ nodes/run.py          # subprocess: compile + run the KTT driver, get timing
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
| `output/branches/<name>/branch_config.json` | `BranchConfig` | Settings the frontend owns (`max_iter`) — see the writer rule below |
| `output/branches/<name>/iterN/state.json` | `IterState` | Per-iteration work products (kernel, params, results, decision) |

Nodes receive a `WorkingState` (composed from all three), then the worker decomposes it back after each step. See `state/types.py` for full field definitions.

**One writer per file.** The worker holds `BranchManifest` in memory for the duration
of a node — an LLM call or a tuner run — and rewrites `branch.json` wholesale
afterwards. So anything the user can edit while a branch runs must live in
`branch_config.json`: a setting kept on the manifest gets reverted by that write
minutes later. The worker picks it up fresh in `_compose_working_state`, alongside the
re-reads of `problem.yaml` and the reference kernel.

`branch_config.json` is created once by the optimizer process, in
`engine/master._init_branch`, and after that only the server writes it —
`api/branches.py`'s config endpoint and the budget grant in change-decision, which is
where `grant_one_more_iteration` is now called from. The revert/continue surgery in
`state/control.py` no longer grants anything itself, so nothing inside a running worker
can overwrite the file. The worker's one remaining write is `_migrate_legacy_max_iter`,
create-if-absent via `os.link`, which cannot land on top of a budget the user just
set.

### Iteration Status Machine

Each iteration progresses through these statuses in order:
`planning → authoring → running → profiling → proposing → deciding → decided`

After `deciding`, `IterState.next_status` — written to `iterN/state.json`, not to the
manifest — drives the branch-level outcome. The worker copies it into `manifest.status`:
- `continue` → increment `current_iter`, new iteration starts at `implementing` (planning runs once per branch); `configuring` if `skip_implement`
- `retry` → increment `current_iter`, new iteration starts at `implementing` with the previous decision/feedback/run_output carried over (fix_errors prompt); `configuring` if `skip_implement`
- `branch` → master spawns sub-strategies (up to `MAX_BRANCH_DEPTH`)
- `stop/success/failed` → terminal

**Path-budget mode is the default.** `config.PATH_BUDGET` is 20 and both entry points
follow it: `cli.py`'s `--path-budget` defaults to that value, and `api/schemas.RunConfig`
reads `MAX_ITERATIONS` / `MAX_BRANCH_DEPTH` / `PATH_BUDGET` from `config` rather than
restating them, so a UI run and `python cli.py` on the same problem run the same budget.
A whole root-to-leaf path therefore shares 20 iterations, `--max-iter` is ignored (the
CLI warns when a non-default one is passed), and `MAX_BRANCH_DEPTH` still bounds nesting.
`--path-budget 0`, or `PATH_BUDGET = 0`, selects depth mode, where `MAX_ITERATIONS` is
the per-branch budget.

**A raised budget propagates.** `engine/master._spawn_sub_branches` sizes children from
the parent's *effective* `max_iter` — the value in `branch_config.json`, which the UI
can raise and which the server's `grant_one_more_iteration` raises when an exhausted
branch is revived — not from `config`. In path-budget mode the grant extends the path
rather than coming out of the children's share: `BranchManifest.path_budget_total`
carries the budget that path actually runs on, starting at `PATH_BUDGET` and growing by
whatever an ancestor was granted beyond its allocated share. It defaults to 0, read as
"use `config.PATH_BUDGET`", so pre-existing manifests keep working.

**`implementing` and `configuring` are dispatch aliases.** `decide.py` and its prompt
still emit them, and `engine/worker.py:_dispatch_status` routes both to
`nodes/author.py`, which does the work of both in one tool loop. The scope distinction
rides on `IterState.authoring_scope` (`skip_implement` → `"config_only"`). Nothing
writes `authoring` to disk, so state files and the frontend see the statuses they
always did — which is why they still appear above. With `AGENTIC_STEPS=False` the two
statuses reach `implement.py` and `configure.py` instead, and mean what they say.

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
| Staged workspace (`.staging/`) | `agentic/workspace.py` |
| Offline replay of a recorded step | `agentic/replay.py` |
| NVRTC check (ctypes → `libnvrtc.so`) | `utils/nvrtc.py` |
| `results.json` queries | `utils/landscape.py` |
| Cross-branch reads | `state/crossbranch.py` |
| Per-problem kernel rules | `utils/rules.py` |

The LLM owns four files — `kernels.cu` plus `region_kernels.cpp` / `region_params.cpp`
/ `region_launcher.cpp` — and never the framework skeleton, which is spliced by
`assemble_framework_cpp`. `configure_node` writes the same three region files, so the
`AGENTIC_STEPS=False` path leaves an iteration in the same shape.
`utils/framework.extract_regions` is the inverse splice: it seeds a step from an
iteration that predates the region files, and `nodes/run._load_regions` uses it to
recover the three bodies out of `framework.cpp` when those files are absent — without
which the reassembled driver would reference `kernel` without ever declaring it.

Knobs in `config.py`:
- `AGENTIC_STEPS` — kill switch back to the two single-shot calls. The only route to
  them; nothing selects them at runtime.
- `STEP_TOOL_BUDGET` — tool calls per step (50); a runaway guard, not a working limit.
- `STEP_TRUNCATION_RETRIES` — truncated replies corrected before the step gives up (3).
- `CROSS_BRANCH_ACCESS` — `errors` / `log` / `index` / `off`. **No level exposes
  another branch's kernel or framework regions.** Parallel branches are parallel bets;
  a branch that can read the leader's code converges on it. Failure analyses prune dead
  ends without supplying a solution, which is the only sharing that pays.

**A turn with no tool call is corrected, not surrendered to.** There are two of them
and they need opposite advice:

- **Truncation** — the reply hit the provider's output token cap before reaching a
  call. `llm.tracked.was_truncated` spots it from `finish_reason` (`length` or
  `max_tokens`, matched case-insensitively so all four providers are covered) or from a
  reply with no content, no call and no stated reason. `agentic/loop.py` and
  `TrackedLLM` share that one predicate instead of keeping a copy each. The model is
  told it was cut off and given an escalating correction (`_TRUNCATION_NUDGES`) that
  shrinks the unit of work. After `STEP_TRUNCATION_RETRIES` the step ends `TRUNCATED`.
- **A stall** — real prose, no call. One nudge.

Thinking models spend the whole cap reasoning about a "write the kernel" prompt often
enough that mistaking it for "this provider cannot use tools" takes the tool loop
offline for the rest of a run.

**A step that cannot author its files fails the iteration.** `nodes/author._fail_iteration`
writes an `[AUTHORING FAILED]` diagnosis to `run_output` and routes to `deciding` — the
same route `implement_node` takes when its call comes back empty, and `decide` reads it
as a generation failure and retries with a smaller ask. It never falls back to the
single-shot calls; with `AGENTIC_STEPS=False` the worker dispatches those directly and
`author_node` is not on the path at all. A step that changed no file has authored
nothing, even when its workspace compiles: `end_step` refuses, and so does the salvage
path that commits an aborted-but-complete workspace. The workspace is seeded with the
previous iteration's files, so committing them untouched re-benchmarks a kernel that
was already measured and spends a budget slot on it.

**`bind_tools` never raises.** For every provider here it attaches the schemas locally
and returns — cerit is `ChatOpenAI` with a custom `base_url`, so a server without tool
support is only discovered by calling it. It surfaces as `BIND_UNSUPPORTED`, which
fails the iteration like any other aborted step.

**Watch out:** a tool-call response has empty `content` by construction. `TrackedLLM.ainvoke`
returns early on `response.tool_calls` — without that, its empty-content retry path
fires five times with backoff on every single tool call.

### Finalize Step

After the branches drain, `engine/master.run_optimization_engine` calls
`engine.finalize.run_finalize(trigger="auto")` when `config.FINALIZE_AUTO` is on and
`should_auto_finalize` agrees: no engine task raised and the monitor did not shut down
with work still queued. A run that ended because its workers died is not a finished
run, so it is not finalized.

`FINALIZE_AUTO` is on by default, so every run that ends by itself finalizes. That
costs up to `FINALIZE_MAX_FINALISTS` x `FINALIZE_TUNER_TIMEOUT` of GPU time per case
(4 x 1000 s as shipped), plus the LLM calls: the selection, one widen step per
finalist and the overview. `--no-finalize`, or `finalize: false` in a UI `RunConfig`,
turns it off for one run. `--finalize-timeout` overrides `FINALIZE_TUNER_TIMEOUT` (per
finalist per case); the other knobs are `FINALIZE_ARCHS` and `FINALIZE_MAX_FINALISTS`.

`--finalize` runs the step alone on an existing `output/`. It refuses while
`output/run.pid` names a live process, and holds that file itself while it runs, so
the API will not start a run or a second finalize over the same manifest. A normal CLI
run still writes no `run.pid` (only the API's runs do), so one in another terminal is
not detected.

**One writer, and never `branches/`.** `output/finalize/finalize.json` is written only
by the finalize process; the API reads it. Everything finalize produces lives under
`output/finalize/`, so a finished run's branches stay byte-for-byte as the workers left
them. `run_finalize` never raises: a failure is recorded as `status="failed"`, and the
next call resumes each finalist from the stage it reached.

### Kernel Rules

`problem.yaml` takes a `rules:` block constraining what a kernel may do — the mechanism
that stops a branch winning by quietly dropping to fp16 accumulation, which still
validates whenever `validation.tolerance` is loose enough.

`text` rules reach every prompt that writes a kernel — `prompts/author.py`,
`prompts/implement.py` and `prompts/fix_errors.py` all render `ctx["rules_block"]` — so
both authoring paths carry them. `forbid` regexes are machine-checked, but only inside
the agentic step (`agentic/tools.py`, before the compile check): **with
`AGENTIC_STEPS=False` nothing enforces them.** A pattern that does not compile is
reported as a violation rather than skipped, so a typo'd regex fails the check loudly
instead of silently disarming the rule it was meant to be. See
`docs/PROBLEM_YAML_GUIDE.md`.

### Tests

```bash
conda activate ktt
python -m pytest tests/ -q                       # ~15 s with recorded run output on disk
python -m pytest tests/ -m "not integration" -q  # skip the real g++/NVRTC link
```

No live LLM calls: `agentic/replay.ScriptedLLM` drives the loop deterministically, and
the same class replays a recorded `step_trace.jsonl`. A scripted turn is a list of
tool calls, a string (prose, no calls), or a dict carrying response metadata —
`{"text": ..., "finish_reason": "length"}` is how a truncated reply is scripted.

The `integration` marker covers the real toolchain — it links the driver against
`libktt.so`, *runs* it, and NVRTC-compiles recorded kernels whose real outcome is
known. Tests needing only `g++` guard on `shutil.which`, so the fast suite is green
on a machine with no compiler.

Many tests replay recorded run data under `problems/*/output/` (e.g.
`problems/covariance/output/`), which is gitignored. When that data is not on disk those
tests skip cleanly, so the passing count depends on what is checked out. Note `problems/*/inputs.hpp` is gitignored — the engine regenerates
it from `inputs.yaml` at every run start (`utils/inputs.ensure_inputs_hpp`), and a test
that reads one (e.g. `test_nvrtc.py`) needs a problem whose header has been generated.

### LLM Provider Configuration

The LLM stack lives in `llm/`: `llm/providers.py` (roster and resolution),
`llm/tracked.py` (the wrappers), `llm/claude.py` (structured output for the Claude
models that reject a forced tool call), `llm/retry.py`, `llm/tokens.py`. `config.py`
re-exports most of it for existing call sites, but defines none of it. All LLM calls go
through `TrackedLLM` / `TrackedStructuredLLM`, which retry with exponential backoff and
track token usage.

Supported providers: `openai`, `anthropic`, `gemini`, `cerit` (OpenAI-compatible endpoint).

Provider selection, highest priority first (same order as README):

1. `--provider <name>` CLI flag (or `provider` in a UI run config)
2. `LLM_PROVIDER` hardcoded in `llm/providers.py` (`None` by default — leave as-is to fall through).
   It has to be that module's global: `_detect_provider` reads it there, so assigning
   `config.LLM_PROVIDER` does nothing, and `config.py` no longer re-exports the name.
3. `LLM_PROVIDER` env var in `.env`
4. Auto-detect from whichever API key is set. When multiple are set, this order wins: **CERIT > Anthropic > OpenAI > Gemini**.

`claude` and `google` are accepted as aliases; any other unrecognised value warns and
falls through to key detection, on both the `--provider` and the env-var route. The
model is set via `--model`; otherwise the provider's default from
`MODELS` in `llm/providers.py` is used.

### Output Directory Layout

A fresh run **archives** the previous one instead of deleting it —
`utils/files.archive_output_dir` renames `output/` to `archive/output_N` (N = highest
existing + 1) and starts with an empty `output/`, carrying `reference_time.json`
across. It used to `rmtree`; the frontend's Run button and `cli.py` without `--resume`
both land there and neither asked first, which cost a 23-hour run. A rename is O(1) at
any size, so there is no cost argument for deleting. A directory with no `branches/`
never held work and is removed rather than numbered.

`python cli.py --dir <problem> --prune-archives` strips only regenerable files from
archived runs (`cacao_in_*.bin`, `cacao_ref_*.bin`, `driver`, `__pycache__`) — ~99% of
an archive's size. Kernels, driver regions, results, decisions, step traces and
`.staging/` are never touched, and `output/` is never touched at all. `.staging/`
survives on disk only when an authoring step aborted before committing, which makes it
the only copy of that attempt. Nothing prunes automatically.

```
problems/<name>/output/
├── context.json                   # shared problem context
├── final_results.json             # best result summary
├── finalize/                      # finalize.json (FinalizeManifest) plus one directory per finalist
└── branches/
    └── <strategy_name>/
        ├── branch.json            # BranchManifest (worker-owned)
        ├── branch_config.json     # BranchConfig (server-owned; see the writer rule)
        ├── control/               # queued stop/revert/message signals, one file each
        ├── iter1/
        │   ├── state.json         # IterState
        │   ├── kernels.cu         # generated kernel(s)
        │   ├── region_kernels.cpp # LLM-authored CACAO:KERNELS region
        │   ├── region_params.cpp  # LLM-authored CACAO:PARAMS region
        │   ├── region_launcher.cpp# LLM-authored CACAO:LAUNCHER region
        │   ├── framework.cpp      # assembled KTT C++ driver (engine skeleton + LLM regions)
        │   ├── inputs.hpp         # copied from the problem dir, staged beside the driver
        │   ├── step_trace.jsonl   # every turn of the authoring loop
        │   ├── driver             # compiled driver binary
        │   ├── cacao_in_*.bin     # tuner input dumps (regenerable)
        │   ├── results.json       # timing results
        │   ├── tuner_output.txt   # KTT stdout/stderr
        │   ├── ncu_profile.csv    # NCU metrics (if profiled)
        │   └── .staging/          # uncommitted authoring work; the only copy if a step aborts
        └── branches/              # sub-branches (recursive)
```

A problem declaring several `cases:` puts each case's `inputs.hpp`, `framework.cpp`,
`driver`, `results.json` and `tuner_output.txt` in `iterN/case_<name>/` instead of at
the iteration root; `kernels.cu` and the region files stay at the root, shared.
(`tuner_output.txt` was missing from this list, which is how `grep_tuner_output` came
to look for it at the root and fail on every multi-case problem.) One case — which
is every problem with no `cases:` block — keeps the layout above exactly.
`utils.cases.primary_case_dir` is what run-level readers anchor on.

### Server + Frontend

`server.py` is a thin entry point that creates the FastAPI app via `api.create_app()`. The server is split into an `api/` package:

| Module | Responsibility |
|--------|---------------|
| `api/__init__.py` | App factory, CORS, router mounting |
| `api/schemas.py` | Pydantic request/response models |
| `api/helpers.py` | Shared state (`_running_problems`, `RunEntry`), utilities |
| `api/problems.py` | Problem CRUD (list, create, update, delete, detail, logs) |
| `api/tree.py` | Tree scanning (with a light `finalize` summary), status (with a per-run `phase`), and GPU endpoints |
| `api/optimizer.py` | `/run`, `/stop`, `/resume` with per-GPU locking |
| `api/branches.py` | Branch control (stop, resume, message, change-decision, config, iterations, delete); all but iterations answer 409 while finalize runs |
| `api/finalize.py` | `POST`/`GET /api/problems/{name}/finalize`: start a finalize-only process, report its state (`not_run`, `finalizing`, `stopped`, `failed`, `stale`, `up_to_date`) |

Multiple problems can run concurrently on different GPUs (per-GPU `RunEntry` registry).

Frontend stack: React 19 + Vite + TypeScript + shadcn/ui + Tailwind v4 + React Flow (`@xyflow/react`) + Zustand. Tree layout uses `dagre`.
