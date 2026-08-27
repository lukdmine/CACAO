# Setup Guide

## Quick Setup (Recommended)

```bash
# 1. Create and activate a Python environment
conda create -n ktt python=3.10 -y
conda activate ktt

# 2. Run the setup script
./setup.sh
```

The setup script handles: dependency installation, CUDA detection, KTT build,
symlink creation, and environment verification. See below for manual steps
if you prefer.

## Manual Setup

### Prerequisites

- **CUDA Toolkit 12.0+** installed at `/usr/local/cuda`
- **NVIDIA GPU** with CUDA support
- **Miniconda/Anaconda** for Python environment management
- **LLM API Key** — one of: Anthropic (Claude), OpenAI, Google (Gemini), or CERIT

## Detailed Setup

### Step 1: Create Python Environment

```bash
# Python 3.10 is required — 3.11+ breaks pybind11 in KTT
conda create -n ktt python=3.10 -y
conda activate ktt
pip install -r requirements.txt
```

### Step 2: Build KTT (C++ library only)

If `libktt.so` is not already built:

```bash
# KTT is pinned to a release tag. utils/framework.py generates C++ against this
# API and the driver is recompiled against these headers every iteration, so the
# pin is a compatibility contract — do not build master.
git clone https://github.com/HiPerCoRe/KTT.git
cd KTT && git checkout v2.3.1

# For a system-installed CUDA (nvcc in /usr/bin, headers in /usr/include), use /usr.
# For a dedicated install, use its root, e.g. /usr/local/cuda.
export CUDA_PATH=/usr

./premake5 gmake
cd Build
make config=release_x86_64 Ktt -j$(nproc)

ls x86_64_Release/*.so   # should show libktt.so
```

> **Do not pass `--python`.** The framework driver is a pure C++ KTT client and
> does not use the Python bindings. Building with them compiles pybind11 into
> `libktt.so` itself, which leaves an unresolvable `libpython` dependency —
> every driver link then fails with ~155 `undefined reference to 'Py*'` errors.
> If you have an existing build made with `--python`, you must delete the stale
> objects (`rm -rf Build/x86_64_Release/obj`) before rebuilding; regenerating
> the makefile alone will silently relink the old Python objects.

### Step 3: Create Symlink

From the project root:

```bash
ln -sf KTT/Build/x86_64_Release/libktt.so libktt.so
```

### Step 4: Configure LLM Provider

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then set one of:

```bash
# Claude (default)
ANTHROPIC_API_KEY=your-key-here

# Or OpenAI
# OPENAI_API_KEY=your-key-here

# Or Google Gemini
# GOOGLE_API_KEY=your-key-here

# Or CERIT (OpenAI-compatible endpoint)
# CERIT_API_KEY=your-key-here
# CERIT_API_BASE=https://your-cerit-endpoint
```

The provider is auto-detected from which key is set.

### Step 5: Verify Setup

```bash
conda activate ktt

# The check that matters: a C++ KTT driver must link. This is what the optimizer
# does every iteration (utils/build.py). It fails if libktt.so was built with --python.
printf '#include <Ktt.h>\nint main() { return 0; }\n' > /tmp/link_test.cpp
g++ -std=c++17 -I"$(pwd)/KTT/Source" /tmp/link_test.cpp "$(pwd)/libktt.so" \
    -Wl,-rpath,"$(pwd)" -o /tmp/link_test && echo "KTT C++ link OK"

python -c "from state import *; print('State module OK')"
python -c "from config import OptimizerConfig; print('Config OK')"
```

## Running the Optimizer (CLI)

```bash
conda activate ktt
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(pwd):$LD_LIBRARY_PATH

python cli.py --dir problems/mmul
```

Options:

```bash
python cli.py --dir problems/mmul --resume         # resume interrupted run
python cli.py --dir problems/mmul --max-iter 3 --max-depth 1
python cli.py --dir problems/mmul --best           # show results without running
python cli.py --dir problems/mmul --provider anthropic --model claude-opus-4-7
```

## Running the Server + Frontend

```bash
# Terminal 1: Backend API server
conda activate ktt
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(pwd):$LD_LIBRARY_PATH
python server.py                    # http://localhost:8003

# Terminal 2: Frontend dev server
cd frontend
npm install
npm run dev                         # http://localhost:5003
```

The frontend polls `http://localhost:8003`. Override ports via `PORT` / `FRONTEND_PORT` / `VITE_API_BASE` env vars (see `.env.example`). Multiple problems can run concurrently on different GPUs.

## Running a Framework Driver Standalone

Each iteration compiles `framework.cpp` into a `driver` binary and runs it. To do
that by hand for a generated iteration:

```bash
ITER=./problems/mmul/output/branches/my_branch/iter_1

g++ -std=c++17 -m64 -O3 -I"$(pwd)/KTT/Source" \
    "$ITER/framework.cpp" "$(pwd)/libktt.so" -Wl,-rpath,"$(pwd)" -o "$ITER/driver"

# driver <platform> <device> <duration_s> <tolerance> <output_base> <kernels.cu> <ref_kernel.cu>
cd "$ITER" && ./driver 0 0 20 1.0 results kernels.cu ../../../../ref_kernel.cu
```

KTT writes `results.json` (it appends the extension), parsed by `utils/results.py`.

## Troubleshooting

### undefined reference to `Py...` when linking the driver
`libktt.so` was built with the Python bindings (`--python`). The framework driver
is pure C++ and does not need them. Rebuild without the flag — and delete the
stale objects first, or make will silently relink the old Python ones:

```bash
cd KTT && CUDA_PATH=/usr ./premake5 gmake
cd Build && rm -rf x86_64_Release/obj
make config=release_x86_64 Ktt -j$(nproc)
```

### ImportError: libnvrtc.so not found
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### ImportError: libktt.so not found
```bash
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
```

### GLIBCXX version not found
Use Python 3.10 (not 3.11+):
```bash
conda create -n ktt python=3.10 -y
```

## Environment Variables Summary

```bash
# Required (one LLM provider)
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, or CERIT_API_KEY + CERIT_API_BASE

# Required for runtime
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(pwd):$LD_LIBRARY_PATH

# For building KTT (one-time)
export CUDA_PATH=/usr/local/cuda
export PYTHON_HEADERS=/path/to/python3.10/include
export PYTHON_LIB=/path/to/libpython3.10.so
```
