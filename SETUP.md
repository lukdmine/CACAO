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

### Step 2: Build KTT with Python Support

If `pyktt.so` and `libktt.so` are not already built:

```bash
cd KTT

export CUDA_PATH=/usr/local/cuda
export PYTHON_HEADERS=/path/to/miniconda3/envs/ktt/include/python3.10
export PYTHON_LIB=/path/to/miniconda3/envs/ktt/lib/libpython3.10.so

./premake5 gmake --python
cd Build
make config=release_x86_64 Ktt -j$(nproc)

ls x86_64_Release/*.so   # should show pyktt.so and libktt.so
```

### Step 3: Create Symlinks

From the project root:

```bash
ln -sf KTT/Build/x86_64_Release/pyktt.so pyktt.so
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
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(pwd):$LD_LIBRARY_PATH

python -c "import pyktt; print('pyktt OK')"
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

## Running the Tuner Standalone

```bash
python tuner.py --working-dir ./problems/mmul/output/branches/my_branch/iter1
```

Or with all options:

```bash
python tuner.py --platform 0 --device 0 --problem ./problem.yaml --params ./params.json --output ./results
```

## Troubleshooting

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

### ModuleNotFoundError: No module named 'pyktt'
Ensure symlinks exist in the project root:
```bash
ls -la pyktt.so libktt.so
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
