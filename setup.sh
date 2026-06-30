#!/usr/bin/env bash
#
# setup.sh — First-time setup for the CUDA Agentic Kernel Optimizer
#
# Automates: venv check, pip install, CUDA detection, KTT clone/build,
# symlink creation, cuda_env.yaml generation, verification.
#
# Usage:
#   ./setup.sh                  # full setup
#   ./setup.sh --skip-ktt       # skip KTT clone/build
#   ./setup.sh --skip-requirements  # skip pip install
#   ./setup.sh --reconfigure    # only regenerate cuda_env.yaml
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Parse flags ─────────────────────────────────────────────────────
SKIP_KTT=false
SKIP_REQUIREMENTS=false
RECONFIGURE=false

for arg in "$@"; do
    case "$arg" in
        --skip-ktt)          SKIP_KTT=true ;;
        --skip-requirements) SKIP_REQUIREMENTS=true ;;
        --reconfigure)       RECONFIGURE=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [--skip-ktt] [--skip-requirements] [--reconfigure]"
            echo ""
            echo "  --skip-ktt          Skip KTT clone and build"
            echo "  --skip-requirements Skip pip install -r requirements.txt"
            echo "  --reconfigure       Only regenerate cuda_env.yaml (skip everything else)"
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================"
echo "  CUDA Agentic Optimizer — Setup"
echo "========================================"
echo ""

# ── Step 1: Check virtual environment ───────────────────────────────
info "Step 1: Checking Python virtual environment..."

VENV_NAME=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV_NAME="$(basename "$VIRTUAL_ENV")"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    VENV_NAME="$CONDA_DEFAULT_ENV"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    VENV_NAME="$(basename "$CONDA_PREFIX")"
fi

if [ -z "$VENV_NAME" ]; then
    error "No Python virtual environment detected."
    echo ""
    echo "Please create and activate one before running this script."
    echo ""
    echo "  Recommended (conda):"
    echo "    conda create -n ktt python=3.10 -y"
    echo "    conda activate ktt"
    echo ""
    echo "  Alternative (venv):"
    echo "    python3 -m venv .venv"
    echo "    source .venv/bin/activate"
    echo ""
    echo "Then re-run: ./setup.sh"
    exit 1
fi

echo ""
info "Detected virtual environment: ${BLUE}$VENV_NAME${NC}"
read -rp "Use this environment? [Y/n] " USE_ENV
USE_ENV="${USE_ENV:-Y}"

if [[ ! "$USE_ENV" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please activate the desired environment and re-run ./setup.sh"
    echo ""
    echo "  Recommended (conda):"
    echo "    conda create -n ktt python=3.10 -y"
    echo "    conda activate ktt"
    echo ""
    echo "  Alternative (venv):"
    echo "    python3 -m venv .venv"
    echo "    source .venv/bin/activate"
    exit 0
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
success "Using Python $PYTHON_VERSION in environment '$VENV_NAME'"

# If --reconfigure, skip to cuda_env.yaml generation
if $RECONFIGURE; then
    info "Reconfigure mode — skipping to environment detection..."
    # Jump to step 7 equivalent
    info "Detecting CUDA environment..."
    python3 -c "
from utils.cuda_env import detect, format_summary, save_config
from pathlib import Path
env = detect(Path('.'))
print(format_summary(env))
for w in env.warnings:
    print(f'[WARN] {w}')
for e in env.errors:
    print(f'[ERROR] {e}')
save_config(env, Path('.'))
print()
print('cuda_env.yaml written.')
"
    exit 0
fi

# ── Step 2: Install Python requirements ─────────────────────────────
if $SKIP_REQUIREMENTS; then
    info "Step 2: Skipping pip install (--skip-requirements)"
else
    info "Step 2: Installing Python requirements..."
    pip install -r requirements.txt
    success "Requirements installed"
fi

# ── Step 3: Detect CUDA installation ────────────────────────────────
info "Step 3: Detecting CUDA installation..."

CUDA_DIR=""

# Check env vars
for VAR in CUDA_PATH CUDA_HOME CUDA_ROOT; do
    VAL="${!VAR:-}"
    if [ -n "$VAL" ] && [ "$VAL" != "/usr" ] && [ -d "$VAL" ] && [ -f "$VAL/bin/nvcc" ]; then
        CUDA_DIR="$VAL"
        success "Found CUDA via \$$VAR: $CUDA_DIR"
        break
    fi
done

# Check nvcc on PATH
if [ -z "$CUDA_DIR" ]; then
    NVCC_PATH=$(command -v nvcc 2>/dev/null || true)
    if [ -n "$NVCC_PATH" ]; then
        NVCC_REAL=$(readlink -f "$NVCC_PATH")
        CANDIDATE_DIR="$(dirname "$(dirname "$NVCC_REAL")")"
        # Reject /usr — means CUDA is system-installed with no single root
        if [ "$CANDIDATE_DIR" != "/usr" ] && [ -d "$CANDIDATE_DIR/include" ]; then
            CUDA_DIR="$CANDIDATE_DIR"
            success "Found CUDA via nvcc on PATH: $CUDA_DIR"
        fi
    fi
fi

# Check common locations
if [ -z "$CUDA_DIR" ]; then
    for CANDIDATE in /usr/local/cuda /usr/local/cuda-12* /opt/cuda /opt/cuda-*; do
        if [ -d "$CANDIDATE" ] && [ -f "$CANDIDATE/bin/nvcc" ]; then
            CUDA_DIR="$CANDIDATE"
            success "Found CUDA at: $CUDA_DIR"
            break
        fi
    done
fi

if [ -z "$CUDA_DIR" ]; then
    # Check if CUDA is system-installed (nvcc on PATH but no dedicated root)
    if command -v nvcc &>/dev/null; then
        CUDA_DIR=""  # No root dir, but tools are available
        NVCC_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "unknown")
        success "CUDA $NVCC_VERSION (system-installed, no dedicated CUDA root)"
    else
        error "CUDA Toolkit not found!"
        echo ""
        echo "  Searched: \$CUDA_PATH, \$CUDA_HOME, \$CUDA_ROOT, PATH,"
        echo "            /usr/local/cuda*, /opt/cuda*"
        echo ""
        echo "  Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
        echo "  Then set: export CUDA_PATH=/path/to/cuda"
        echo "  And re-run: ./setup.sh"
        exit 1
    fi
else
    NVCC_VERSION=$("$CUDA_DIR/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "unknown")
    success "CUDA $NVCC_VERSION at $CUDA_DIR"
fi

# ── Step 4: Check optional tools ────────────────────────────────────
info "Step 4: Checking optional tools..."

# ncu
NCU_PATH=$(command -v ncu 2>/dev/null || true)
if [ -z "$NCU_PATH" ] && [ -n "$CUDA_DIR" ] && [ -f "$CUDA_DIR/bin/ncu" ]; then
    NCU_PATH="$CUDA_DIR/bin/ncu"
fi
if [ -n "$NCU_PATH" ]; then
    success "ncu found: $NCU_PATH"
else
    warn "ncu (Nsight Compute) not found — profiling will be skipped"
    echo "  Install: https://developer.nvidia.com/nsight-compute"
fi

# nvidia-smi
NVIDIA_SMI_PATH=$(command -v nvidia-smi 2>/dev/null || true)
if [ -n "$NVIDIA_SMI_PATH" ]; then
    success "nvidia-smi found: $NVIDIA_SMI_PATH"
else
    warn "nvidia-smi not found — GPU detection will be limited"
fi

# gcc
GCC_PATH=$(command -v gcc 2>/dev/null || true)
if [ -n "$GCC_PATH" ]; then
    GCC_VER=$(gcc --version 2>/dev/null | head -1 || echo "unknown")
    success "gcc found: $GCC_PATH ($GCC_VER)"
else
    error "gcc not found — required for compiling CPU reference implementations"
    echo "  Install: sudo apt install gcc  (or equivalent for your distro)"
    exit 1
fi

# ── Step 5: Clone & build KTT ───────────────────────────────────────
if $SKIP_KTT; then
    info "Step 5: Skipping KTT build (--skip-ktt)"
else
    info "Step 5: Setting up KTT..."

    # Clone (pinned to last known working commit — upstream master has broken Python bindings)
    KTT_COMMIT="931c4157"
    if [ -d "KTT" ]; then
        success "KTT directory already exists — skipping clone"
    else
        info "Cloning KTT..."
        git clone https://github.com/HiPerCoRe/KTT.git
        cd KTT && git checkout "$KTT_COMMIT" && cd ..
        success "KTT cloned (pinned to $KTT_COMMIT)"
    fi

    # Auto-detect Python headers and lib
    PYTHON_HEADERS=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
    PYTHON_LIBDIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    PYTHON_LDLIB=$(python3 -c "import sysconfig; v=sysconfig.get_config_var('LDLIBRARY'); print(v if v else '')")

    if [ -z "$PYTHON_LDLIB" ]; then
        # Fallback: look for libpython3.X.so in LIBDIR
        PYTHON_LDLIB="libpython${PYTHON_VERSION}.so"
    fi
    PYTHON_LIB="$PYTHON_LIBDIR/$PYTHON_LDLIB"

    # If the reported lib doesn't exist (e.g. conda gave us .a), search for .so
    if [ ! -f "$PYTHON_LIB" ]; then
        # Try common alternative locations
        SEARCH_DIRS=("$PYTHON_LIBDIR" "${CONDA_PREFIX:-/nonexistent}/lib")
        for SDIR in "${SEARCH_DIRS[@]}"; do
            for CANDIDATE_LIB in "$SDIR/libpython${PYTHON_VERSION}.so" "$SDIR/libpython${PYTHON_VERSION}m.so"; do
                if [ -f "$CANDIDATE_LIB" ]; then
                    PYTHON_LIB="$CANDIDATE_LIB"
                    break 2
                fi
            done
        done
    fi

    info "Python headers: $PYTHON_HEADERS"
    info "Python lib:     $PYTHON_LIB"

    if [ ! -f "$PYTHON_LIB" ]; then
        warn "Python shared library (.so) not found at $PYTHON_LIB"
        echo ""
        if [ -n "${CONDA_PREFIX:-}" ]; then
            echo "  For conda, reinstall Python with the shared library enabled:"
            echo "    conda install -c conda-forge python=${PYTHON_VERSION} --force-reinstall"
            echo ""
            echo "  This installs the .so shared library needed by KTT's Python bindings."
            echo "  (The default conda python may only include the static .a library.)"
        else
            echo "  Install the Python dev package:"
            echo "    sudo apt install python${PYTHON_VERSION}-dev"
        fi
        echo ""
        error "Cannot build KTT without Python shared library. Fix the above and re-run."
        exit 1
    fi

    # Build
    info "Building KTT with Python support..."
    cd KTT

    # KTT's build system needs CUDA_PATH to find libraries.
    # For system-installed CUDA (no dedicated root), /usr works.
    export CUDA_PATH="${CUDA_DIR:-/usr}"
    export PYTHON_HEADERS
    export PYTHON_LIB

    PREMAKE_VERSION="5.0.0-beta8"

    # Function to download premake5 pre-built binary
    download_premake5() {
        PREMAKE_URL="https://github.com/premake/premake-core/releases/download/v${PREMAKE_VERSION}/premake-${PREMAKE_VERSION}-linux.tar.gz"
        if command -v curl &>/dev/null; then
            curl -sL "$PREMAKE_URL" -o premake5.tar.gz
        elif command -v wget &>/dev/null; then
            wget -q "$PREMAKE_URL" -O premake5.tar.gz
        else
            return 1
        fi
        tar xzf premake5.tar.gz
        rm -f premake5.tar.gz
        chmod +x premake5
    }

    # Function to build premake5 from source (fallback for old glibc)
    build_premake5_from_source() {
        info "Building premake5 from source (this may take a minute)..."
        rm -rf premake5-build-tmp
        mkdir premake5-build-tmp && cd premake5-build-tmp

        # Clone the specific tag (more reliable than downloading zip from GitHub releases)
        if git clone --depth 1 --branch "v${PREMAKE_VERSION}" https://github.com/premake/premake-core.git src 2>/dev/null; then
            info "Cloned premake5 source"
        else
            error "Failed to clone premake5 source"
            cd ..
            rm -rf premake5-build-tmp
            return 1
        fi

        cd src
        if make -f Bootstrap.mak linux -j"$(nproc)"; then
            info "premake5 compiled"
        else
            error "premake5 compilation failed"
            cd ../..
            rm -rf premake5-build-tmp
            return 1
        fi

        if [ ! -f bin/release/premake5 ]; then
            error "premake5 binary not found after build"
            cd ../..
            rm -rf premake5-build-tmp
            return 1
        fi

        cp bin/release/premake5 ../../premake5
        chmod +x ../../premake5
        cd ../..
        rm -rf premake5-build-tmp
    }

    if [ ! -f premake5 ]; then
        info "premake5 not found — downloading pre-built binary..."
        if download_premake5; then
            success "premake5 ${PREMAKE_VERSION} downloaded"
        else
            warn "Failed to download premake5 binary"
        fi
    fi

    # Test if premake5 binary works (may fail on old glibc)
    PREMAKE_OK=false
    if [ -f premake5 ] && ./premake5 --version &>/dev/null; then
        PREMAKE_OK=true
    fi

    if ! $PREMAKE_OK; then
        warn "Pre-built premake5 binary is incompatible with this system (likely old glibc)"
        echo "  Falling back to building premake5 from source..."
        rm -f premake5

        if ! command -v make &>/dev/null; then
            error "make is required to build premake5 from source"
            echo "  Install: sudo apt install build-essential"
            cd "$SCRIPT_DIR"
            exit 1
        fi
        if ! command -v unzip &>/dev/null; then
            error "unzip is required to build premake5 from source"
            echo "  Install: sudo apt install unzip"
            cd "$SCRIPT_DIR"
            exit 1
        fi

        # premake5's Linux build links against libuuid (src/host/os_uuid.c).
        # On stripped-down systems the uuid/uuid.h header is often missing.
        # Try conda-forge libuuid (no sudo required) when we're in a conda env.
        if ! echo '#include <uuid/uuid.h>' | cc -E -x c - &>/dev/null; then
            info "uuid/uuid.h not found — libuuid is required to build premake5"
            if [ -n "${CONDA_PREFIX:-}" ] && command -v conda &>/dev/null; then
                info "Installing libuuid from conda-forge into \$CONDA_PREFIX..."
                if conda install -y -c conda-forge libuuid &>/dev/null; then
                    success "libuuid installed via conda-forge"
                else
                    error "conda install libuuid failed"
                    echo "  Try manually: conda install -c conda-forge libuuid"
                    cd "$SCRIPT_DIR"
                    exit 1
                fi
            else
                error "libuuid development headers are required (uuid/uuid.h)"
                echo "  With sudo:     sudo apt install uuid-dev"
                echo "  With conda:    conda install -c conda-forge libuuid"
                cd "$SCRIPT_DIR"
                exit 1
            fi
        fi

        # Make conda-provided headers/libs visible to cc and the bootstrap binary.
        if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/include/uuid/uuid.h" ]; then
            export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
            export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
            export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi

        if build_premake5_from_source; then
            success "premake5 ${PREMAKE_VERSION} built from source"
        else
            error "Failed to build premake5 from source"
            echo "  You can try installing premake5 manually:"
            echo "    sudo apt install premake5"
            echo "  Or build it: https://premake.github.io/download"
            cd "$SCRIPT_DIR"
            exit 1
        fi
    fi

    ./premake5 gmake --python
    cd Build
    make config=release_x86_64 Ktt -j"$(nproc)"
    cd "$SCRIPT_DIR"

    # Verify build
    if [ -f "KTT/Build/x86_64_Release/pyktt.so" ] && [ -f "KTT/Build/x86_64_Release/libktt.so" ]; then
        success "KTT built successfully"
    else
        error "KTT build did not produce expected .so files"
        echo "  Check KTT/Build/ for build output"
        exit 1
    fi

    # Create symlinks
    info "Creating symlinks..."
    ln -sf KTT/Build/x86_64_Release/pyktt.so pyktt.so
    ln -sf KTT/Build/x86_64_Release/libktt.so libktt.so
    success "Symlinks created: pyktt.so, libktt.so"
fi

# ── Step 6: Verify pyktt import ─────────────────────────────────────
info "Step 6: Verifying pyktt import..."

if [ -n "$CUDA_DIR" ]; then
    export LD_LIBRARY_PATH="${CUDA_DIR}/lib64:$(pwd):${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$(pwd):${LD_LIBRARY_PATH:-}"
fi

if python3 -c "import pyktt; print('pyktt loaded successfully')" 2>/dev/null; then
    success "pyktt import OK"
else
    error "Failed to import pyktt"
    echo ""
    echo "  This might be a Python version issue. KTT's pybind11 bindings"
    echo "  are known to work best with Python 3.10."
    echo ""
    echo "  Current Python: $PYTHON_VERSION"
    echo ""
    echo "  Try: conda create -n ktt python=3.10 -y && conda activate ktt"
    echo "  Then re-run: ./setup.sh --skip-ktt"
    exit 1
fi

# ── Step 7: Generate cuda_env.yaml ──────────────────────────────────
info "Step 7: Generating cuda_env.yaml..."

python3 -c "
from utils.cuda_env import detect, save_config
from pathlib import Path
env = detect(Path('.'))
save_config(env, Path('.'))
print('cuda_env.yaml written')
"
success "cuda_env.yaml generated"

# ── Step 8: Print summary ───────────────────────────────────────────
echo ""
echo "========================================"
echo "  Setup Complete"
echo "========================================"
echo ""

python3 -c "
from utils.cuda_env import get_env, format_summary
env = get_env()
print(format_summary(env))
"

echo ""

# GPU detection
python3 -c "
from utils.gpu_info import detect_gpus, format_gpu_table
gpus = detect_gpus()
if gpus:
    print(format_gpu_table(gpus))
else:
    print('No GPUs detected (nvidia-smi/ncu may not be available)')
" 2>/dev/null || warn "GPU detection skipped"

echo ""
echo "  Python:      $PYTHON_VERSION ($VENV_NAME)"
echo "  CUDA:        $NVCC_VERSION"
echo ""
success "Ready to run!"
echo ""
echo "  Set your LLM API key in .env (see .env.example)"
echo ""
echo "  CLI:       python cli.py --dir problems/mmul"
echo "  Server:    python server.py              # http://localhost:8003"
echo "  Frontend:  cd frontend && npm install && npm run dev  # http://localhost:5003"
echo ""
