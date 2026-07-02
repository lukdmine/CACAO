"""
CUDA environment detection and configuration.

Resolves paths to CUDA tools (nvcc, ncu, nvidia-smi, gcc/g++) via:
1. Explicit overrides from cuda_env.yaml
2. Environment variables (CUDA_PATH, CUDA_HOME, CUDA_ROOT)
3. PATH lookup (which/shutil.which)
4. Common install locations
5. HPC module system probe
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Common CUDA install locations to search
_CUDA_SEARCH_PATHS = [
    "/usr/local/cuda",
    "/usr/local/cuda-12",
    "/usr/local/cuda-12.0",
    "/usr/local/cuda-12.1",
    "/usr/local/cuda-12.2",
    "/usr/local/cuda-12.3",
    "/usr/local/cuda-12.4",
    "/usr/local/cuda-12.5",
    "/usr/local/cuda-12.6",
    "/opt/cuda",
]


@dataclass
class CudaEnv:
    """Resolved CUDA environment paths. All fields are Optional — None means not found."""

    cuda_path: Optional[Path] = None
    nvcc: Optional[Path] = None
    ncu: Optional[Path] = None
    nvidia_smi: Optional[Path] = None
    gcc: Optional[Path] = None
    gpp: Optional[Path] = None  # g++
    cuda_include: Optional[Path] = None

    _warnings: list[str] = field(default_factory=list, repr=False)
    _errors: list[str] = field(default_factory=list, repr=False)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def errors(self) -> list[str]:
        return list(self._errors)


def _find_cuda_path_from_env() -> Optional[Path]:
    """Check CUDA_PATH, CUDA_HOME, CUDA_ROOT env vars."""
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        val = os.environ.get(var)
        if val and Path(val).is_dir() and Path(val) != Path("/usr"):
            return Path(val)
    return None


def _find_cuda_path_from_nvcc() -> Optional[Path]:
    """Derive CUDA path from nvcc location on PATH."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc is typically at <cuda_path>/bin/nvcc
        cuda_path = Path(nvcc).resolve().parent.parent
        # Reject /usr — that means CUDA is system-installed (no single root)
        if cuda_path == Path("/usr"):
            return None
        if (cuda_path / "include").is_dir():
            return cuda_path
    return None


def _find_cuda_path_from_common_locations() -> Optional[Path]:
    """Search common CUDA install directories."""
    # Also search for versioned dirs we didn't list
    search = list(_CUDA_SEARCH_PATHS)
    for pattern_dir in [Path("/usr/local"), Path("/opt")]:
        if pattern_dir.is_dir():
            for child in sorted(pattern_dir.iterdir(), reverse=True):
                if (
                    child.name.startswith("cuda")
                    and child.is_dir()
                    and str(child) not in search
                ):
                    search.append(str(child))

    for path_str in search:
        p = Path(path_str)
        if p.is_dir() and (p / "bin" / "nvcc").exists():
            return p
    return None


def _find_cuda_path_from_modules() -> Optional[Path]:
    """Try HPC module system to find CUDA."""
    try:
        result = subprocess.run(
            ["bash", "-c", "module list 2>&1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Check if any cuda module is loaded
        if "cuda" in result.stdout.lower():
            # Try to get the CUDA path from the module
            result2 = subprocess.run(
                [
                    "bash",
                    "-c",
                    "module show cuda 2>&1 | grep -i 'CUDA_PATH\\|CUDA_HOME\\|prepend-path.*PATH'",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result2.stdout.splitlines():
                for token in line.split():
                    p = Path(token)
                    if p.is_dir() and (p / "bin" / "nvcc").exists():
                        return p
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _find_tool(
    name: str, cuda_path: Optional[Path] = None, subdir: str = "bin"
) -> Optional[Path]:
    """Find a tool binary: check cuda_path/bin first, then PATH."""
    if cuda_path:
        candidate = cuda_path / subdir / name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def _find_cuda_include(cuda_path: Optional[Path]) -> Optional[Path]:
    """Find CUDA include directory containing headers."""
    candidates = []
    if cuda_path:
        candidates.append(cuda_path / "include")

    # Also check env var override
    env_path = os.environ.get("CUDA_PATH", "")
    if env_path:
        candidates.append(Path(env_path) / "include")

    candidates.extend(
        [
            Path("/usr/local/cuda/include"),
            Path("/usr/include"),
        ]
    )

    for path in candidates:
        # Check for cuda_runtime.h (universal) or mma.h (tensor cores)
        if path.is_dir() and any(
            (path / h).exists() for h in ("cuda_runtime.h", "mma.h")
        ):
            return path
    return None


def _load_config_file(project_root: Path) -> dict:
    """Load cuda_env.yaml if it exists. Returns empty dict if not found."""
    config_path = project_root / "cuda_env.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            return data
        except Exception:
            return {}
    return {}


def detect(project_root: Optional[Path] = None) -> CudaEnv:
    """
    Detect the CUDA environment. Returns a CudaEnv with resolved paths.

    Discovery order for each tool:
    1. cuda_env.yaml explicit path
    2. CUDA_PATH / CUDA_HOME / CUDA_ROOT env vars
    3. which <tool> (PATH lookup)
    4. Common install locations
    5. HPC module system probe
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    config = _load_config_file(project_root)
    env = CudaEnv()

    # --- CUDA path ---
    if "cuda_path" in config:
        p = Path(config["cuda_path"])
        if p.is_dir():
            env.cuda_path = p
        else:
            env._errors.append(f"cuda_env.yaml: cuda_path '{p}' does not exist")

    if not env.cuda_path:
        env.cuda_path = (
            _find_cuda_path_from_env()
            or _find_cuda_path_from_nvcc()
            or _find_cuda_path_from_common_locations()
            or _find_cuda_path_from_modules()
        )

    # cuda_path is None when CUDA is system-installed (/usr/bin/nvcc etc.)
    # That's fine — individual tools are still found via PATH.
    # Only error if nvcc is also missing (checked below).

    # --- nvcc ---
    if "nvcc" in config:
        p = Path(config["nvcc"])
        env.nvcc = p if p.exists() else None
        if not env.nvcc:
            env._errors.append(f"cuda_env.yaml: nvcc path '{p}' does not exist")
    else:
        env.nvcc = _find_tool("nvcc", env.cuda_path)

    if not env.nvcc:
        env._errors.append(
            "nvcc not found.\n"
            "  Searched: CUDA_PATH/bin, PATH, common CUDA locations\n"
            "  Fix: Install CUDA Toolkit, or set nvcc path in cuda_env.yaml"
        )

    # --- ncu (optional) ---
    if "ncu" in config:
        p = Path(config["ncu"])
        env.ncu = p if p.exists() else None
        if not env.ncu:
            env._warnings.append(f"cuda_env.yaml: ncu path '{p}' does not exist")
    else:
        env.ncu = _find_tool("ncu", env.cuda_path)

    if not env.ncu:
        env._warnings.append(
            "ncu (Nsight Compute) not found — profiling will be skipped.\n"
            "  Fix: Install NVIDIA Nsight Compute, or set ncu path in cuda_env.yaml"
        )

    # --- nvidia-smi (optional) ---
    if "nvidia_smi" in config:
        p = Path(config["nvidia_smi"])
        env.nvidia_smi = p if p.exists() else None
    else:
        env.nvidia_smi = _find_tool("nvidia-smi")

    if not env.nvidia_smi:
        env._warnings.append(
            "nvidia-smi not found — GPU auto-detection will be limited.\n"
            "  This usually comes with the NVIDIA driver, not the CUDA Toolkit."
        )

    # --- gcc ---
    if "gcc" in config:
        p = Path(config["gcc"])
        env.gcc = p if p.exists() else None
    else:
        env.gcc = _find_tool("gcc") or _find_tool("cc")

    if not env.gcc:
        env._errors.append(
            "gcc not found — required for compiling CPU reference implementations.\n"
            "  Fix: Install gcc (e.g., apt install gcc, dnf install gcc)"
        )

    # --- g++ ---
    env.gpp = _find_tool("g++") or _find_tool("c++")

    # --- CUDA include dir ---
    if "cuda_include" in config:
        p = Path(config["cuda_include"])
        env.cuda_include = p if p.is_dir() else None
    else:
        env.cuda_include = _find_cuda_include(env.cuda_path)

    if not env.cuda_include:
        env._warnings.append(
            "CUDA include directory not found — tensor core kernels may fail to compile.\n"
            "  Fix: Set cuda_include in cuda_env.yaml or install CUDA Toolkit development headers"
        )

    return env


def get_ncu_tmpdir() -> Path:
    """Return a stable, per-user temp directory for NCU (created if missing).

    NVIDIA Nsight Compute serializes profiling through a single fixed lock file
    at ``$TMPDIR/nsight-compute-lock`` (``/tmp/nsight-compute-lock`` by default).
    On a shared machine the first user to profile creates that file, and the
    kernel's ``fs.protected_regular`` hardening then blocks every *other* user
    from opening it in the world-writable sticky ``/tmp`` — even at mode 666 —
    so their ncu run dies with ``InterprocessLockFailed`` and exit code 9.

    Giving each user their own TMPDIR gives each their own lock file (and keeps
    NCU's within-user serialization intact), which avoids the cross-user clash.
    """
    # tempfile.gettempdir() honors an already-set TMPDIR, else falls back to /tmp.
    try:
        uid = os.getuid()
    except AttributeError:  # non-POSIX (e.g. Windows)
        uid = os.environ.get("USER", "default")
    tmpdir = Path(tempfile.gettempdir()) / f"ncu-{uid}"
    tmpdir.mkdir(mode=0o700, exist_ok=True)
    return tmpdir


def get_subprocess_env(env: Optional[CudaEnv] = None) -> dict:
    """Build an environment dict for subprocesses with LD_LIBRARY_PATH set correctly."""
    if env is None:
        env = get_env()
    sub_env = os.environ.copy()
    parts = []
    if env.cuda_path:
        lib64 = env.cuda_path / "lib64"
        if lib64.is_dir():
            parts.append(str(lib64))
    # Add project root so libktt.so symlink is found
    project_root = Path(__file__).parent.parent
    parts.append(str(project_root))
    existing = sub_env.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    sub_env["LD_LIBRARY_PATH"] = ":".join(parts)
    # Force POSIX numeric locale so tools like NCU always use '.' as the
    # decimal separator, regardless of the host machine's regional settings.
    sub_env["LC_NUMERIC"] = "C"
    # Redirect NCU's fixed-path lock file to a per-user dir so profiling doesn't
    # collide between users sharing this machine (see get_ncu_tmpdir).
    sub_env["TMPDIR"] = str(get_ncu_tmpdir())
    return sub_env


def format_summary(env: CudaEnv) -> str:
    """Format the detected environment as a human-readable summary."""
    lines = ["CUDA Environment:"]

    def _status(val, label):
        if val:
            return f"  {label}: {val}"
        return f"  {label}: NOT FOUND"

    lines.append(_status(env.cuda_path, "CUDA path"))
    lines.append(_status(env.nvcc, "nvcc"))
    lines.append(_status(env.ncu, "ncu"))
    lines.append(_status(env.nvidia_smi, "nvidia-smi"))
    lines.append(_status(env.gcc, "gcc"))
    lines.append(_status(env.gpp, "g++"))
    lines.append(_status(env.cuda_include, "CUDA include"))

    return "\n".join(lines)


def save_config(env: CudaEnv, project_root: Path):
    """Save detected environment to cuda_env.yaml."""
    data = {}
    if env.cuda_path:
        data["cuda_path"] = str(env.cuda_path)
    if env.nvcc:
        data["nvcc"] = str(env.nvcc)
    if env.ncu:
        data["ncu"] = str(env.ncu)
    if env.nvidia_smi:
        data["nvidia_smi"] = str(env.nvidia_smi)
    if env.gcc:
        data["gcc"] = str(env.gcc)
    if env.gpp:
        data["gpp"] = str(env.gpp)
    if env.cuda_include:
        data["cuda_include"] = str(env.cuda_include)

    config_path = project_root / "cuda_env.yaml"
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# --- Module-level singleton ---
_env: Optional[CudaEnv] = None


def get_env(project_root: Optional[Path] = None) -> CudaEnv:
    """Get or initialize the global CudaEnv singleton."""
    global _env
    if _env is None:
        _env = detect(project_root)
    return _env
