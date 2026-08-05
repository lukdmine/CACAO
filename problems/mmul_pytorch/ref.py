"""PyTorch (GPU) reference: matrix multiply C = A @ B.

All buffers are the matrices in ROW-MAJOR (C) order: generated CUDA kernels must
write C as C_flat[i*kSizeN + j] = C[i,j]. KTT validates the flat buffer
element-wise against this reference's output.

PyTorch is an OPTIONAL dependency: only problems whose ref.py imports torch
need it installed. Install into the interpreter that runs the reference (the
ktt conda env):
    /home/u550615/miniconda3/envs/ktt/bin/pip install torch \\
        --index-url https://download.pytorch.org/whl/cu124

The reference is split into two functions so CACAO can time just the GPU op:

  - ``prepare_input(scalars, buffers)`` does the numpy->device (H2D) transfer.
    It is NOT timed.
  - ``gemm_reference(prepared)`` is the pure GPU matmul. CACAO times calls to
    it via torch.cuda.Event. It returns a device tensor; the runner does the
    D2H (``.cpu().numpy()``) outside the timed region.
"""

try:
    import torch
except ImportError as e:
    raise ImportError(
        "ref.py needs PyTorch, an optional dependency. Install it into the "
        "interpreter that runs the reference (the ktt conda env):\n"
        "  /home/u550615/miniconda3/envs/ktt/bin/pip install torch "
        "--index-url https://download.pytorch.org/whl/cu124"
    ) from e


def prepare_input(scalars, buffers):
    """numpy -> device tensors. The flat buffers ARE row-major matrices, so a
    reshape view is the only interpretation they need."""
    M, N, K = scalars["kSizeM"], scalars["kSizeN"], scalars["kSizeK"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tA = torch.from_numpy(buffers["mat_a"].reshape(M, K)).to(device)
    tB = torch.from_numpy(buffers["mat_b"].reshape(K, N)).to(device)
    return tA, tB


def gemm_reference(prepared):
    tA, tB = prepared
    return (tA @ tB).flatten()
