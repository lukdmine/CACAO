"""Precise GPU-only reference timing via ``torch.cuda.Event``.

Invoked once at startup by ``engine/master.py`` (python references only) AFTER a
dump tool has materialized the real ``cacao_in_<name>.bin`` from ``inputs.hpp``.
Loads ref.py, calls ``prepare_input(scalars, buffers)`` (H2D — not timed), warms
up, then times single calls of the reference function with CUDA events and
prints ``CACAO_REF_TIME_US=<min µs>`` (the marker keeps the parse robust against
any stray stdout from torch or ref.py).

ref.py contract for precise timing:
  - ``prepare_input(scalars, buffers) -> prepared``  (H2D + setup; not timed)
  - ``<reference_function>(prepared) -> result``     (the GPU op; timed)

The reference function should return a GPU tensor whose memory layout already
matches the validated buffer's order (the runner converts via ``.cpu().numpy()``);
D2H transfer is deliberately excluded from the timed region.

If ref.py has no ``prepare_input``, or torch/CUDA is unavailable, the script
exits non-zero so the caller falls back to KTT's coarse wall-clock reference time.

Usage:
    python -m utils.torch_ref_timer --inputs inputs.yaml --ref ref.py \
        --function gemm_reference --inputs-dir <dir with cacao_in_*.bin>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WARMUP_ITERS = 5
_TIMED_ITERS = 5


def main() -> int:
    p = argparse.ArgumentParser(
        description="Precise GPU reference timing (torch.cuda.Event)"
    )
    p.add_argument("--inputs", required=True, help="path to inputs.yaml")
    p.add_argument("--ref", required=True, help="path to ref.py")
    p.add_argument("--function", required=True, help="reference function name in ref.py")
    p.add_argument(
        "--inputs-dir", required=True, help="dir containing cacao_in_<name>.bin"
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not available", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 3

    from utils.python_ref_runner import load_inputs, load_ref_module

    mod = load_ref_module(Path(args.ref))
    prepare_input = getattr(mod, "prepare_input", None)
    if not callable(prepare_input):
        print(
            "ref.py has no prepare_input — cannot time precisely", file=sys.stderr
        )
        return 4
    ref_func = getattr(mod, args.function, None)
    if not callable(ref_func):
        print(f"ref.py has no callable '{args.function}'", file=sys.stderr)
        return 5

    scalars, buffers, _ = load_inputs(Path(args.inputs), Path(args.inputs_dir))

    # H2D + any setup — NOT timed.
    prepared = prepare_input(scalars, buffers)
    torch.cuda.synchronize()

    # Warmup so caches / autotune settle before the timed calls.
    for _ in range(_WARMUP_ITERS):
        ref_func(prepared)
    torch.cuda.synchronize()

    # Time single calls with CUDA events (GPU-only; excludes H2D/D2H); the min is
    # the least noisy estimate of pure device time. First-write-wins downstream,
    # so one unlucky blip would otherwise skew every speedup of the whole run.
    times = []
    for _ in range(_TIMED_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ref_func(prepared)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"CACAO_REF_TIME_US={min(times) * 1000.0}")  # ms -> µs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
