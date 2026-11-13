"""Lowest GPU architecture a finished kernel compiles for.

The kernel is compiled with a real-architecture option (``sm_XX``) for each target, at
three kinds of configuration: the low one (every ``USE_`` toggle off), that one with
each toggle switched on in turn, and the winning configuration. The toggle rows say
which feature raised the floor, so a caller can offer a variant without it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from models.finalize import ArchReport
from utils import nvrtc


def _lowest_compiling(
    kernel_src: str,
    config: Dict[str, str],
    base_options: List[str],
    archs: Sequence[int],
    label: str,
) -> Optional[int]:
    for arch in sorted(archs):
        options = base_options + [nvrtc.real_arch_option(arch)]
        if nvrtc.compile_source(kernel_src, config, options, label).ok:
            return arch
    return None


def check_archs(
    kernel_src: str,
    params_src: str,
    *,
    winning_config: Dict[str, str],
    cuda_include: str,
    scalar_defines: List[str],
    archs: Sequence[int],
) -> ArchReport:
    if not archs:
        return ArchReport(skipped="NVRTC unavailable or reports no supported architectures")

    params = nvrtc.parse_parameters(params_src)
    base = nvrtc.sample_configurations(params)[0][1]
    options = [f"-I{cuda_include}"] + list(scalar_defines)

    per_toggle: Dict[str, Optional[int]] = {}
    for name in params:
        if name.startswith("USE_"):
            per_toggle[name] = _lowest_compiling(
                kernel_src, {**base, name: "1"}, options, archs, f"{name}=1"
            )

    return ArchReport(
        archs=sorted(archs),
        base_min_arch=_lowest_compiling(kernel_src, base, options, archs, "base"),
        winner_min_arch=_lowest_compiling(
            kernel_src, dict(winning_config), options, archs, "winner"
        ),
        per_toggle=per_toggle,
    )
