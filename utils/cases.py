"""Case list, layout and budget resolution.

Kept out of utils/inputs.py: that module renders inputs.hpp and is already 628 lines,
and none of this is codegen. The one rule worth stating up front is the layout rule —
a problem with a single case (declared or implicit) writes its artifacts at the
iteration root, exactly as every run before this feature did.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from models.cases import CaseSpec


def case_list(meta: dict) -> List[CaseSpec]:
    """The problem's cases, in declared order. ``meta`` is the parsed problem.yaml.

    A problem with no ``cases:`` block gets one implicit case named ``default``, whose
    resolution is an identity copy of the declared inputs — so every existing problem
    keeps behaving exactly as it did.
    """
    raw = (meta or {}).get("cases") or []
    if not raw:
        return [CaseSpec()]

    cases = [CaseSpec.model_validate(c) for c in raw]
    names = [c.name for c in cases]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"Duplicate case name(s) {dupes} in problem.yaml — each case names a "
            "directory, so names must be unique"
        )
    return cases


def case_dir(iter_dir, cases: List[CaseSpec], case: CaseSpec) -> Path:
    """Where this case's driver, header and results live.

    One case -> the iteration root, which is the layout every problem had before this
    feature and which resume, revert, archive pruning and the tree API all read. More
    than one -> a subdirectory each, symmetrically. Putting the primary at the root and
    the rest in subdirectories would place the same artifact in two different locations
    depending on a field the reader has to go look up.
    """
    iter_dir = Path(iter_dir)
    if len(cases) <= 1:
        return iter_dir
    return iter_dir / f"case_{case.name}"


def case_duration(meta: dict, case: CaseSpec) -> Optional[float]:
    """Tuning budget in seconds for this case, or None to use the engine default."""
    if case.duration_s is not None:
        return float(case.duration_s)
    value = ((meta or {}).get("tuning") or {}).get("duration_s")
    return None if value is None else float(value)


def stage_case(problem_dir, iter_dir, spec, meta: dict, cases, case, regions: dict) -> Path:
    """Lay out one case's build directory and return it.

    Writes, into the case directory:
      * ``inputs.hpp``  — generated from the case-resolved spec, so its constexpr sizes,
        -D macros and init=file paths are this case's.
      * ``inputs.yaml`` — the resolved spec. The driver's cwd is this directory and
        utils/inputs bakes ``--inputs inputs.yaml`` into the python-reference command,
        so the reference runner reads this case's scalars with no flag and no new code.
      * ``framework.cpp`` — identical text for every case: the grid expression and the
        three LLM regions are shared, and only the constexprs in inputs.hpp differ.
      * ``kernels.cu`` — copied from the iteration root, which is where the LLM's single
        kernel source lives.

    For a single-case problem the case directory IS the iteration root, so this writes
    exactly the files the pre-multi-case pipeline wrote, in the same place.
    """
    import shutil

    import yaml as _yaml

    from utils.framework import assemble_framework_cpp
    from utils.inputs import generate_inputs_hpp

    iter_dir = Path(iter_dir)
    cdir = case_dir(iter_dir, cases, case)
    cdir.mkdir(parents=True, exist_ok=True)

    resolved = spec.for_case(case)
    reference = (meta or {}).get("reference")

    (cdir / "inputs.hpp").write_text(
        generate_inputs_hpp(resolved, reference, problem_dir), encoding="utf-8"
    )
    (cdir / "inputs.yaml").write_text(
        _yaml.safe_dump(
            resolved.model_dump(by_alias=True, exclude_none=True), sort_keys=False
        ),
        encoding="utf-8",
    )
    (cdir / "framework.cpp").write_text(
        assemble_framework_cpp(meta, regions), encoding="utf-8"
    )

    kernels = iter_dir / "kernels.cu"
    if cdir != iter_dir and kernels.exists():
        shutil.copyfile(kernels, cdir / "kernels.cu")

    if reference and str(reference.get("type", "cuda")).lower() == "python":
        ref_py = Path(problem_dir) / reference.get("file", "ref.py")
        if ref_py.exists():
            shutil.copyfile(ref_py, cdir / "ref.py")

    return cdir
