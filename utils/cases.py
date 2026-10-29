"""Case list, layout, staging and budget resolution.

The layout rule is worth stating up front: a problem with a single case (declared or
implicit) writes its artifacts at the iteration root, exactly as every run before this
feature did.
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


def primary_case_dir(problem_dir, iter_dir) -> Path:
    """The iteration directory holding the anchor results.json.

    Identity for a single-case problem, i.e. every problem that declares no ``cases:``
    block. Readers that summarise a run — the tree API, the results merge — need one
    results.json per iteration to point at, and it is the primary case's.

    Never raises: a summary is not worth failing a run over.
    """
    iter_dir = Path(iter_dir)
    try:
        import yaml as _yaml

        meta = _yaml.safe_load(
            (Path(problem_dir) / "problem.yaml").read_text(encoding="utf-8")
        ) or {}
        cases = case_list(meta)
        return case_dir(iter_dir, cases, cases[0])
    except Exception:
        return iter_dir


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


def stage_engine_files(problem_dir, iter_dir, meta: dict) -> str:
    """Place the files the engine owns beside the ones the LLM writes, and return
    ``inputs.hpp``'s text.

    inputs.hpp is generated once per run from inputs.yaml, so the LLM neither owns it
    nor can author it, and its absence is not something an iteration can recover from —
    hence the raise rather than a route to the fix loop. Shared by the agentic and the
    single-shot authoring paths, which need exactly the same files staged.
    """
    import shutil

    from utils.problem import reference_spec

    problem_dir = Path(problem_dir)
    iter_dir = Path(iter_dir)

    inputs_src = problem_dir / "inputs.hpp"
    if not inputs_src.exists():
        raise FileNotFoundError(
            f"{inputs_src} not found — the problem's I/O boundary was never generated. "
            "It comes from inputs.yaml at the start of a run; check that inputs.yaml exists."
        )
    shutil.copyfile(inputs_src, iter_dir / "inputs.hpp")

    ref = reference_spec(meta)
    if ref.type == "python":
        ref_py = problem_dir / ref.file
        if not ref_py.exists():
            raise FileNotFoundError(
                f"{ref_py} not found — python reference file required by problem.yaml"
            )
        shutil.copyfile(ref_py, iter_dir / "ref.py")
        shutil.copyfile(problem_dir / "inputs.yaml", iter_dir / "inputs.yaml")

    return inputs_src.read_text(encoding="utf-8")


def stage_case(problem_dir, iter_dir, spec, meta: dict, cases, case, regions: dict) -> Path:
    """Lay out one case's build directory and return it.

    The staged ``inputs.yaml`` is the resolved spec, not the declared one: the driver's
    cwd is this directory and utils/inputs bakes ``--inputs inputs.yaml`` into the
    python-reference command, so the reference runner reads this case's scalars with no
    flag and no new code.

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
    # An all-empty regions dict assembles a skeleton whose KERNELS region never declares
    # `kernel`, which the rest of the template references. Never write that over a driver
    # that is already here — the regions could not be recovered, not proven absent.
    framework = cdir / "framework.cpp"
    if any((regions or {}).values()) or not framework.exists():
        framework.write_text(assemble_framework_cpp(meta, regions), encoding="utf-8")

    kernels = iter_dir / "kernels.cu"
    if cdir != iter_dir and kernels.exists():
        shutil.copyfile(kernels, cdir / "kernels.cu")

    if reference and str(reference.get("type", "cuda")).lower() == "python":
        ref_py = Path(problem_dir) / reference.get("file", "ref.py")
        if ref_py.exists():
            shutil.copyfile(ref_py, cdir / "ref.py")

    return cdir
