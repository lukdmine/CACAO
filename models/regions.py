"""Pydantic model for the three LLM-authored framework-file regions.

Used by configure_node to structure LLM output. The three bodies are spliced into
the engine skeleton by utils.framework.assemble_framework_cpp(). See
docs/FRAMEWORK_FILE_SPEC.md §9.
"""

from pydantic import BaseModel, Field


class FrameworkRegions(BaseModel):
    """The three C++ region bodies the LLM fills in framework.cpp."""

    kernels: str = Field(
        ...,
        description=(
            "Body of the CACAO:KERNELS region. For each __global__ in kernels.cu: "
            "AddKernelDefinitionFromFile(\"func\", kernelFile, ndRange, ktt::DimensionVector()). "
            "Create the kernel and assign it to a variable named exactly `kernel` "
            "(ktt::KernelId kernel = tuner.CreateSimpleKernel(...) for one kernel, or "
            "CreateCompositeKernel(name, {defs}) for a pipeline). Register any intermediate "
            "scratch buffers with tuner.AddArgumentVector(std::vector<T>(size), ReadWrite). "
            "Call tuner.SetArguments(def, {...}) for every definition, matching that "
            "__global__'s parameter order, using boundary handles in.<name> (e.g. in.mat_a) "
            "and scratch ids. `tuner`, `in`, `ndRange`, `kernelFile` are in scope; scalars "
            "are host consts."
        ),
    )
    params: str = Field(
        ...,
        description=(
            "Body of the CACAO:PARAMS region. tuner.AddParameter(kernel, \"NAME\", "
            "std::vector<uint64_t>{...}) for each tuning parameter (these become -D macros "
            "in the kernel). Add tuner.AddConstraint(kernel, {names}, lambda) to prune "
            "illegal combinations and tuner.AddThreadModifier(...) to scale grid/block from "
            "parameters. Uses `kernel` and the definition ids from the KERNELS region."
        ),
    )
    launcher: str = Field(
        "",
        description=(
            "Body of the CACAO:LAUNCHER region. For a multi-kernel pipeline or custom "
            "launch, tuner.SetLauncher(kernel, [captures](ktt::ComputeInterface& ci){ ... "
            "RunKernel sequence ... }). Leave EMPTY for a single kernel that uses thread "
            "modifiers (KTT's default launcher handles it)."
        ),
    )
