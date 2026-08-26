"""The launcher contract — the part of the framework file that compiles either way.

Every other mistake in a region is caught by g++ or NVRTC. These are not:

* ``RunKernel`` takes a ``KernelDefinitionId``. Passing the composite ``KernelId``
  compiles cleanly and silently runs a single definition. One real run spent three
  iterations on this, with the tuner reporting ``Launching configuration 2 / 27 for
  kernel precompute_init_kernel`` while the rest of the pipeline never executed.
* A reference that loops to a fixed point needs a launcher that loops to a fixed
  point. A single round compiles, runs, and returns the wrong answer. Three parallel
  branches on a flood-fill problem each launched once and each failed every
  configuration on validation.

So this block is examples, not prose: the distinctions are ones a model gets right by
seeing them and wrong by being told about them.

Every example here has been compiled and run against the real tuner. The first draft
of the convergence loop used ``UploadBuffer`` and failed all six configurations with
``Buffer for argument with id 2 already exists`` — teaching an API from its header
documentation is not the same as knowing it works. See
tests/test_launcher_reference.py, which builds and runs these patterns.
"""

LAUNCHER_RULES = """## Launcher contract (region_launcher.cpp)

**Single kernel driven by thread modifiers: leave this region EMPTY.** KTT's default
launcher runs it.

Otherwise `SetLauncher`, and the `RunKernel` sequence IS the schedule. Every call
blocks; there is one compute queue, so no overlap.

```cpp
// RunKernel takes a KernelDefinitionId — NOT the KernelId of the composite kernel.
// Capture the definition ids. Passing `kernel` here COMPILES and then runs only one
// definition, which shows up as a validation failure, never as a compile error.
tuner.SetLauncher(kernel, [defA, defB](ktt::ComputeInterface& ci) {
    const auto& cfg = ci.GetCurrentConfiguration();
    const uint64_t tile =
        ktt::ParameterPair::GetParameterValue<uint64_t>(cfg.GetPairs(), "TILE");
    ci.RunKernel(defA, ktt::DimensionVector(/*global*/), ktt::DimensionVector(/*block*/));
    ci.RunKernel(defB);   // base sizes; runs after A
});
```

`GetParameterValue`'s type appears only in the return type, so `<uint64_t>` is
REQUIRED — omitting it does not compile.

### Iterative algorithms (BFS, flood fill, relaxation, anything run to a fixed point)

If the reference loops until nothing changes, ONE round of kernel launches is wrong —
it propagates one level and stops. Loop in the launcher and read the flag back:

```cpp
tuner.SetLauncher(kernel, [expandDef, changedId](ktt::ComputeInterface& ci) {
    const auto& cfg = ci.GetCurrentConfiguration();
    const uint64_t maxRounds =
        ktt::ParameterPair::GetParameterValue<uint64_t>(cfg.GetPairs(), "MAX_ROUNDS");
    unsigned int changed = 1u;
    for (uint64_t round = 0; changed != 0u && round < maxRounds; ++round) {
        const unsigned int zero = 0u;
        ci.UpdateBuffer(changedId, &zero, sizeof(zero));   // reset the flag
        ci.RunKernel(expandDef);
        ci.DownloadBuffer(changedId, &changed, sizeof(changed));
    }
});
```

Use `UpdateBuffer(id, &value, sizeof(value))` to write into a buffer KTT already
manages. **Do NOT use `UploadBuffer` for this** — it is for user-managed arguments and
here fails every configuration at launch with `Buffer for argument with id N already
exists`, which is reported as a run failure and not as anything resembling its cause.

`DownloadBuffer(ArgumentId, void*, size_t)` is how the host observes convergence.

Bound the loop so a buggy kernel cannot hang the tuner — but bound it by the WORST
case, not a convenient number. A bound that is merely large stops the propagation
early, and an unconverged result is not an error: it is a wrong answer that fails
validation with no indication of why. On a 2048x2048 flood fill, 4096 rounds converge
and 2048 do not. Never make the bound a tuning parameter whose small values are
insufficient — the tuner will report those configurations as validation failures and
the cause is invisible in the output.

Reading back every round costs a synchronisation, so a parameter for
rounds-per-readback (do K kernel launches between checks) is a legitimate thing to
tune; the round *limit* is not.

**Do not attempt grid-wide synchronisation inside one kernel instead.**
`cg::this_grid().sync()` compiles and fails at runtime. A block-local "changed" flag
makes blocks terminate independently and produces exactly this class of validation
failure.

**Global size units follow `global_size_type` in problem.yaml**: `cuda` → grid in
blocks; `opencl` → total thread count, with KTT computing grid = global/block. `ndRange`
uses the same convention.
"""
