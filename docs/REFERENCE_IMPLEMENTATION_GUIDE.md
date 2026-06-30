# Reference Implementation Guide

This guide explains how to implement reference kernels so validation is consistent between CUDA and CPU references.

## Standard ABI

Function signatures contain only vector pointer arguments, in `vectors:` order. Scalars are injected as `-D` compiler flags and are available as compile-time constants in both CUDA and CPU reference functions.

For a problem defined like this:

```yaml
scalars:
  - name: N
    dtype: int
    value: 1024
  - name: K
    dtype: int
    value: 64

vectors:
  - name: A
    dtype: float
    size: N * K
    access: read
  - name: B
    dtype: float
    size: N
    access: write
```

the standard function signatures are:

```cuda
extern "C" __global__ void kernel(const float* A, float* B)
```

```c
void reference(float* A, float* B)
```

Scalars `N` and `K` are available in both as compile-time `#define` constants.

## Why this convention

- scalars are injected as `-D` compiler flags, so they are compile-time constants in both the kernel and the reference
- function signatures contain only the data that varies at runtime (vector pointers)
- it avoids false validation mismatches caused by ABI order differences

## CUDA Reference Rules

For `reference.type: cuda`:

```yaml
reference:
  type: cuda
  file: ref_kernel.cu
  function: reference
  block_x: 8
  block_y: 8
  block_z: 1
```

Rules:
- use `extern "C" __global__`
- only vector (pointer) arguments, in `vectors:` order from problem.yaml
- scalars are available as compile-time `#define` constants — do not include them as function parameters
- keep the implementation simple and correct rather than optimized
- `block_x`, `block_y`, and `block_z` define the reference launch shape (`block_z` defaults to `1`)

Example:

```cuda
extern "C" __global__ void reference(const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = x[i] * 2.0f;
    }
}
```

Here `N` is a compile-time constant injected via `-DN=1024`.

## CPU Reference Rules

For `reference.type: cpu_c`:

```yaml
reference:
  type: cpu_c
  file: ref_cpu.c
  function: reference
```

Rules:
- only vector (pointer) arguments, in `vectors:` order from problem.yaml
- scalars are available as compile-time `#define` constants via `-D` gcc flags — do not include them as function parameters
- keep vector order exactly the same as `vectors:` in `problem.yaml`
- for C++ sources, export the function with `extern "C"`

Example C reference:

```c
#ifdef __cplusplus
extern "C" {
#endif

void reference(const float* x, float* y) {
    for (int i = 0; i < N; ++i) {
        y[i] = x[i] * 2.0f;
    }
}

#ifdef __cplusplus
}
#endif
```

Here `N` is a compile-time constant injected via `-DN=1024`.

## Mapping from `problem.yaml`

Given:

```yaml
scalars:
  - name: M
  - name: N

vectors:
  - name: A
  - name: B
  - name: C
```

the expected function signature is:

```text
(A, B, C)
```

`M` and `N` are `#define` constants available throughout the function body.

## Output Buffers in CPU References

For CPU references:
- `read` vectors contain the initialized input data
- the validated output vector is provided as the callback output buffer
- non-validated write vectors are provided as temporary zeroed buffers

That means the CPU reference should compute all outputs it logically owns, even if only one output is currently validated.

## Common Mistakes

1. Adding scalar parameters to the function signature — scalars are `#define` constants, not arguments.
2. Reordering vector arguments relative to `vectors:` in `problem.yaml`.
3. Forgetting `extern "C"` for C++ CPU references.
4. Validating one output while leaving other output buffers semantically unimplemented.

## Checklist

- [ ] `kernel.function` matches the CUDA symbol name
- [ ] `reference.function` matches the reference symbol name
- [ ] function signatures contain only vector pointer arguments in `vectors:` order
- [ ] scalars are used as compile-time constants (not function parameters)
- [ ] validated outputs are fully written
- [ ] tolerance matches expected numeric error
