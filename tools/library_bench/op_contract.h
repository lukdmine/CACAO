// op_contract.h — THE contract between the fixed timing harness (harness.cu)
// and a pluggable op (ops/<name>.cu). An op is the ONLY library-specific code
// in the tool: it receives the problem's real input buffers (host memory, by
// name) plus the scalars, does its own H2D/descriptor/algo-sweep setup, and
// exposes the single call the harness times.
//
// Everything except op_run() is untimed. op_run() must issue the SAME work
// every call (identical input, identical computation) — the harness reports
// min over reps, and correctness is checked against bench_op_<output>.bin
// produced after a sanity run.
#pragma once

#include <cstddef>

struct CacaoOpInput {
    const char* name;    // buffer name from inputs.yaml
    const char* dtype;   // "float" or "int" (both 4 bytes)
    size_t elems;        // element count
    const void* data;    // host data, exactly as dumped from the problem's generators
};

struct CacaoOpScalar {
    const char* name;    // scalar name from inputs.yaml
    int is_float;        // 0: int (value holds it exactly), 1: float
    double value;
};

struct CacaoOpSpec {
    const CacaoOpInput* inputs;   // read + readwrite buffers, in inputs.yaml order
    int num_inputs;
    const CacaoOpScalar* scalars; // every scalar, in inputs.yaml order
    int num_scalars;
    const char* output_name;      // the problem's validated buffer (what the gate checks)
    const char* output_dtype;
    size_t output_elems;
};

// H2D copies, descriptors, workspace, ALGORITHM SWEEP: pick the fastest legal
// algorithm here, untimed, and settle all state op_run() needs.
void op_setup(const CacaoOpSpec& spec);

// The one timed call: issue the op on the current stream and return
// (the harness synchronizes via CUDA events per rep).
void op_run();

// Copy the op's result for the validated buffer into dst (host memory,
// output_elems * 4 bytes). Called after one op_run(), untimed — used by the
// verification gate, which refuses to seed a baseline on mismatch.
void op_output(void* dst);

// Optional cleanup hook counterpart — free everything op_setup allocated.
void op_teardown();

// Ops may print one informational stderr line as CACAO_BENCH_INFO=<chosen
// algorithm, achieved TFLOP/s, ...> — the bench driver passes it through.
