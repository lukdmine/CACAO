// ops/passthrough.cu — the null op: device-to-device copy of the single input
// into the validated output. No vendor library involved; it exists to prove
// the whole pipeline (dump -> generate -> nvcc -> run -> verify -> time ->
// parse -> seed) end to end before any real library op lands. Tests pair it
// with a ref kernel that computes the same copy.
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "op_contract.h"

static void* g_in = nullptr;
static void* g_out = nullptr;
static size_t g_bytes = 0;

void op_setup(const CacaoOpSpec& spec) {
    if (spec.num_inputs != 1) {
        fprintf(stderr, "passthrough: expects exactly 1 input buffer, got %d\n", spec.num_inputs);
        exit(2);
    }
    const size_t elem = std::strcmp(spec.output_dtype, "float") == 0 ? sizeof(float) : sizeof(int);
    g_bytes = spec.output_elems * elem;
    if (cudaMalloc(&g_in, g_bytes) != cudaSuccess || cudaMalloc(&g_out, g_bytes) != cudaSuccess) {
        fprintf(stderr, "passthrough: cudaMalloc(%zu) failed\n", g_bytes);
        exit(2);
    }
    if (cudaMemcpy(g_in, spec.inputs[0].data, g_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "passthrough: H2D failed\n");
        exit(2);
    }
}

void op_run() {
    cudaMemcpyAsync(g_out, g_in, g_bytes, cudaMemcpyDeviceToDevice);
}

void op_output(void* dst) {
    if (cudaMemcpy(dst, g_out, g_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "passthrough: D2H failed\n");
        exit(2);
    }
}

void op_teardown() {
    cudaFree(g_in);
    cudaFree(g_out);
    g_in = g_out = nullptr;
}
