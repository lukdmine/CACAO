// harness.cu — FIXED timing core; never edited per problem or per library.
//
// Problem-specifics arrive through the generated bench_spec.h (buffer/scalar
// tables, ref launch geometry); the reference kernel runs via the generated
// bench_ref_launch.cu; the library-specific code implements op_contract.h.
//
// Session layout (one GPU context, nothing else on the device):
//   1. host-load dumped inputs (cacao_in_<name>.bin), H2D every buffer
//   2. run the problem's reference kernel, D2H + write bench_ref_<valid>.bin
//   3. op_setup()  (untimed: H2D, descriptors, algo sweep)
//   4. one sanity op_run() -> op_output() -> bench_op_<valid>.bin (the gate)
//   5. warmup op_run()s, then per-rep CUDA-event timing; print
//      CACAO_REF_TIME_US=<min_of_reps> on stdout, median on stderr.
//
// argv: bench <device> <input_dir> <warmup> <reps> [out_dir]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "op_contract.h"
#include "bench_spec.h"

// Generated launcher: runs the problem's reference kernel (the trusted oracle).
extern "C" void cacao_launch_ref(void* const* d, unsigned gx, unsigned gy, unsigned gz,
                                 unsigned bx, unsigned by, unsigned bz);

#define CUDA_OK(call)                                                                  \
    do {                                                                               \
        cudaError_t e_ = (call);                                                       \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(e_),        \
                    __FILE__, __LINE__);                                               \
            exit(2);                                                                   \
        }                                                                              \
    } while (0)

static size_t dtype_size(const char* dtype) {
    return std::strcmp(dtype, "float") == 0 ? sizeof(float) : sizeof(int);
}

static std::vector<char> read_input_file(const std::string& path, size_t expect_bytes) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s — input dump must run first\n", path.c_str());
        exit(2);
    }
    std::vector<char> buf(expect_bytes);
    const size_t got = fread(buf.data(), 1, expect_bytes, f);
    fclose(f);
    if (got != expect_bytes) {
        fprintf(stderr, "%s: expected %zu bytes, read %zu\n", path.c_str(), expect_bytes, got);
        exit(2);
    }
    return buf;
}

static void write_file(const std::string& path, const void* data, size_t bytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(2);
    }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: bench <device> <input_dir> <warmup> <reps> [out_dir]\n");
        return 2;
    }
    const int device = atoi(argv[1]);
    const std::string in_dir = argv[2];
    const int warmup = atoi(argv[3]);
    const int reps = atoi(argv[4]);
    const std::string out_dir = (argc > 5) ? argv[5] : ".";

    CUDA_OK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "CACAO_BENCH_GPU=%s (device %d)\n", prop.name, device);

    using namespace cacao_bench;
    const BufferInfo& valid = kBuffers[kValidatedIndex];
    const size_t valid_bytes = valid.elems * dtype_size(valid.dtype);

    // ---- host load + H2D every buffer (write-only ones start zeroed) ----
    std::vector<std::vector<char>> host(kNumBuffers);
    std::vector<void*> dev(kNumBuffers, nullptr);
    std::vector<CacaoOpInput> inputs;
    for (int i = 0; i < kNumBuffers; i++) {
        const size_t bytes = kBuffers[i].elems * dtype_size(kBuffers[i].dtype);
        if (kBuffers[i].access == 1) {  // write-only
            host[i].assign(bytes, 0);
        } else {
            host[i] = read_input_file(in_dir + "/cacao_in_" + kBuffers[i].name + ".bin", bytes);
        }
        CUDA_OK(cudaMalloc(&dev[i], bytes));
        CUDA_OK(cudaMemcpy(dev[i], host[i].data(), bytes, cudaMemcpyHostToDevice));
        if (kBuffers[i].access != 1) {
            inputs.push_back(CacaoOpInput{kBuffers[i].name, kBuffers[i].dtype,
                                          kBuffers[i].elems, host[i].data()});
        }
    }

    // ---- 1. reference kernel (trusted oracle) -> bench_ref_<valid>.bin ----
    unsigned gx, gy, gz;
    const unsigned bx = kRefBlock[0], by = kRefBlock[1], bz = kRefBlock[2];
    if (kGlobalSizeIsOpenCL) {  // KTT OpenCL style: grid exprs = total thread counts
        if (kRefGrid[0] % bx || kRefGrid[1] % by || kRefGrid[2] % bz) {
            fprintf(stderr, "opencl-style grid %zu,%zu,%zu not divisible by block %u,%u,%u\n",
                    kRefGrid[0], kRefGrid[1], kRefGrid[2], bx, by, bz);
            return 2;
        }
        gx = (unsigned)(kRefGrid[0] / bx);
        gy = (unsigned)(kRefGrid[1] / by);
        gz = (unsigned)(kRefGrid[2] / bz);
    } else {  // KTT CUDA style: grid exprs are already block counts
        gx = (unsigned)kRefGrid[0];
        gy = (unsigned)kRefGrid[1];
        gz = (unsigned)kRefGrid[2];
    }
    cacao_launch_ref(dev.data(), gx, gy, gz, bx, by, bz);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    {
        std::vector<char> out(valid_bytes);
        CUDA_OK(cudaMemcpy(out.data(), dev[kValidatedIndex], valid_bytes, cudaMemcpyDeviceToHost));
        write_file(out_dir + "/bench_ref_" + valid.name + ".bin", out.data(), valid_bytes);
    }

    // ---- 2. op: setup (untimed), sanity run, gate output ----
    std::vector<CacaoOpScalar> scalars;
    for (int i = 0; i < kNumScalars; i++)
        scalars.push_back(CacaoOpScalar{kScalars[i].name, kScalars[i].is_float, kScalars[i].value});
    const CacaoOpSpec spec{inputs.data(), (int)inputs.size(), scalars.data(), (int)scalars.size(),
                           valid.name, valid.dtype, valid.elems};
    op_setup(spec);
    CUDA_OK(cudaGetLastError());
    op_run();
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    {
        std::vector<char> out(valid_bytes);
        op_output(out.data());
        write_file(out_dir + "/bench_op_" + valid.name + ".bin", out.data(), valid_bytes);
    }

    // ---- 3. warmup + per-rep CUDA-event timing (min wins) ----
    for (int i = 0; i < warmup; i++)
        op_run();
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    std::vector<float> ms(reps);
    for (int i = 0; i < reps; i++) {
        CUDA_OK(cudaEventRecord(start));
        op_run();
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        CUDA_OK(cudaEventElapsedTime(&ms[i], start, stop));
    }
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));

    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const float min_us = sorted.front() * 1000.0f;
    const float median_us = sorted[sorted.size() / 2] * 1000.0f;
    printf("CACAO_REF_TIME_US=%.3f\n", min_us);
    fprintf(stderr, "CACAO_BENCH_MEDIAN_US=%.3f (min over %d reps after %d warmup)\n",
            median_us, reps, warmup);

    op_teardown();
    for (void* p : dev)
        cudaFree(p);
    return 0;
}
