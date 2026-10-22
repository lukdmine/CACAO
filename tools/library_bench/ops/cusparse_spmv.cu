// ops/cusparse_spmv.cu — cuSPARSE SpMV (CSR x dense vector, fp32) for the
// cusparse_spmv-shaped problem contract: inputs = values/col_indices/row_ptr/x;
// output y = a x x.
//
// op_setup sweeps every CSR SpMV algorithm cuSPARSE offers (untimed, measured
// with events on the default stream) and keeps the winner — "library at its
// best" is exact, not guessed. The harness's timed reps then measure only
// that winner.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "op_contract.h"

#define CUSPARSE_OK(call)                                                            \
    do {                                                                             \
        cusparseStatus_t s_ = (call);                                                 \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "cusparse error %s at %s:%d\n",                           \
                    cusparseGetErrorString(s_), __FILE__, __LINE__);                  \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

#define CUDA_OK(call)                                                                \
    do {                                                                             \
        cudaError_t e_ = (call);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(e_),       \
                    __FILE__, __LINE__);                                              \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

static const CacaoOpInput* find_input(const CacaoOpSpec& spec, const char* name) {
    for (int i = 0; i < spec.num_inputs; i++)
        if (std::strcmp(spec.inputs[i].name, name) == 0)
            return &spec.inputs[i];
    fprintf(stderr, "cusparse_spmv: input %s not found\n", name);
    exit(2);
}

static long find_scalar(const CacaoOpSpec& spec, const char* name) {
    for (int i = 0; i < spec.num_scalars; i++)
        if (std::strcmp(spec.scalars[i].name, name) == 0)
            return (long)spec.scalars[i].value;
    fprintf(stderr, "cusparse_spmv: scalar %s not found\n", name);
    exit(2);
}

static cusparseHandle_t g_handle;
static void *g_values, *g_colidx, *g_rowptr, *g_x, *g_y, *g_ws;
static cusparseSpMatDescr_t g_a;
static cusparseDnVecDescr_t g_xvec, g_yvec;
static cusparseSpMVAlg_t g_winner;
static long g_out_elems;

static const struct { cusparseSpMVAlg_t alg; const char* name; } kAlgs[] = {
    {CUSPARSE_SPMV_ALG_DEFAULT, "ALG_DEFAULT"},
    {CUSPARSE_SPMV_CSR_ALG1, "SPMV_CSR_ALG1"},
    {CUSPARSE_SPMV_CSR_ALG2, "SPMV_CSR_ALG2"},
};

void op_setup(const CacaoOpSpec& spec) {
    const long m = find_scalar(spec, "SPMV_M");
    const long nnz = find_scalar(spec, "SPMV_NNZ");
    g_out_elems = m;

    const size_t val_b = find_input(spec, "values")->elems * sizeof(float);
    const size_t idx_b = find_input(spec, "col_indices")->elems * sizeof(int);
    const size_t ptr_b = find_input(spec, "row_ptr")->elems * sizeof(int);
    const size_t x_b = find_input(spec, "x")->elems * sizeof(float);
    CUDA_OK(cudaMalloc(&g_values, val_b));
    CUDA_OK(cudaMalloc(&g_colidx, idx_b));
    CUDA_OK(cudaMalloc(&g_rowptr, ptr_b));
    CUDA_OK(cudaMalloc(&g_x, x_b));
    CUDA_OK(cudaMalloc(&g_y, (size_t)m * sizeof(float)));
    CUDA_OK(cudaMemcpy(g_values, find_input(spec, "values")->data, val_b, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_colidx, find_input(spec, "col_indices")->data, idx_b, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_rowptr, find_input(spec, "row_ptr")->data, ptr_b, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_x, find_input(spec, "x")->data, x_b, cudaMemcpyHostToDevice));

    CUSPARSE_OK(cusparseCreate(&g_handle));
    CUSPARSE_OK(cusparseCreateCsr(&g_a, m, m, nnz, g_rowptr, g_colidx, g_values,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_OK(cusparseCreateDnVec(&g_xvec, m, g_x, CUDA_R_32F));
    CUSPARSE_OK(cusparseCreateDnVec(&g_yvec, m, g_y, CUDA_R_32F));

    const float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    float best_ms = 1e30f;
    int best_i = -1;

    // Untimed algorithm sweep: 3 warmup + 10 timed reps per algorithm, min wins.
    // The results legitimately depend on the data distribution, so this runs on
    // the REAL inputs, after H2D — not on a synthetic guess. Per-rep time is
    // reported (elapsed / 10).
    for (size_t i = 0; i < sizeof(kAlgs) / sizeof(kAlgs[0]); i++) {
        size_t ws_bytes = 0;
        CUSPARSE_OK(cusparseSpMV_bufferSize(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, g_a, g_xvec, &beta, g_yvec,
                                            CUDA_R_32F, kAlgs[i].alg, &ws_bytes));
        void* ws = nullptr;
        if (ws_bytes)
            CUDA_OK(cudaMalloc(&ws, ws_bytes));
        for (int r = 0; r < 3; r++)
            CUSPARSE_OK(cusparseSpMV(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                     g_a, g_xvec, &beta, g_yvec, CUDA_R_32F,
                                     kAlgs[i].alg, ws));
        cudaEventRecord(ev0);
        for (int r = 0; r < 10; r++)
            CUSPARSE_OK(cusparseSpMV(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                     g_a, g_xvec, &beta, g_yvec, CUDA_R_32F,
                                     kAlgs[i].alg, ws));
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        if (ms < best_ms) {
            if (g_ws)
                cudaFree(g_ws);  // evict the previous best's workspace
            g_ws = ws;
            best_ms = ms;
            best_i = (int)i;
        } else if (ws) {
            cudaFree(ws);
        }
        fprintf(stderr, "CUSPARSE sweep: %-18s %.1f us/rep (avg of 10)\n", kAlgs[i].name,
                ms * 100.0f);
    }
    if (best_i < 0) {
        fprintf(stderr, "cusparse_spmv: no algorithm succeeded\n");
        exit(2);
    }
    g_winner = kAlgs[best_i].alg;
    fprintf(stderr, "CACAO_BENCH_INFO=algo=%s avg10_us=%.1f\n", kAlgs[best_i].name,
            best_ms * 100.0f);
}

void op_run() {
    const float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_OK(cusparseSpMV(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             g_a, g_xvec, &beta, g_yvec, CUDA_R_32F, g_winner, g_ws));
}

void op_output(void* dst) {
    CUDA_OK(cudaMemcpy(dst, g_y, (size_t)g_out_elems * sizeof(float), cudaMemcpyDeviceToHost));
}

void op_teardown() {
    if (g_ws) cudaFree(g_ws);
    cusparseDestroySpMat(g_a);
    cusparseDestroyDnVec(g_xvec);
    cusparseDestroyDnVec(g_yvec);
    cusparseDestroy(g_handle);
    cudaFree(g_values); cudaFree(g_colidx); cudaFree(g_rowptr);
    cudaFree(g_x); cudaFree(g_y);
}
