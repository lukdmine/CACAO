// ops/cusparse_spmm.cu — cuSPARSE SpMM (CSR x dense, fp32) for the
// cusparse_spmm-shaped problem contract: inputs = values/col_indices/row_ptr/b;
// output c = a x b, row-major.
//
// op_setup sweeps every CSR algorithm cuSPARSE offers (untimed, measured with
// events on a side stream) and keeps the winner — "library at its best" is
// exact, not guessed. The harness's 100-rep min then times only that winner.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    fprintf(stderr, "cusparse_spmm: input %s not found\n", name);
    exit(2);
}

static long find_scalar(const CacaoOpSpec& spec, const char* name) {
    for (int i = 0; i < spec.num_scalars; i++)
        if (std::strcmp(spec.scalars[i].name, name) == 0)
            return (long)spec.scalars[i].value;
    fprintf(stderr, "cusparse_spmm: scalar %s not found\n", name);
    exit(2);
}

static cusparseHandle_t g_handle;
static void *g_values, *g_colidx, *g_rowptr, *g_b, *g_c, *g_ws;
static cusparseSpMatDescr_t g_a;
static cusparseDnMatDescr_t g_bmat, g_cmat;
static cusparseSpMMAlg_t g_winner;
static long g_out_elems;

static const struct { cusparseSpMMAlg_t alg; const char* name; } kAlgs[] = {
    {CUSPARSE_SPMM_ALG_DEFAULT, "ALG_DEFAULT"},
    {CUSPARSE_SPMM_CSR_ALG1, "SPMM_CSR_ALG1"},
    {CUSPARSE_SPMM_CSR_ALG2, "SPMM_CSR_ALG2"},
    {CUSPARSE_SPMM_CSR_ALG3, "SPMM_CSR_ALG3"},
};

void op_setup(const CacaoOpSpec& spec) {
    const long m = find_scalar(spec, "SPMM_M");
    const long k = find_scalar(spec, "SPMM_K");
    const long n = find_scalar(spec, "SPMM_N");
    const long nnz = find_scalar(spec, "SPMM_NNZ");
    g_out_elems = m * n;

    const size_t val_b = find_input(spec, "values")->elems * sizeof(float);
    const size_t idx_b = find_input(spec, "col_indices")->elems * sizeof(int);
    const size_t ptr_b = find_input(spec, "row_ptr")->elems * sizeof(int);
    const size_t b_b = find_input(spec, "b")->elems * sizeof(float);
    CUDA_OK(cudaMalloc(&g_values, val_b));
    CUDA_OK(cudaMalloc(&g_colidx, idx_b));
    CUDA_OK(cudaMalloc(&g_rowptr, ptr_b));
    CUDA_OK(cudaMalloc(&g_b, b_b));
    CUDA_OK(cudaMalloc(&g_c, (size_t)g_out_elems * sizeof(float)));
    CUDA_OK(cudaMemcpy(g_values, find_input(spec, "values")->data, val_b, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_colidx, find_input(spec, "col_indices")->data, idx_b, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_rowptr, find_input(spec, "row_ptr")->data, ptr_b, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_b, find_input(spec, "b")->data, b_b, cudaMemcpyHostToDevice));

    CUSPARSE_OK(cusparseCreate(&g_handle));
    CUSPARSE_OK(cusparseCreateCsr(&g_a, m, k, nnz, g_rowptr, g_colidx, g_values,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_OK(cusparseCreateDnMat(&g_bmat, k, n, n, g_b, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_OK(cusparseCreateDnMat(&g_cmat, m, n, n, g_c, CUDA_R_32F, CUSPARSE_ORDER_ROW));

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
        CUSPARSE_OK(cusparseSpMM_bufferSize(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                            g_a, g_bmat, &beta, g_cmat, CUDA_R_32F,
                                            kAlgs[i].alg, &ws_bytes));
        void* ws = nullptr;
        if (ws_bytes)
            CUDA_OK(cudaMalloc(&ws, ws_bytes));
        for (int r = 0; r < 3; r++)
            CUSPARSE_OK(cusparseSpMM(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, g_a, g_bmat,
                                     &beta, g_cmat, CUDA_R_32F, kAlgs[i].alg, ws));
        cudaEventRecord(ev0);
        for (int r = 0; r < 10; r++)
            CUSPARSE_OK(cusparseSpMM(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, g_a, g_bmat,
                                     &beta, g_cmat, CUDA_R_32F, kAlgs[i].alg, ws));
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
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    g_winner = kAlgs[best_i].alg;
    fprintf(stderr, "CACAO_BENCH_INFO=algo=%s avg10_us=%.1f\n", kAlgs[best_i].name,
            best_ms * 100.0f);
}

void op_run() {
    const float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_OK(cusparseSpMM(g_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, g_a, g_bmat,
                             &beta, g_cmat, CUDA_R_32F, g_winner, g_ws));
}

void op_output(void* dst) {
    CUDA_OK(cudaMemcpy(dst, g_c, (size_t)g_out_elems * sizeof(float), cudaMemcpyDeviceToHost));
}

void op_teardown() {
    cudaFree(g_values);
    cudaFree(g_colidx);
    cudaFree(g_rowptr);
    cudaFree(g_b);
    cudaFree(g_c);
    if (g_ws)
        cudaFree(g_ws);
    cusparseDestroySpMat(g_a);
    cusparseDestroyDnMat(g_bmat);
    cusparseDestroyDnMat(g_cmat);
    cusparseDestroy(g_handle);
}
