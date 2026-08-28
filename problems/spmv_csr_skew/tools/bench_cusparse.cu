// Measure the incumbent: cuSPARSE CSR SpMV on the binaries fetch_matrices.py wrote.
//
//     make -C problems/spmv_csr_skew/tools
//     problems/spmv_csr_skew/tools/bench_cusparse            # every matrix present
//     problems/spmv_csr_skew/tools/bench_cusparse mawi       # just one
//
// The call shape is the one the thesis benchmarked against (eval/cusparse_spmv.cu):
// cusparseSpMV with CUSPARSE_SPMV_ALG_DEFAULT, the workspace allocated OUTSIDE the timed
// region, timed with CUDA events. cusparseSpMV_preprocess is deliberately not called --
// the comparison is single-launch SpMV on raw CSR with nothing precomputed, which is the
// rule the tuned kernel plays by too.
//
// It also recomputes y on the CPU and checks it exactly. That is a smoke test of the
// binaries themselves: the inputs are built so fp32 SpMV is order-independent, so any
// nonzero difference here means the .bin files are wrong, not that the GPU rounded.

#include <cusparse.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_CUDA(f)  do { cudaError_t e = (f); if (e != cudaSuccess) {                 \
        std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                     \
                     cudaGetErrorString(e)); return 1; } } while (0)

#define CHECK_SP(f)    do { cusparseStatus_t s = (f); if (s != CUSPARSE_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuSPARSE %s:%d: %s\n", __FILE__, __LINE__,                  \
                     cusparseGetErrorString(s)); return 1; } } while (0)

namespace {

const char* kNames[] = {"transient", "rail4284", "mawi", "GL7d19", "kron_g500"};

template <typename T>
std::vector<T> read_bin(const std::string& path, size_t count)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t bytes = static_cast<size_t>(f.tellg());
    if (bytes != count * sizeof(T))
        throw std::runtime_error(path + ": holds " + std::to_string(bytes) +
                                 " bytes, expected " + std::to_string(count * sizeof(T)));
    std::vector<T> v(count);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}

// Reads one integer field out of the flat meta.json fetch_matrices.py writes.
long meta_field(const std::string& text, const char* key)
{
    const std::string pattern = std::string("\"") + key + "\":";
    const size_t at = text.find(pattern);
    if (at == std::string::npos)
        throw std::runtime_error(std::string("meta.json has no ") + key);
    return std::strtol(text.c_str() + at + pattern.size(), nullptr, 10);
}

// MUST match gen_x in the generated inputs.hpp, or the timing is of a different problem.
std::vector<float> make_x(long ncols, long xmod)
{
    std::vector<float> v(static_cast<size_t>(ncols));
    for (size_t i = 0; i < v.size(); ++i) {
        std::uint64_t h = static_cast<std::uint64_t>(i) + 0x9E3779B97F4A7C15ull;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
        h ^= h >> 31;
        v[i] = static_cast<float>(1u + static_cast<unsigned>(
                   h % static_cast<std::uint64_t>(xmod)));
    }
    return v;
}

cusparseSpMVAlg_t kAlg = CUSPARSE_SPMV_ALG_DEFAULT;
const char* kAlgName = "ALG_DEFAULT";

int run(const std::string& dir, const std::string& name, int reps)
{
    std::ifstream mf(dir + "/" + name + ".meta.json");
    if (!mf) return 2;  // not fetched; skip quietly
    const std::string meta((std::istreambuf_iterator<char>(mf)),
                            std::istreambuf_iterator<char>());

    const long nrows = meta_field(meta, "NROWS");
    const long ncols = meta_field(meta, "NCOLS");
    const long nnz   = meta_field(meta, "NNZ");
    const long xmod  = meta_field(meta, "XMOD");

    auto h_rowptr = read_bin<int>  (dir + "/" + name + ".rowptr.bin", nrows + 1);
    auto h_col    = read_bin<int>  (dir + "/" + name + ".col.bin",    nnz);
    auto h_val    = read_bin<float>(dir + "/" + name + ".val.bin",    nnz);
    auto h_x      = make_x(ncols, xmod);

    int *d_rowptr = nullptr, *d_col = nullptr;
    float *d_val = nullptr, *d_x = nullptr, *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_rowptr, sizeof(int)   * (nrows + 1)));
    CHECK_CUDA(cudaMalloc(&d_col,    sizeof(int)   * nnz));
    CHECK_CUDA(cudaMalloc(&d_val,    sizeof(float) * nnz));
    CHECK_CUDA(cudaMalloc(&d_x,      sizeof(float) * ncols));
    CHECK_CUDA(cudaMalloc(&d_y,      sizeof(float) * nrows));
    CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowptr.data(), sizeof(int) * (nrows + 1),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col, h_col.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_val.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), sizeof(float) * ncols, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, sizeof(float) * nrows));

    cusparseHandle_t handle = nullptr;
    CHECK_SP(cusparseCreate(&handle));

    cusparseSpMatDescr_t A;
    cusparseDnVecDescr_t X, Y;
    CHECK_SP(cusparseCreateCsr(&A, nrows, ncols, nnz, d_rowptr, d_col, d_val,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_SP(cusparseCreateDnVec(&X, ncols, d_x, CUDA_R_32F));
    CHECK_SP(cusparseCreateDnVec(&Y, nrows, d_y, CUDA_R_32F));

    const float alpha = 1.0f, beta = 0.0f;
    const cusparseSpMVAlg_t alg = kAlg;
    size_t bufBytes = 0;
    CHECK_SP(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A,
                                     X, &beta, Y, CUDA_R_32F, alg, &bufBytes));
    void* buf = nullptr;
    if (bufBytes) CHECK_CUDA(cudaMalloc(&buf, bufBytes));

    // warm-up (also produces the y we verify)
    CHECK_SP(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, X, &beta,
                          Y, CUDA_R_32F, alg, buf));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_y(static_cast<size_t>(nrows));
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, sizeof(float) * nrows, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    double worst = 0.0;
    for (long r = 0; r < nrows; ++r) {
        float sum = 0.0f;
        for (int j = h_rowptr[r]; j < h_rowptr[r + 1]; ++j)
            sum += h_val[j] * h_x[h_col[j]];
        const double d = std::abs(static_cast<double>(sum) - h_y[r]);
        if (d > worst) worst = d;
        if (d != 0.0) ++bad;
    }

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    std::vector<double> us;
    us.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CHECK_CUDA(cudaEventRecord(t0));
        CHECK_SP(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, X,
                              &beta, Y, CUDA_R_32F, alg, buf));
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        us.push_back(ms * 1000.0);
    }
    std::sort(us.begin(), us.end());

    const double traffic = 4.0 * (nrows + 1) + 8.0 * nnz + 4.0 * nrows + 4.0 * ncols;
    const double median = us[us.size() / 2];
    std::printf("%-12s %-12s rows=%-10ld nnz=%-11ld  min %9.2f us  median %9.2f us"
                "  -> %6.1f GB/s of %.1f MiB compulsory\n",
                name.c_str(), kAlgName, nrows, nnz, us.front(), median,
                traffic / (median * 1e-6) / 1e9, traffic / 1048576.0);
    if (bad)
        std::printf("             !! %zu of %ld rows differ from the CPU result "
                    "(worst %.3g) -- the .bin files are wrong\n", bad, nrows, worst);

    cudaFree(buf); cudaFree(d_y); cudaFree(d_x); cudaFree(d_val);
    cudaFree(d_col); cudaFree(d_rowptr);
    cusparseDestroyDnVec(Y); cusparseDestroyDnVec(X); cusparseDestroySpMat(A);
    cusparseDestroy(handle);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string dir = std::string(argv[0]).substr(0, std::string(argv[0]).rfind('/'))
                          + "/../inputs";
    const int reps = 30;

    std::vector<std::string> names;
    for (int i = 1; i < argc; ++i) names.emplace_back(argv[i]);
    if (names.empty()) for (const char* n : kNames) names.emplace_back(n);

    struct { cusparseSpMVAlg_t a; const char* n; } algs[] = {
        {CUSPARSE_SPMV_ALG_DEFAULT, "ALG_DEFAULT"},
        {CUSPARSE_SPMV_CSR_ALG1,    "CSR_ALG1"},
        {CUSPARSE_SPMV_CSR_ALG2,    "CSR_ALG2"},
    };
    std::printf("cuSPARSE CSR SpMV, %d reps, workspace allocated outside the timed "
                "region\n\n", reps);
    int failures = 0;
    for (const auto& n : names) {
        for (const auto& g : algs) {
            kAlg = g.a; kAlgName = g.n;
            try {
                const int rc = run(dir, n, reps);
                if (rc == 2) { std::printf("%-12s (not fetched)\n", n.c_str()); break; }
                else if (rc)  ++failures;
            } catch (const std::exception& e) {
                std::printf("%-12s %-12s ERROR: %s\n", n.c_str(), g.n, e.what());
                ++failures;
            }
        }
    }
    std::printf("\nPaste the medians into problems/spmv_csr_skew/output/"
                "reference_time.json:\n"
                "  the first case in problem.yaml is the flat \"reference_time_us\" key;\n"
                "  every other case goes under \"cases\" by name.\n");
    return failures ? 1 : 0;
}
