// ops/cudnn_conv2d.cu — cuDNN v9 graph-API conv2d (NCHW fp32, zero-padding
// cross-correlation) for the conv2d_cudnn problem.
//
// cuDNN v9 removed the legacy cudnnConvolutionForward API, so this goes
// through cudnn_frontend's Graph. op_setup builds plans for EVERY engine the
// heuristics return (BuildPlanPolicy_t::ALL), times them all on the real
// inputs, and keeps the winner — "library at its best" is exact.
//
// Written against cudnn_frontend v1.x for cuDNN 9.x (build_plans /
// get_execution_plan_count / execute_plan_at_index). Headers + libcudnn are a
// user-managed prerequisite (see tools/library_bench/README.md): if your
// frontend version renamed a method, the nvcc error points right at it —
// this file is the only cudnn-specific code in the tool.
//
// fp32 compute only (set_compute_data_type FLOAT; TF32 engines are not
// considered) so the comparison is fp32 vs fp32.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include "op_contract.h"

namespace fe = cudnn_frontend;

#define CUDA_OK(call)                                                                \
    do {                                                                             \
        cudaError_t e_ = (call);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(e_),       \
                    __FILE__, __LINE__);                                              \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

#define CUDNN_OK(call)                                                               \
    do {                                                                             \
        cudnnStatus_t s_ = (call);                                                    \
        if (s_ != CUDNN_STATUS_SUCCESS) {                                             \
            fprintf(stderr, "cudnn error %s at %s:%d\n", cudnnGetErrorString(s_),     \
                    __FILE__, __LINE__);                                              \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

// A plan that executed fine at sweep time turning bad mid-timing is a hard
// error: the harness would otherwise time an empty stream as ~0 us.
#define FE_OK(call)                                                                  \
    do {                                                                             \
        auto status_ = (call);                                                        \
        if (status_.is_bad()) {                                                       \
            fprintf(stderr, "cudnn_frontend error %s at %s:%d\n",                     \
                    status_.get_message().c_str(), __FILE__, __LINE__);               \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

static const CacaoOpInput* find_input(const CacaoOpSpec& spec, const char* name) {
    for (int i = 0; i < spec.num_inputs; i++)
        if (std::strcmp(spec.inputs[i].name, name) == 0)
            return &spec.inputs[i];
    fprintf(stderr, "cudnn_conv2d: input %s not found\n", name);
    exit(2);
}

static long find_scalar(const CacaoOpSpec& spec, const char* name) {
    for (int i = 0; i < spec.num_scalars; i++)
        if (std::strcmp(spec.scalars[i].name, name) == 0)
            return (long)spec.scalars[i].value;
    fprintf(stderr, "cudnn_conv2d: scalar %s not found\n", name);
    exit(2);
}

static cudnnHandle_t g_handle;
static std::shared_ptr<fe::graph::Graph> g_graph;
static std::unordered_map<int64_t, void*> g_pack;
static std::vector<void*> g_plan_ws;      // per-plan workspace (empty if none needed)
static int g_best_plan = -1;
static void* g_x, *g_w, *g_y;
static long g_out_elems;

static float time_plan(int idx, void* ws) {
    // one warmup, then 5 timed reps; returns min
    cudaEvent_t ev0, ev1;
    CUDA_OK(cudaEventCreate(&ev0));
    CUDA_OK(cudaEventCreate(&ev1));
    std::vector<float> ms(5);
    if (g_graph->execute_plan_at_index(g_handle, g_pack, ws, idx).is_bad()) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        return 1e30f;  // plan not runnable on this device — out of the race
    }
    CUDA_OK(cudaDeviceSynchronize());
    for (float& m : ms) {
        CUDA_OK(cudaEventRecord(ev0));
        if (g_graph->execute_plan_at_index(g_handle, g_pack, ws, idx).is_bad()) {
            cudaEventRecord(ev1);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
            return 1e30f;  // plan turned bad mid-sweep — out of the race
        }
        CUDA_OK(cudaEventRecord(ev1));
        CUDA_OK(cudaEventSynchronize(ev1));
        CUDA_OK(cudaEventElapsedTime(&m, ev0, ev1));
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    float best = ms[0];
    for (float m : ms) best = m < best ? m : best;
    return best;
}

void op_setup(const CacaoOpSpec& spec) {
    const long n = find_scalar(spec, "CONV_N"), c = find_scalar(spec, "CONV_C");
    const long h = find_scalar(spec, "CONV_H"), w = find_scalar(spec, "CONV_W");
    const long k = find_scalar(spec, "CONV_K"), r = find_scalar(spec, "CONV_R");
    const long s = find_scalar(spec, "CONV_S");
    const long stride = find_scalar(spec, "CONV_STRIDE"), pad = find_scalar(spec, "CONV_PAD");
    const long ho = find_scalar(spec, "CONV_HO"), wo = find_scalar(spec, "CONV_WO");
    g_out_elems = n * k * ho * wo;

    const size_t x_bytes = find_input(spec, "input")->elems * sizeof(float);
    const size_t w_bytes = find_input(spec, "filter")->elems * sizeof(float);
    CUDA_OK(cudaMalloc(&g_x, x_bytes));
    CUDA_OK(cudaMalloc(&g_w, w_bytes));
    CUDA_OK(cudaMalloc(&g_y, (size_t)g_out_elems * sizeof(float)));
    CUDA_OK(cudaMemcpy(g_x, find_input(spec, "input")->data, x_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_w, find_input(spec, "filter")->data, w_bytes, cudaMemcpyHostToDevice));

    CUDNN_OK(cudnnCreate(&g_handle));

    g_graph = std::make_shared<fe::graph::Graph>();
    g_graph->set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = g_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_uid(1).set_data_type(fe::DataType_t::FLOAT)
                                 .set_dim({n, c, h, w})
                                 .set_stride({c * h * w, h * w, w, 1}));
    auto W = g_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_uid(2).set_data_type(fe::DataType_t::FLOAT)
                                 .set_dim({k, c, r, s})
                                 .set_stride({c * r * s, r * s, s, 1}));
    auto conv_attr = fe::graph::Conv_fprop_attributes()
                         .set_padding({pad, pad}).set_stride({stride, stride}).set_dilation({1, 1});
    auto Y = g_graph->conv_fprop(X, W, conv_attr);
    Y->set_uid(3).set_output(true).set_data_type(fe::DataType_t::FLOAT)
        .set_dim({n, k, ho, wo}).set_stride({k * ho * wo, ho * wo, wo, 1});
    g_pack = {{1, g_x}, {2, g_w}, {3, g_y}};

    fe::error_t status;
    const auto fail_at = [](const char* stage, fe::error_t& e) {
        fprintf(stderr, "cudnn graph build failed at %s: %s\n", stage,
                e.get_message().c_str());
        exit(2);
    };
    if ((status = g_graph->validate()).is_bad())
        fail_at("validate", status);
    if ((status = g_graph->build_operation_graph(g_handle)).is_bad())
        fail_at("build_operation_graph", status);
    if ((status = g_graph->create_execution_plans({fe::HeurMode_t::A})).is_bad())
        fail_at("create_execution_plans", status);
    if ((status = g_graph->check_support(g_handle)).is_bad())
        fail_at("check_support", status);
    if ((status = g_graph->build_plans(g_handle, fe::BuildPlanPolicy_t::ALL)).is_bad())
        fail_at("build_plans", status);

    const int plans = (int)g_graph->get_execution_plan_count();
    fprintf(stderr, "cuDNN: %d engine plan(s) from heuristics A\n", plans);
    float best = 1e30f;
    for (int i = 0; i < plans; i++) {
        const size_t ws_bytes = g_graph->get_workspace_size_plan_at_index(i);
        void* ws = nullptr;
        if (ws_bytes)
            CUDA_OK(cudaMalloc(&ws, ws_bytes));
        g_plan_ws.push_back(ws);
        const float ms = time_plan(i, ws);
        fprintf(stderr, "cuDNN plan %d: %.1f us\n", i, ms * 1000.0f);
        if (ms < best) {
            best = ms;
            g_best_plan = i;
        }
    }
    if (g_best_plan < 0) {
        fprintf(stderr, "cuDNN: no runnable engine plan\n");
        exit(2);
    }
    fprintf(stderr, "CACAO_BENCH_INFO=engine_plan=%d/%d min5_us=%.1f\n", g_best_plan,
            plans, best * 1000.0f);
}

void op_run() {
    FE_OK(g_graph->execute_plan_at_index(g_handle, g_pack, g_plan_ws[g_best_plan],
                                         g_best_plan));
}

void op_output(void* dst) {
    CUDA_OK(cudaMemcpy(dst, g_y, (size_t)g_out_elems * sizeof(float), cudaMemcpyDeviceToHost));
}

void op_teardown() {
    for (void* ws : g_plan_ws)
        if (ws)
            cudaFree(ws);
    cudaFree(g_x);
    cudaFree(g_w);
    cudaFree(g_y);
    g_graph.reset();
    cudnnDestroy(g_handle);
}
